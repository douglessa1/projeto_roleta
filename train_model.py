import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import joblib
import os
from threading import Lock
import xgboost as xgb
import database as db
import analysis
import warnings

warnings.filterwarnings("ignore")

# --- PARÂMETROS ---
MIN_RODADAS = 800
MODEL_DIR = "ml_models"
MODEL_FILENAME = 'multiclass_roulette_model.pkl'
SCALER_FILENAME = 'multiclass_scaler.pkl'
FEATURES_ORDER_FILENAME = 'feature_order.pkl'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)
FEATURES_ORDER_PATH = os.path.join(MODEL_DIR, FEATURES_ORDER_FILENAME)


def detect_and_drop_perfect_leaks(X: pd.DataFrame, y: pd.Series, verbose: bool = True):
    """
    Detecta colunas em X que são exatamente iguais ao target y (vazamento perfeito)
    ou que reproduzem o alvo de forma íntegra e as remove.
    Retorna (X_clean, dropped_columns_list)
    """
    dropped = []
    # alinhar índices para segurança
    Xc = X.copy().reset_index(drop=True)
    yc = y.copy().reset_index(drop=True)

    for col in Xc.columns:
        col_series = Xc[col]
        # compara igualdade direta (com preenchimento para NaN)
        if col_series.fillna(-9999).equals(yc.fillna(-9999)):
            dropped.append(col)
            continue

        # para colunas numéricas inteiras ou booleanas, testar igualdade após cast
        try:
            if pd.api.types.is_integer_dtype(col_series) or pd.api.types.is_bool_dtype(col_series):
                if (col_series.fillna(-9999).astype(int) == yc.fillna(-9999).astype(int)).all():
                    dropped.append(col)
                    continue
        except Exception:
            pass

    if dropped and verbose:
        print(f"⚠️  [TREINO] Detectadas colunas com vazamento perfeito e removidas: {dropped}")

    Xc.drop(columns=dropped, inplace=True, errors='ignore')
    return Xc, dropped


def train_and_save_model(lock: Lock = None):
    """
    Busca o histórico, gera features, treina um modelo multiclasse (0-36)
    e salva o modelo treinado e o normalizador, COM ALINHAMENTO ROBUSTO.
    """

    # 1. Obter Histórico Bruto do Supabase
    print("1. [TREINO] Obtendo histórico bruto...")
    try:
        client = db.get_supabase_client()
        history_list = db.get_raw_history(client, limit=2000)
    except Exception as e:
        print(f"❌ [TREINO] Erro ao conectar ao DB/buscar histórico: {e}")
        return

    if len(history_list) < MIN_RODADAS:
        print(f"❌ [TREINO] Histórico Insuficiente: {len(history_list)}/{MIN_RODADAS}.")
        return

    # 2. Gerar Features Avançadas
    print("2. [TREINO] Gerando features (filtros) do histórico...")
    # full_features_df contém a coluna 'number' (resultado N_t) e features F(N_t, N_t-1, ...)
    full_features_df = analysis.gerar_features_avancadas(history_list)

    if full_features_df is None or full_features_df.empty:
        print("❌ [TREINO] Falha ao gerar features (DataFrame vazio).")
        return

    # 3. Preparação Rigorosa de Dados para o ML (ALINHAMENTO FINAL)
    print("3. [TREINO] Preparando dados para o modelo (prevenindo vazamento)...")

    # Target (Y) é o resultado que queremos prever: N_t
    y_raw = full_features_df['number'].reset_index(drop=True)

    # Features (X) são as colunas F(N_t, N_t-1, ...)
    X_raw = full_features_df.drop(columns=['number']).reset_index(drop=True)

    # -------------------------
    # CORREÇÃO ROBUSTA DE VAZAMENTO:
    # As features X_raw[i] foram calculadas USANDO o resultado N_t (potencialmente).
    # Precisamos usar X_raw[i-1] para prever Y_raw[i].
    # -------------------------

    # 1. Desloca as Features X_raw em 1 (para que X[i] = F(N_{i-1}, N_{i-2},...))
    X_shifted = X_raw.shift(1)

    # 2. Remove a primeira linha NaN e realinha (garantindo que X[0] preveja y[1])
    X = X_shifted.iloc[1:].reset_index(drop=True)
    y = y_raw.iloc[1:].reset_index(drop=True)

    # 3. Detecta e remove vazamentos perfeitos (colunas que reproduzem o target)
    X, dropped_cols = detect_and_drop_perfect_leaks(X, y)

    # 4. Limpeza e Tipagem
    X.fillna(0, inplace=True)
    y = y.astype(int)

    if X.empty:
        print("❌ [TREINO] Após remoção/shift o conjunto de features ficou vazio.")
        return

    feature_columns_in_order = X.columns.tolist()

    # 4. Treino com Holdout para avaliação realista
    print(f"4. [TREINO] Dividindo em treino/teste para avaliação (Amostras totais: {len(X)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Normalização baseada no conjunto de treino
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=37,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.8,
        subsample=0.8
    )

    print(f"5. [TREINO] Treinando Modelo XGBoost (treino: {len(X_train)}, teste: {len(X_test)})...")
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"❌ [TREINO] Falha no fit do modelo: {e}")
        return

    # Avaliação realista no conjunto de teste
    y_pred_test = model.predict(X_test_scaled)
    y_pred_proba_test = model.predict_proba(X_test_scaled)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    loss_test = log_loss(y_test, y_pred_proba_test)

    print(f"5. [TREINO] Avaliação (Holdout) -> Acurácia: {accuracy_test:.4f} | Log Loss: {loss_test:.4f}")

    # Alerta se ainda houver acurácia suspeitamente alta
    if accuracy_test > 0.5:
        print("⚠️  [TREINO] Acurácia no conjunto de teste está alta (>0.5). Verifique possíveis fontes adicionais de vazamento nas features geradas.")

    # 6. Re-treina em TODO o conjunto (treino + teste) para produção e salva o modelo final
    print("6. [TREINO] Re-treinando modelo em todo o conjunto e salvando artefatos finais...")
    # refit scaler e modelo em todo o X
    scaler_full = StandardScaler()
    scaler_full.fit(X)
    X_scaled_full = scaler_full.transform(X)

    model_full = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=37,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.8,
        subsample=0.8
    )

    try:
        model_full.fit(X_scaled_full, y)
    except Exception as e:
        print(f"❌ [TREINO] Falha no fit final do modelo: {e}")
        return

    # 7. Verificação pós-treino (cheque rápido)
    y_pred_full = model_full.predict(X_scaled_full)
    y_pred_proba_full = model_full.predict_proba(X_scaled_full)
    acc_full = accuracy_score(y, y_pred_full)
    loss_full = log_loss(y, y_pred_proba_full)

    print(f"7. [TREINO] Performance (treinado em todo o conjunto) -> Acurácia treinada: {acc_full:.4f} | Log Loss: {loss_full:.4f}")
    if acc_full > 0.95:
        print("⚠️  [TREINO] Acurácia extremamente alta no conjunto de treino completo. Isso pode indicar overfitting ou ainda existir vazamento nas features (investigue).")

    # 8. Salvamento
    print("8. [TREINO] Salvando arquivos do modelo...")

    class DummyLock:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_value, traceback): pass

    if lock is None:
        lock = DummyLock()

    with lock:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        joblib.dump(model_full, MODEL_PATH)
        joblib.dump(scaler_full, SCALER_PATH)
        joblib.dump(feature_columns_in_order, FEATURES_ORDER_PATH)

    print(f"✅ [TREINO] Modelo, Scaler e Ordem de Features salvos.")


if __name__ == '__main__':
    print("--- Executando Treinamento Manual ---")
    manual_lock = Lock()
    train_and_save_model(lock=manual_lock)
