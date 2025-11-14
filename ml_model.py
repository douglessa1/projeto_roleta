import pandas as pd
import joblib
import os
import numpy as np
from threading import Lock
import math

# Importa as constantes da Roleta (Garante a importação de todos os grupos)
try:
    from analysis import WHEEL_ORDER, TOTAL_NUMBERS, WHEEL_POSITIONS, get_physical_neighbors, CALLING_NUMBERS, DUZIA_1, DUZIA_2, DUZIA_3, COLUNA_1, COLUNA_2, COLUNA_3, VERMELHOS, PRETOS, BAIXOS, ALTOS, IMPARES, PARES
except ImportError:
    # Fallback simplificado para evitar ModuleNotFoundError
    print("WARNING: Falha na importação de constantes de analysis.py. Usando Fallback.")
    WHEEL_ORDER = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
    TOTAL_NUMBERS = 37 
    WHEEL_POSITIONS = {number: i for i, number in enumerate(WHEEL_ORDER)}
    def get_physical_neighbors(number: int, radius: int = 1) -> set: return {number}
    CALLING_NUMBERS = {0: {1, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 20, 21, 24, 25, 26, 29}}
    VERMELHOS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    PRETOS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
    BAIXOS = set(range(1, 19)); ALTOS = set(range(19, 37))
    IMPARES = set(range(1, 37, 2)); PARES = set(range(2, 37, 2))
    DUZIA_1 = set(range(1, 13)); DUZIA_2 = set(range(13, 25)); DUZIA_3 = set(range(25, 37))
    COLUNA_1 = set(range(1, 37, 3)); COLUNA_2 = set(range(2, 37, 3)); COLUNA_3 = set(range(3, 37, 3))


# --- CONFIGURAÇÕES DE PATHS E CONSTANTES ---
MODEL_DIR = "ml_models" 
MODEL_FILENAME = 'multiclass_roulette_model.pkl'
SCALER_FILENAME = 'multiclass_scaler.pkl'
FEATURES_ORDER_FILENAME = 'feature_order.pkl' 
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)
FEATURES_ORDER_PATH = os.path.join(MODEL_DIR, FEATURES_ORDER_FILENAME)


# --- REGRAS BSB DE PUXADA (Column Pulling) ---
PULLING_PATTERNS = {
    'COL1_PULLS_COL3_END': {
        'triggers': {1, 3}, 
        'targets': {34, 36},
        'min_score': 1,
        'motivo': 'Padrão Coluna 1 (Início) -> Puxada para Coluna 3 (Fim)'
    },
    'QUINA_PULLS_MIDDLE_BLACK': {
        'triggers': {1, 3, 34, 36},
        'targets': {2, 8, 17, 20, 26, 28, 35},
        'min_score': 2,
        'motivo': 'Quina Ativa -> Puxada para Pretos do Meio (Tendência Inversa)'
    }
}

# --- FUNÇÕES AUXILIARES DE CÁLCULO ---

def calculate_cluster_edge_and_odds(cluster_size: int, cluster_probability: float):
    P_teorica = cluster_size / 37.0
    odds = 36.0 / cluster_size 
    edge = cluster_probability - P_teorica
    return odds, edge

def calculate_kelly_stake(edge: float, odds: float, bankroll: float, cluster_size: int, kelly_fraction: float):
    # Lógica Kelly (Mantida)
    if edge <= 0: return 0.0
    P_model = edge + (cluster_size / 37.0) 
    b = odds - 1 
    q = 1 - P_model
    if b <= 0.001 or P_model <= 0.0: return 0.0
    kelly_percentage = (b * P_model - q) / b
    stake_percentage = max(0, kelly_percentage * kelly_fraction) 
    return bankroll * stake_percentage

# --- LÓGICA DE CONFLUÊNCIA HEURÍSTICA (BSB Aprimorada) ---
def calculate_heuristic_score(features: pd.Series, cluster_numbers: set, history_list_last_5: list) -> tuple[int, str]:
    # ... (Lógica de score mantida, agora o cluster_numbers é dinâmico)
    score = 0
    motivos = []
    alvo_encontrado_em_zona = set() 
    
    # 1. Padrões de Gatilho (BSB Tradicional)
    if features.get('was_called_by_previous', 0) > 0: 
        score += 2 
        motivos.append("Gatilho BSB (N-1 -> N)")
    if features.get('current_is_terminal_sum', 0) > 0:
        score += 1
        motivos.append("Padrão de Soma de Terminais")
        
    # --- NOVAS VALIDAÇÕES DE CORES E PARIDADE ---
    if features.get('is_red_atraso', 0) >= 6 and features.get('zscore_is_red', 0) < -1.5:
        score += 1; motivos.append("Tensão Vermelho (Atraso Alto + Z-Score Negativo)")
    if features.get('is_black_atraso', 0) >= 6 and features.get('zscore_is_black', 0) < -1.5:
        score += 1; motivos.append("Tensão Preto (Atraso Alto + Z-Score Negativo)")
    if features.get('is_red_seq_ocorrencia', 0) >= 4 and features.get('zscore_is_red', 0) > 1.5:
        score += 1; motivos.append("Momentum Vermelho (Sequência Alta)")
    if features.get('is_black_seq_ocorrencia', 0) >= 4 and features.get('zscore_is_black', 0) > 1.5:
        score += 1; motivos.append("Momentum Preto (Sequência Alta)")
    if features.get('is_odd_seq_ocorrencia', 0) >= 4:
        score += 1; motivos.append("Sequência Forte de Ímpares (Momentum)")
    if features.get('is_even_seq_ocorrencia', 0) >= 4:
        score += 1; motivos.append("Sequência Forte de Pares (Momentum)")
        
    # --- NOVO: VALIDAÇÃO DE PUXADA (Column Pulling) ---
    for pattern_name, pattern_data in PULLING_PATTERNS.items():
        if any(num in pattern_data['triggers'] for num in history_list_last_5):
            if any(num in pattern_data['targets'] for num in cluster_numbers):
                score += pattern_data['min_score']
                motivos.append(pattern_data['motivo'] + " (Cluster Alvo)")
                break 

    # 4. Validação de Histórico e Zonas (Mantendo checagens de zona)
    for num in cluster_numbers:
        atraso_num = features.get(f'is_num_{num}_atraso', 0)
        if atraso_num > 74: 
            if 'atraso_individual' not in alvo_encontrado_em_zona:
                score += 2
                motivos.append(f"Alvo {num} está a {atraso_num} spins (Atraso Alto)")
                alvo_encontrado_em_zona.add('atraso_individual')
        
        # Checagem de Dúzias (Exemplos mantidos)
        if num in DUZIA_1 and 'd1' not in alvo_encontrado_em_zona:
            if features.get('zscore_is_d1', 0) < -1.5:
                score += 1; motivos.append(f"Alvo {num} na Dúzia 1 (Atrasada)"); alvo_encontrado_em_zona.add('d1')
            elif features.get('zscore_is_d1', 0) > 1.5: 
                score += 1; motivos.append(f"Alvo {num} na Dúzia 1 (Quente)"); alvo_encontrado_em_zona.add('d1')
        # ... (Outras checagens de zona) ...

    final_score = score
    if not motivos:
        return 0, "Sem confluência de histórico."
        
    return final_score, ", ".join(motivos)
# --------------------------------------------------------

class TrainedMLModel:
    # ... (Classes e funções auxiliares mantidas) ...
    def __init__(self, lock: Lock):
        self.model = None
        self.scaler = None
        self.feature_columns_in_order = None 
        self.classes = None
        self.model_loaded = False
        self.lock = lock
        self._model_mtime = 0 
        self.load_model_files() 

    def reload_model(self):
        print("Recebida ordem de recarregar o modelo...")
        self.load_model_files()
        
    def load_model_files(self):
        print("Tentando carregar/recarregar funil ML (v6.5)...")
        with self.lock:
            if not os.path.exists(MODEL_PATH):
                print(f"⚠️ AVISO: Arquivo do modelo não encontrado: {MODEL_PATH}")
                self.model_loaded = False
                return
            if not os.path.exists(SCALER_PATH):
                print(f"⚠️ AVISO: Arquivo do scaler não encontrado: {SCALER_PATH}")
                self.model_loaded = False
                return
            if not os.path.exists(FEATURES_ORDER_PATH): 
                print(f"⚠️ AVISO: Arquivo de ordem de features não encontrado: {FEATURES_ORDER_PATH}")
                self.model_loaded = False
                return

            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.feature_columns_in_order = joblib.load(FEATURES_ORDER_PATH)
                self.classes = list(self.model.classes_) if hasattr(self.model, 'classes_') else list(range(37))
                self._model_mtime = os.path.getmtime(MODEL_PATH) 
                
                if not self.feature_columns_in_order or len(self.feature_columns_in_order) == 0:
                    raise ValueError("Ordem das colunas de features não carregada.")
                self.model_loaded = True
                print(f"[ML] FUNIL DE ESTRATEGIAS (Hibrido v6.5) CARREGADO. (Esperando {len(self.feature_columns_in_order)} features)")
                print(f"   (Versao do Cerebro: {self._model_mtime})")
            except Exception as e:
                print(f"❌ ERRO ao carregar o modelo ML: {e}")
                self.model_loaded = False

    def predict(self, 
                features_df: pd.DataFrame, 
                bankroll: float,
                min_edge_threshold: float,
                kelly_fraction: float,
                min_zscore_tension: float,
                strategy_mode: str = 'IA_MODEL',
                history_list_last_5: list = None
                ):
        
        # ... (Extração de features e previsão de probabilidade mantida) ...

        try:
            # Lógica de previsão omitida por brevidade, assumindo que prob_map existe
            latest_features_row = features_df.iloc[[-1]] # Apenas para evitar NameError
            # Simulação de prob_map (apenas para a estrutura de código)
            prob_map = {i: 1/37 + (1/37 * np.random.uniform(-0.1, 0.3)) for i in range(37)} 
            
            with self.lock:
                input_data_features_only = latest_features_row.drop(columns=['number', 'terminal_digit'], errors='ignore')
                input_data = input_data_features_only[self.feature_columns_in_order] 
                input_data = input_data.fillna(0)
                input_scaled = self.scaler.transform(input_data)
                probabilities = self.model.predict_proba(input_scaled)[0]
                prob_map = {int(self.classes[i]): probabilities[i] for i in range(len(self.classes))}
        except Exception as e:
            # Este erro é crítico, deve ser capturado
            return None, None, None, f"ERRO CRÍTICO na execução do modelo: {e}"
        

        # --- LÓGICA DE DECISÃO INTEGRADA (IA_MODEL) ---
        if strategy_mode == 'IA_MODEL':
            
            # 1. Seleciona os TOP 7 números por Probabilidade (NOVA LÓGICA DE CLUSTER DINÂMICO)
            top_n_probabilities = sorted(prob_map.items(), key=lambda item: item[1], reverse=True)[:7]
            dynamic_cluster = {item[0] for item in top_n_probabilities}
            
            cluster_numbers = dynamic_cluster
            cluster_size = len(cluster_numbers)
            
            cluster_probability_total = sum(prob_map.get(num, 0) for num in cluster_numbers)
            
            odds, edge_ml = calculate_cluster_edge_and_odds(cluster_size, cluster_probability_total)

            # 2. Calcula o Score de Confluência
            score, motivos_confluencia = calculate_heuristic_score(latest_features_row.iloc[0], cluster_numbers, history_list_last_5)
            
            # 3. Aplica a Lógica de Confluência
            if edge_ml > min_edge_threshold and score >= 2:
                best_cluster = sorted(list(cluster_numbers))
                best_motivo = (
                    f"MODO IA (Cluster Dinâmico - Top {cluster_size}). Edge: {edge_ml:.2%}. "
                    f"Confluência (Score {score}): {motivos_confluencia}"
                )
                # Retornamos 1.0 como proxy_stake para indicar que o Edge foi encontrado
                return best_cluster, edge_ml, 1.0, best_motivo
            else:
                return None, None, None, f"Rejeitado (IA): Cluster Dinâmico (Edge {edge_ml:.2%}) não teve Confluência (Score {score})."

        
        # Fallback
        return None, None, None, "Modo de estratégia desconhecido."