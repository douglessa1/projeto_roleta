# douglessa1/projeto_roleta/projeto_roleta-4eb8af59f00aad63289b5a75b94bcc4e4e852c83/ml_model.py
import pandas as pd
import joblib
import os
import numpy as np
from threading import Lock
import math
import typing # Import typing

# Importa as constantes da Roleta 
try:
    from analysis import WHEEL_ORDER, TOTAL_NUMBERS, WHEEL_POSITIONS, get_physical_neighbors, CALLING_NUMBERS
except ImportError:
    # Fallback (omitido por brevidade)
    WHEEL_ORDER = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
    TOTAL_NUMBERS = 37 
    WHEEL_POSITIONS = {number: i for i, number in enumerate(WHEEL_ORDER)}
    def get_physical_neighbors(number: int, radius: int = 1) -> set:
        neighbors = {number}
        if number in WHEEL_POSITIONS:
            pos = WHEEL_POSITIONS[number]
            for i in range(-radius, radius + 1):
                index = (pos + i) % TOTAL_NUMBERS
                neighbors.add(WHEEL_ORDER[index])
        return neighbors
    CALLING_NUMBERS = { 0: {1, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 20, 21, 24, 25, 26, 29}, 1: {0, 27, 21, 20, 34}, 2: {22, 28, 31, 36}, 3: {3, 7, 12, 19}, 4: {7, 10, 19, 21}, 5: {0, 27, 21, 20, 34}, 6: {9, 27, 28, 34}, 7: {3, 19, 21, 28, 29}, 8: {0, 8, 21, 15, 28}, 9: {0, 6, 22, 30}, 10: {0, 4, 7, 13, 17}, 11: {3, 15, 22, 24}, 12: {9, 16, 21, 27, 32}, 13: {0, 28, 31}, 14: {0, 5, 20, 27, 23, 32}, 15: {0, 11, 22}, 16: {0, 12, 19, 24}, 17: {0, 10, 13, 20, 21, 29}, 18: {11, 15, 22}, 19: {3, 4, 7, 16}, 20: {0, 15, 17, 27}, 21: {0, 7, 12, 23, 27}, 22: {2, 11, 15, 22}, 23: {12, 21, 27, 32}, 24: {0, 19, 35}, 25: {18, 25, 26, 36}, 26: {0, 10, 12, 15, 25, 28, 29, 32}, 27: {12, 21, 20, 32}, 28: {13, 31}, 29: {0, 13, 17, 20, 24}, 30: {4, 7, 9, 11, 12}, 31: set(), 32: set(), 33: set(), 34: {6, 27}, 35: {24}, 36: {2, 25}, }


# --- CORREÇÃO (v6.0.1): Importa a classe do novo arquivo ---
try:
    from adaptive_system import AdaptiveThresholdSystem
except ImportError:
    print("WARNING: adaptive_system.py não encontrado. Usando 'object' as fallback.")
    # Se adaptive_system.py não existir, define como um 'object' genérico
    class AdaptiveThresholdSystem(object):
        def get_recent_performance(self) -> float:
            return 0.5 # Retorna neutro


# --- CONFIGURAÇÃO DE VIZINHANÇA ---
RADIUS_VIZINHOS = 3 
WHEEL_CLUSTERS = {
    n: get_physical_neighbors(n, radius=RADIUS_VIZINHOS) | {n} 
    for n in range(TOTAL_NUMBERS)
}

# --- MAPEAMENTOS DE ZONAS ---
ZONAS_MAPEAMENTO = {
    'is_d1': set(range(1, 13)), 'is_d2': set(range(13, 25)), 'is_d3': set(range(25, 37)),
    'is_c1': set(range(1, 37, 3)), 'is_c2': set(range(2, 37, 3)), 'is_c3': set(range(3, 37, 3)),
    'is_red': {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36},
    'is_black': {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35},
    'is_low': set(range(1, 19)), 'is_high': set(range(19, 37)),
    'is_odd': set(range(1, 37, 2)), 'is_even': set(range(2, 37, 2)),
}

# --- CONFIGURAÇÕES DO MODELO (PATHs) ---
MODEL_DIR = "ml_models" 

# --- CORREÇÃO (v6.0.3): Garante que o nome do arquivo está correto (sem 'moodel') ---
MODEL_FILENAME = 'multiclass_roulette_model.pkl'
SCALER_FILENAME = 'multiclass_scaler.pkl'
FEATURES_ORDER_FILENAME = 'feature_order.pkl' 

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)
FEATURES_ORDER_PATH = os.path.join(MODEL_DIR, FEATURES_ORDER_FILENAME) 

# --- NOVO (v6.0 - Req 1): PESOS DINÂMICOS ---
SCORE_WEIGHTS = {
    'gatilho_bsb': 2.5, # Gatilho BSB é o sinal mais forte
    'soma_terminal': 1.5,
    'confluencia_features': 1.0, # Peso por cada sinal de confluência (do analysis.py)
    
    'tensao_zona': {
        'zscore_extremo': 2.0, # |z| > 2.5
        'zscore_moderado': 1.5, # |z| > 1.8
        'zscore_leve': 1.0 # |z| > 1.2
    },
    'atraso_alvo': {
        'critico': 2.5, # > 100 spins
        'alto': 2.0, # > 75 spins
        'moderado': 1.5 # > 50 spins
    },
    'padroes_sequencia': {
        'sequencia_longa': 2.0, # 6+
        'sequencia_media': 1.5, # 4–5
        'sequencia_curta': 1.0 # 3
    },
    'momentum_zscore': 1.5 # Peso se o Z-score da zona está em momentum
}


# --- FUNÇÕES AUXILIARES DE LÓGICA (v6.0) ---

def calculate_cluster_edge_and_odds(cluster_size: int, cluster_probability: float):
    P_teorica = cluster_size / 37.0
    odds = 36.0 / cluster_size 
    edge = cluster_probability - P_teorica
    return odds, edge

# NOVO (v6.0 - Req 1): Cálculo de Score com Pesos Dinâmicos
def calculate_dynamic_confluence_score(features: pd.Series, cluster_numbers: set) -> tuple[float, str]:
    """
    Calcula um "Score de Confluência" dinâmico (float)
    usando os pesos (SCORE_WEIGHTS) e as features (v6.0).
    """
    score = 0.0
    motivos = []
    
    # 1. Padrões de Gatilho (Estes são os mais fortes)
    if features.get('was_called_by_previous', 0) > 0:
        score += SCORE_WEIGHTS['gatilho_bsb']
        motivos.append(f"GatilhoBSB(x{SCORE_WEIGHTS['gatilho_bsb']})")
        
    if features.get('current_is_terminal_sum', 0) > 0:
        score += SCORE_WEIGHTS['soma_terminal']
        motivos.append(f"SomaTerminal(x{SCORE_WEIGHTS['soma_terminal']})")

    # 2. Confluência de Features (Aprendida pelo ML)
    confluence_count = features.get('feature_confluence_count', 0)
    if confluence_count > 0:
        peso = SCORE_WEIGHTS['confluencia_features'] * confluence_count
        score += peso
        motivos.append(f"Confluencia({confluence_count}x{SCORE_WEIGHTS['confluencia_features']})")

    # 3. Validação de Histórico (Atraso/Tensão de Zona e Alvo)
    alvo_encontrado_em_zona = set() # Para não repetir motivos

    for num in cluster_numbers:
        # Verifica Atraso Individual (Sinal Forte)
        atraso_num = features.get(f'is_num_{num}_atraso', 0)
        if atraso_num > 100:
            if 'atraso_critico' not in alvo_encontrado_em_zona:
                score += SCORE_WEIGHTS['atraso_alvo']['critico']
                motivos.append(f"AtrasoAlvo(x{SCORE_WEIGHTS['atraso_alvo']['critico']})")
                alvo_encontrado_em_zona.add('atraso_critico')
        elif atraso_num > 75:
            if 'atraso_alto' not in alvo_encontrado_em_zona:
                score += SCORE_WEIGHTS['atraso_alvo']['alto']
                motivos.append(f"AtrasoAlvo(x{SCORE_WEIGHTS['atraso_alvo']['alto']})")
                alvo_encontrado_em_zona.add('atraso_alto')
        elif atraso_num > 50:
            if 'atraso_mod' not in alvo_encontrado_em_zona:
                score += SCORE_WEIGHTS['atraso_alvo']['moderado']
                motivos.append(f"AtrasoAlvo(x{SCORE_WEIGHTS['atraso_alvo']['moderado']})")
                alvo_encontrado_em_zona.add('atraso_mod')
        
        # --- Validação de Tensão (Z-Score) e Momentum de Zonas ---
        for zona_key, zona_set in ZONAS_MAPEAMENTO.items():
            if num in zona_set and zona_key not in alvo_encontrado_em_zona:
                zscore = features.get(f'zscore_{zona_key}', 0)
                zscore_abs = abs(zscore)
                
                # Pesa pela Tensão (Atraso/Quente)
                if zscore_abs > 2.5:
                    score += SCORE_WEIGHTS['tensao_zona']['zscore_extremo']
                    motivos.append(f"ZonaExtrema(x{SCORE_WEIGHTS['tensao_zona']['zscore_extremo']})")
                    alvo_encontrado_em_zona.add(zona_key)
                elif zscore_abs > 1.8:
                    score += SCORE_WEIGHTS['tensao_zona']['zscore_moderado']
                    motivos.append(f"ZonaMod(x{SCORE_WEIGHTS['tensao_zona']['zscore_moderado']})")
                    alvo_encontrado_em_zona.add(zona_key)
                elif zscore_abs > 1.2:
                    score += SCORE_WEIGHTS['tensao_zona']['zscore_leve']
                    motivos.append(f"ZonaLeve(x{SCORE_WEIGHTS['tensao_zona']['zscore_leve']})")
                    alvo_encontrado_em_zona.add(zona_key)
                
                # Pesa pelo Momentum (se o Z-score está crescendo)
                zscore_mom = features.get(f'zscore_{zona_key}_momentum', 0)
                if zscore_mom > 0 and (f"{zona_key}_mom" not in alvo_encontrado_em_zona):
                    score += SCORE_WEIGHTS['momentum_zscore']
                    motivos.append(f"Momentum(x{SCORE_WEIGHTS['momentum_zscore']})")
                    alvo_encontrado_em_zona.add(f"{zona_key}_mom")

    # 4. Validação de Sequência
    seq_longa, seq_media, seq_curta = False, False, False
    for fcol in ['is_red_seq_ocorrencia', 'is_black_seq_ocorrencia', 'is_low_seq_ocorrencia', 'is_high_seq_ocorrencia', 'is_even_seq_ocorrencia', 'is_odd_seq_ocorrencia']:
        seq_len = features.get(fcol, 0)
        if seq_len >= 6: seq_longa = True
        elif seq_len >= 4: seq_media = True
        elif seq_len == 3: seq_curta = True
    
    if seq_longa:
        score += SCORE_WEIGHTS['padroes_sequencia']['sequencia_longa']
        motivos.append(f"SeqL(x{SCORE_WEIGHTS['padroes_sequencia']['sequencia_longa']})")
    elif seq_media:
        score += SCORE_WEIGHTS['padroes_sequencia']['sequencia_media']
        motivos.append(f"SeqM(x{SCORE_WEIGHTS['padroes_sequencia']['sequencia_media']})")
    elif seq_curta:
        score += SCORE_WEIGHTS['padroes_sequencia']['sequencia_curta']
        motivos.append(f"SeqC(x{SCORE_WEIGHTS['padroes_sequencia']['sequencia_curta']})")

    if not motivos:
        return 0.0, "Sem confluência de histórico."
        
    return round(score, 2), ", ".join(motivos)

# NOVO (v6.0 - Req 2): Cálculo de Volatilidade e Threshold Dinâmico
def calculate_table_volatility(features_df: pd.DataFrame) -> float:
    """Calcula a volatilidade recente da mesa (Std Dev dos Z-Scores)."""
    try:
        # Pega os Z-scores das últimas 10 rodadas
        zscore_cols = [col for col in features_df.columns if col.startswith('zscore_is_') and not col.endswith('_momentum')]
        recent_zscores = features_df.iloc[-10:][zscore_cols]
        # Calcula a média do Desvio Padrão de todos os Z-scores
        volatility = recent_zscores.std().mean()
        return volatility if pd.notna(volatility) else 1.0
    except Exception:
        return 1.0 # Retorna volatilidade neutra em caso de erro

def calculate_dynamic_threshold(base_threshold: float, volatilidade_mesa: float, performance_recente: float) -> float:
    """Calcula o threshold de ativação dinâmico (Req 2)."""
    
    # Ajuste de Volatilidade: Mais volátil -> Mais cauteloso (threshold maior)
    if volatilidade_mesa > 1.8:
        ajuste_volatilidade = 0.5
    elif volatilidade_mesa < 0.8:
        ajuste_volatilidade = -0.3
    else:
        ajuste_volatilidade = 0.0
        
    # Ajuste de Performance: Performance ruim -> Mais cauteloso (threshold maior)
    if performance_recente < 0.4: # Menos de 40% de acerto recente
        ajuste_performance = 0.4
    elif performance_recente > 0.6: # Mais de 60% de acerto recente
        ajuste_performance = -0.2
    else:
        ajuste_performance = 0.0
        
    # O threshold base (ex: 2.0) é ajustado
    final_threshold = base_threshold + ajuste_volatilidade + ajuste_performance
    
    # Garante um limite mínimo
    return max(1.5, final_threshold)

# NOVO (v6.0 - Req 3): Cálculo de Confiança Híbrida
def calculate_hybrid_confidence(prob_ml: float, score_heuristico: float, volatilidade: float) -> float:
    """Calcula o Score de Confiança Combinado (Req 3)."""
    
    # Normaliza os pesos
    # prob_ml (já é 0-1)
    # score_heuristico (normalizamos assumindo que um score "bom" é ~5.0)
    peso_ml = prob_ml
    peso_heuristico = min(1.0, score_heuristico / 5.0) # Normaliza o score heurístico
    
    fator_concordancia = 1.0
    
    # Se ML e Heurística concordam fortemente, aumenta a confiança
    if prob_ml > 0.05 and score_heuristico >= 3.0: # 0.05 é alto (3x a base de 1/37)
        fator_concordancia = 1.2 # Bônus de 20%
    # Se discordam (ML baixo, Heurística baixa), reduz
    elif prob_ml < 0.02 and score_heuristico <= 1:
        fator_concordancia = 0.8
        
    # Média ponderada dos pesos, multiplicada pela concordância
    confiança_bruta = (peso_ml + peso_heuristico) / 2 * fator_concordancia
    
    # Ajuste final pela volatilidade (mais volátil, menos confiança)
    confiança_final = confiança_bruta * (1.0 / (1.0 + volatilidade * 0.2))
    
    return min(1.0, confiança_final) # Garante que fique entre 0 e 1

# --------------------------------------------------------

class TrainedMLModel:
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
        print("Tentando carregar/recarregar funil ML (v6.0)...")
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
                print(f"🧠 FUNIL DE ESTRATÉGIAS (Híbrido v6.0) CARREGADO. (Esperando {len(self.feature_columns_in_order)} features)")
                print(f"   (Versão do Cérebro: {self._model_mtime})")
            except Exception as e:
                print(f"❌ ERRO ao carregar o modelo ML: {e}")
                self.model_loaded = False


    # --- FUNÇÃO DE PREDIÇÃO (v6.0 - HÍBRIDA) ---
    def predict(self, 
                features_df: pd.DataFrame, 
                # --- CORREÇÃO (v6.0.2): Usa aspas (Forward Reference) ---
                adaptive_system: "AdaptiveThresholdSystem", 
                config: dict # Traz 'base_threshold', etc.
                ):
        
        # --- CORREÇÃO (v6.0.4): Lógica de verificação robusta ---
        # 0. Verificação de Saúde e Recarregamento
        
        # Se o modelo não está carregado, tenta carregar
        if not self.model_loaded:
            print("Predict: Modelo não carregado. Tentando carregar...")
            self.load_model_files()
            # Se falhar de novo, retorna
            if not self.model_loaded:
                 return None, 0.0, 0.0, 0.0, 0.0, "Rejeitado: Modelo ML não carregado. (Execute train_model.py)"
        
        # Se o modelo ESTÁ carregado, checa por atualizações
        try:
            current_disk_mtime = os.path.getmtime(MODEL_PATH)
            if current_disk_mtime != self._model_mtime:
                print(f"DETECTADA NOVA VERSÃO DO 'CÉREBRO' NO DISCO! (Disco: {current_disk_mtime} vs Memória: {self._model_mtime})")
                self.load_model_files()
        except FileNotFoundError:
            # Isso não deve acontecer se self.model_loaded=True, mas por segurança
             print(f"AVISO: Modelo ML carregado, mas {MODEL_PATH} não encontrado no disco.")
        except Exception as e:
            print(f"Erro ao verificar a versão do modelo: {e}")
        # --- FIM DA CORREÇÃO ---
        
        try:
            latest_features_row = features_df.iloc[[-1]]
            number_gatilho = latest_features_row['number'].iloc[0]
            if pd.isna(number_gatilho) or number_gatilho == -1:
                 return None, 0.0, 0.0, 0.0, 0.0, "Rejeitado: Não há número gatilho (N_t) disponível."
            number_gatilho = int(number_gatilho)
        except Exception as e:
             return None, 0.0, 0.0, 0.0, 0.0, f"Erro ao extrair features: {e}"

        # 1. Execução do ML (Probabilidades)
        try:
            input_data_features_only = latest_features_row.drop(columns=['number'], errors='ignore')
            
            # Garante que colunas que faltam (ex: 'terminal_digit') não quebrem
            cols_presentes = [col for col in self.feature_columns_in_order if col in input_data_features_only.columns]
            input_data = input_data_features_only[cols_presentes]
            
            # Re-adiciona colunas faltantes (se o treino teve e o real não) com 0
            cols_faltantes = set(self.feature_columns_in_order) - set(cols_presentes)
            for col in cols_faltantes:
                input_data[col] = 0
            
            input_data = input_data[self.feature_columns_in_order] # Garante a ordem
            
            input_data = input_data.fillna(0)
            
            with self.lock:
                input_scaled = self.scaler.transform(input_data)
                probabilities = self.model.predict_proba(input_scaled)[0]
            prob_map = {int(self.classes[i]): probabilities[i] for i in range(len(self.classes))}
        except Exception as e:
            print(f"❌ ERRO CRÍTICO na previsão do ML: {e}")
            return None, 0.0, 0.0, 0.0, 0.0, "ERRO CRÍTICO na execução do modelo."
            

        # --- LÓGICA DE DECISÃO HÍBRIDA (v6.0) ---

        # 2. Encontra o Pico do ML (o número com maior probabilidade)
        pico_number_ml = max(prob_map, key=prob_map.get)
        prob_pico_ml = prob_map[pico_number_ml]
        
        cluster_numbers = WHEEL_CLUSTERS.get(pico_number_ml)
        if not cluster_numbers:
            return None, 0.0, 0.0, 0.0, 0.0, f"Rejeitado: Pico do ML {pico_number_ml} inválido."
            
        cluster_size = len(cluster_numbers)
        cluster_probability_total = sum(prob_map.get(num, 0) for num in cluster_numbers)
        odds, edge_ml = calculate_cluster_edge_and_odds(cluster_size, cluster_probability_total)

        # 3. Calcula o Score de Confluência Dinâmico (Req 1)
        score_heuristico, motivos_confluencia = calculate_dynamic_confluence_score(
            latest_features_row.iloc[0], 
            cluster_numbers
        )
        
        # 4. Calcula o Threshold Dinâmico (Req 2)
        volatilidade_mesa = calculate_table_volatility(features_df)
        
        if isinstance(adaptive_system, AdaptiveThresholdSystem):
            performance_recente = adaptive_system.get_recent_performance()
        else:
            print("WARNING: adaptive_system não é uma instância válida. Usando performance neutra (0.5).")
            performance_recente = 0.5
            
        base_threshold = config.get('base_threshold', 2.0)
        
        dynamic_threshold = calculate_dynamic_threshold(
            base_threshold, 
            volatilidade_mesa, 
            performance_recente
        )

        # 5. Calcula a Confiança Híbrida (Req 3)
        hybrid_confidence = calculate_hybrid_confidence(
            cluster_probability_total, 
            score_heuristico, 
            volatilidade_mesa
        )

        # 6. Decisão Final (Híbrida)
        min_edge = config.get('min_edge_threshold', 0.001)

        # O threshold base é 2.0 (de 1.5 a 3.0). A confiança é 0-1.
        # Dividimos o threshold por 10 (ex: 2.0 -> 0.2) para comparar com a confiança
        if hybrid_confidence > (dynamic_threshold / 10.0) and edge_ml > min_edge:
            
            stake_proxy = 1.0 # Indica aposta
            
            best_cluster = sorted(list(cluster_numbers))
            best_motivo = (
                f"HÍBRIDO v6.0 (Pico ML {pico_number_ml}). "
                f"Conf: {hybrid_confidence*100:.1f}% (T: {dynamic_threshold*10:.1f}%). "
                f"Edge: {edge_ml:.2%}. "
                f"ScoreH: {score_heuristico:.1f}. "
                f"Vol: {volatilidade_mesa:.2f}. "
                f"Motivos: {motivos_confluencia}"
            )
            
            # Retorna todos os dados para log (Req 6)
            return best_cluster, edge_ml, hybrid_confidence, dynamic_threshold, score_heuristico, best_motivo
        
        else:
            # Rejeitado
            motivo_rejeicao = (
                f"Rejeitado (Híbrido v6.0): "
                f"Conf: {hybrid_confidence*100:.1f}% (T: {dynamic_threshold*10:.1f}%). "
                f"Edge: {edge_ml:.2%}. "
                f"ScoreH: {score_heuristico:.1f}."
            )
            return None, edge_ml, hybrid_confidence, dynamic_threshold, score_heuristico, motivo_rejeicao