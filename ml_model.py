import pandas as pd
import joblib
import os
import numpy as np
from threading import Lock
import math

# Importa as constantes da Roleta 
try:
    from analysis import WHEEL_ORDER, TOTAL_NUMBERS, WHEEL_POSITIONS, get_physical_neighbors, CALLING_NUMBERS
except ImportError:
    # Fallback (omitido por brevidade, mas está no seu arquivo)
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


# --- CONFIGURAÇÃO DE VIZINHANÇA ---
RADIUS_VIZINHOS = 3 
WHEEL_CLUSTERS = {
    n: get_physical_neighbors(n, radius=RADIUS_VIZINHOS) | {n} 
    for n in range(TOTAL_NUMBERS)
}

# --- MAPEAMENTOS DE ZONAS (v5.8) ---
VERMELHOS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
PRETOS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
BAIXOS = set(range(1, 19))
ALTOS = set(range(19, 37))
IMPARES = set(range(1, 37, 2))
PARES = set(range(2, 37, 2))
DUZIA_1 = set(range(1, 13))
DUZIA_2 = set(range(13, 25))
DUZIA_3 = set(range(25, 37))
COLUNA_1 = set(range(1, 37, 3))
COLUNA_2 = set(range(2, 37, 3))
COLUNA_3 = set(range(3, 37, 3))

# --- CONFIGURAÇÕES DO MODELO (PATHs) ---
MODEL_DIR = "ml_models" 
MODEL_FILENAME = 'multiclass_roulette_model.pkl'
SCALER_FILENAME = 'multiclass_scaler.pkl'
FEATURES_ORDER_FILENAME = 'feature_order.pkl' 
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)
FEATURES_ORDER_PATH = os.path.join(MODEL_DIR, FEATURES_ORDER_FILENAME) 

# --- FUNÇÕES AUXILIARES ---

def calculate_cluster_edge_and_odds(cluster_size: int, cluster_probability: float):
    P_teorica = cluster_size / 37.0
    odds = 36.0 / cluster_size 
    edge = cluster_probability - P_teorica
    return odds, edge

def calculate_kelly_stake(edge: float, odds: float, bankroll: float, cluster_size: int, kelly_fraction: float):
    if edge <= 0: return 0.0
    P_model = edge + (cluster_size / 37.0) 
    b = odds - 1 
    q = 1 - P_model
    if b <= 0.001 or P_model <= 0.0: return 0.0
    kelly_percentage = (b * P_model - q) / b
    stake_percentage = max(0, kelly_percentage * kelly_fraction) 
    return bankroll * stake_percentage

# --- LÓGICA DE CONFLUÊNCIA HEURÍSTICA (v5.8 - Zonas Quentes e Frias) ---
def calculate_heuristic_score(features: pd.Series, cluster_numbers: set) -> tuple[int, str]:
    """
    Calcula um "Score de Confluência" para o Modo LUCAS_BSB,
    validando o histórico (v5.8 - Quente e Frio).
    """
    score = 0
    motivos = []
    
    # 1. Padrões de Gatilho (Estes são os mais fortes)
    if features.get('was_called_by_previous', 0) > 0: # O N_t foi chamado pelo N_t-1
        score += 2 # Peso maior
        motivos.append("Gatilho BSB (N-1 -> N)")
    if features.get('current_is_terminal_sum', 0) > 0: # O N_t completou uma soma de terminal
        score += 1
        motivos.append("Padrão de Soma de Terminais")

    # 2. Validação de Histórico (Atraso/Tensão de Zona)
    alvo_encontrado_em_zona = set() # Para não repetir motivos

    for num in cluster_numbers:
        # Verifica Atraso Individual (Sinal Forte)
        atraso_num = features.get(f'is_num_{num}_atraso', 0)
        if atraso_num > 74: # Muito atrasado (mais de 2x a média)
            if 'atraso_individual' not in alvo_encontrado_em_zona:
                score += 2
                motivos.append(f"Alvo {num} está a {atraso_num} spins (Atraso Alto)")
                alvo_encontrado_em_zona.add('atraso_individual')
        
        # --- CORREÇÃO v5.8: Adiciona Zonas QUENTES (Tendência) e FRIAS (Atraso) ---
        
        # Dúzias
        if num in DUZIA_1 and 'd1' not in alvo_encontrado_em_zona:
            if features.get('zscore_is_d1', 0) < -1.5:
                score += 1; motivos.append(f"Alvo {num} na Dúzia 1 (Atrasada)"); alvo_encontrado_em_zona.add('d1')
            elif features.get('zscore_is_d1', 0) > 1.5: 
                score += 1; motivos.append(f"Alvo {num} na Dúzia 1 (Quente)"); alvo_encontrado_em_zona.add('d1')
        if num in DUZIA_2 and 'd2' not in alvo_encontrado_em_zona:
            if features.get('zscore_is_d2', 0) < -1.5:
                score += 1; motivos.append(f"Alvo {num} na Dúzia 2 (Atrasada)"); alvo_encontrado_em_zona.add('d2')
            elif features.get('zscore_is_d2', 0) > 1.5: 
                score += 1; motivos.append(f"Alvo {num} na Dúzia 2 (Quente)"); alvo_encontrado_em_zona.add('d2')
        if num in DUZIA_3 and 'd3' not in alvo_encontrado_em_zona:
            if features.get('zscore_is_d3', 0) < -1.5:
                score += 1; motivos.append(f"Alvo {num} na Dúzia 3 (Atrasada)"); alvo_encontrado_em_zona.add('d3')
            elif features.get('zscore_is_d3', 0) > 1.5: 
                score += 1; motivos.append(f"Alvo {num} na Dúzia 3 (Quente)"); alvo_encontrado_em_zona.add('d3')

        # Colunas
        if num in COLUNA_1 and 'c1' not in alvo_encontrado_em_zona:
            if features.get('zscore_is_c1', 0) < -1.5:
                score += 1; motivos.append(f"Alvo {num} na Coluna 1 (Atrasada)"); alvo_encontrado_em_zona.add('c1')
            elif features.get('zscore_is_c1', 0) > 1.5: 
                score += 1; motivos.append(f"Alvo {num} na Coluna 1 (Quente)"); alvo_encontrado_em_zona.add('c1')
        if num in COLUNA_2 and 'c2' not in alvo_encontrado_em_zona:
            if features.get('zscore_is_c2', 0) < -1.5:
                score += 1; motivos.append(f"Alvo {num} na Coluna 2 (Atrasada)"); alvo_encontrado_em_zona.add('c2')
            elif features.get('zscore_is_c2', 0) > 1.5: 
                score += 1; motivos.append(f"Alvo {num} na Coluna 2 (Quente)"); alvo_encontrado_em_zona.add('c2')
        if num in COLUNA_3 and 'c3' not in alvo_encontrado_em_zona:
            if features.get('zscore_is_c3', 0) < -1.5:
                score += 1; motivos.append(f"Alvo {num} na Coluna 3 (Atrasada)"); alvo_encontrado_em_zona.add('c3')
            elif features.get('zscore_is_c3', 0) > 1.5: 
                score += 1; motivos.append(f"Alvo {num} na Coluna 3 (Quente)"); alvo_encontrado_em_zona.add('c3')

    # 3. Validação de Tendência (Zonas Quentes / Sequências)
    if features.get('is_odd_seq_ocorrencia', 0) >= 4:
        score += 1
        motivos.append(f"Sequência de {int(features.get('is_odd_seq_ocorrencia'))} Ímpares")
    if features.get('is_even_seq_ocorrencia', 0) >= 4:
        score += 1
        motivos.append(f"Sequência de {int(features.get('is_even_seq_ocorrencia'))} Pares")
        
    for num in cluster_numbers:
        if num in IMPARES and features.get('zscore_is_odd', 0) > 1.5 and 'quente_impar' not in alvo_encontrado_em_zona:
            score += 1; motivos.append(f"Alvo {num} nos Ímpares (Quente)"); alvo_encontrado_em_zona.add('quente_impar')
        if num in PARES and features.get('zscore_is_even', 0) > 1.5 and 'quente_par' not in alvo_encontrado_em_zona:
            score += 1; motivos.append(f"Alvo {num} nos Pares (Quente)"); alvo_encontrado_em_zona.add('quente_par')
        if num in VERMELHOS and features.get('zscore_is_red', 0) > 1.5 and 'quente_vermelho' not in alvo_encontrado_em_zona: 
            score += 1; motivos.append(f"Alvo {num} nos Vermelhos (Quente)"); alvo_encontrado_em_zona.add('quente_vermelho')
        if num in PRETOS and features.get('zscore_is_black', 0) > 1.5 and 'quente_preto' not in alvo_encontrado_em_zona: 
            score += 1; motivos.append(f"Alvo {num} nos Pretos (Quente)"); alvo_encontrado_em_zona.add('quente_preto')
        if num in BAIXOS and features.get('zscore_is_low', 0) > 1.5 and 'quente_baixo' not in alvo_encontrado_em_zona: 
            score += 1; motivos.append(f"Alvo {num} nos Baixos (Quente)"); alvo_encontrado_em_zona.add('quente_baixo')
        if num in ALTOS and features.get('zscore_is_high', 0) > 1.5 and 'quente_alto' not in alvo_encontrado_em_zona: 
            score += 1; motivos.append(f"Alvo {num} nos Altos (Quente)"); alvo_encontrado_em_zona.add('quente_alto')

    final_score = score
    if not motivos:
        return 0, "Sem confluência de histórico."
        
    return final_score, ", ".join(motivos)
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
        print("Tentando carregar/recarregar funil ML...")
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
                print(f"🧠 FUNIL DE ESTRATÉGIAS (ML v5.11) CARREGADO. (Esperando {len(self.feature_columns_in_order)} features)")
                print(f"   (Versão do Cérebro: {self._model_mtime})")
            except Exception as e:
                print(f"❌ ERRO ao carregar o modelo ML: {e}")
                self.model_loaded = False


    def predict(self, 
                features_df: pd.DataFrame, 
                bankroll: float,
                min_edge_threshold: float,
                kelly_fraction: float,
                min_zscore_tension: float,
                strategy_mode: str = 'IA_MODEL' 
                ):
        
        try:
            current_disk_mtime = os.path.getmtime(MODEL_PATH)
            if current_disk_mtime != self._model_mtime:
                print(f"DETECTADA NOVA VERSÃO DO 'CÉREBRO' NO DISCO! (Disco: {current_disk_mtime} vs Memória: {self._model_mtime})")
                self.load_model_files()
        except Exception as e:
            print(f"Erro ao verificar a versão do modelo: {e}")
            self.model_loaded = False
        
        if not self.model_loaded or features_df is None or features_df.empty:
            return None, None, None, "Rejeitado: Modelo ML não carregado ou features vazias."
        
        # --- LÓGICA DE TEMPO REAL (v5.10): Usa o último número ---
        try:
            latest_features_row = features_df.iloc[[-1]]
            number_gatilho = latest_features_row['number'].iloc[0]
            if pd.isna(number_gatilho) or number_gatilho == -1:
                 return None, None, None, "Rejeitado: Não há número gatilho (N_t) disponível."
            number_gatilho = int(number_gatilho)
        except IndexError:
             return None, None, None, "Rejeitado: Histórico muito curto para identificar N_t."
        except Exception as e:
             return None, None, None, f"Erro ao extrair features: {e}"
        # ---------------------------------------------------------

        try:
            input_data_features_only = latest_features_row.drop(columns=['number', 'terminal_digit'], errors='ignore')
            input_data = input_data_features_only[self.feature_columns_in_order] 
            input_data = input_data.fillna(0)
            
            with self.lock:
                input_scaled = self.scaler.transform(input_data)
                probabilities = self.model.predict_proba(input_scaled)[0]
            prob_map = {int(self.classes[i]): probabilities[i] for i in range(len(self.classes))}
        except Exception as e:
            print(f"❌ ERRO CRÍTICO na previsão do ML: {e}")
            print(f"   (Esperava {len(self.feature_columns_in_order)} colunas, mas as features atuais podem estar dessincronizadas)")
            return None, None, None, "ERRO CRÍTICO na execução do modelo."
            

        # --- LÓGICA DE DECISÃO INTEGRADA (v5.11 - Conforme o seu prompt) ---

        # Modo 1: LUCAS_BSB (Puro Heurística - BSB-First)
        # 1. Encontra o gatilho BSB
        # 2. Pontua os alvos com base na confluência (Zonas Quentes/Frias, Atrasos)
        # 3. Ignora o ML
        if strategy_mode == 'LUCAS_BSB':
            alvos_heurísticos = CALLING_NUMBERS.get(number_gatilho, set())
            if not alvos_heurísticos:
                return None, None, None, f"Rejeitado (BSB): Gatilho {number_gatilho} não possui alvos BSB definidos."
            
            best_score = -1
            best_cluster_bsb = None
            best_motivo_bsb = ""
            pico_vencedor_bsb = None

            for pico_number in alvos_heurísticos:
                cluster_numbers = WHEEL_CLUSTERS.get(pico_number)
                if not cluster_numbers: continue
                
                score, motivos_confluencia = calculate_heuristic_score(latest_features_row.iloc[0], cluster_numbers)
                
                if score > best_score:
                    best_score = score
                    best_cluster_bsb = sorted(list(cluster_numbers))
                    best_motivo_bsb = motivos_confluencia
                    pico_vencedor_bsb = pico_number
            
            if best_score >= 2: 
                heuristic_edge = best_score / 5.0 # Confiança baseada no Score (5=100%)
                best_motivo = (
                    f"MODO BSB (Gatilho {number_gatilho}). Foco no Pico {pico_vencedor_bsb}. "
                    f"Confluência (Score {best_score}): {best_motivo_bsb}"
                )
                return best_cluster_bsb, heuristic_edge, 1.0, best_motivo 
            else:
                return None, None, None, f"Rejeitado (BSB): Gatilho {number_gatilho} não teve Confluência (Score: {best_score})."

        # Modo 2: IA_MODEL (Confluência - ML-First)
        # 1. O ML prevê o alvo (Pico_ML)
        # 2. O alvo é validado por heurísticas (Score de Confluência)
        # 3. Aposta se (Probabilidade_ML > Limite) E (Score_Heurístico >= 2)
        elif strategy_mode == 'IA_MODEL':
            
            # 1. Encontra o Pico do ML (o número com maior probabilidade)
            pico_number_ml = max(prob_map, key=prob_map.get)
            cluster_numbers = WHEEL_CLUSTERS.get(pico_number_ml)
            if not cluster_numbers:
                return None, None, None, f"Rejeitado (IA): Pico do ML {pico_number_ml} inválido."
                
            cluster_size = len(cluster_numbers)
            cluster_probability_total = sum(prob_map.get(num, 0) for num in cluster_numbers)
            odds, edge_ml = calculate_cluster_edge_and_odds(cluster_size, cluster_probability_total)

            # 2. Calcula o Score de Confluência para esse alvo do ML
            score, motivos_confluencia = calculate_heuristic_score(latest_features_row.iloc[0], cluster_numbers)
            
            # 3. Aplica a Lógica de Confluência (Ponto 3 do seu prompt)
            if edge_ml > min_edge_threshold and score >= 2:
                best_cluster = sorted(list(cluster_numbers))
                best_motivo = (
                    f"MODO IA (Pico ML {pico_number_ml}). Edge: {edge_ml:.2%}. "
                    f"Confluência (Score {score}): {motivos_confluencia}"
                )
                return best_cluster, edge_ml, 1.0, best_motivo
            else:
                return None, None, None, f"Rejeitado (IA): Pico ML {pico_number_ml} (Edge {edge_ml:.2%}) não teve Confluência (Score {score})."

        
        # Fallback
        return None, None, None, "Modo de estratégia desconhecido."

    

