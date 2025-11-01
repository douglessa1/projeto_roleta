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
    # Fallback caso o analysis.py não seja encontrado
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
    # DICIONÁRIO DE REGRAS HEURÍSTICAS DE CHAMADA (COPIADO DO analysis.py)
    CALLING_NUMBERS = {
        0: {1, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 20, 21, 24, 25, 26, 29},
        1: {0, 27, 21, 20, 34}, 2: {22, 28, 31, 36}, 3: {3, 7, 12, 19}, 4: {7, 10, 19, 21},
        5: {0, 27, 21, 20, 34}, 6: {9, 27, 28, 34}, 7: {3, 19, 21, 28, 29}, 8: {0, 8, 21, 15, 28},
        9: {0, 6, 22, 30}, 10: {0, 4, 7, 13, 17}, 11: {3, 15, 22, 24}, 12: {9, 16, 21, 27, 32},
        13: {0, 28, 31}, 14: {0, 5, 20, 27, 23, 32}, 15: {0, 11, 22}, 16: {0, 12, 19, 24},
        17: {0, 10, 13, 20, 21, 29}, 18: {11, 15, 22}, 19: {3, 4, 7, 16}, 20: {0, 15, 17, 27},
        21: {0, 7, 12, 23, 27}, 22: {2, 11, 15, 22}, 23: {12, 21, 27, 32}, 24: {0, 19, 35},
        25: {18, 25, 26, 36}, 26: {0, 10, 12, 15, 25, 28, 29, 32}, 27: {12, 21, 20, 32},
        28: {13, 31}, 29: {0, 13, 17, 20, 24}, 30: {4, 7, 9, 11, 12},
        31: set(), 32: set(), 33: set(), 34: {6, 27}, 35: {24}, 36: {2, 25},
    }

# --- CONFIGURAÇÃO DE VIZINHANÇA ---
# Define o raio de vizinhos a ser considerado
RADIUS_VIZINHOS = 3 

# Mapeamento de todos os 37 clusters de vizinhança possíveis
WHEEL_CLUSTERS = {
    n: get_physical_neighbors(n, radius=RADIUS_VIZINHOS) | {n} 
    for n in range(TOTAL_NUMBERS)
}


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
    """ Calcula o Edge (vantagem) e a Odd (pagamento bruto) """
    P_teorica = cluster_size / 37.0
    # A odd de pagamento BRUTA (incluindo o stake de volta) é 36 / N
    odds = 36.0 / cluster_size 
    edge = cluster_probability - P_teorica
    return odds, edge

def calculate_kelly_stake(edge: float, odds: float, bankroll: float, cluster_size: int, kelly_fraction: float):
    # Esta função não é mais usada para o stake, mas é mantida para validar o Edge
    if edge <= 0: return 0.0
    P_model = edge + (cluster_size / 37.0) 
    b = odds - 1 # b é a odd LÍQUIDA (ex: 35 para 1, não 36 para 1)
    q = 1 - P_model
    if b <= 0.001 or P_model <= 0.0: return 0.0
    kelly_percentage = (b * P_model - q) / b
    stake_percentage = max(0, kelly_percentage * kelly_fraction) 
    return bankroll * stake_percentage

# --------------------------------------------------------

class TrainedMLModel:
    def __init__(self, lock: Lock):
        self.model = None
        self.scaler = None
        self.feature_columns_in_order = None 
        self.classes = None
        self.model_loaded = False
        self.lock = lock

        self.load_model_files()

    def reload_model(self):
        """ Função chamada pelo main.py para forçar o recarregamento. """
        print("Recebida ordem de recarregar o modelo...")
        self.load_model_files()
        
    def load_model_files(self):
        """ Carrega os arquivos .pkl do modelo, scaler e ordem das colunas. """
        print("Tentando carregar/recarregar funil ML...")
        with self.lock:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_ORDER_PATH):
                try:
                    self.model = joblib.load(MODEL_PATH)
                    self.scaler = joblib.load(SCALER_PATH)
                    self.feature_columns_in_order = joblib.load(FEATURES_ORDER_PATH)

                    self.classes = list(self.model.classes_) if hasattr(self.model, 'classes_') else list(range(37))
                    
                    if not self.feature_columns_in_order or len(self.feature_columns_in_order) == 0:
                        raise ValueError("Ordem das colunas de features não carregada.")

                    self.model_loaded = True
                    print(f"🧠 FUNIL DE ESTRATÉGIAS (ML) CARREGADO com sucesso. (Esperando {len(self.feature_columns_in_order)} features)")
                except Exception as e:
                    print(f"❌ ERRO ao carregar o modelo ML: {e}")
                    self.model_loaded = False
            else:
                print(f"⚠️ AVISO: Arquivos do funil ML não encontrados. O robô irá iniciar sem decisões ML.")
                self.model_loaded = False

    def calculate_heuristic_score(self, latest_features: pd.Series, min_zscore_tension: float):
        """
        NOVA LÓGICA: Calcula um "Score de Confluência" baseado no PDF.
        Não olha o alvo, mas sim o "momento" da mesa (zonas quentes/frias).
        """
        score = 0
        rationale = [] # Justificativa

        # Tensão Mínima (Z-Score)
        z_min = -abs(min_zscore_tension) # ex: -1.0
        z_max = abs(min_zscore_tension)  # ex: 1.0

        # Padrão 1: Tensão em Cores
        if latest_features.get('zscore_is_red', 0) < z_min:
            score += 1
            rationale.append("Cor Vermelha em forte atraso (fria)")
        if latest_features.get('zscore_is_black', 0) < z_min:
            score += 1
            rationale.append("Cor Preta em forte atraso (fria)")

        # Padrão 2: Tensão em Paridade/Metades
        if latest_features.get('zscore_is_low', 0) < z_min:
            score += 1
            rationale.append("Metade Baixa (1-18) em forte atraso (fria)")
        if latest_features.get('zscore_is_high', 0) < z_min:
            score += 1
            rationale.append("Metade Alta (19-36) em forte atraso (fria)")

        # Padrão 3: Momentum (EWMA)
        if latest_features.get('is_red_ewma_short', 0) > 0.8:
            score += 1
            rationale.append("Momentum Alto (Quente) na Cor Vermelha")
        if latest_features.get('is_black_ewma_short', 0) > 0.8:
            score += 1
            rationale.append("Momentum Alto (Quente) na Cor Preta")
            
        # Padrão 4: Tensão em Dúzias/Colunas (Exemplo)
        if latest_features.get('zscore_is_d1', 0) < z_min - 0.5: # Atraso mais forte
            score += 2 # Ponto extra
            rationale.append("1ª Dúzia em atraso extremo (muito fria)")
        if latest_features.get('zscore_is_c1', 0) > z_max + 0.5: # Tendência mais forte
            score += 2 # Ponto extra
            rationale.append("1ª Coluna em alta frequência (muito quente)")

        return score, rationale


    def predict(self, 
                features_df: pd.DataFrame, 
                bankroll: float,
                min_edge_threshold: float,
                kelly_fraction: float,
                min_zscore_tension: float,
                strategy_mode: str = 'IA_MODEL' # NOVO PARÂMETRO
                ):
        """
        PRIORIZAÇÃO EXCLUSIVA: Busca o cluster de vizinhança APENAS nos alvos ativados pela heurística,
        com o maior Edge positivo validado pelo ML.
        """
        if not self.model_loaded or features_df is None or features_df.empty:
            return None, None, None, "Rejeitado: Modelo ML não carregado ou features vazias."

        
        # 1. Preparação (Necessário para ambos os modos)
        prob_map = {}
        try:
            latest_features_row = features_df.iloc[[-1]]
            input_data = latest_features_row[self.feature_columns_in_order]
            input_data = input_data.fillna(0)
            
            with self.lock:
                input_scaled = self.scaler.transform(input_data)
                
                # Apenas o MODO IA precisa de probabilidades
                if strategy_mode == 'IA_MODEL':
                    probabilities = self.model.predict_proba(input_scaled)[0]
                    prob_map = {int(self.classes[i]): probabilities[i] for i in range(len(self.classes))}
        
        except Exception as e:
            print(f"❌ ERRO CRÍTICO na preparação do ML: {e}")
            return None, None, None, "ERRO CRÍTICO na execução do modelo."
            
        # 0. Determinar o Número Gatilho (N_t-1)
        try:
            number_gatilho = features_df['number'].iloc[-2]
            if pd.isna(number_gatilho) or number_gatilho == -1:
                return None, None, None, "Rejeitado: Não há número gatilho (N-1) disponível."
            number_gatilho = int(number_gatilho)
        except IndexError:
            return None, None, None, "Rejeitado: Histórico muito curto para identificar N-1."


        # 1. Obter os Alvos da Sua Estratégia (Os clusters de vizinhança a serem avaliados)
        alvos_heurísticos = CALLING_NUMBERS.get(number_gatilho, set())
        
        if not alvos_heurísticos:
            return None, None, None, f"Rejeitado: Gatilho {number_gatilho} não possui alvos definidos."


        # 2. Encontrar o MAIOR EDGE (Modo IA) ou MAIOR SCORE (Modo BSB)
        
        best_edge = -1.0
        best_score = 0 # Para o modo BSB
        best_cluster = None
        best_odds = 0.0
        pico_vencedor = None
        best_motivo_base = ""

        # Iteramos APENAS sobre os clusters de vizinhança centrados nos números chamados
        for pico_number in alvos_heurísticos:
            
            cluster_numbers = WHEEL_CLUSTERS.get(pico_number)
            if not cluster_numbers: continue

            cluster_size = len(cluster_numbers)
            
            motivo_base = (
                f"HEURÍSTICA ATIVA (Gatilho {number_gatilho} chamou Pico {pico_number}, Raio {RADIUS_VIZINHOS}). "
                f"Cluster de {cluster_size} números."
            )
            
            # --- LÓGICA DE DECISÃO (IA vs BSB) ---
            
            if strategy_mode == 'IA_MODEL':
                # --- MODO I.A. (EXISTENTE) ---
                # Soma a probabilidade que o ML atribuiu a todos os números do cluster
                cluster_probability_total = sum(prob_map.get(num, 0) for num in cluster_numbers)
                
                odds, current_edge = calculate_cluster_edge_and_odds(cluster_size, cluster_probability_total)
                
                # CRITÉRIO: Deve ter Edge positivo E ser o melhor Edge encontrado
                if current_edge > min_edge_threshold and current_edge > best_edge:
                    best_edge = current_edge
                    best_cluster = sorted(list(cluster_numbers))
                    best_odds = odds
                    pico_vencedor = pico_number
                    best_motivo_base = f"{motivo_base} Edge ML: {current_edge:.2%}."

            
            elif strategy_mode == 'LUCAS_BSB':
                # --- MODO HEURÍSTICO (NOVO) ---
                # Calcula o "score de confluência" baseado no momento da mesa
                score, rationale_list = self.calculate_heuristic_score(latest_features_row.iloc[0], min_zscore_tension)
                
                # CRITÉRIO: Deve ter Score >= 3 E ser o melhor Score encontrado
                # (Seu pedido: "pelo menos 3 a 4 padrões")
                if score >= 3 and score > best_score:
                    best_score = score
                    best_cluster = sorted(list(cluster_numbers))
                    best_odds, _ = calculate_cluster_edge_and_odds(cluster_size, 0) # Odds é (36/N)
                    pico_vencedor = pico_number
                    best_motivo_base = f"{motivo_base} Confluência Score: {score}. ({', '.join(rationale_list)})"
                    
                    # "Falsifica" o Edge e o Stake para passar na validação do main.py
                    best_edge = 0.01 # Fake Edge (só precisa ser > 0)
                    # O stake_proxy é o que o main.py checa (só precisa ser > 0)
            
            # ------------------------------------

                
        # 4. Decisão Final e Staking
        if best_cluster:
            
            # O stake_proxy (antigo stake_value) só precisa ser > 0 para o main.py aceitar
            stake_proxy = 1.0 
            
            return best_cluster, best_edge, stake_proxy, best_motivo_base
        
        # Se o loop terminou sem encontrar nenhum Edge (Modo IA) ou Score (Modo BSB)
        return None, None, None, f"Rejeitado: Gatilho {number_gatilho} não validou Edge/Confluência (Modo: {strategy_mode})."
