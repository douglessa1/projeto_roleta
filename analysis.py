import pandas as pd
import numpy as np

# --- CONSTANTES GLOBAIS DA ROLETA ---
VERMELHOS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
PRETOS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
TOTAL_NUMBERS = 37 # 0 a 36

# Mapeamento da Ordem Física da Roleta
WHEEL_ORDER = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5,
    24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]
WHEEL_POSITIONS = {number: i for i, number in enumerate(WHEEL_ORDER)}

# Grupos Customizados
GRUPOS_CUSTOMIZADOS = {
    '1C_IC_V': {9, 12}, '1C_IC_P': {8, 10, 11},
    '1C_S_V': {1, 3, 5, 7}, '1C_S_P': {2, 4, 6},
    '2C_IC_V': {18, 21, 16, 19}, '2C_IC_P': {13, 17, 20},
    '2C_S_V': {14, 23}, '2C_S_P': {15, 22, 24},
    '3C_IC_V': {27, 30}, '3C_IC_P': {26, 28, 29, 31},
    '3C_S_V': {25, 32, 34, 36}, '3C_S_P': {33, 35},
}

# --- DICIONÁRIO DE REGRAS HEURÍSTICAS DE CHAMADA (Gatilhos BSB) ---
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

# --- FUNÇÕES DE UTILIADADE (v5.3 - Tempo Real) ---

def calcular_sequencia_consecutiva(series: pd.Series) -> pd.Series:
    """Calcula a contagem de ocorrências consecutivas ATUAIS (Tempo Real)."""
    if series.empty or series.isnull().all():
        return pd.Series(0, index=series.index)
    series_filled = series.fillna(-1)
    seq = series_filled.groupby((series_filled != series_filled.shift()).cumsum()).cumcount() + 1
    # Retorna o dado ATUAL (sem shift)
    return seq.fillna(0).reindex(series.index)

def calcular_atraso(series: pd.Series) -> pd.Series:
    """Calcula há quantas rodadas um evento ocorreu pela última vez (Tempo Real)."""
    s = series.copy().fillna(0)
    ocorrencias = s.where(s == 1)
    idx_series = pd.Series(range(len(s)), index=s.index)
    last_occurrence_idx = idx_series.where(ocorrencias == 1).ffill().fillna(-1)
    atraso = idx_series - last_occurrence_idx
    # Retorna o dado ATUAL (sem shift)
    final_atraso = atraso.fillna(len(series)).astype(int)
    return final_atraso.reindex(series.index)

def calcular_z_score(series: pd.Series, expected_prob: float, window: int = 37) -> pd.Series:
    """Calcula o Z-score de desvio da frequência esperada para a janela especificada."""
    series_filled = series.fillna(0)
    counts = series_filled.rolling(window=window, min_periods=window).sum()
    expected_mean = expected_prob * window
    expected_std = np.sqrt(window * expected_prob * (1 - expected_prob))
    if expected_std < 1e-6: 
        return pd.Series(np.where(counts.isna(), np.nan, 0), index=series.index) 
    z_score = (counts - expected_mean) / expected_std
    return z_score.fillna(0).reindex(series.index) 

def get_physical_neighbors(number: int, radius: int = 1) -> set:
    """Retorna o conjunto de números que são vizinhos físicos de 'number' no volante."""
    if pd.isna(number) or number not in WHEEL_POSITIONS:
        return set()
    try:
        num_int = int(number)
        pos = WHEEL_POSITIONS[num_int]
    except (ValueError, KeyError):
        return set()
    neighbors = set()
    for i in range(-radius, radius + 1):
        if i == 0: continue 
        index = (pos + i) % TOTAL_NUMBERS
        neighbors.add(WHEEL_ORDER[index])
    return neighbors

# --- FUNÇÃO AUXILIAR PARA HEURÍSTICA (Gatilho BSB) ---
# CORREÇÃO v5.3.1: Re-adicionando a função que faltava
def check_heuristic_call_lagged(current_num, previous_num):
    """ Checa se N_t (current) foi um alvo chamado por N_t-1 (previous) """
    if pd.isna(current_num) or previous_num == -1:
        return 0
    
    current_num_int = int(current_num)
    previous_num_int = int(previous_num)
    
    called_targets = CALLING_NUMBERS.get(previous_num_int, set())
    if current_num_int in called_targets:
        return 1
            
    if previous_num_int in {13, 17, 24, 29} and current_num_int == 0:
        return 1
    if previous_num_int in {6, 34} and current_num_int in {6, 34}:
        return 1
        
    return 0

# --- GERAÇÃO PRINCIPAL DE FEATURES (v5.3) ---

def gerar_features_avancadas(history_list: list) -> pd.DataFrame:
    
    min_len = 37 
    if len(history_list) < min_len:
        print(f"  (Analysis): Histórico ({len(history_list)}) menor que o mínimo ({min_len}) para features avançadas.")
        return pd.DataFrame() 

    history_df = pd.DataFrame(history_list, columns=['number']).reset_index(drop=True) 
    
    # N_t (Número Atual)
    N_t = history_df['number']
    N_t_1 = history_df['number'].shift(1).fillna(-1).astype(int)
    
    new_features = {}

    # --- 1. FEATURES BÁSICAS E CATEGÓRICAS (USANDO N_t) ---
    
    # Padrões de Números Individuais (para Momentum)
    for i in range(TOTAL_NUMBERS):
        new_features[f'is_num_{i}'] = N_t.apply(lambda x: 1 if x == i else 0)

    new_features['is_red'] = N_t.apply(lambda x: 1 if x in VERMELHOS else 0)
    new_features['is_black'] = N_t.apply(lambda x: 1 if x in PRETOS else 0)
    new_features['is_zero'] = N_t.apply(lambda x: 1 if x == 0 else 0)
    new_features['is_low'] = N_t.apply(lambda x: 1 if 1 <= x <= 18 else 0)
    new_features['is_high'] = N_t.apply(lambda x: 1 if 19 <= x <= 36 else 0) 
    new_features['is_even'] = N_t.apply(lambda x: 1 if pd.notna(x) and x != 0 and x % 2 == 0 else 0)
    new_features['is_odd'] = N_t.apply(lambda x: 1 if pd.notna(x) and x % 2 != 0 else 0)
    new_features['is_d1'] = N_t.apply(lambda x: 1 if 1 <= x <= 12 else 0)
    new_features['is_d2'] = N_t.apply(lambda x: 1 if 13 <= x <= 24 else 0)
    new_features['is_d3'] = N_t.apply(lambda x: 1 if 25 <= x <= 36 else 0)
    new_features['is_c1'] = N_t.apply(lambda x: 1 if pd.notna(x) and x != 0 and x % 3 == 1 else 0)
    new_features['is_c2'] = N_t.apply(lambda x: 1 if pd.notna(x) and x != 0 and x % 3 == 2 else 0)
    new_features['is_c3'] = N_t.apply(lambda x: 1 if pd.notna(x) and x != 0 and x % 3 == 0 else 0)
    
    new_features['terminal_digit'] = N_t.apply(lambda x: x % 10 if pd.notna(x) else -1)
    
    for nome_feature, numeros_set in GRUPOS_CUSTOMIZADOS.items():
        new_features[f'is_{nome_feature}'] = N_t.apply(lambda x: 1 if pd.notna(x) and x in numeros_set else 0)

    # --- 2. FEATURES DE HEURÍSTICA EM TEMPO REAL (v5.3) ---
    
    N_t_2 = history_df['number'].shift(2).fillna(-1)
    T_t = new_features['terminal_digit']
    T_t_1 = T_t.shift(1).fillna(-1)
    T_t_2 = T_t.shift(2).fillna(-1)

    # NOVO (v5.3): O padrão é se o *número atual* (T_t) é a soma dos dois anteriores
    T_sum_mod_10 = (T_t_1 + T_t_2).apply(lambda x: x % 10 if x >= 0 else -1)
    is_terminal_sum_pattern = (T_t == T_sum_mod_10).astype(int)
    is_terminal_sum_pattern[(T_t_1 == -1) | (T_t_2 == -1)] = 0
    new_features['current_is_terminal_sum'] = is_terminal_sum_pattern # Renomeado

    # NOVO (v5.3): Verifica se o número *atual* (N_t) é um gatilho BSB
    new_features['current_is_bsb_trigger'] = N_t.apply(lambda x: 1 if pd.notna(x) and int(x) in CALLING_NUMBERS else 0)
    
    # NOVO (v5.3): Verifica se o número *atual* (N_t) foi chamado pelo *anterior* (N_t_1)
    heuristic_data = pd.DataFrame({'current_num': N_t, 'previous_num': N_t_1})
    def apply_heuristic_call_simple(row):
        # A função check_heuristic_call_lagged agora está definida fora
        return check_heuristic_call_lagged(row['current_num'], row['previous_num'])
    new_features['was_called_by_previous'] = heuristic_data.apply(apply_heuristic_call_simple, axis=1) # Renomeado


    # --- 3. FEATURES DE MOMENTUM, ATRASO, Z-SCORE (EM TEMPO REAL) ---
    
    binary_features = [col for col in new_features if col.startswith('is_')]
    alpha_ewma_short = 0.5 
    alpha_ewma_medium = 0.2 

    for feature in binary_features:
        series = pd.Series(new_features[feature], index=history_df.index)
        
        if not series.empty:
            # CORREÇÃO v5.2: Usa as funções de Tempo Real (sem shift)
            new_features[f'{feature}_seq_ocorrencia'] = calcular_sequencia_consecutiva(series) * series
            new_features[f'{feature}_seq_nao_ocorrencia'] = calcular_sequencia_consecutiva(1 - series) * (1 - series)
            new_features[f'{feature}_atraso'] = calcular_atraso(series)
            new_features[f'{feature}_ewma_short'] = series.ewm(alpha=alpha_ewma_short, adjust=False).mean().fillna(0)
            new_features[f'{feature}_ewma_medium'] = series.ewm(alpha=alpha_ewma_medium, adjust=False).mean().fillna(0)

    # Z-Score
    features_para_zscore = {}
    for nome_grupo, grupo_set in GRUPOS_CUSTOMIZADOS.items():
         prob = len(grupo_set) / 37
         features_para_zscore[f'is_{nome_grupo}'] = prob
    for i in range(TOTAL_NUMBERS):
        features_para_zscore[f'is_num_{i}'] = 1/37
    
    # Adiciona as probabilidades para os binários simples
    simple_binary_probs = {
        'is_red': 18/37, 'is_black': 18/37, 'is_low': 18/37, 'is_high': 18/37,
        'is_even': 18/37, 'is_odd': 18/37, 'is_d1': 12/37, 'is_d2': 12/37, 'is_d3': 12/37, 
        'is_c1': 12/37, 'is_c2': 12/37, 'is_c3': 12/37
    }
    features_para_zscore.update(simple_binary_probs)
         
    for feature, prob in features_para_zscore.items():
        if feature in new_features:
            new_features[f'zscore_{feature}'] = calcular_z_score(pd.Series(new_features[feature], index=history_df.index), prob)

    # --- 4. FINALIZAÇÃO ---
    
    # Remove a coluna temporária 'terminal_digit'
    if 'terminal_digit' in new_features:
        del new_features['terminal_digit']
        
    history_df_cleaned = history_df.copy()

    features_df_to_concat = pd.DataFrame(new_features, index=history_df.index)
    history_df = pd.concat([history_df_cleaned, features_df_to_concat], axis=1)

    # Filtra as linhas iniciais que não têm Lags/Z-Scores completos.
    final_df = history_df.iloc[min_len - 1:].reset_index(drop=True).dropna(axis=0)
    
    final_df = final_df.copy() 
    
    print(f"  (Analysis v5.3.1 - Tempo Real): Features geradas. Shape final: {final_df.shape}")
    return final_df