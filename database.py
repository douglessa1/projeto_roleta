# douglessa1/projeto_roleta/projeto_roleta-4eb8af59f00aad63289b5a75b94bcc4e4e852c83/database.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import pandas as pd

# load_dotenv() # Removido. O main.py carrega o .env

supabase: Client = None

def get_supabase_client() -> Client:
    """ Cria e retorna o cliente Supabase. """
    global supabase
    if supabase is None:
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise EnvironmentError("SUPABASE_URL e SUPABASE_KEY são obrigatórios no .env")
        supabase = create_client(url, key)
    return supabase

def get_config(client: Client):
    """ Busca a configuração, incluindo o status do bot e a estratégia ativa. """
    try:
        # --- CORREÇÃO DE CACHE ---
        data = client.table("config").select("*").eq("id", 1).limit(1).execute()
        
        if data.data:
            config_data = data.data[0]
        else:
            raise Exception("Tabela 'config' está vazia ou inacessível.")
        # --- FIM DA CORREÇÃO ---
            
        # Garante que os campos críticos tenham valores padrao robustos
        config_data['saldo_simulado'] = float(config_data.get('saldo_simulado', 1000.00))
        config_data['min_edge_threshold'] = float(config_data.get('min_edge_threshold', 0.001)) 
        config_data['kelly_fraction'] = float(config_data.get('kelly_fraction', 0.5))
        config_data['min_zscore_tension'] = float(config_data.get('min_zscore_tension', 1.0))
        config_data['chip_value'] = float(config_data.get('chip_value', 0.50))
        config_data['bot_status'] = config_data.get('bot_status', 'RUNNING') 
        # 'active_strategy' será removido do 'predict', mas mantido aqui por enquanto
        config_data['active_strategy'] = config_data.get('active_strategy', 'IA_MODEL') 
        
        # NOVO (v6.0): Thresholds Dinâmicos (lidos do DB)
        config_data['base_threshold'] = float(config_data.get('base_threshold', 2.0))
        
        return config_data
        
    except Exception as e:
        print(f"Erro ao buscar config (Usando Defaults): {e}")
        return {
            'saldo_simulado': 1000.00,
            'min_edge_threshold': 0.001,
            'kelly_fraction': 0.5,
            'min_zscore_tension': 1.0,
            'chip_value': 0.50,
            'bot_status': 'RUNNING',
            'active_strategy': 'IA_MODEL',
            'base_threshold': 2.0
        }

def get_latest_pending_decision(client: Client):
    """ Busca a última aposta que ainda está PENDENTE. """
    try:
        data = client.table("decisions").select("*").eq("status", "PENDENTE").order("timestamp", desc=True).limit(1).execute()
        if data.data:
            return data.data[0]
        return None
    except Exception as e:
        print(f"Erro ao buscar decisão pendente: {e}")
        return None

def get_raw_history(client: Client, limit=2000):
    """ Busca os últimos N resultados brutos. """
    try:
        data = client.table("raw_results").select("number").order("timestamp", desc=True).limit(limit).execute()
        return [item['number'] for item in reversed(data.data)] # Inverte para o mais antigo primeiro
    except Exception as e:
        print(f"Erro ao buscar histórico bruto: {e}")
        return []

def get_all_decisions(client: Client, strategy_filter: str = "TODAS"):
    """ 
    Busca todas as decisões para o cálculo de KPIs.
    O filtro de estratégia é mantido para compatibilidade com o dashboard,
    mas a lógica v6.0 não usa mais 'strategy_mode'.
    """
    try:
        query = client.table("decisions").select(
            "stake_investido, resultado_financeiro, status, timestamp, strategy_mode"
        )
        
        # O dashboard ainda pode tentar filtrar por 'IA_MODEL' ou 'LUCAS_BSB'
        # Vamos manter a compatibilidade
        if strategy_filter and strategy_filter != "TODAS":
            # Se a nova lógica não gravar 'strategy_mode', filtramos por 'edge' > 0
            if strategy_filter == 'IA_MODEL':
                 query = query.gt('edge', 0) # Assumindo que BSB tem edge=0
            # Se o filtro for BSB, pode não retornar nada se o campo não existir mais
            # O ideal é refatorar o dashboard, mas por hora mantemos
            else:
                 query = query.eq('strategy_mode', strategy_filter) 
            
        data = query.order("timestamp", desc=True).execute()
        return data.data
    except Exception as e:
        print(f"Erro ao buscar todas as decisões: {e}")
        return []

def get_performance_history(client: Client, limit=1000):
    """ (NOVO v6.0) Busca o histórico de performance para o AdaptiveSystem. """
    try:
        data = client.table("performance_log").select(
            "success, hybrid_confidence, p_l"
        ).order("timestamp", desc=True).limit(limit).execute()
        return data.data
    except Exception as e:
        print(f"Erro ao buscar histórico de performance: {e}")
        return []


def insert_new_result(client: Client, number: int):
    """ Insere o último resultado capturado. """
    try:
        client.table("raw_results").insert({"number": number}).execute()
    except Exception as e:
        print(f"Erro ao inserir resultado bruto: {e}")
        
def insert_new_decision(client: Client, decision_data: dict):
    """ Registra uma nova decisão de aposta (Híbrida). """
    try:
        client.table("decisions").insert(decision_data).execute()
    except Exception as e:
        print(f"Erro ao inserir nova decisão: {e}")

def update_decision_status(client: Client, decision_id: int, update_data: dict):
    """ Atualiza uma decisão PENDENTE para GREEN ou RED. """
    try:
        client.table("decisions").update(update_data).eq("id", decision_id).execute()
    except Exception as e:
        print(f"Erro ao atualizar decisão: {e}")

def update_saldo_simulado(client: Client, novo_saldo: float):
    """ Atualiza o saldo simulado na tabela config. """
    try:
        client.table("config").update({"saldo_simulado": novo_saldo}).eq("id", 1).execute()
    except Exception as e:
        print(f"Erro ao atualizar saldo: {e}")

# NOVO (v6.0 - Req 6): Log de Métricas de Performance
def insert_performance_log(client: Client, log_data: dict):
    """
    Insere um registro na tabela 'performance_log' para
    análise de performance e aprendizado adaptativo.
    """
    try:
        # Garante que os campos Padrão (not-null) existam
        log_data.setdefault('success', False)
        log_data.setdefault('hybrid_confidence', 0.0)
        log_data.setdefault('p_l', 0.0)
        client.table("performance_log").insert(log_data).execute()
    except Exception as e:
        print(f"Erro ao inserir performance log: {e}")

def update_adaptive_thresholds(client: Client, new_base_threshold: float, new_recent_performance: float):
    """ (NOVO v6.0) Salva os thresholds recalibrados no DB. """
    try:
        update_data = {
            "base_threshold": new_base_threshold,
            "bot_recent_performance": new_recent_performance
        }
        client.table("config").update(update_data).eq("id", 1).execute()
        print(f"ADAPTIVE: Limiares atualizados no DB -> Base: {new_base_threshold:.2f}, Perf: {new_recent_performance:.2f}")
    except Exception as e:
        print(f"Erro ao salvar thresholds adaptativos: {e}")