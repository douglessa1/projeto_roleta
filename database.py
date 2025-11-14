import os
from supabase import create_client, Client
from dotenv import load_dotenv
import pandas as pd

# ----------------------------------------------------
# ⚠️ VARIÁVEL TEMPORÁRIA: Este valor deve ser obtido do token JWT após o login (FUTURO)
DEFAULT_USER_ID = 1 
# ----------------------------------------------------

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

def get_config(client: Client, user_id: int = DEFAULT_USER_ID):
    """ Busca a configuração, incluindo o status do bot e a estratégia ativa. """
    try:
        # FILTRO POR USER_ID
        data = client.table("config_v2").select("*").eq("id", 1).eq("user_id", user_id).limit(1).execute()
        
        if data.data:
            config_data = data.data[0]
        else:
            # Se não houver config para o user_id, use defaults robustos, mas alerte.
            print(f"⚠️ AVISO: Nenhuma configuração encontrada para user_id {user_id}. Usando defaults iniciais.")
            return get_default_config()

        config_data['saldo_simulado'] = float(config_data.get('saldo_simulado') or 1000.00)
        config_data['min_edge_threshold'] = float(config_data.get('min_edge_threshold') or 0.001)
        config_data['kelly_fraction'] = float(config_data.get('kelly_fraction') or 0.5)
        config_data['min_zscore_tension'] = float(config_data.get('min_zscore_tension') or 1.0)
        config_data['chip_value'] = float(config_data.get('chip_value') or 0.50)
        
        active_strategy = config_data.get('active_strategy', 'IA_MODEL')
        if active_strategy not in ['IA_MODEL', 'LUCAS_BSB']:
            print(f"⚠️ AVISO: Modo de estratégia '{active_strategy}' é inválido. Usando 'IA_MODEL' como fallback.")
            config_data['active_strategy'] = 'IA_MODEL'
        else:
            config_data['active_strategy'] = active_strategy
        
        config_data['bot_status'] = config_data.get('bot_status', 'RUNNING') 
        
        return config_data
        
    except Exception as e:
        print(f"Erro ao buscar config (Usando Defaults): {e}")
        return get_default_config()

def get_default_config():
     return {
        'saldo_simulado': 1000.00,
        'min_edge_threshold': 0.001,
        'kelly_fraction': 0.5,
        'min_zscore_tension': 1.0,
        'chip_value': 0.50,
        'bot_status': 'RUNNING',
        'active_strategy': 'IA_MODEL'
    }

def get_latest_pending_decision(client: Client, user_id: int = DEFAULT_USER_ID):
    """ Busca a última aposta que ainda está PENDENTE. """
    try:
        # FILTRO POR USER_ID
        data = client.table("decisions_v2").select("*").eq("status", "PENDENTE").eq("user_id", user_id).order("timestamp", desc=True).limit(1).execute()
        if data.data:
            return data.data[0]
        return None
    except Exception as e:
        print(f"Erro ao buscar decisão pendente: {e}")
        return None

def get_raw_history(client: Client, limit=100, user_id: int = DEFAULT_USER_ID):
    """ Busca os últimos N resultados brutos. """
    try:
        # FILTRO POR USER_ID
        data = client.table("raw_results_v2").select("number").eq("user_id", user_id).order("timestamp", desc=True).limit(limit).execute()
        return [item['number'] for item in reversed(data.data)]
    except Exception as e:
        print(f"Erro ao buscar histórico bruto: {e}")
        return []

def get_all_decisions(client: Client, strategy_filter: str = "TODAS", user_id: int = DEFAULT_USER_ID):
    """ 
    Busca todas as decisões para o cálculo de KPIs, 
    OPCIONALMENTE filtrando por estratégia.
    """
    try:
        query = client.table("decisions_v2").select(
            "stake_investido, resultado_financeiro, status, timestamp, strategy_mode"
        ).eq("user_id", user_id) # FILTRO POR USER_ID
        
        if strategy_filter and strategy_filter != "TODAS":
            query = query.eq('strategy_mode', strategy_filter)
            
        data = query.order("timestamp", desc=True).execute()
        return data.data
    except Exception as e:
        print(f"Erro ao buscar todas as decisões: {e}")
        return []

def insert_new_result(client: Client, number: int, user_id: int = DEFAULT_USER_ID):
    """ Insere o último resultado capturado. """
    try:
        # INSERE O USER_ID JUNTO
        client.table("raw_results_v2").insert({"number": number, "user_id": user_id}).execute()
    except Exception as e:
        print(f"Erro ao inserir resultado bruto: {e}")
        
def insert_new_decision(client: Client, decision_data: dict, user_id: int = DEFAULT_USER_ID):
    """ Registra uma nova decisão de aposta (Kelly/Edge). """
    try:
        # INSERE O USER_ID JUNTO
        decision_data['user_id'] = user_id
        client.table("decisions_v2").insert(decision_data).execute()
    except Exception as e:
        print(f"Erro ao inserir nova decisão: {e}")

def update_decision_status(client: Client, decision_id: int, update_data: dict, user_id: int = DEFAULT_USER_ID):
    """ Atualiza uma decisão PENDENTE para GREEN ou RED. """
    try:
        # FILTRO POR USER_ID
        client.table("decisions_v2").update(update_data).eq("id", decision_id).eq("user_id", user_id).execute()
    except Exception as e:
        print(f"Erro ao atualizar decisão: {e}")

def update_saldo_simulado(client: Client, novo_saldo: float, user_id: int = DEFAULT_USER_ID):
    """ Atualiza o saldo simulado na tabela config. """
    try:
        # FILTRO POR USER_ID
        client.table("config_v2").update({"saldo_simulado": novo_saldo}).eq("id", 1).eq("user_id", user_id).execute()
    except Exception as e:
        print(f"Erro ao atualizar saldo: {e}")