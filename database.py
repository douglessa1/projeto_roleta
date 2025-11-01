import os
from supabase import create_client, Client
from dotenv import load_dotenv # Este import é necessário, mas a chamada não.
import pandas as pd

# load_dotenv() # <-- REMOVIDO DAQUI. Será chamado pelo main.py

supabase: Client = None

def get_supabase_client() -> Client:
    """ Cria e retorna o cliente Supabase. """
    global supabase
    if supabase is None:
        # Esta é a forma CORRETA de ler as variáveis do .env
        url: str = os.environ.get("https://ugajvctklhmfvtujecyx.supabase.co")
        key: str = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVnYWp2Y3RrbGhtZnZ0dWplY3l4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTIxNjU0MywiZXhwIjoyMDc2NzkyNTQzfQ.p77XT7dtwXmUI0hnYkzhcb5wdI8LcZ0KTt22SDzbLok")
        if not url or not key:
            raise EnvironmentError("SUPABASE_URL e SUPABASE_KEY são obrigatórios no .env")
        supabase = create_client(url, key)
    return supabase

def get_config(client: Client):
# ... (O restante do arquivo database.py continua exatamente como a v4.4) ...
    """ Busca a configuração, incluindo o status do bot e a estratégia ativa. """
    try:
        data = client.table("config").select("*").eq("id", 1).single().execute()
        config_data = data.data
        
        # Garante que os campos críticos tenham valores padrao robustos
        config_data['saldo_simulado'] = float(config_data.get('saldo_simulado', 1000.00))
        config_data['min_edge_threshold'] = float(config_data.get('min_edge_threshold', 0.001)) 
        config_data['kelly_fraction'] = float(config_data.get('kelly_fraction', 0.5))
        config_data['min_zscore_tension'] = float(config_data.get('min_zscore_tension', 1.0))
        config_data['chip_value'] = float(config_data.get('chip_value', 0.50))
        config_data['bot_status'] = config_data.get('bot_status', 'RUNNING') # NOVO
        config_data['active_strategy'] = config_data.get('active_strategy', 'IA_MODEL') # NOVO
        
        return config_data
    except Exception as e:
        print(f"Erro ao buscar config (Usando Defaults): {e}")
        # Retorna um dicionário de defaults se o acesso falhar
        return {
            'saldo_simulado': 1000.00,
            'min_edge_threshold': 0.001,
            'kelly_fraction': 0.5,
            'min_zscore_tension': 1.0,
            'chip_value': 0.50,
            'bot_status': 'RUNNING',
            'active_strategy': 'IA_MODEL'
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
    Busca todas as decisões para o cálculo de KPIs, 
    OPCIONALMENTE filtrando por estratégia.
    """
    try:
        # CORREÇÃO: Seleciona todas as colunas necessárias
        query = client.table("decisions").select(
            "stake_investido, resultado_financeiro, status, timestamp, strategy_mode"
        )
        
        # NOVO: Adiciona o filtro de estratégia se ele for fornecido e não for "TODAS"
        if strategy_filter and strategy_filter != "TODAS":
            query = query.eq('strategy_mode', strategy_filter)
            
        data = query.order("timestamp", desc=True).execute()
        return data.data
    except Exception as e:
        print(f"Erro ao buscar todas as decisões: {e}")
        return []

def insert_new_result(client: Client, number: int):
    """ Insere o último resultado capturado. """
    try:
        client.table("raw_results").insert({"number": number}).execute()
    except Exception as e:
        print(f"Erro ao inserir resultado bruto: {e}")
        
def insert_new_decision(client: Client, decision_data: dict):
    """ Registra uma nova decisão de aposta (Kelly/Edge). """
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

