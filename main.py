import time
import uvicorn
from fastapi import FastAPI, Query
from threading import Thread, Lock
from contextlib import asynccontextmanager
import pandas as pd
import os
import numpy as np 
from fastapi.middleware.cors import CORSMiddleware 
import re
from pydantic import BaseModel 
from starlette.responses import JSONResponse

# --- CORREÇÃO DE ERRO .env ---
from dotenv import load_dotenv
load_dotenv() 
# -----------------------------

import database as db
import analysis
from scraper import RouletteScraper
from ml_model import TrainedMLModel
# import train_model # <-- REMOVIDO: Não vamos mais treinar a partir daqui
from gemini_client import generate_content 

# --- CONFIGURAÇÃO ---
# RODADAS_ATE_PROXIMO_TREINO = 4 # <-- REMOVIDO
# --------------------

# --- INICIALIZAÇÃO DOS COMPONENTES DO AGENTE ---
db_client = None
scraper = None
ml_logic = None
model_lock = None
# rounds_since_last_train = 0 # <-- REMOVIDO

try:
    db_client = db.get_supabase_client()
    scraper = RouletteScraper() 
    model_lock = Lock() 
    
    if not os.path.exists("ml_models"):
        os.makedirs("ml_models")
        
    ml_logic = TrainedMLModel(model_lock) 
    print("Agente Analista inicializado com sucesso.")
except Exception as e:
    print(f"Erro fatal na inicialização: {e}")
    exit(1)

def log(mensagem):
    """Função de log centralizada."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {mensagem}")

# --- VALIDAÇÃO KELLY STAKING COM STAKE FIXO ---

def validar_e_atualizar_saldo_kelly(latest_result: int):
    log(f"Validando resultado: {latest_result}")
    decisao_pendente = db.get_latest_pending_decision(db_client)
    if not decisao_pendente:
        log("Nenhuma aposta pendente (Kelly).")
        return True 
    config = db.get_config(db_client)
    saldo_atual = float(config['saldo_simulado'])
    dec_id = decisao_pendente['id']
    try:
        stake_investido = float(decisao_pendente['stake_investido'])
        odds_pagamento = float(decisao_pendente['odds_pagamento'])
    except KeyError:
        log(f"AVISO: Aposta {dec_id} é LEGADO (faltam chaves Kelly). Marcando como 'LIMPO' para fechar o ciclo.")
        db.update_decision_status(db_client, dec_id, { 'status': 'LIMPO_LEGADO', 'resultado_financeiro': 0.0, 'numero_vencedor': latest_result })
        return True 
    aposta_sugerida_raw = decisao_pendente['aposta_sugerida']
    if isinstance(aposta_sugerida_raw, str):
        aposta_sugerida_list = aposta_sugerida_raw.replace('{', '').replace('}', '').split(',')
    elif isinstance(aposta_sugerida_raw, list):
        aposta_sugerida_list = aposta_sugerida_raw
    else:
        log("ERRO DE TIPAGEM: Aposta sugerida não é string nem lista. Fechando ciclo.")
        db.update_decision_status(db_client, dec_id, {'status': 'TIPAGEM_FALHA'})
        return True
    aposta_sugerida_int = [int(n.strip()) for n in aposta_sugerida_list if n.strip().isdigit()]
    is_win = latest_result in aposta_sugerida_int
    if is_win:
        lucro_bruto = stake_investido * odds_pagamento
        lucro_liquido = lucro_bruto - stake_investido 
        novo_saldo = saldo_atual + lucro_liquido # Correção de P&L
        status = 'GREEN'
    else:
        lucro_liquido = -stake_investido
        novo_saldo = saldo_atual + lucro_liquido
        status = 'RED'
    db.update_decision_status(db_client, dec_id, { 'status': status, 'resultado_financeiro': float(lucro_liquido), 'numero_vencedor': latest_result })
    db.update_saldo_simulado(db_client, novo_saldo)
    log(f"{status}! Resultado (Líquido): {lucro_liquido:+.2f}. Novo Saldo: {novo_saldo:.2f}")
    return True

def decidir_nova_aposta(config: dict):
    if db.get_latest_pending_decision(db_client):
        log("Já há aposta pendente. Aguardando validação.")
        return
    active_strategy = config.get('active_strategy', 'IA_MODEL') 
    log(f"Buscando oportunidade (Modo: {active_strategy})...")
    history_list = db.get_raw_history(db_client, limit=100)
    saldo_atual = float(config['saldo_simulado'])
    min_edge = float(config.get('min_edge_threshold', 0.001))
    kelly_fraction = float(config.get('kelly_fraction', 0.5))
    min_zscore = float(config.get('min_zscore_tension', 1.0))
    chip_value = float(config.get('chip_value', 0.50))
    if len(history_list) < 37: 
        log(f"Histórico insuficiente para análise complexa ({len(history_list)}/37).")
        return
    features_df = analysis.gerar_features_avancadas(history_list)
    if features_df is None or features_df.empty:
        log("Features insuficientes para análise.")
        return
    
    # (v5.11) A lógica de decisão agora é muito mais inteligente
    aposta, edge, stake_proxy, motivo = ml_logic.predict(
        features_df, 
        bankroll=saldo_atual,
        min_edge_threshold=min_edge,
        kelly_fraction=kelly_fraction,
        min_zscore_tension=min_zscore,
        strategy_mode=active_strategy 
    ) 
    
    if stake_proxy is not None and stake_proxy > 0:
        cluster_size = len(aposta)
        final_stake_value = cluster_size * chip_value
        if final_stake_value > saldo_atual:
            log(f"ML: Stake fixo ({final_stake_value:.2f}) excede o saldo ({saldo_atual:.2f}). Rejeitado.")
            return
        odds_pagamento = 36.0 / cluster_size
        aposta_str_list = [str(n) for n in aposta]
        nova_decisao = {
            'status': 'PENDENTE',
            'aposta_sugerida': '{' + ','.join(aposta_str_list) + '}',
            'motivo_decisao': motivo,
            'edge': float(edge),  
            'stake_investido': float(final_stake_value), 
            'odds_pagamento': float(odds_pagamento),
            'strategy_mode': active_strategy 
        }
        db.insert_new_decision(db_client, nova_decisao)
        # Log detalhado (Ponto 6 do seu prompt)
        log(f"NOVA APOSTA FIXA REGISTRADA (Modo: {active_strategy}):")
        log(f"   Cluster: {aposta}")
        log(f"   Confiança (Edge/Score): {edge*100:.2f}%")
        log(f"   Motivo: {motivo}")
    else:
        log(f"ML (Modo: {active_strategy}): Nenhum Edge/Confluência significativa encontrada nesta rodada.")

def run_agente_analista_cycle():
    """ O CICLO DE AÇÃO RIGOROSO. """
    while True:
        try:
            config = db.get_config(db_client)
            bot_status = config.get('bot_status', 'RUNNING')
            if bot_status != 'RUNNING':
                log("Bot está PAUSADO (via dashboard). Aguardando 10s...")
                time.sleep(10) 
                continue 
            latest_result = scraper.get_latest_result() 
            if latest_result is None:
                time.sleep(5) 
                continue
            db.insert_new_result(db_client, latest_result)
            log(f"Resultado Registrado: {latest_result}")
            validar_e_atualizar_saldo_kelly(latest_result)
            decidir_nova_aposta(config)
            
            # --- LÓGICA DE AUTO-TREINAMENTO REMOVIDA (v4.7) ---
            
        except Exception as e:
            log(f"ERRO CRÍTICO NO CICLO PRINCIPAL: {e}")
            time.sleep(30) 

# --- ENDPOINTS DO FASTAPI PARA O DASHBOARD ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("Iniciando Agente Analista em background...")
    agente_thread = Thread(target=run_agente_analista_cycle, daemon=True)
    agente_thread.start()
    yield
    log("Encerrando scraper...")
    if scraper:
        scraper.close()

app = FastAPI(lifespan=lifespan)

# --- CONFIGURAÇÃO CORS ---
origins = ["*"] 
app.add_middleware( CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"], )

# Modelos Pydantic para receber dados
class BotTogglePayload(BaseModel):
    status: str 
class UserConfigUpdate(BaseModel):
    chip_value: float = None
    kelly_fraction: float = None
    min_edge_threshold: float = None
    active_strategy: str = None

# --- ENDPOINTS DE CONTROLE E SEGURANÇA ---

@app.get("/api/config/frontend")
async def get_frontend_config():
    """ Injeta as chaves públicas (seguras) do .env para o frontend. """
    try:
        public_url = os.environ.get("SUPABASE_URL")
        public_key = os.environ.get("SUPABASE_ANON_KEY") 
        if not public_url or not public_key:
            log("ERRO DE SEGURANÇA: SUPABASE_URL ou SUPABASE_ANON_KEY não encontrados no .env")
            raise ValueError("SUPABASE_URL ou SUPABASE_ANON_KEY não encontrados no .env")
        return JSONResponse({
            "SUPABASE_URL": public_url,
            "SUPABASE_ANON_KEY": public_key
        })
    except Exception as e:
        log(f"Erro ao buscar config do frontend: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/bot/toggle")
async def toggle_bot_status(payload: BotTogglePayload):
    try:
        new_status = payload.status
        if new_status not in ['RUNNING', 'PAUSED']:
            return {"status_code": 400, "message": "Status inválido"}
        db.supabase.table("config").update({"bot_status": new_status}).eq("id", 1).execute()
        log(f"*** STATUS DO BOT ALTERADO PARA: {new_status} ***")
        return {"status_code": 200, "message": f"Bot status set to {new_status}", "new_status": new_status}
    except Exception as e:
        log(f"Erro ao atualizar status do bot: {e}")
        return {"status_code": 500, "message": "Erro interno"}

@app.post("/api/settings/update")
async def update_user_settings(config: UserConfigUpdate):
    try:
        update_data = config.dict(exclude_unset=True)
        if not update_data:
            return {"status_code": 400, "message": "Nenhum dado enviado"}
        db.supabase.table("config").update(update_data).eq("id", 1).execute()
        log(f"Configurações atualizadas pelo usuário: {update_data}")
        return {"status_code": 200, "message": "Configurações atualizadas"}
    except Exception as e:
        log(f"Erro ao atualizar settings: {e}")
        return {"status_code": 500, "message": "Erro interno"}
    
@app.get("/api/kpis")
async def get_kpis(strategy: str = Query("TODAS")):
    try:
        config = db.get_config(db_client)
        decisions_data = db.get_all_decisions(db_client, strategy_filter=strategy) 
        
        # O rounds_since_train não existe mais
        
        if not decisions_data:
             return {
                "saldo_simulado": float(config.get('saldo_simulado', 1000.0)),
                "total_investido": 0.0,
                "p_l": 0.0,
                "roi": 0.0,
                "active_strategy": config.get('active_strategy', 'IA_MODEL'),
                "bot_status": config.get('bot_status', 'RUNNING') 
            }
        df_decisions = pd.DataFrame(decisions_data)
        df_concluido = df_decisions[df_decisions['status'].isin(['GREEN', 'RED'])].copy()
        if df_concluido.empty:
            return {
                "saldo_simulado": float(config.get('saldo_simulado', 1000.0)),
                "total_investido": 0.0,
                "p_l": 0.0,
                "roi": 0.0,
                "active_strategy": config.get('active_strategy', 'IA_MODEL'),
                "bot_status": config.get('bot_status', 'RUNNING')
            }
        total_investido = df_concluido['stake_investido'].astype(float).sum()
        p_l = df_concluido['resultado_financeiro'].astype(float).sum()
        roi = (p_l / total_investido) * 100 if total_investido > 0 else 0
        return {
            "saldo_simulado": float(config['saldo_simulado']),
            "total_investido": round(total_investido, 2),
            "p_l": round(p_l, 2),
            "roi": round(roi, 2),
            "active_strategy": config.get('active_strategy', 'IA_MODEL'),
            "bot_status": config.get('bot_status', 'RUNNING')
        }
    except Exception as e:
        log(f"Erro ao buscar KPIs: {e}")
        return {"status_code": 500, "message": "Erro ao processar dados de KPI"}

@app.get("/api/performance-history")
async def get_performance_history(strategy: str = Query("TODAS")):
    try:
        decisions_data = db.get_all_decisions(db_client, strategy_filter=strategy) 
        if not decisions_data:
            return {"labels": [0], "data": [0.0]}
        df = pd.DataFrame(decisions_data)
        df = df.sort_values(by='timestamp', ascending=True)
        df_concluido = df[df['status'].isin(['GREEN', 'RED'])].copy()
        if df_concluido.empty:
             return {"labels": [0], "data": [0.0]}
        df_concluido['p_l'] = df_concluido['resultado_financeiro'].astype(float)
        df_concluido['p_l_acumulado'] = df_concluido['p_l'].cumsum()
        labels = list(range(1, len(df_concluido) + 1))
        data = df_concluido['p_l_acumulado'].tolist()
        labels.insert(0, 0)
        data.insert(0, 0.0)
        return {"labels": labels, "data": data}
    except Exception as e:
        log(f"Erro ao gerar histórico de performance: {e}")
        return {"labels": [0], "data": [0.0]}

@app.get("/api/analysis/latest")
async def get_latest_analysis():
    try:
        history_list = db.get_raw_history(db_client, limit=100) 
        if len(history_list) < 37:
            return {"status_code": 404, "message": "Histórico insuficiente para Z-Score."}
        features_df = analysis.gerar_features_avancadas(history_list)
        if features_df.empty:
            return {"status_code": 404, "message": "Features não geradas."}
        latest_features_raw = features_df.iloc[-1].to_dict()
        dashboard_data = {}
        for k, v in latest_features_raw.items():
            if k.startswith('zscore_is_') or k.startswith('is_') or k.endswith('_atraso') or k.endswith('_ewma_short'):
                if isinstance(v, (np.float32, np.float64)): dashboard_data[k] = float(v)
                elif isinstance(v, (np.int64)): dashboard_data[k] = int(v)
                else: dashboard_data[k] = v
        return {"status_code": 200, "data": dashboard_data}
    except Exception as e:
        log(f"Erro ao buscar análise mais recente: {e}")
        return {"status_code": 500, "message": "Erro na análise estatística."}

@app.get("/api/triggers/active")
async def get_active_triggers():
    try:
        data = db.get_latest_pending_decision(db_client) 
        return {"status_code": 200, "active_trigger": data}
    except Exception as e:
        log(f"Erro ao buscar triggers: {e}")
        return {"status_code": 500, "message": "Erro ao buscar triggers ativos."}

@app.get("/api/ai/gemini-analysis")
async def get_gemini_analysis():
    try:
        analysis_data_response = await get_latest_analysis()
        if analysis_data_response["status_code"] != 200:
            return {"status_code": 400, "message": "Dados de análise indisponíveis."}
        analysis_data = analysis_data_response["data"]
        prompt_text = "Analise os indicadores estatísticos de roleta (Z-Score=desvio, EWMA=momentum) e forneça um resumo de 3 frases sobre a tendência atual da mesa. Foco nos Z-Scores mais extremos (acima de 2.0 ou abaixo de -2.0) e nos EWMA mais altos. Dados:"
        formatted_data = {k: v for k, v in analysis_data.items() if k.startswith('zscore') or k.endswith('_ewma_short')}
        prompt_text += str(formatted_data) 
        gemini_response = generate_content(prompt_text) 
        return {"status_code": 200, "analysis": gemini_response}
    except Exception as e:
        return {"status_code": 500, "message": f"Erro na chamada Gemini: {e}"}

if __name__ == "__main__":
    log("Iniciando servidor FastAPI/Uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

