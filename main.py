# douglessa1/projeto_roleta/projeto_roleta-4eb8af59f00aad63289b5a75b94bcc4e4e852c83/main.py
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
from gemini_client import generate_content 

# --- CORREÇÃO (v6.0.1): Importa a classe do novo arquivo ---
from adaptive_system import AdaptiveThresholdSystem
# --------------------------------------------------------


# --- INICIALIZAÇÃO DOS COMPONENTES DO AGENTE ---
db_client = None
scraper = None
ml_logic = None
model_lock = None
adaptive_system = None # (v6.0)

try:
    db_client = db.get_supabase_client()
    scraper = RouletteScraper() 
    model_lock = Lock() 
    
    if not os.path.exists("ml_models"):
        os.makedirs("ml_models")
        
    ml_logic = TrainedMLModel(model_lock) 
    
    # (v6.0) Inicializa o sistema adaptativo
    adaptive_system = AdaptiveThresholdSystem(db_client) 
    
    print("Agente Analista (Híbrido v6.0) inicializado com sucesso.")
except Exception as e:
    print(f"Erro fatal na inicialização: {e}")
    exit(1)

def log(mensagem):
    """Função de log centralizada."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {mensagem}")

# --- VALIDAÇÃO E LOG (v6.0) ---

def validar_e_atualizar_saldo_kelly(latest_result: int):
    log(f"Validando resultado: {latest_result}")
    decisao_pendente = db.get_latest_pending_decision(db_client)
    if not decisao_pendente:
        log("Nenhuma aposta pendente.")
        return True 
        
    config = db.get_config(db_client)
    saldo_atual = float(config['saldo_simulado'])
    dec_id = decisao_pendente['id']
    
    try:
        stake_investido = float(decisao_pendente['stake_investido'])
        odds_pagamento = float(decisao_pendente['odds_pagamento'])
        aposta_sugerida_raw = decisao_pendente['aposta_sugerida']
        
        # (v6.0) Pega os dados de log
        hybrid_conf = float(decisao_pendente.get('hybrid_confidence', 0.0))
        edge = float(decisao_pendente.get('edge', 0.0))
        heuristic_score = float(decisao_pendente.get('heuristic_score', 0.0))
        dynamic_threshold = float(decisao_pendente.get('dynamic_threshold', 0.0))

    except Exception as e:
        log(f"AVISO: Aposta {dec_id} é LEGADO ou dados inválidos ({e}). Marcando como 'LIMPO' para fechar o ciclo.")
        db.update_decision_status(db_client, dec_id, { 'status': 'LIMPO_LEGADO', 'resultado_financeiro': 0.0, 'numero_vencedor': latest_result })
        return True 

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
        novo_saldo = saldo_atual + lucro_liquido
        status = 'GREEN'
    else:
        lucro_liquido = -stake_investido
        novo_saldo = saldo_atual + lucro_liquido
        status = 'RED'
        
    # Atualiza a aposta
    db.update_decision_status(db_client, dec_id, { 'status': status, 'resultado_financeiro': float(lucro_liquido), 'numero_vencedor': latest_result })
    db.update_saldo_simulado(db_client, novo_saldo)
    
    log(f"{status}! Resultado (Líquido): {lucro_liquido:+.2f}. Novo Saldo: {novo_saldo:.2f}")

    # --- NOVO (v6.0 - Req 5 & 6): Atualiza o Sistema Adaptativo e Loga a Performance ---
    try:
        # 1. Atualiza o sistema adaptativo
        adaptive_system.update_performance(is_win, hybrid_conf, lucro_liquido)
        
        # 2. Prepara o log de performance
        log_data = {
            'decision_id': dec_id,
            'success': is_win,
            'p_l': float(lucro_liquido),
            'hybrid_confidence': hybrid_conf,
            'dynamic_threshold': dynamic_threshold,
            'edge_ml': edge,
            'heuristic_score': heuristic_score,
            'recent_bot_performance': adaptive_system.get_recent_performance(),
            'stake': stake_investido,
            'odds': odds_pagamento
        }
        
        # 3. Insere o log no DB
        db.insert_performance_log(db_client, log_data)
        log(f"PERFORMANCE LOG: Registrado (Conf: {hybrid_conf*100:.1f}%)")
        
    except Exception as e:
        log(f"ERRO no Log de Performance: {e}")

    return True

# --- DECISÃO DE APOSTA (v6.0 - Híbrido) ---
def decidir_nova_aposta(config: dict):
    if db.get_latest_pending_decision(db_client):
        log("Já há aposta pendente. Aguardando validação.")
        return

    log(f"Buscando oportunidade (Modo: Híbrido v6.0)...")
    history_list = db.get_raw_history(db_client, limit=100)
    saldo_atual = float(config['saldo_simulado'])
    chip_value = float(config.get('chip_value', 0.50))

    if len(history_list) < 37: 
        log(f"Histórico insuficiente para análise complexa ({len(history_list)}/37).")
        return
        
    features_df = analysis.gerar_features_avancadas(history_list)
    if features_df is None or features_df.empty:
        log("Features insuficientes para análise.")
        return
    
    # (v6.0) A lógica de decisão agora é unificada e usa o sistema adaptativo
    aposta, edge, hybrid_conf, dyn_thresh, score_h, motivo = ml_logic.predict(
        features_df, 
        adaptive_system=adaptive_system,
        config=config
    ) 
    
    if aposta is not None:
        # Aposta foi aprovada pela lógica híbrida
        cluster_size = len(aposta)
        final_stake_value = cluster_size * chip_value # Usando stake fixo
        
        if final_stake_value > saldo_atual:
            log(f"APOSTA REJEITADA: Stake fixo ({final_stake_value:.2f}) excede o saldo ({saldo_atual:.2f}).")
            return
            
        odds_pagamento = 36.0 / cluster_size
        aposta_str_list = [str(n) for n in aposta]
        
        nova_decisao = {
            'status': 'PENDENTE',
            'aposta_sugerida': '{' + ','.join(aposta_str_list) + '}',
            'motivo_decisao': motivo,
            
            # (v6.0) Campos de Log
            'edge': float(edge),
            'hybrid_confidence': float(hybrid_conf),
            'dynamic_threshold': float(dyn_thresh),
            'heuristic_score': float(score_h),
            
            'stake_investido': float(final_stake_value), 
            'odds_pagamento': float(odds_pagamento),
            'strategy_mode': 'HIBRIDO_V6' # Novo modo
        }
        
        db.insert_new_decision(db_client, nova_decisao)
        
        log(f"NOVA APOSTA (Híbrida v6.0) REGISTRADA:")
        log(f"   Cluster: {aposta}")
        log(f"   Confiança: {hybrid_conf*100:.1f}% (Threshold: {dyn_thresh*10:.1f}%)")
        log(f"   Stake: R$ {final_stake_value:.2f}")
        log(f"   Motivo: {motivo}")
    else:
        # Aposta foi rejeitada pela lógica
        log(f"Nenhuma oportunidade (Híbrido v6.0): {motivo}")

def run_agente_analista_cycle():
    """ O CICLO DE AÇÃO RIGOROSO (v6.0). """
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
            
            # 1. Valida a aposta anterior (e loga a performance)
            validar_e_atualizar_saldo_kelly(latest_result)
            
            # 2. Decide a próxima aposta (usando o sistema híbrido)
            decidir_nova_aposta(config)
            
        except Exception as e:
            log(f"ERRO CRÍTICO NO CICLO PRINCIPAL: {e}")
            time.sleep(30) 

# --- ENDPOINTS DO FASTAPI (Sem mudanças significativas, exceto remoção de 'train') ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("Iniciando Agente Analista (Híbrido v6.0) em background...")
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
    active_strategy: str = None # Mantido para o dashboard
    base_threshold: float = None # (v6.0)

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
        
        # (v6.0) Atualiza o threshold base se enviado
        if 'base_threshold' in update_data and adaptive_system:
            # Pega a performance recente atual para salvar junto
            config_db = db.get_config(db_client)
            db.update_adaptive_thresholds(
                db_client, 
                update_data['base_threshold'], 
                config_db.get('bot_recent_performance', 0.5)
            )
            
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
        
        # (v6.0) Pega a performance recente do sistema adaptativo
        recent_performance = adaptive_system.get_recent_performance()
        
        if not decisions_data:
             return {
                "saldo_simulado": float(config.get('saldo_simulado', 1000.0)),
                "total_investido": 0.0,
                "p_l": 0.0,
                "roi": 0.0,
                # (v6.0) Usamos a performance como 'rounds_since_train'
                "rounds_since_train": round(recent_performance * 100), 
                "active_strategy": "HIBRIDO_V6", # Modo é fixo agora
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
                "rounds_since_train": round(recent_performance * 100),
                "active_strategy": "HIBRIDO_V6",
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
            "rounds_since_train": round(recent_performance * 100), # (v6.0) Retorna % de acerto
            "active_strategy": "HIBRIDO_V6",
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
        
        # (v6.0) Adiciona os novos campos de confluência
        dashboard_data = {}
        for k, v in latest_features_raw.items():
            if k.startswith('zscore_') or k.startswith('is_') or k.endswith('_atraso') or k.endswith('_ewma_short') or k.startswith('signal_') or k == 'feature_confluence_count':
                if isinstance(v, (np.float32, np.float64)): dashboard_data[k] = float(v)
                elif isinstance(v, (np.int64, np.bool_)): dashboard_data[k] = int(v)
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
        
        # (v6.0) Atualiza o prompt para incluir os novos sinais
        prompt_text = (
            "Analise os indicadores (Z-Score=desvio, Confluence=soma de sinais) e "
            "forneça um resumo de 3 frases sobre a tendência. "
            "Foco nos Z-Scores extremos (> 2.0 ou < -2.0) e no 'feature_confluence_count'. Dados:"
        )
        formatted_data = {
            k: v for k, v in analysis_data.items() 
            if k.startswith('zscore') or k.endswith('_ewma_short') or k.startswith('signal_') or k == 'feature_confluence_count'
        }
        prompt_text += str(formatted_data) 
        gemini_response = generate_content(prompt_text) 
        return {"status_code": 200, "analysis": gemini_response}
    except Exception as e:
        return {"status_code": 500, "message": f"Erro na chamada Gemini: {e}"}

if __name__ == "__main__":
    log("Iniciando servidor FastAPI/Uvicorn (Híbrido v6.0)...")
    uvicorn.run(app, host="127.0.0.1", port=8000)