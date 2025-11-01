import os
import time
import re
from dotenv import load_dotenv
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from queue import Queue
import threading

# --- CARREGAMENTO DAS VARIÁVEIS .env ---
# (Apenas para Supabase, não precisamos de login aqui)
load_dotenv()

# --- CONFIGURAÇÃO ---
# URL do Histórico da Roleta Brasileira no TipMiner
URL_TIPMINER = "https://www.tipminer.com/br/historico/pragmatic/roleta-brasileira"
# SELETOR CORRETO para os números no histórico do TipMiner (confirmado por você)
SELETOR_HISTORICO_TIPMINER = ".cell__result"
# --------------------

# --- CLASSE PRINCIPAL ---
class RouletteScraper:
    def __init__(self):
        print(f"🚀 Scraper iniciado (MODO TIPMINER - {URL_TIPMINER})")
        self.results_queue = Queue()
        self.ultimo_numero_real = -1
        self.driver = None

        self.browser_thread = threading.Thread(target=self.run_browser, daemon=True)
        self.browser_thread.start()

    # --- Função de Extração (Adaptada para TipMiner com o seletor correto) ---
    def extrair_numeros_tipminer(self) -> list[int]:
        """Extrai os resultados da Roleta Brasileira do TipMiner."""
        numeros = []
        try:
            # Espera até 10 segundos para que os elementos do histórico estejam presentes
            wait = WebDriverWait(self.driver, 10)
            # Usa o SELETOR_HISTORICO_TIPMINER correto que você encontrou: .cell__result
            elementos = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, SELETOR_HISTORICO_TIPMINER)))

            # Pega o texto de cada elemento encontrado
            textos_encontrados = [] # Para Debug
            for e in elementos:
                t = e.text.strip()
                textos_encontrados.append(t) # DEBUG
                # Valida se é um número entre 0 e 36
                if t.isdigit():
                    num_int = int(t)
                    if 0 <= num_int <= 36:
                        numeros.append(num_int)

            # DEBUG: Mostra todos os textos que o seletor pegou
            # print(f"   Textos extraídos: {textos_encontrados[:10]}...") # Mostra os 10 primeiros

            if not numeros:
                # print("⚠️ Nenhum número válido (0-36) encontrado nos elementos.") # Log Redundante
                return []

            # A ordem no TipMiner HTML (com .cell__result) parece ser do mais RECENTE para o mais antigo.

            # Retorna apenas os últimos 50 (para consistência)
            return numeros[:50]

        except Exception as e:
            print(f"❌ ERRO na extração do TipMiner: {e}")
            return []

    # --- Função Principal do Navegador (Adaptada para TipMiner) ---
    def run_browser(self):
        """Executa o navegador Undetected Chromedriver e monitora o TipMiner."""
        try:
            print("Iniciando Undetected Chromedriver...")
            options = uc.ChromeOptions()
            options.add_argument("--headless=new") # Roda invisível
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36')
            options.add_argument('--lang=pt-BR')

            self.driver = uc.Chrome(options=options)
            self.driver.implicitly_wait(10) # Espera implícita

            print(f"🎯 Acessando {URL_TIPMINER}")
            self.driver.get(URL_TIPMINER)
            print("   Página acessada. Aguardando 8s...")
            time.sleep(8) # Aumenta a espera inicial

            print("🔍 Monitorando resultados do TipMiner...")
            while True:
                try:
                    numeros = self.extrair_numeros_tipminer()
                    if numeros:
                        # O primeiro número na lista do TipMiner é o mais recente
                        ultimo = numeros[0]
                        if ultimo != self.ultimo_numero_real:
                            self.ultimo_numero_real = ultimo
                            print(f"🎲 Novo número (TipMiner): {ultimo}")
                            self.results_queue.put(ultimo)
                        # else:
                            # print(f"Monitorando TipMiner... Último: {ultimo}")
                    else:
                         print("⚠️ Nenhum número detectado no TipMiner. Recarregando...")
                         time.sleep(3)
                         self.driver.refresh()
                         time.sleep(8) # Espera recarregar

                    # Verifica o TipMiner a cada 10 segundos (pode ajustar)
                    time.sleep(10)

                except Exception as e:
                    if "disconnected" in str(e).lower() or "no such window" in str(e).lower() or "target window already closed" in str(e).lower():
                         print("Navegador fechado.")
                         break
                    print(f"⚠️ Erro no loop: {e}. Recarregando...")
                    try:
                        self.driver.refresh()
                    except Exception as reload_e:
                         print(f"Falha crítica ao recarregar: {reload_e}")
                         break
                    time.sleep(15) # Espera mais tempo após erro

        except Exception as e:
            print(f"❌ ERRO CRÍTICO no Undetected Chromedriver: {e}")
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                except Exception:
                    pass
            print("🛑 Navegador encerrado.")
            self.results_queue.put(None) # Sinaliza fim para main.py

    # --- Interface com main.py (Não muda) ---
    def get_latest_result(self) -> int | None:
        """Pega o próximo resultado da fila."""
        try:
            # Aumentamos o timeout para dar mais margem ao TipMiner
            novo_numero = self.results_queue.get(timeout=60)
            if novo_numero is None:
                return None
            return novo_numero
        except Exception:
            print("⏳ Nenhum novo resultado do TipMiner na fila. Aguardando...")
            return None

    def close(self):
        """Envia sinal para a thread do navegador encerrar."""
        self.results_queue.put(None)
        if self.driver:
             try:
                 self.driver.quit()
             except Exception:
                 pass
        print("🧩 Scraper encerrado.")

