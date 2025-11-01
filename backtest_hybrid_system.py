# backtest_auto_setup.py
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import analysis
    from ml_model import TrainedMLModel
    from database import get_supabase_client, get_raw_history, get_config, update_saldo_simulado
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Módulos principais não disponíveis: {e}")
    ML_AVAILABLE = False

class AutoSetupBacktest:
    def __init__(self, initial_bankroll=1000.0):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.results = []
        
        # Configurações padrão
        self.config = {
            'min_edge_threshold': 0.001,
            'kelly_fraction': 0.5,
            'min_zscore_tension': 1.0,
            'chip_value': 0.50
        }
        
        self.ml_available = ML_AVAILABLE
        if self.ml_available:
            try:
                from threading import Lock
                self.model_lock = Lock()
                self.ml_model = TrainedMLModel(self.model_lock)
                self.ml_available = self.ml_model.model_loaded
            except Exception as e:
                print(f"⚠️  Erro ao carregar modelo ML: {e}")
                self.ml_available = False

    def load_recent_data(self, limit=500):
        """Carrega dados recentes para validação rápida"""
        try:
            client = get_supabase_client()
            raw_data = get_raw_history(client, limit=limit)
            return raw_data if raw_data else self.generate_sample_data(200)
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return self.generate_sample_data(200)

    def generate_sample_data(self, n_spins=200):
        """Gera dados de exemplo para validação"""
        np.random.seed(42)
        return list(np.random.randint(0, 37, n_spins))

    def quick_validation_test(self, historical_data, strategy_mode, test_spins=100):
        """Teste rápido de validação com poucos spins"""
        if len(historical_data) < 150:
            return {"error": f"Dados insuficientes: {len(historical_data)}/150"}
        
        # Usar apenas os últimos test_spins spins para teste rápido
        test_data = historical_data[-test_spins:]
        start_index = 50  # Começar após ter histórico suficiente para features
        
        results = {
            'bets': 0, 'wins': 0, 'total_stake': 0, 'total_profit': 0,
            'bankroll_progression': [], 'edge_samples': []
        }
        
        temp_bankroll = self.initial_bankroll
        
        for i in range(start_index, len(test_data)):
            history_up_to_i = test_data[:i]
            next_spin = test_data[i]
            
            if len(history_up_to_i) < 37:
                continue
                
            # Simular decisão (versão simplificada)
            aposta, edge, motivo = self.simulate_quick_decision(history_up_to_i, strategy_mode)
            
            if aposta and len(aposta) > 0:
                stake = len(aposta) * self.config['chip_value']
                
                if stake > temp_bankroll:
                    continue
                    
                won = next_spin in aposta
                profit = (stake * (36.0 / len(aposta)) - stake) if won else -stake
                temp_bankroll += profit
                
                results['bets'] += 1
                results['total_stake'] += stake
                results['total_profit'] += profit
                if won: results['wins'] += 1
                if edge: results['edge_samples'].append(edge)
                results['bankroll_progression'].append(temp_bankroll)
        
        if results['bets'] > 0:
            results['win_rate'] = results['wins'] / results['bets']
            results['roi'] = (results['total_profit'] / results['total_stake']) * 100
            results['avg_edge'] = np.mean(results['edge_samples']) if results['edge_samples'] else 0
            results['final_bankroll'] = temp_bankroll
            results['profit'] = results['total_profit']
        else:
            results['win_rate'] = 0
            results['roi'] = 0
            results['avg_edge'] = 0
            results['final_bankroll'] = temp_bankroll
            results['profit'] = 0
            
        return results

    def simulate_quick_decision(self, history, strategy_mode):
        """Versão simplificada da decisão para validação rápida"""
        try:
            if strategy_mode == 'IA_MODEL' and self.ml_available:
                features_df = analysis.gerar_features_avancadas(history)
                if not features_df.empty:
                    aposta, edge, _, motivo = self.ml_model.predict(
                        features_df, self.bankroll, 0.001, 0.5, 1.0, strategy_mode
                    )
                    return aposta, edge, motivo
            
            # Fallback para heurística BSB simples
            current_num = history[-1]
            bsb_triggers = {
                0: [1, 5, 8, 9, 10], 1: [0, 27, 21], 2: [22, 28, 31],
                3: [3, 7, 12], 4: [7, 10, 19], 5: [0, 27, 21],
                32: [12, 35, 3], 34: [6, 27], 36: [2, 25]
            }
            
            if current_num in bsb_triggers:
                cluster = bsb_triggers[current_num]
                return cluster, 0.015, f"BSB {current_num}"
                
            return None, None, "Sem gatilho"
            
        except Exception as e:
            return None, None, f"Erro: {str(e)}"

    def evaluate_strategy_performance(self, historical_data):
        """Avalia ambas as estratégias e recomenda a melhor"""
        print("🔍 Avaliando estratégias...")
        
        strategies = ['IA_MODEL', 'LUCAS_BSB']
        results = {}
        
        for strategy in strategies:
            print(f"  📊 Testando {strategy}...")
            result = self.quick_validation_test(historical_data, strategy, test_spins=100)
            results[strategy] = result
        
        # Análise comparativa
        valid_results = {k: v for k, v in results.items() if 'error' not in v and v['bets'] > 10}
        
        if not valid_results:
            print("❌ Nenhuma estratégia teve desempenho válido")
            return 'IA_MODEL', results  # Fallback
        
        # Escolher melhor estratégia baseada em múltiplos fatores
        best_strategy = None
        best_score = -float('inf')
        
        for strategy, data in valid_results.items():
            # Score composto: ROI + WinRate + Número de apostas
            roi_score = data['roi']
            win_rate_score = data['win_rate'] * 100
            bet_count_score = min(data['bets'] / 20, 1.0) * 50  # Normalizar para 50 pontos max
            edge_score = data['avg_edge'] * 1000
            
            total_score = roi_score + win_rate_score + bet_count_score + edge_score
            
            print(f"    {strategy}: ROI={data['roi']:.2f}%, WinRate={data['win_rate']:.2%}, "
                  f"Apostas={data['bets']}, Score={total_score:.2f}")
            
            if total_score > best_score:
                best_score = total_score
                best_strategy = strategy
        
        print(f"✅ Estratégia recomendada: {best_strategy} (Score: {best_score:.2f})")
        return best_strategy, results

    def update_bot_configuration(self, recommended_strategy, performance_data):
        """Atualiza a configuração do bot baseado nos resultados"""
        try:
            client = get_supabase_client()
            
            # Buscar configuração atual
            current_config = get_config(client)
            
            # Preparar atualizações
            updates = {
                'active_strategy': recommended_strategy,
                'auto_setup_timestamp': datetime.now().isoformat(),
                'auto_setup_results': json.dumps(performance_data, default=str)
            }
            
            # Ajustar parâmetros baseado no desempenho
            strategy_data = performance_data.get(recommended_strategy, {})
            
            if 'roi' in strategy_data:
                if strategy_data['roi'] > 20:
                    updates['min_edge_threshold'] = max(0.002, self.config['min_edge_threshold'])
                    updates['kelly_fraction'] = min(0.7, self.config['kelly_fraction'] + 0.1)
                    print("  🚀 Condições favoráveis - aumentando agressividade")
                elif strategy_data['roi'] < 0:
                    updates['min_edge_threshold'] = 0.005  # Mais conservador
                    updates['kelly_fraction'] = 0.3  # Kelly mais fraco
                    print("  🛡️  Condições desfavoráveis - modo conservador")
            
            # Aplicar atualizações
            client.table("config").update(updates).eq("id", 1).execute()
            
            print(f"✅ Configuração atualizada:")
            print(f"   Estratégia: {recommended_strategy}")
            print(f"   Min Edge: {updates.get('min_edge_threshold', 'mantido')}")
            print(f"   Kelly Fraction: {updates.get('kelly_fraction', 'mantido')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao atualizar configuração: {e}")
            return False

    def run_auto_setup(self):
        """Executa setup automático completo"""
        print("🚀 INICIANDO SETUP AUTOMÁTICO DO BOT")
        print("="*50)
        
        # Carregar dados recentes
        historical_data = self.load_recent_data(limit=300)
        print(f"📊 Dados carregados: {len(historical_data)} spins")
        
        # Avaliar estratégias
        recommended_strategy, performance_data = self.evaluate_strategy_performance(historical_data)
        
        # Atualizar configuração
        success = self.update_bot_configuration(recommended_strategy, performance_data)
        
        # Resumo final
        print("\n" + "="*50)
        print("📋 RESUMO DO SETUP AUTOMÁTICO")
        print("="*50)
        
        for strategy, data in performance_data.items():
            if 'error' not in data:
                status = "✅ RECOMENDADA" if strategy == recommended_strategy else "📊 TESTADA"
                print(f"{status} {strategy}:")
                print(f"   ROI: {data.get('roi', 0):.2f}%")
                print(f"   Win Rate: {data.get('win_rate', 0):.2%}")
                print(f"   Apostas: {data.get('bets', 0)}")
                print(f"   Lucro: R$ {data.get('profit', 0):.2f}")
            else:
                print(f"❌ {strategy}: {data['error']}")
        
        print(f"\n🎯 Estratégia ativa: {recommended_strategy}")
        print(f"⏰ Setup concluído em: {datetime.now().strftime('%H:%M:%S')}")
        
        return recommended_strategy, performance_data, success

def setup_bot_automatically():
    """Função principal para setup automático"""
    setup = AutoSetupBacktest(initial_bankroll=1000.0)
    return setup.run_auto_setup()

class PeriodicValidator:
    def __init__(self):
        self.last_validation = None
        self.validation_interval = timedelta(hours=6)  # Revalidar a cada 6 horas
    
    def should_revalidate(self):
        if not self.last_validation:
            return True
        
        client = get_supabase_client()
        config = get_config(client)
        last_auto_setup = config.get('auto_setup_timestamp')
        
        if not last_auto_setup:
            return True
        
        try:
            last_time = datetime.fromisoformat(last_auto_setup.replace('Z', '+00:00'))
            return datetime.now().utcnow() - last_time > self.validation_interval
        except:
            return True
    
    def run_periodic_validation(self):
        if self.should_revalidate():
            print("🔄 Executando revalidação periódica...")
            setup_bot_automatically()
            self.last_validation = datetime.now()

if __name__ == "__main__":
    setup_bot_automatically()