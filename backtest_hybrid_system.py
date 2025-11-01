# backtest_hybrid_system.py
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
from threading import Lock
import warnings
warnings.filterwarnings('ignore')

# Adiciona o diretório atual ao path para importar seus módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import analysis
    from ml_model import TrainedMLModel, WHEEL_CLUSTERS
    from database import get_supabase_client, get_raw_history, get_all_decisions
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Alguns módulos não disponíveis: {e}")
    ML_AVAILABLE = False

class HybridSystemBacktest:
    def __init__(self, initial_bankroll=1000.0):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.results = []
        self.detailed_logs = []
        
        # Configurações do sistema (simulando a tabela config)
        self.config = {
            'min_edge_threshold': 0.001,
            'kelly_fraction': 0.5,
            'min_zscore_tension': 1.0,
            'chip_value': 0.50
        }
        
        # Inicializar modelo ML se disponível
        self.ml_available = ML_AVAILABLE
        if self.ml_available:
            try:
                self.model_lock = Lock()
                self.ml_model = TrainedMLModel(self.model_lock)
                self.ml_available = self.ml_model.model_loaded
                if self.ml_available:
                    print("✅ Modelo ML carregado com sucesso")
                else:
                    print("⚠️  Modelo ML não pôde ser carregado")
            except Exception as e:
                print(f"⚠️  Erro ao carregar modelo ML: {e}")
                self.ml_available = False
        else:
            print("⚠️  Módulos ML não disponíveis - usando modo heurístico apenas")
        
        # Métricas de performance
        self.performance_metrics = {
            'total_spins': 0,
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'total_wagered': 0.0,
            'total_won': 0.0,
            'total_lost': 0.0,
            'max_drawdown': 0.0,
            'peak_bankroll': initial_bankroll,
            'win_rate': 0.0,
            'net_profit': 0.0,
            'roi_percent': 0.0
        }

    def load_historical_data(self, days_back=30, limit=5000):
        """Carrega dados históricos do Supabase ou usa dados de exemplo"""
        try:
            if ML_AVAILABLE:
                client = get_supabase_client()
                raw_data = get_raw_history(client, limit=limit)
                
                if raw_data and len(raw_data) > 100:
                    print(f"✅ Carregados {len(raw_data)} spins históricos do Supabase")
                    return raw_data
            
            # Fallback para dados de exemplo
            print("✅ Gerando dados de exemplo para teste...")
            return self.generate_sample_data(1000)
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados históricos: {e}")
            print("✅ Gerando dados de exemplo para teste...")
            return self.generate_sample_data(1000)

    def generate_sample_data(self, n_spins=1000):
        """Gera dados de exemplo realistas para teste"""
        np.random.seed(42)
        # Gera dados mais realistas com alguns padrões
        data = []
        for i in range(n_spins):
            # Adiciona alguns padrões simples para teste
            if i > 50 and i % 37 == 0:
                data.append(0)  # zero a cada 37 spins
            elif i > 20 and len(data) > 0 and data[-1] in [1, 3, 5]:
                data.append(np.random.choice([2, 4, 6]))  # Padrão de cores alternadas
            else:
                data.append(np.random.randint(0, 37))
        return data

    def simulate_ml_decision(self, features_df, strategy_mode, current_bankroll):
        """Simula a decisão do ML usando o modelo treinado"""
        if not self.ml_available:
            return None, None, None, "Modelo ML não disponível"
        
        try:
            aposta, edge, stake_proxy, motivo = self.ml_model.predict(
                features_df,
                bankroll=current_bankroll,
                min_edge_threshold=self.config['min_edge_threshold'],
                kelly_fraction=self.config['kelly_fraction'],
                min_zscore_tension=self.config['min_zscore_tension'],
                strategy_mode=strategy_mode
            )
            return aposta, edge, stake_proxy, motivo
        except Exception as e:
            return None, None, None, f"Erro no ML: {str(e)}"

    def simulate_heuristic_decision(self, current_number, history):
        """Simula decisão heurística baseada em regras BSB simplificadas"""
        try:
            # Regras BSB simplificadas para backtest
            bsb_triggers = {
                0: [1, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 20, 21, 24, 25, 26, 29],
                1: [0, 27, 21, 20, 34],
                2: [22, 28, 31, 36],
                3: [3, 7, 12, 19],
                4: [7, 10, 19, 21],
                5: [0, 27, 21, 20, 34],
                32: [12, 35, 3, 26, 0],
                34: [6, 27],
                36: [2, 25]
            }
            
            if current_number in bsb_triggers:
                targets = bsb_triggers[current_number]
                # Pegar os primeiros 3-5 números como cluster
                cluster_size = min(len(targets), 5)
                cluster = targets[:cluster_size]
                edge = 0.015  # Edge simulado para heurística
                return cluster, edge, 1.0, f"Gatilho BSB {current_number} -> {cluster}"
            
            return None, None, None, "Sem gatilhos BSB"
            
        except Exception as e:
            return None, None, None, f"Erro heurística: {str(e)}"

    def calculate_stake(self, edge, cluster_size, strategy_mode):
        """Calcula valor da aposta baseado na estratégia"""
        if strategy_mode == 'IA_MODEL' and edge > 0:
            # Stake baseado em Kelly fractionado para IA
            odds = 36.0 / cluster_size
            P_model = edge + (cluster_size / 37.0)
            b = odds - 1
            q = 1 - P_model
            
            if b > 0.001 and P_model > 0.0:
                kelly_percentage = (b * P_model - q) / b
                stake_percentage = max(0.01, min(0.05, kelly_percentage * 0.25))  # Limitado a 1-5%
                return self.bankroll * stake_percentage
        
        # Stake fixo para BSB ou fallback
        return cluster_size * self.config['chip_value']

    def execute_backtest(self, historical_data, strategy_mode='IA_MODEL', start_index=100):
        """Executa backtest completo"""
        
        print(f"🎯 Iniciando backtest - Modo: {strategy_mode}")
        print(f"💰 Bankroll inicial: R$ {self.bankroll:.2f}")
        print(f"📊 Spins para análise: {len(historical_data)}")
        
        for i in range(start_index, len(historical_data)):
            if i % 500 == 0 and i > start_index:
                print(f"⚡ Processando spin {i}/{len(historical_data)}...")
                print(f"   Bankroll atual: R$ {self.bankroll:.2f}")
                print(f"   Apostas realizadas: {self.performance_metrics['total_bets']}")
            
            # Histórico até o spin atual
            history_up_to_i = historical_data[:i]
            next_spin_result = historical_data[i]  # N+1 que queremos prever
            
            # Pular se histórico insuficiente
            if len(history_up_to_i) < 37:
                continue
            
            current_number = history_up_to_i[-1]  # Número mais recente (N)
            
            # Simular decisão baseada na estratégia
            if strategy_mode == 'IA_MODEL' and self.ml_available:
                try:
                    features_df = analysis.gerar_features_avancadas(history_up_to_i)
                    if not features_df.empty:
                        aposta, edge, stake_proxy, motivo = self.simulate_ml_decision(
                            features_df, strategy_mode, self.bankroll
                        )
                    else:
                        aposta, edge, stake_proxy, motivo = None, None, None, "Features vazias"
                except Exception as e:
                    aposta, edge, stake_proxy, motivo = None, None, None, f"Erro features: {e}"
            else:
                # Usar heurística BSB
                aposta, edge, stake_proxy, motivo = self.simulate_heuristic_decision(
                    current_number, history_up_to_i
                )
            
            # Processar aposta se válida
            if aposta and len(aposta) > 0:
                cluster_size = len(aposta)
                
                # Calcular stake
                stake = self.calculate_stake(edge or 0, cluster_size, strategy_mode)
                
                # Verificar bankroll
                if stake <= 0 or stake > self.bankroll:
                    continue
                
                # Verificar resultado
                won = next_spin_result in aposta
                payout = stake * (36.0 / cluster_size) if won else 0
                profit = payout - stake
                
                # Atualizar bankroll e métricas
                self.bankroll += profit
                self.performance_metrics['total_bets'] += 1
                self.performance_metrics['total_wagered'] += stake
                
                if won:
                    self.performance_metrics['wins'] += 1
                    self.performance_metrics['total_won'] += payout
                else:
                    self.performance_metrics['losses'] += 1
                    self.performance_metrics['total_lost'] += stake
                
                # Atualizar drawdown
                if self.bankroll > self.performance_metrics['peak_bankroll']:
                    self.performance_metrics['peak_bankroll'] = self.bankroll
                
                current_drawdown = (self.performance_metrics['peak_bankroll'] - self.bankroll) / self.performance_metrics['peak_bankroll'] * 100
                if current_drawdown > self.performance_metrics['max_drawdown']:
                    self.performance_metrics['max_drawdown'] = current_drawdown
                
                # Registrar resultado
                result_entry = {
                    'spin_index': i,
                    'current_number': current_number,
                    'next_number': next_spin_result,
                    'aposta': aposta,
                    'stake': stake,
                    'edge': edge or 0,
                    'won': won,
                    'profit': profit,
                    'bankroll': self.bankroll,
                    'motivo': motivo,
                    'strategy_mode': strategy_mode
                }
                
                self.results.append(result_entry)
                
                # Log para apostas significativas
                if abs(profit) > 20 or (i % 100 == 0 and self.performance_metrics['total_bets'] > 0):
                    status = "✅ GREEN" if won else "❌ RED"
                    edge_pct = f"{edge*100:.2f}%" if edge else "N/A"
                    print(f"{status}! Spin {i}: {aposta} | Stake: R$ {stake:.2f} | Profit: R$ {profit:.2f} | Edge: {edge_pct}")
            
            self.performance_metrics['total_spins'] += 1
        
        return self.generate_report()

    def generate_report(self):
        """Gera relatório completo do backtest"""
        
        if self.performance_metrics['total_bets'] == 0:
            return {"error": "Nenhuma aposta realizada durante o backtest"}
        
        # Calcular métricas básicas
        win_rate = self.performance_metrics['wins'] / self.performance_metrics['total_bets']
        net_profit = self.performance_metrics['total_won'] - self.performance_metrics['total_wagered']
        roi = (net_profit / self.performance_metrics['total_wagered']) * 100 if self.performance_metrics['total_wagered'] > 0 else 0
        
        # Atualizar métricas
        self.performance_metrics.update({
            'win_rate': win_rate,
            'net_profit': net_profit,
            'roi_percent': roi,
            'final_bankroll': self.bankroll,
            'total_return': ((self.bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        })
        
        # Métricas por aposta
        if self.performance_metrics['total_bets'] > 0:
            self.performance_metrics['avg_bet_size'] = self.performance_metrics['total_wagered'] / self.performance_metrics['total_bets']
            self.performance_metrics['profit_per_bet'] = net_profit / self.performance_metrics['total_bets']
        else:
            self.performance_metrics['avg_bet_size'] = 0
            self.performance_metrics['profit_per_bet'] = 0
        
        report = {
            'performance_metrics': self.performance_metrics,
            'strategy_analysis': self.analyze_strategy_performance(),
            'risk_metrics': self.calculate_risk_metrics(),
            'streaks': self.calculate_streaks(),
            'bankroll_progression': self.get_bankroll_progression(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.print_detailed_report(report)
        self.save_report_to_file(report)
        
        return report

    def analyze_strategy_performance(self):
        """Analisa performance por tipo de estratégia e motivo"""
        strategy_stats = {}
        
        for result in self.results:
            strategy_key = result['strategy_mode']
            motivo = result['motivo']
            
            if strategy_key not in strategy_stats:
                strategy_stats[strategy_key] = {
                    'bets': 0, 'wins': 0, 'total_stake': 0, 'total_profit': 0,
                    'motivos': {}
                }
            
            stats = strategy_stats[strategy_key]
            stats['bets'] += 1
            stats['total_stake'] += result['stake']
            stats['total_profit'] += result['profit']
            
            # Contar por motivo
            motivo_key = motivo.split(':')[0] if ':' in motivo else motivo[:30]
            if motivo_key not in stats['motivos']:
                stats['motivos'][motivo_key] = 0
            stats['motivos'][motivo_key] += 1
            
            if result['won']:
                stats['wins'] += 1
        
        # Calcular métricas para cada estratégia
        for key, stats in strategy_stats.items():
            if stats['bets'] > 0:
                stats['win_rate'] = stats['wins'] / stats['bets']
                stats['roi'] = (stats['total_profit'] / stats['total_stake']) * 100 if stats['total_stake'] > 0 else 0
                stats['avg_profit_per_bet'] = stats['total_profit'] / stats['bets']
                stats['avg_bet_size'] = stats['total_stake'] / stats['bets']
        
        return strategy_stats

    def calculate_risk_metrics(self):
        """Calcula métricas de risco avançadas"""
        if len(self.results) < 2:
            return {}
        
        profits = [r['profit'] for r in self.results]
        winning_profits = [p for p in profits if p > 0]
        losing_profits = [p for p in profits if p < 0]
        
        volatility = np.std(profits) if profits else 0
        avg_profit = np.mean(profits) if profits else 0
        
        return {
            'volatility': volatility,
            'sharpe_ratio': (avg_profit / volatility) if volatility > 0 else 0,
            'profit_factor': abs(sum(winning_profits) / abs(sum(losing_profits))) if losing_profits else float('inf'),
            'expectancy': avg_profit,
            'avg_win': np.mean(winning_profits) if winning_profits else 0,
            'avg_loss': np.mean(losing_profits) if losing_profits else 0,
            'largest_win': max(winning_profits) if winning_profits else 0,
            'largest_loss': min(losing_profits) if losing_profits else 0
        }

    def calculate_streaks(self):
        """Calcula sequências de vitórias/derrotas"""
        if not self.results:
            return {}
        
        streaks = {'win_streaks': [], 'loss_streaks': [], 'current_streak': 0, 'current_type': None}
        current_streak = 0
        current_type = None
        
        for result in self.results:
            if result['won']:
                if current_type == 'win':
                    current_streak += 1
                else:
                    if current_type == 'loss' and current_streak > 0:
                        streaks['loss_streaks'].append(current_streak)
                    current_streak = 1
                    current_type = 'win'
            else:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    if current_type == 'win' and current_streak > 0:
                        streaks['win_streaks'].append(current_streak)
                    current_streak = 1
                    current_type = 'loss'
        
        # Última sequência
        if current_streak > 0:
            if current_type == 'win':
                streaks['win_streaks'].append(current_streak)
            else:
                streaks['loss_streaks'].append(current_streak)
        
        streaks['longest_win_streak'] = max(streaks['win_streaks']) if streaks['win_streaks'] else 0
        streaks['longest_loss_streak'] = max(streaks['loss_streaks']) if streaks['loss_streaks'] else 0
        streaks['avg_win_streak'] = np.mean(streaks['win_streaks']) if streaks['win_streaks'] else 0
        streaks['avg_loss_streak'] = np.mean(streaks['loss_streaks']) if streaks['loss_streaks'] else 0
        
        return streaks

    def get_bankroll_progression(self):
        """Retorna progressão do bankroll"""
        if not self.results:
            return {'spins': [0], 'bankroll': [self.initial_bankroll]}
        
        spins = [0]
        bankrolls = [self.initial_bankroll]
        
        for result in self.results:
            spins.append(result['spin_index'])
            bankrolls.append(result['bankroll'])
        
        return {'spins': spins, 'bankroll': bankrolls}

    def print_detailed_report(self, report):
        """Imprime relatório detalhado no console"""
        print("\n" + "="*80)
        print("📊 RELATÓRIO COMPLETO DE BACKTEST - SISTEMA HÍBRIDO")
        print("="*80)
        
        metrics = report['performance_metrics']
        risk = report['risk_metrics']
        streaks = report['streaks']
        
        print(f"\n💵 PERFORMANCE FINANCEIRA:")
        print(f"   Bankroll Final:    R$ {metrics['final_bankroll']:,.2f}")
        print(f"   Lucro Líquido:     R$ {metrics['net_profit']:,.2f}")
        print(f"   ROI:               {metrics['roi_percent']:.2f}%")
        print(f"   Retorno Total:     {metrics['total_return']:.2f}%")
        print(f"   Total Apostado:    R$ {metrics['total_wagered']:,.2f}")
        
        print(f"\n🎯 MÉTRICAS DE ACERTO:")
        print(f"   Apostas Totais:    {metrics['total_bets']}")
        print(f"   Taxa de Acerto:    {metrics['win_rate']:.2%}")
        print(f"   Vitórias:          {metrics['wins']}")
        print(f"   Derrotas:          {metrics['losses']}")
        
        print(f"\n⚡ MÉTRICAS POR APOSTA:")
        print(f"   Aposta Média:      R$ {metrics.get('avg_bet_size', 0):.2f}")
        print(f"   Lucro/Aposta:      R$ {metrics.get('profit_per_bet', 0):.2f}")
        
        print(f"\n📈 MÉTRICAS DE RISCO:")
        print(f"   Drawdown Máximo:   {metrics['max_drawdown']:.2f}%")
        print(f"   Volatilidade:      R$ {risk.get('volatility', 0):.2f}")
        print(f"   Expectativa:       R$ {risk.get('expectancy', 0):.2f}")
        print(f"   Fator de Profit:   {risk.get('profit_factor', 0):.2f}")
        
        print(f"\n🔥 SEQUÊNCIAS:")
        print(f"   Maior Vitórias:    {streaks.get('longest_win_streak', 0)}")
        print(f"   Maior Derrotas:    {streaks.get('longest_loss_streak', 0)}")
        print(f"   Média Vitórias:    {streaks.get('avg_win_streak', 0):.1f}")
        print(f"   Média Derrotas:    {streaks.get('avg_loss_streak', 0):.1f}")

        # Análise por estratégia
        strategy_analysis = report['strategy_analysis']
        if strategy_analysis:
            print(f"\n🎮 ANÁLISE POR ESTRATÉGIA:")
            for strategy, stats in strategy_analysis.items():
                print(f"   {strategy}:")
                print(f"     Apostas: {stats['bets']} | Win Rate: {stats.get('win_rate', 0):.2%}")
                print(f"     ROI: {stats.get('roi', 0):.2f}% | Lucro: R$ {stats['total_profit']:.2f}")
                print(f"     Aposta Média: R$ {stats.get('avg_bet_size', 0):.2f}")

        print(f"\n⏰ TEMPO DE EXECUÇÃO:")
        print(f"   Spins Processados: {metrics['total_spins']}")
        print(f"   Timestamp:         {report['timestamp']}")

    def save_report_to_file(self, report):
        """Salva relatório em arquivo JSON"""
        try:
            filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Converter para formato serializável
            serializable_report = {}
            for key, value in report.items():
                if key == 'performance_metrics':
                    serializable_report[key] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                                              for k, v in value.items()}
                elif key == 'risk_metrics':
                    serializable_report[key] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                                              for k, v in value.items()}
                else:
                    serializable_report[key] = value
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_report, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Relatório salvo como: {filename}")
            
        except Exception as e:
            print(f"⚠️  Erro ao salvar relatório: {e}")

def run_comprehensive_backtest():
    """Executa backtest abrangente para todos os modos"""
    
    print("🚀 INICIANDO BACKTEST COMPREENSIVO DO SISTEMA HÍBRIDO")
    print("="*60)
    
    # Carregar dados
    backtester = HybridSystemBacktest(initial_bankroll=1000.0)
    historical_data = backtester.load_historical_data(limit=2000)
    
    if not historical_data:
        print("❌ Não foi possível carregar dados para backtest")
        return
    
    # Testar diferentes estratégias
    strategies = ['IA_MODEL', 'LUCAS_BSB']
    results = {}
    
    for strategy in strategies:
        print(f"\n🎯 Testando estratégia: {strategy}")
        print("-" * 40)
        
        # Reset backtester para nova estratégia
        backtester = HybridSystemBacktest(initial_bankroll=1000.0)
        
        # Executar backtest
        report = backtester.execute_backtest(
            historical_data, 
            strategy_mode=strategy,
            start_index=100
        )
        
        results[strategy] = report
    
    # Comparação final
    print("\n" + "="*80)
    print("🏆 COMPARAÇÃO FINAL ENTRE ESTRATÉGIAS")
    print("="*80)
    
    comparison_data = []
    for strategy, report in results.items():
        if 'error' not in report:
            metrics = report['performance_metrics']
            comparison_data.append({
                'Estratégia': strategy,
                'Lucro Líquido': f"R$ {metrics['net_profit']:.2f}",
                'ROI': f"{metrics['roi_percent']:.2f}%",
                'Win Rate': f"{metrics['win_rate']:.2%}",
                'Apostas': metrics['total_bets'],
                'Drawdown Máx': f"{metrics['max_drawdown']:.2f}%"
            })
    
    # Mostrar tabela comparativa
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        
        # Determinar melhor estratégia
        try:
            best_strategy = max(comparison_data, key=lambda x: float(x['ROI'].replace('%', '')))
            print(f"\n🏅 MELHOR ESTRATÉGIA: {best_strategy['Estratégia']} (ROI: {best_strategy['ROI']})")
        except:
            print(f"\n🏅 Análise completa concluída")
    
    return results

if __name__ == "__main__":
    # Executar backtest completo
    results = run_comprehensive_backtest()
    
    # Opção para backtest individual
    print("\n" + "="*60)
    try:
        individual_test = input("Deseja executar um backtest individual? (s/n): ")
        
        if individual_test.lower() == 's':
            strategy = input("Escolha a estratégia (IA_MODEL/LUCAS_BSB): ").strip() or "LUCAS_BSB"
            bankroll = float(input("Bankroll inicial (ex: 1000): ") or "1000")
            
            backtester = HybridSystemBacktest(initial_bankroll=bankroll)
            historical_data = backtester.load_historical_data()
            
            if historical_data:
                backtester.execute_backtest(historical_data, strategy_mode=strategy)
    except:
        print("✅ Backtest principal concluído!")