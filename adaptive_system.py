# douglessa1/projeto_roleta/projeto_roleta-4eb8af59f00aad63289b5a75b94bcc4e4e852c83/adaptive_system.py
import database as db

class AdaptiveThresholdSystem:
    def __init__(self, client: db.Client, calibration_interval=100):
        self.client = client
        self.calibration_interval = calibration_interval
        self.performance_history = [] # (success, confidence, p_l)
        self.recent_performance_metric = 0.5 # Começa neutro (50%)
        self.spins_since_recalibration = 0
        
        # Carrega o histórico de performance do DB ao iniciar
        self._load_performance_history()

    def _load_performance_history(self):
        try:
            history_data = db.get_performance_history(self.client, limit=self.calibration_interval)
            # O histórico vem do mais recente para o mais antigo
            for item in reversed(history_data):
                self.performance_history.append((
                    item.get('success', False),
                    item.get('hybrid_confidence', 0.0),
                    item.get('p_l', 0.0)
                ))
            
            if self.performance_history:
                # Calcula a performance (taxa de acerto)
                success_rate = sum(1 for s, c, p in self.performance_history if s) / len(self.performance_history)
                self.recent_performance_metric = success_rate
                
            print(f"ADAPTIVE: Sistema carregado com {len(self.performance_history)} registros. Performance inicial: {self.recent_performance_metric:.2%}")
                
        except Exception as e:
            print(f"ADAPTIVE: Erro ao carregar histórico de performance: {e}")

    def update_performance(self, success: bool, confidence: float, p_l: float):
        """ Atualiza o histórico com o último resultado. """
        try:
            self.performance_history.append((success, confidence, p_l))
            self.spins_since_recalibration += 1
            
            # Mantém o histórico com o tamanho da janela
            if len(self.performance_history) > self.calibration_interval:
                self.performance_history.pop(0)
            
            # Recalcula a métrica de performance (média móvel simples de acerto)
            success_rate = sum(1 for s, c, p in self.performance_history if s) / len(self.performance_history)
            self.recent_performance_metric = success_rate
            
            # Verifica se é hora de recalibrar os thresholds
            if self.spins_since_recalibration >= self.calibration_interval:
                self._recalibrar_thresholds()
                self.spins_since_recalibration = 0
                
        except Exception as e:
            print(f"ADAPTIVE: Erro ao atualizar performance: {e}")

    def _recalibrar_thresholds(self):
        """
        (Req 5 - Placeholder) Recalcula o 'base_threshold' ideal.
        Esta função pode ser expandida para otimizar o ROI.
        """
        print(f"ADAPTIVE: Recalibrando thresholds... (Performance atual: {self.recent_performance_metric:.2%})")
        # Lógica de recalibração (Exemplo simples):
        
        config = db.get_config(self.client)
        current_base_threshold = config.get('base_threshold', 2.0)
        
        if self.recent_performance_metric > 0.55:
            new_base_threshold = max(1.5, current_base_threshold - 0.1) # Mais agressivo
        elif self.recent_performance_metric < 0.45:
            new_base_threshold = min(3.0, current_base_threshold + 0.1) # Mais conservador
        else:
            new_base_threshold = current_base_threshold # Mantém
            
        if new_base_threshold != current_base_threshold:
            # Salva o novo threshold no DB
            db.update_adaptive_thresholds(self.client, new_base_threshold, self.recent_performance_metric)
        else:
             print("ADAPTIVE: Threshold base mantido.")

    def get_recent_performance(self) -> float:
        """ Retorna a métrica de performance recente (taxa de acerto). """
        return self.recent_performance_metric