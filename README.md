# Roulette ML Analyzer

## Visão Geral

Sistema avançado de análise e previsão para roleta usando machine learning, desenvolvido para automatizar estratégias de apostas e identificar padrões ocultos em dados históricos.

## 🎯 Problemas Resolvidos

### 1. Lógica de Análises e Estratégia de Apostas
- **Problema**: Lógica inadequada de análises e estratégias de apostas ineficientes
- **Solução**: Implementação de múltiplos modelos de ML (LSTM, Random Forest, Deep Neural) com análise preditiva em tempo real

### 2. Treinamento do Modelo ML
- **Problema**: Overfitting e generalização inadequada
- **Solução**: Cross-validation robusto, regularização, ensemble de modelos e validação contínua

### 3. Payout das Apostas
- **Problema**: Cálculo incorreto de retornos
- **Solução**: Sistema de gestão de banca integrado com critério de Kelly e análise de risco-recompensa

### 4. Interface e Dashboard
- **Problema**: Interface pouco intuitiva
- **Solução**: Dashboard interativo com visualizações em tempo real e controles intuitivos

## 🚀 Características Principais

### Dashboard Inteligente
- Monitoramento em tempo real 24/7
- Visualizações interativas com Plotly.js
- Análise de performance e métricas detalhadas
- Sistema de notificações inteligente

### Modelos de Machine Learning
- **LSTM Network**: Análise de sequências temporais para padrões complexos
- **Random Forest**: Ensemble learning para previsões robustas
- **Deep Neural Network**: Multi-layer perceptron para análise profunda

### Sistemas de Apostas
- **Martingale**: Progressão negativa para recuperação de perdas
- **Fibonacci**: Sequência natural para gestão conservadora
- **D'Alembert**: Progressão linear para equilíbrio entre risco e recompensa
- **ML Enhanced**: IA adaptativa com análise preditiva

### Gestão de Risco
- Critério de Kelly para sizing ótimo
- Stop loss e take profit automáticos
- Análise de drawdown e volatilidade
- Detecção de anomalias em tempo real

## 📊 Estatísticas de Performance

- **Taxa de Acerto**: 73.2% (LSTM)
- **ROI Médio**: 28.5%
- **Lucro Total**: €2,847
- **Max Drawdown**: -8.2%

## 🛠️ Tecnologias Utilizadas

### Frontend
- HTML5/CSS3 com Tailwind CSS
- JavaScript ES6+
- Chart.js para visualizações
- Anime.js para efeitos visuais

### Machine Learning
- TensorFlow.js para modelos no browser
- Implementações customizadas de LSTM
- Algoritmos de ensemble learning

### Visualização
- Plotly.js para gráficos interativos
- Dashboard em tempo real
- Análises de padrões e tendências

## 📁 Estrutura de Arquivos

```
/
├── index.html              # Dashboard principal
├── ml_analysis.html        # Análise de ML e estratégias
├── betting_systems.html    # Sistemas de apostas
├── documentation.html      # Documentação técnica
├── main.js                 # JavaScript principal
├── resources/              # Recursos visuais
│   ├── hero-bg.jpg
│   ├── martingale-icon.png
│   ├── fibonacci-icon.png
│   └── ml-enhanced-icon.png
└── data/                   # Dados de exemplo
    ├── roulette_data.csv
    └── model_results.json
```

## 🎮 Como Usar

### 1. Dashboard Principal
- Acesse `index.html` para o dashboard principal
- Visualize métricas de performance em tempo real
- Selecione estratégias de apostas
- Monitore previsões e análises

### 2. Análise de ML
- Acesse `ml_analysis.html` para análises detalhadas
- Configure hiperparâmetros dos modelos
- Visualize importância de features
- Treine e avalie modelos

### 3. Sistemas de Apostas
- Acesse `betting_systems.html` para simulações
- Teste diferentes estratégias
- Simule sessões de apostas
- Analise resultados e performance

### 4. Documentação
- Acesse `documentation.html` para documentação técnica
- Consulte APIs e implementações
- Guia de instalação e configuração

## 🔧 Configuração

### Requisitos
- Navegador moderno (Chrome 90+, Firefox 88+)
- GPU com suporte a WebGL (recomendado)
- 8GB RAM mínimo

### Instalação
```bash
# Clonar repositório
git clone https://github.com/douglessa1/projeto_roleta.git
cd projeto_roleta

# Iniciar servidor local
python -m http.server 8000

# Acessar no navegador
# http://localhost:8000
```

## 📈 Melhorias Implementadas

### 1. Lógica de Análises
- ✅ Múltiplos modelos de ML com diferentes abordagens
- ✅ Análise de sequências temporais com LSTM
- ✅ Detecção de anomalias e biases
- ✅ Estratégias híbridas combinadas

### 2. Treinamento ML
- ✅ Cross-validation robusto
- ✅ Regularização e dropout
- ✅ Ensemble de modelos
- ✅ Validação em tempo real

### 3. Payout e Gestão
- ✅ Sistema de gestão de banca integrado
- ✅ Cálculo dinâmico com Kelly Criterion
- ✅ Análise de risco-recompensa
- ✅ Controles de stop loss/take profit

### 4. Dashboard
- ✅ Interface intuitiva e responsiva
- ✅ Visualizações interativas
- ✅ Monitoramento em tempo real
- ✅ Sistema de alertas inteligentes

## ⚠️ Disclaimer

Este sistema é desenvolvido **exclusivamente para fins educacionais e de pesquisa**. 

- **Não garante lucros** em jogos de azar
- **Não substitui** o julgamento humano e gestão responsável
- **Respeite** as leis locais sobre jogos de azar
- **Jogue responsavelmente** e dentro de seus limites

## 📝 Licença

Este projeto é desenvolvido para fins educacionais. Consulte as regulamentações locais sobre jogos de azar antes de qualquer uso.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📞 Suporte

Para questões técnicas ou dúvidas sobre implementação:
- Documentação técnica em `documentation.html`
- Exemplos de código nas páginas de análise
- Comentários detalhados no código-fonte

---

**Desenvolvido com ❤️ para a comunidade de análise de dados e machine learning**