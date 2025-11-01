// Dashboard.tsx
import React from "react";
import {
  getDashboardData,
  KpiCard,
  StrategyPerformance,
  SuggestedBet,
  BankrollSimulation,
  ContextData,
} from "./dashboard";

const Dashboard: React.FC = () => {
  const { kpis, strategies, simulation, context } = getDashboardData();

  return (
    <div>
      {/* KPIs */}
      <section>
        <h2>KPIs</h2>
        <div style={{ display: "flex", gap: "16px" }}>
          {kpis.map((kpi: KpiCard, idx: number) => (
            <div key={kpi.title} style={{ border: "1px solid #ccc", padding: 12 }}>
              <h3>{kpi.title}</h3>
              <p>{kpi.value}</p>
              <span style={{ color: kpi.color }}>{kpi.trend}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Estratégias */}
      <section>
        <h2>Estratégias</h2>
        {strategies.map((strategy: StrategyPerformance) => (
          <div key={strategy.id} style={{ marginBottom: 24, border: "1px solid #eee", padding: 12 }}>
            <h3>
              {strategy.name} [{strategy.type}]
            </h3>
            <div>
              Confiança: {(strategy.confidence * 100).toFixed(1)}%
              <div>
                Tags:{" "}
                {strategy.tags.map((tag: string) => (
                  <span key={tag} style={{ marginRight: 8, background: "#eee", padding: "2px 6px", borderRadius: 4 }}>
                    {tag}
                  </span>
                ))}
              </div>
              <div>
                Amostras: {strategy.kpis.samples} | Hit Rate: {(strategy.kpis.hitRate * 100).toFixed(2)}% | Prob/Spin:{" "}
                {(strategy.kpis.probPerSpin * 100).toFixed(2)}% | Aposta Média: R$ {strategy.kpis.avgStake.toFixed(2)} |
                ROI: {(strategy.kpis.roi * 100).toFixed(2)}% | Lucro: R$ {strategy.kpis.profit.toFixed(2)}
              </div>
              <div>
                Histórico:{" "}
                {strategy.history.map(
                  (
                    h: {
                      cycle: number;
                      stake: number;
                      result: "Win" | "Red" | "Neutral";
                      bankroll: number;
                    },
                    i: number
                  ) => (
                    <span
                      key={i}
                      title={`Ciclo ${h.cycle} - ${h.result} - Bankroll: ${h.bankroll}`}
                      style={{
                        marginRight: 4,
                        color: h.result === "Win" ? "green" : h.result === "Red" ? "red" : "gray",
                        fontWeight: "bold",
                      }}
                    >
                      {h.result === "Win" ? "W" : h.result === "Red" ? "R" : "-"}
                    </span>
                  )
                )}
              </div>
              {strategy.suggestedBet && (
                <div style={{ marginTop: 8, background: "#f9f9f9", padding: 8, borderRadius: 4 }}>
                  <strong>Sugestão:</strong> {strategy.suggestedBet.name} | Risco: {strategy.suggestedBet.risk} | Aposta: R${" "}
                  {strategy.suggestedBet.stake.toFixed(2)} | Retorno Esperado: R${" "}
                  {strategy.suggestedBet.expectedReturn.toFixed(2)} | Probabilidade:{" "}
                  {(strategy.suggestedBet.probability * 100).toFixed(2)}%
                  <div>
                    <em>{strategy.suggestedBet.rationale}</em>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </section>

      {/* Simulação */}
      <section>
        <h2>Simulação de Bankroll</h2>
        <div>
          Banco Inicial: R$ {simulation.initialBank.toFixed(2)}<br />
          Total Apostado: R$ {simulation.totalStaked.toFixed(2)}<br />
          Lucro Líquido: R$ {simulation.netProfit.toFixed(2)}<br />
          ROI Composto: {(simulation.roiCompound * 100).toFixed(2)}%<br />
          <em>{simulation.interpretation}</em>
        </div>
      </section>

      {/* Contexto */}
      <section>
        <h2>Contexto Atual</h2>
        <div>
          Último Número: {context.latestNumber} <br />
          Zona Quente: {context.hotZone} <br />
          Sequência: {context.sequence} <br />
          Insights: {context.insights} <br />
          Strip: {context.historyStrip.map((n: number, i: number) => (
            <span key={i} style={{ marginRight: 4 }}>
              {n}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Dashboard;