// dashboard.ts

// --- TIPAGEM ---

export interface KpiCard {
  title: string;
  value: string;
  trend: string; // Ex: +5.2% vs. Month
  color: 'emerald' | 'amber' | 'rose' | 'indigo';
}

export interface StrategyPerformance {
  id: number;
  name: string;
  type: 'IA' | 'Manual';
  confidence: number; // 0.0 to 1.0
  tags: string[];
  kpis: {
    samples: number;
    hitRate: number; // 0.0 to 1.0
    probPerSpin: number; // 0.0 to 1.0 (Probabilidade de aposta)
    avgStake: number;
    roi: number; // 0.0 to 1.0
    profit: number;
  };
  history: { cycle: number; stake: number; result: 'Win' | 'Red' | 'Neutral'; bankroll: number }[];
  suggestedBet: SuggestedBet | null;
}

export interface SuggestedBet {
  name: string;
  risk: 'Low' | 'Medium' | 'High';
  stake: number;
  expectedReturn: number;
  probability: number; // Probabilidade de ganho
  rationale: string;
}

export interface BankrollSimulation {
  initialBank: number;
  totalStaked: number;
  netProfit: number;
  roiCompound: number;
  interpretation: string;
}

export interface ContextData {
  latestNumber: number | null;
  hotZone: string;
  sequence: string;
  insights: string;
  historyStrip: number[];
}

// --- DADOS MOCK DINÂMICOS ---

const generateRandomKPI = (base: number, variance: number) => {
  return parseFloat((base + Math.random() * variance * (Math.random() > 0.5 ? 1 : -1)).toFixed(2));
};

const generatePerformanceHistory = (): StrategyPerformance['history'] => {
  const history: StrategyPerformance['history'] = [];
  let bankroll = 1000;
  for (let i = 1; i <= 6; i++) {
    const result = Math.random() > 0.6 ? 'Win' : 'Red';
    const stake = generateRandomKPI(10, 5);
    const payout = 35; // Aposta Straight
    
    let profit = 0;
    if (result === 'Win') {
        profit = stake * payout;
    } else if (result === 'Red') {
        profit = -stake;
    }
    
    bankroll = parseFloat((bankroll + profit).toFixed(2));
    
    history.push({ cycle: i, stake: parseFloat(stake.toFixed(2)), result, bankroll });
  }
  return history;
};


export const getDashboardData = () => {
  // --- CARDS KPIS ---
  const kpis: KpiCard[] = [
    { title: "Projeção Principal", value: `${generateRandomKPI(150, 50).toFixed(0)} Giros`, trend: "+5.2% vs. Mês", color: 'emerald' },
    { title: "Confiança Média", value: `${generateRandomKPI(0.85, 0.05).toFixed(2)}`, trend: "-1.1% vs. Último", color: 'amber' },
    { title: "ROI Composto", value: `${generateRandomKPI(0.18, 0.05).toFixed(2)}`, trend: "+12.4% vs. Semana", color: 'indigo' },
    { title: "Hit Rate Médio", value: `${generateRandomKPI(0.04, 0.01).toFixed(3)}`, trend: "-0.5% vs. Mês", color: 'rose' },
  ];

  // --- ESTRATÉGIAS ---
  const strategies: StrategyPerformance[] = [
    {
      id: 1,
      name: "Heurística 8/0 (Vizinhos)",
      type: 'IA',
      confidence: 0.92,
      tags: ['Tensão', 'Volante', 'Pós-Gatilho'],
      kpis: {
        samples: 120,
        hitRate: 0.052,
        probPerSpin: 0.25,
        avgStake: 15.50,
        roi: 0.25,
        profit: 350.50,
      },
      history: generatePerformanceHistory(),
      suggestedBet: {
        name: 'Vizinhos do 8',
        risk: 'High',
        stake: 5.00,
        expectedReturn: 175.00,
        probability: 0.055,
        rationale: "Gatilho 17 saiu. O ML identificou o número 8 como o pico de probabilidade na região. Alto Edge validado. Raio 3.",
      },
    },
    {
      id: 2,
      name: "Padrão Terminal 5",
      type: 'IA',
      confidence: 0.78,
      tags: ['Momentum', 'Terminal'],
      kpis: {
        samples: 45,
        hitRate: 0.031,
        probPerSpin: 0.15,
        avgStake: 12.00,
        roi: 0.10,
        profit: 85.20,
      },
      history: generatePerformanceHistory(),
      suggestedBet: null, // Sem aposta no momento
    },
     {
      id: 3,
      name: "Tensão C3 (Coluna)",
      type: 'Manual',
      confidence: 0.65,
      tags: ['Atraso', 'Recuperação'],
      kpis: {
        samples: 200,
        hitRate: 0.324,
        probPerSpin: 0.80,
        avgStake: 2.00,
        roi: 0.05,
        profit: 15.00,
      },
      history: generatePerformanceHistory(),
      suggestedBet: null, // Sem aposta no momento
    },
  ];

  // --- SIMULAÇÃO ---
  const simulation: BankrollSimulation = {
    initialBank: 1000,
    totalStaked: 5320.50,
    netProfit: 185.30,
    roiCompound: 0.185,
    interpretation: "O sistema demonstrou forte resiliência em ciclos de baixa, mantendo o ROI acima do esperado (15%) devido ao acionamento estratégico de Kelly Criterion nos picos de confiança.",
  };

  // --- CONTEXTO ---
  const context: ContextData = {
    latestNumber: 16,
    hotZone: 'Tier-Volante Superior (19, 4, 21)',
    sequence: '3 Vermelhos/Altos, Ping-Pong Par/Ímpar',
    insights: 'O Z-Score do Low está em -2.5, indicando forte atraso. A recuperação é iminente, mas o ML foca no cluster de vizinhança 28/35.',
    historyStrip: [32, 1, 15, 26, 4, 21, 2, 19, 3, 34, 0, 16, 12, 5, 23],
  };

  return { kpis, strategies, simulation, context };
};