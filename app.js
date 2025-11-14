// frontend logic for Roleta Analyst (dark futurista)
// expects same-origin API: /api/*

let token = localStorage.getItem("token") || null;
const apiBase = "/api";

function setAuthHeader(headers = {}) {
  if (token) headers["Authorization"] = "Bearer " + token;
  return headers;
}

/* ------------------- LOGIN / LOGOUT ------------------- */
async function login() {
  const user = document.getElementById("login-user").value || "";
  const pass = document.getElementById("login-pass").value || "";
  const form = new URLSearchParams();
  form.append("username", user);
  form.append("password", pass);

  try {
    const r = await fetch(apiBase + "/auth/login", { method: "POST", body: form });
    if (!r.ok) {
      alert("Usuário ou senha incorretos.");
      return;
    }
    const j = await r.json();
    token = j.access_token;
    localStorage.setItem("token", token);
    onLoginSuccess(user);
  } catch (e) {
    alert("Erro ao conectar ao servidor.");
  }
}

function logout() {
  token = null;
  localStorage.removeItem("token");
  document.getElementById("login-block").style.display = "";
  document.getElementById("user-info").style.display = "none";
  location.reload();
}

function onLoginSuccess(username){
  document.getElementById("login-block").style.display = "none";
  document.getElementById("user-info").style.display = "";
  document.getElementById("username-label").innerText = username;
  carregarKPIs();
  carregarGrafico();
  carregarAnalise();
  carregarLogs();
  setInterval(carregarKPIs, 8000);
  setInterval(carregarAnalise, 12000);
  setInterval(carregarGrafico, 15000);
  setInterval(carregarLogs, 5000);
}

/* ------------------- NAV ------------------- */
document.querySelectorAll(".nav-btn").forEach(btn=>{
  btn.addEventListener("click", ()=> {
    document.querySelectorAll(".nav-btn").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    const p = btn.getAttribute("data-panel");
    document.querySelectorAll(".panel").forEach(panel => panel.classList.remove("active"));
    const el = document.getElementById("panel-" + p);
    if (el) el.classList.add("active");
  });
});

/* ------------------- KPIs ------------------- */
async function carregarKPIs(){
  if (!token) return;
  try {
    const r = await fetch(apiBase + "/kpis", { headers: setAuthHeader() });
    if (!r.ok) {
      // if unauthorized, clear login
      if (r.status === 401) { logout(); alert("Sessão expirada. Faça login novamente."); }
      return;
    }
    const j = await r.json();
    document.getElementById("saldo").innerText = "R$ " + Number(j.saldo_simulado || 0).toFixed(2);
    document.getElementById("pl").innerText = "R$ " + Number(j.p_l || 0).toFixed(2);
    document.getElementById("roi").innerText = (Number(j.roi || 0)).toFixed(2) + "%";
    const btn = document.getElementById("bot-toggle");
    if ((j.bot_status || "PAUSED") === "RUNNING") {
      btn.innerText = "⏸ Pausar"; btn.classList.add("running"); btn.classList.remove("paused");
      document.getElementById("bot-indicator").className = "indicator running";
      document.getElementById("bot-indicator").innerText = "RUNNING";
    } else {
      btn.innerText = "▶ Iniciar"; btn.classList.remove("running"); btn.classList.add("paused");
      document.getElementById("bot-indicator").className = "indicator paused";
      document.getElementById("bot-indicator").innerText = "PAUSED";
    }
  } catch (e) {
    console.error("KPIs error", e);
  }
}

/* ------------------- BOT TOGGLE ------------------- */
async function toggleBot(){
  if (!token) return alert("Faça login primeiro.");
  const btn = document.getElementById("bot-toggle");
  const newStatus = btn.classList.contains("running") ? "PAUSED" : "RUNNING";
  try {
    const r = await fetch(apiBase + "/bot/toggle", {
      method: "POST",
      headers: setAuthHeader({"Content-Type":"application/json"}),
      body: JSON.stringify({ status: newStatus })
    });
    if (r.ok) carregarKPIs();
  } catch (e) { console.error(e) }
}

/* ------------------- PERFORMANCE CHART ------------------- */
async function carregarGrafico(){
  if (!token) return;
  try {
    const r = await fetch(apiBase + "/performance-history", { headers: setAuthHeader() });
    if (!r.ok) return;
    const j = await r.json();
    Plotly.newPlot("chart", [{ x: j.labels || [], y: j.data || [], type: "scatter", line: { shape: "spline" } }], { margin:{t:20}, paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)" });
  } catch(e) { console.error(e) }
}

/* ------------------- ANALYSIS / GEMINI ------------------- */
async function carregarAnalise(){
  if (!token) return;
  try {
    const r = await fetch(apiBase + "/ai/gemini-analysis", { headers: setAuthHeader() });
    if (!r.ok) return;
    const j = await r.json();
    if (j.status_code === 200) document.getElementById("analysis-box").innerText = j.analysis;
    else document.getElementById("analysis-box").innerText = "Nenhuma análise.";
  } catch(e){ console.error(e) }
}

/* ------------------- CONFLUENCE / DECISIONS ------------------- */
async function carregarConfluence(){
  // placeholder: usa /analysis/latest se existir
  if (!token) return;
  try {
    const r = await fetch(apiBase + "/analysis/latest", { headers: setAuthHeader() });
    if (!r.ok) { document.getElementById("confluence").innerText = "API indisponível"; return; }
    const j = await r.json();
    if (j.status_code === 200) {
      const data = j.data;
      // exibir alguns campos relevantes
      const keys = Object.keys(data).filter(k=>k.startsWith("zscore")||k.endsWith("_ewma_short")).slice(0,10);
      document.getElementById("confluence").innerHTML = keys.map(k=>`<div><strong>${k}</strong>: ${data[k]}</div>`).join("");
    } else {
      document.getElementById("confluence").innerText = "Nenhum dado.";
    }
  } catch(e){ console.error(e) }
}

/* ------------------- DECISIONS / JOGADAS MANUAIS ------------------- */
async function carregarDecisions(){
  // placeholder: GET /api/performance-history utilizado como demo
  if (!token) return;
  try {
    const r = await fetch(apiBase + "/performance-history", { headers: setAuthHeader() });
    if (!r.ok) return;
    const j = await r.json();
    document.getElementById("decisions-list").innerText = JSON.stringify(j);
  } catch(e){ console.error(e) }
}

function criarJogada(){
  const nums = document.getElementById("manual-numbers").value.trim();
  const stake = parseFloat(document.getElementById("manual-stake").value) || 0;
  if (!nums) return alert("Informe números.");
  // aqui apenas mostra localmente — você pode chamar o endpoint de decisões se existir
  alert("Jogada criada: " + nums + " stake: " + stake);
}

/* ------------------- LOGS ------------------- */
async function carregarLogs(){
  if (!token) return;
  try {
    const r = await fetch(apiBase + "/ui/logs", { headers: setAuthHeader() });
    if (!r.ok) return;
    const j = await r.json();
    const lines = (j.logs || []).slice(-200).reverse();
    document.getElementById("log-box").innerHTML = lines.map(l=>`<div>${l}</div>`).join("");
  } catch(e){ console.error(e) }
}

/* ------------------- CONFIG FRONTEND TEST ------------------- */
async function testFrontendConfig(){
  try {
    const r = await fetch(apiBase + "/config/frontend");
    const j = await r.json();
    document.getElementById("api-base").innerText = location.origin;
    document.getElementById("api-base-2").innerText = location.origin;
    document.getElementById("spin-interval").innerText = j.SPIN_INTERVAL_SECONDS || j.spin_interval || "-";
    alert("Config frontend carregada.");
  } catch(e){ alert("Erro ao testar config frontend."); }
}

/* ------------------- UI INIT ------------------- */
(function init(){
  document.getElementById("api-base").innerText = "/";
  if (token) {
    // try to validate token by calling /kpis
    carregarKPIs().then(()=> {
      document.getElementById("login-block").style.display = "none";
      document.getElementById("user-info").style.display = "";
    }).catch(()=>{});
  }
})();
