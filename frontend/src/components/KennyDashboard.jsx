import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================================
// KENNY — AI Options Day Trading Dashboard
// api.luckykenny.cloud (REST) + ws.luckykenny.cloud (WebSocket)
// ============================================================================

const API = "https://api.luckykenny.cloud";
const WS = "wss://ws.luckykenny.cloud";
const SYMS = ["SPY","QQQ","TSLA","NVDA","META","AAPL","GOOGL","MSFT","AMZN","AMD","NFLX"];

// ── Hooks ──

function useVerdicts() {
  const [active, setActive] = useState([]);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ verdicts_today:0, go_count:0, skip_count:0, active_count:0, wins:0, losses:0, win_rate:0, pnl:0 });
  const [wsOk, setWsOk] = useState(false);
  const ws = useRef(null);
  const rc = useRef(0);

  const load = useCallback(async () => {
    try {
      const [a, h, s] = await Promise.all([
        fetch(`${API}/api/verdicts/latest`).then(r => r.ok ? r.json() : {}),
        fetch(`${API}/api/verdicts?limit=30`).then(r => r.ok ? r.json() : {}),
        fetch(`${API}/api/verdicts/stats`).then(r => r.ok ? r.json() : {}),
      ]);
      setActive(Array.isArray(a) ? a : a.verdicts || []);
      setHistory(Array.isArray(h) ? h : h.verdicts || []);
      if (s.totals) {
        const go = s.go_verdicts || {};
        setStats(prev => ({ ...prev, go_count: go.total_go || 0, verdicts_today: go.total_go || 0 }));
      } else {
        setStats(s);
      }
    } catch {}
  }, []);

  useEffect(() => {
    load();
    const poll = setInterval(load, 12000);
    function connectWs() {
      try {
        const s = new WebSocket(WS);
        ws.current = s;
        s.onopen = () => { setWsOk(true); rc.current = 0; };
        s.onmessage = e => { try { const m = JSON.parse(e.data); if (m.type === "verdict") load(); } catch {} };
        s.onclose = () => { setWsOk(false); setTimeout(connectWs, Math.min(1e3 * 2 ** rc.current, 30e3)); rc.current++; };
        s.onerror = () => s.close();
      } catch { setTimeout(connectWs, 5e3); }
    }
    connectWs();
    return () => { clearInterval(poll); ws.current?.close(); };
  }, [load]);

  return { active, history, stats, wsOk };
}

function useHealth() {
  const [h, setH] = useState(null);
  useEffect(() => {
    const check = () => fetch(`${API}/health`).then(r => r.ok ? r.json() : null).then(setH).catch(() => setH(null));
    check(); const p = setInterval(check, 10000); return () => clearInterval(p);
  }, []);
  return h;
}

// ── Main ──

export default function KennyDashboard() {
  const { active, history, stats, wsOk } = useVerdicts();
  const health = useHealth();
  const [now, setNow] = useState(new Date());
  const [selectedSym, setSelectedSym] = useState(null);
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(t); }, []);

  const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  const mktOpen = et.getDay() > 0 && et.getDay() < 6 && (et.getHours()*60+et.getMinutes()) >= 570 && (et.getHours()*60+et.getMinutes()) < 960;
  const dbOk = health?.components?.database?.status === "healthy";
  const mlOk = health?.components?.ml_models?.status && health.components.ml_models.status !== "not_loaded";

  const filteredActive = (selectedSym ? active.filter(v => v.symbol === selectedSym) : active).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  const filteredHistory = selectedSym ? history.filter(v => v.symbol === selectedSym) : history;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
        :root {
          --bg-0: #08090d; --bg-1: #0e1018; --bg-2: #141720; --bg-3: #1a1e2a;
          --bg-hover: #1f2435; --border: #1e2334; --border-bright: #2a3045;
          --t1: #eef0f4; --t2: #9ba1b0; --t3: #5c6274;
          --green: #00e676; --green-d: rgba(0,230,118,.1); --green-g: rgba(0,230,118,.25);
          --red: #ff5252; --red-d: rgba(255,82,82,.1);
          --blue: #448aff; --blue-d: rgba(68,138,255,.1);
          --amber: #ffc107; --amber-d: rgba(255,193,7,.1);
          --mono: 'IBM Plex Mono', monospace; --sans: 'DM Sans', sans-serif;
        }
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: var(--bg-0); color: var(--t1); font-family: var(--sans); -webkit-font-smoothing: antialiased; overflow: hidden; }

        .app { display: flex; flex-direction: column; height: 100vh; }

        /* ── TOP BAR ── */
        .topbar {
          display: flex; align-items: center; justify-content: space-between;
          height: 42px; padding: 0 16px; background: var(--bg-1);
          border-bottom: 1px solid var(--border); flex-shrink: 0; z-index: 50;
        }
        .topbar-left { display: flex; align-items: center; gap: 10px; }
        .logo { font-family: var(--mono); font-weight: 700; font-size: 15px; color: var(--green); letter-spacing: 3px; }
        .mkt-badge {
          font-family: var(--mono); font-size: 10px; font-weight: 600;
          padding: 2px 8px; border-radius: 3px; letter-spacing: .5px;
        }
        .mkt-open { background: var(--green-d); color: var(--green); }
        .mkt-closed { background: var(--red-d); color: var(--red); }
        .topbar-right { display: flex; align-items: center; gap: 14px; }
        .dots { display: flex; align-items: center; gap: 10px; }
        .dot-item { display: flex; align-items: center; gap: 4px; font-family: var(--mono); font-size: 10px; color: var(--t3); }
        .dot { width: 5px; height: 5px; border-radius: 50%; }
        .d-g { background: var(--green); box-shadow: 0 0 5px var(--green-g); }
        .d-r { background: var(--red); }
        .d-a { background: var(--amber); }
        .clock { font-family: var(--mono); font-size: 11px; color: var(--t2); font-weight: 500; }

        /* ── STATS BAR ── */
        .statsbar {
          display: flex; align-items: center; height: 36px; padding: 0 16px;
          background: var(--bg-1); border-bottom: 1px solid var(--border);
          gap: 2px; flex-shrink: 0;
        }
        .sstat {
          display: flex; align-items: center; gap: 5px;
          padding: 0 12px; height: 100%;
          font-family: var(--mono); font-size: 11px;
          border-right: 1px solid var(--border);
        }
        .sstat:last-child { border-right: none; }
        .sstat-label { color: var(--t3); font-size: 9px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
        .sstat-val { font-weight: 700; color: var(--t1); }
        .sstat-val.g { color: var(--green); }
        .sstat-val.r { color: var(--red); }
        .sstat-val.b { color: var(--blue); }
        .sstat-val.m { color: var(--t3); }

        /* ── MAIN LAYOUT ── */
        .main { display: flex; flex: 1; overflow: hidden; }

        /* ── SIDEBAR (WATCHLIST) ── */
        .sidebar {
          width: 72px; background: var(--bg-1); border-right: 1px solid var(--border);
          display: flex; flex-direction: column; flex-shrink: 0; overflow-y: auto;
        }
        .sidebar::-webkit-scrollbar { width: 0; }
        .sb-title {
          font-family: var(--mono); font-size: 8px; font-weight: 600;
          text-transform: uppercase; letter-spacing: 1.5px; color: var(--t3);
          text-align: center; padding: 10px 0 6px; border-bottom: 1px solid var(--border);
        }
        .sym-btn {
          display: flex; flex-direction: column; align-items: center; justify-content: center;
          padding: 8px 4px; border: none; background: transparent; cursor: pointer;
          border-bottom: 1px solid var(--border); transition: background .1s;
          min-height: 52px; border-left: 2px solid transparent;
        }
        .sym-btn:hover { background: var(--bg-hover); }
        .sym-btn.active { background: var(--bg-3); border-left-color: var(--green); }
        .sym-name { font-family: var(--mono); font-size: 11px; font-weight: 700; color: var(--t1); }
        .sym-dir { font-family: var(--mono); font-size: 9px; font-weight: 600; margin-top: 2px; }
        .sym-dir.c { color: var(--green); }
        .sym-dir.p { color: var(--red); }
        .sym-dir.n { color: var(--t3); }

        /* ── CENTER PANEL ── */
        .center { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

        .active-section {
          flex: 1; min-height: 0; overflow-y: auto; padding: 16px 20px;
          background: var(--bg-0);
        }
        .active-section::-webkit-scrollbar { width: 3px; }
        .active-section::-webkit-scrollbar-track { background: transparent; }
        .active-section::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        .sec-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
        .sec-head h2 {
          font-family: var(--mono); font-size: 10px; font-weight: 600;
          text-transform: uppercase; letter-spacing: 2px; color: var(--t3);
        }
        .sec-count {
          font-family: var(--mono); font-size: 9px; padding: 1px 6px;
          border-radius: 8px; background: var(--bg-3); color: var(--t3);
        }
        .sec-line { flex: 1; height: 1px; background: var(--border); }

        /* Empty state */
        .empty {
          display: flex; flex-direction: column; align-items: center; justify-content: center;
          padding: 60px 20px; text-align: center;
        }
        .empty-pulse {
          width: 48px; height: 48px; border-radius: 50%;
          background: var(--green-d); display: flex; align-items: center; justify-content: center;
          margin-bottom: 16px; animation: pulse 2.5s infinite;
        }
        .empty-pulse-inner {
          width: 12px; height: 12px; border-radius: 50%; background: var(--green); opacity: .6;
        }
        @keyframes pulse {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.15); opacity: .7; }
        }
        .empty-text { font-size: 13px; color: var(--t2); font-weight: 500; }
        .empty-sub { font-size: 11px; color: var(--t3); margin-top: 4px; }

        /* Verdict card */
        .v-card {
          background: var(--bg-2); border: 1px solid var(--border);
          border-radius: 8px; margin-bottom: 8px; overflow: hidden;
          transition: border-color .15s, background .15s;
        }
        .v-card:hover { border-color: var(--border-bright); background: var(--bg-hover); }
        .v-card.go-c { border-left: 3px solid var(--green); }
        .v-card.go-p { border-left: 3px solid var(--red); }
        .v-card.skip { border-left: 3px solid var(--t3); opacity: .65; }
        .v-inner { padding: 14px 16px; }

        .v-row1 { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
        .v-left { display: flex; align-items: center; gap: 8px; }
        .v-sym { font-family: var(--mono); font-size: 18px; font-weight: 700; }
        .v-dir {
          font-family: var(--mono); font-size: 11px; font-weight: 700;
          padding: 2px 8px; border-radius: 3px;
        }
        .v-dir.c { background: var(--green-d); color: var(--green); }
        .v-dir.p { background: var(--red-d); color: var(--red); }
        .v-tag {
          font-family: var(--mono); font-size: 10px; font-weight: 700;
          padding: 2px 8px; border-radius: 3px; letter-spacing: 1px;
        }
        .v-tag.go { background: var(--green-d); color: var(--green); border: 1px solid rgba(0,230,118,.15); }
        .v-tag.sk { background: rgba(92,98,116,.12); color: var(--t3); }

        .v-right { display: flex; align-items: baseline; gap: 6px; }
        .v-conf { font-family: var(--mono); font-size: 20px; font-weight: 700; }
        .v-conf.hi { color: var(--green); }
        .v-conf.md { color: var(--amber); }
        .v-conf.lo { color: var(--t3); }
        .v-pct { font-family: var(--mono); font-size: 11px; color: var(--t3); }
        .v-age { font-family: var(--mono); font-size: 10px; color: var(--t3); margin-left: 8px; }

        .v-plan { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-bottom: 10px; }
        .v-plan-item { background: rgba(0,0,0,.3); border-radius: 4px; padding: 6px 8px; }
        .v-plan-label { font-family: var(--mono); font-size: 8px; color: var(--t3); text-transform: uppercase; letter-spacing: 1px; }
        .v-plan-val { font-family: var(--mono); font-size: 14px; font-weight: 700; margin-top: 2px; }
        .v-plan-val.entry { color: var(--t1); }
        .v-plan-val.tp { color: var(--green); }
        .v-plan-val.sl { color: var(--red); }
        .v-plan-val.rr { color: var(--blue); }

        .v-tfs { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
        .tf {
          font-family: var(--mono); font-size: 9px; font-weight: 600;
          padding: 2px 6px; border-radius: 3px;
        }
        .tf.c { background: var(--green-d); color: var(--green); }
        .tf.p { background: var(--red-d); color: var(--red); }
        .tf.h { background: rgba(92,98,116,.08); color: var(--t3); }
        .v-agree { font-family: var(--mono); font-size: 9px; color: var(--t3); margin-left: 4px; }
        .v-reason { font-size: 11px; color: var(--t3); margin-top: 8px; font-style: italic; }
        .v-result { font-family: var(--mono); font-size: 11px; font-weight: 700; margin-top: 6px; }
        .v-result.w { color: var(--green); }
        .v-result.l { color: var(--red); }

        /* ── RIGHT PANEL ── */
        .right-panel {
          width: 320px; background: var(--bg-1); border-left: 1px solid var(--border);
          display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden;
        }
        .right-head {
          padding: 10px 14px; border-bottom: 1px solid var(--border);
          font-family: var(--mono); font-size: 10px; font-weight: 600;
          text-transform: uppercase; letter-spacing: 2px; color: var(--t3);
          display: flex; align-items: center; justify-content: space-between;
        }
        .right-body { flex: 1; overflow-y: auto; padding: 8px; }
        .right-body::-webkit-scrollbar { width: 3px; }
        .right-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        .mv {
          display: flex; align-items: center; gap: 8px; padding: 8px 10px;
          border-radius: 6px; margin-bottom: 4px; transition: background .1s;
          cursor: default; border-left: 2px solid transparent;
        }
        .mv:hover { background: var(--bg-hover); }
        .mv.go-c { border-left-color: var(--green); }
        .mv.go-p { border-left-color: var(--red); }
        .mv.sk { border-left-color: var(--t3); opacity: .6; }
        .mv-sym { font-family: var(--mono); font-size: 12px; font-weight: 700; color: var(--t1); width: 42px; }
        .mv-dir { font-family: var(--mono); font-size: 10px; font-weight: 700; width: 36px; }
        .mv-dir.c { color: var(--green); }
        .mv-dir.p { color: var(--red); }
        .mv-verdict { font-family: var(--mono); font-size: 9px; font-weight: 700; width: 30px; }
        .mv-verdict.go { color: var(--green); }
        .mv-verdict.sk { color: var(--t3); }
        .mv-conf { font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--t2); flex: 1; text-align: right; }
        .mv-time { font-family: var(--mono); font-size: 9px; color: var(--t3); width: 50px; text-align: right; }
        .right-empty { text-align: center; padding: 30px 16px; font-size: 11px; color: var(--t3); font-family: var(--mono); }

        /* ── FILTER BAR ── */
        .filter-bar {
          display: flex; align-items: center; gap: 8px; padding: 6px 20px;
          background: var(--bg-2); border-bottom: 1px solid var(--border);
          font-family: var(--mono); font-size: 11px; color: var(--t2);
        }
        .filter-clear {
          background: var(--bg-3); border: 1px solid var(--border);
          color: var(--t2); padding: 2px 10px; border-radius: 4px;
          font-family: var(--mono); font-size: 10px; cursor: pointer;
        }
        .filter-clear:hover { background: var(--bg-hover); color: var(--t1); }

        .foot {
          height: 22px; display: flex; align-items: center; justify-content: center;
          background: var(--bg-1); border-top: 1px solid var(--border);
          font-family: var(--mono); font-size: 9px; color: var(--t3);
          letter-spacing: .5px; flex-shrink: 0;
        }

        @media (max-width: 900px) {
          .right-panel { display: none; }
          .sidebar { width: 56px; }
          .sym-name { font-size: 9px; }
        }
      `}</style>

      <div className="app">
        <div className="topbar">
          <div className="topbar-left">
            <span className="logo">KENNY</span>
            <span className={`mkt-badge ${mktOpen ? "mkt-open" : "mkt-closed"}`}>
              {mktOpen ? "LIVE" : "CLOSED"}
            </span>
          </div>
          <div className="topbar-right">
            <div className="dots">
              <span className="dot-item"><span className={`dot ${health ? "d-g" : "d-r"}`}/> API</span>
              <span className="dot-item"><span className={`dot ${dbOk ? "d-g" : "d-r"}`}/> DB</span>
              <span className="dot-item"><span className={`dot ${mlOk ? "d-g" : "d-a"}`}/> ML</span>
              <span className="dot-item"><span className={`dot ${wsOk ? "d-g" : "d-a"}`}/> WS</span>
            </div>
            <span className="clock">
              {et.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit", hour12: true })} ET
            </span>
          </div>
        </div>

        <div className="statsbar">
          <Stat l="Verdicts" v={stats.verdicts_today} />
          <Stat l="GO" v={stats.go_count} c="g" />
          <Stat l="Skip" v={stats.skip_count} c="m" />
          <Stat l="Active" v={stats.active_count} c="b" />
          <Stat l="Wins" v={stats.wins} c="g" />
          <Stat l="Losses" v={stats.losses} c="r" />
          <Stat l="WR" v={stats.win_rate ? `${stats.win_rate.toFixed(0)}%` : "—"} c={stats.win_rate >= 60 ? "g" : "m"} />
          <Stat l="P&L" v={stats.pnl ? `$${stats.pnl > 0 ? "+" : ""}${stats.pnl.toFixed(0)}` : "$0"} c={stats.pnl > 0 ? "g" : stats.pnl < 0 ? "r" : "m"} />
        </div>

        <div className="main">
          <div className="sidebar">
            <div className="sb-title">Watch</div>
            <button className={`sym-btn ${!selectedSym ? "active" : ""}`} onClick={() => setSelectedSym(null)}>
              <span className="sym-name" style={{ color: "var(--t2)" }}>ALL</span>
            </button>
            {(health?.symbols || SYMS).map(s => (
              <button key={s} className={`sym-btn ${selectedSym === s ? "active" : ""}`}
                onClick={() => setSelectedSym(selectedSym === s ? null : s)}>
                <span className="sym-name">{s}</span>
                <span className="sym-dir n">—</span>
              </button>
            ))}
          </div>

          <div className="center">
            {selectedSym && (
              <div className="filter-bar">
                Filtering: <strong>{selectedSym}</strong>
                <button className="filter-clear" onClick={() => setSelectedSym(null)}>Clear</button>
              </div>
            )}
            <div className="active-section">
              <div className="sec-head">
                <h2>Active Verdicts</h2>
                <span className="sec-count">{filteredActive.length}</span>
                <div className="sec-line" />
              </div>
              {filteredActive.length > 0 ? (
                filteredActive.map((v, i) => <VCard key={v.id || i} v={v} isActive />)
              ) : (
                <div className="empty">
                  <div className="empty-pulse"><div className="empty-pulse-inner" /></div>
                  <div className="empty-text">No active verdicts</div>
                  <div className="empty-sub">
                    {mktOpen ? "Scanning — GO signals appear here in real-time" : "Market closed — verdicts generate during trading hours"}
                  </div>
                </div>
              )}
              <div className="sec-head" style={{ marginTop: 24 }}>
                <h2>Recent</h2>
                <span className="sec-count">{filteredHistory.length}</span>
                <div className="sec-line" />
              </div>
              {filteredHistory.length > 0 ? (
                filteredHistory.slice(0, 10).map((v, i) => <VCard key={v.id || i} v={v} />)
              ) : (
                <div style={{ textAlign: "center", padding: 20, color: "var(--t3)", fontFamily: "var(--mono)", fontSize: 11 }}>
                  No verdict history
                </div>
              )}
            </div>
          </div>

          <div className="right-panel">
            <div className="right-head">
              <span>History</span>
              <span className="sec-count">{filteredHistory.length}</span>
            </div>
            <div className="right-body">
              {filteredHistory.length > 0 ? (
                filteredHistory.map((v, i) => <MiniVerdict key={v.id || i} v={v} />)
              ) : (
                <div className="right-empty">No history yet</div>
              )}
            </div>
          </div>
        </div>

        <div className="foot">
          KENNY v1.0 — 32 Models · 4 TF · 11 Symbols{health?.version && ` · ${health.version}`}
        </div>
      </div>
    </>
  );
}

// ── Components ──

function Stat({ l, v, c = "" }) {
  return (
    <div className="sstat">
      <span className="sstat-label">{l}</span>
      <span className={`sstat-val ${c}`}>{v}</span>
    </div>
  );
}

function VCard({ v, isActive }) {
  const go = v.verdict === "GO";
  const call = v.direction === "CALL";
  const cls = go ? (call ? "go-c" : "go-p") : "skip";
  const conf = (v.confidence?.toFixed?.(1) || v.confidence);
  const confCls = conf >= 80 ? "hi" : conf >= 65 ? "md" : "lo";
  const age = v.timestamp ? Math.floor((Date.now() - new Date(v.timestamp).getTime()) / 60000) : 0;
  const align = v.timeframeAlignment || v.timeframe_alignment || {};

  return (
    <div className={`v-card ${cls}`}>
      <div className="v-inner">
        <div className="v-row1">
          <div className="v-left">
            <span className="v-sym">{v.symbol}</span>
            <span className={`v-dir ${call ? "c" : "p"}`}>{v.direction}</span>
            <span className={`v-tag ${go ? "go" : "sk"}`}>{v.verdict}</span>
          </div>
          <div className="v-right">
            <span className={`v-conf ${confCls}`}>{conf}</span>
            <span className="v-pct">%</span>
            {isActive && age > 0 && <span className="v-age">{age}m</span>}
          </div>
        </div>
        {go && (
          <div className="v-plan">
            <div className="v-plan-item"><div className="v-plan-label">Entry</div><div className="v-plan-val entry">${Number(v.entry).toFixed(2)}</div></div>
            <div className="v-plan-item"><div className="v-plan-label">Target</div><div className="v-plan-val tp">${Number(v.target).toFixed(2)}</div></div>
            <div className="v-plan-item"><div className="v-plan-label">Stop</div><div className="v-plan-val sl">${Number(v.stop).toFixed(2)}</div></div>
            <div className="v-plan-item"><div className="v-plan-label">R:R</div><div className="v-plan-val rr">{v.riskReward || v.risk_reward}</div></div>
          </div>
        )}
        <div className="v-tfs">
          {Object.entries(align).map(([tf, dir]) => (
            <span key={tf} className={`tf ${dir === "CALL" ? "c" : dir === "PUT" ? "p" : "h"}`}>{tf}:{dir}</span>
          ))}
          {(v.modelsAgreeing || v.models_agreeing) != null && (
            <span className="v-agree">{v.modelsAgreeing || v.models_agreeing}/4</span>
          )}
        </div>
        {v.reason && <div className="v-reason">{v.reason}</div>}
        {v.result && (
          <div className={`v-result ${v.result === "WIN" ? "w" : "l"}`}>
            {v.result}{v.pnl != null && ` ($${v.pnl > 0 ? "+" : ""}${v.pnl.toFixed(2)})`}
            {(v.closeReason || v.close_reason) && (
              <span style={{ color: "var(--t3)", fontWeight: 400, marginLeft: 6 }}>— {v.closeReason || v.close_reason}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function MiniVerdict({ v }) {
  const go = v.verdict === "GO";
  const call = v.direction === "CALL";
  const cls = go ? (call ? "go-c" : "go-p") : "sk";
  const conf = v.confidence?.toFixed?.(0) || v.confidence;
  const time = v.timestamp ? new Date(v.timestamp).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true }) : "";

  return (
    <div className={`mv ${cls}`}>
      <span className="mv-sym">{v.symbol}</span>
      <span className={`mv-dir ${call ? "c" : "p"}`}>{v.direction}</span>
      <span className={`mv-verdict ${go ? "go" : "sk"}`}>{v.verdict}</span>
      <span className="mv-conf">{conf}%</span>
      <span className="mv-time">{time}</span>
    </div>
  );
}