import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================================
// KENNY — Trade Verdict Dashboard (Production)
// Wired to live backend API endpoints
// ============================================================================

const API_BASE = "";
const WS_BASE = "wss://d1864183bmmxem.cloudfront.net";

const SYMBOLS = ["SPY", "QQQ", "AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "AMD", "TSLA"];

// ============================================================================
// API HOOKS
// ============================================================================

function useVerdicts() {
  const [active, setActive] = useState([]);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({
    verdicts_today: 0, go_count: 0, skip_count: 0,
    active_count: 0, wins: 0, losses: 0, win_rate: 0, pnl: 0,
  });
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectRef = useRef(0);

  const fetchVerdicts = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/verdicts`);
      const data = await res.json();
      setActive(data.active || []);
      setHistory(data.history || []);
      setConnected(true);
    } catch (e) {
      setConnected(false);
    }
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/verdicts/stats`);
        setConnected(true);
      const data = await res.json();
      setStats(data);
    } catch (e) { /* silent */ }
  }, []);

  // WebSocket connection
  useEffect(() => {
    let alive = true;

    function connect() {
      if (!alive) return;
      const ws = new WebSocket(`${WS_BASE}/ws/signals`);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectRef.current = 0;
      };

      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);

          if (msg.type === "connected") {
            // Initial state from server
            if (msg.activeVerdicts) setActive(msg.activeVerdicts);
            if (msg.stats) setStats(msg.stats);
          } else if (msg.type === "verdict") {
            // New verdict arrived
            if (msg.status === "active" && msg.verdict === "GO") {
              setActive(prev => {
                const filtered = prev.filter(v => v.symbol !== msg.symbol);
                return [msg, ...filtered];
              });
            } else if (msg.status === "closed") {
              setActive(prev => prev.filter(v => v.symbol !== msg.symbol));
              setHistory(prev => [msg, ...prev].slice(0, 50));
            }
            // Refresh stats
            fetchStats();
          } else if (msg.type === "heartbeat") {
            ws.send("ping");
          }
        } catch (e) { /* ignore */ }
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        if (alive) {
          const delay = Math.min(3000 * (reconnectRef.current + 1), 15000);
          reconnectRef.current++;
          setTimeout(connect, delay);
        }
      };

      ws.onerror = () => ws.close();
    }

    connect();

    // Poll as fallback every 5 seconds
    const pollInterval = setInterval(() => {
      fetchVerdicts();
      fetchStats();
    }, 5000);

    return () => {
      alive = false;
      clearInterval(pollInterval);
      if (wsRef.current) wsRef.current.close();
    };
  }, [fetchVerdicts, fetchStats]);

  // Initial fetch
  useEffect(() => {
    fetchVerdicts();
    fetchStats();
  }, [fetchVerdicts, fetchStats]);

  return { active, history, stats, connected };
}

// ============================================================================
// COMPONENTS
// ============================================================================

function TimeframeBar({ alignment, confidences }) {
  const tfs = ["1min", "5min", "15min", "1hr"];
  return (
    <div className="tf-bar">
      {tfs.map((tf) => {
        const dir = alignment?.[tf] || "HOLD";
        const conf = confidences?.[tf] || 0;
        const cls = dir === "CALL" ? "tf-call" : dir === "PUT" ? "tf-put" : "tf-hold";
        return (
          <div key={tf} className={`tf-cell ${cls}`}>
            <span className="tf-label">{tf}</span>
            <span className="tf-dir">
              {dir === "HOLD" || dir === "STALE" ? "—" : dir === "CALL" ? "▲" : "▼"}
            </span>
            {conf > 0 && <span className="tf-conf">{conf.toFixed(0)}%</span>}
          </div>
        );
      })}
    </div>
  );
}

function VerdictCard({ verdict }) {
  const isGo = verdict.verdict === "GO";
  const isCall = verdict.direction === "CALL";

  return (
    <div className={`verdict-card ${isGo ? (isCall ? "verdict-call" : "verdict-put") : "verdict-skip"}`}>
      <div className="verdict-header">
        <div className="verdict-left">
          <span className={`verdict-badge ${isGo ? "badge-go" : "badge-skip"}`}>
            {isGo ? "GO" : "SKIP"}
          </span>
          <span className="verdict-symbol">{verdict.symbol}</span>
          <span className={`verdict-direction ${isCall ? "dir-call" : "dir-put"}`}>
            {isCall ? "▲ CALL" : "▼ PUT"}
          </span>
        </div>
        <div className="verdict-right">
          <span className="verdict-confidence">{verdict.confidence?.toFixed?.(1) || verdict.confidence}%</span>
          <span className="verdict-age">{verdict.age || ""}</span>
        </div>
      </div>

      <TimeframeBar
        alignment={verdict.timeframeAlignment}
        confidences={verdict.tfConfidences}
      />

      {isGo && verdict.entry > 0 && (
        <div className="trade-plan">
          <div className="plan-row">
            <div className="plan-item">
              <span className="plan-label">Entry</span>
              <span className="plan-value">${verdict.entry?.toFixed?.(2) || verdict.entry}</span>
            </div>
            <div className="plan-item">
              <span className="plan-label">Target</span>
              <span className="plan-value plan-target">${verdict.target?.toFixed?.(2) || verdict.target}</span>
            </div>
            <div className="plan-item">
              <span className="plan-label">Stop</span>
              <span className="plan-value plan-stop">${verdict.stop?.toFixed?.(2) || verdict.stop}</span>
            </div>
            <div className="plan-item">
              <span className="plan-label">R:R</span>
              <span className="plan-value plan-rr">{verdict.riskReward}</span>
            </div>
          </div>
        </div>
      )}

      <p className="verdict-reason">{verdict.reason}</p>
    </div>
  );
}

function HistoryRow({ item }) {
  const isWin = item.result === "WIN";
  const isLoss = item.result === "LOSS";
  const isCall = item.direction === "CALL";
  const pnlVal = item.pnl || 0;
  const conf = item.confidence?.toFixed?.(1) || item.confidence || "—";

  // Parse time
  let timeStr = "—";
  if (item.timestamp || item.closedAt) {
    try {
      const d = new Date(item.closedAt || item.timestamp);
      timeStr = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
    } catch (e) { /* ignore */ }
  }

  return (
    <div className="history-row">
      <span className="hist-time">{timeStr}</span>
      <span className="hist-symbol">{item.symbol}</span>
      <span className={`hist-dir ${isCall ? "dir-call" : "dir-put"}`}>
        {isCall ? "▲" : "▼"} {item.direction}
      </span>
      <span className="hist-confidence">{conf}%</span>
      <span className={`hist-verdict ${item.verdict === "GO" ? "badge-go-sm" : "badge-skip-sm"}`}>
        {item.verdict}
      </span>
      <span className={`hist-result ${isWin ? "result-win" : isLoss ? "result-loss" : "result-skip"}`}>
        {item.result || "—"}
      </span>
      <span className={`hist-pnl ${pnlVal > 0 ? "pnl-pos" : pnlVal < 0 ? "pnl-neg" : "pnl-none"}`}>
        {pnlVal !== 0 ? `${pnlVal > 0 ? "+" : ""}$${pnlVal.toFixed(0)}` : "—"}
      </span>
      <span className="hist-reason">{item.closeReason || item.reason?.slice(0, 40) || "—"}</span>
    </div>
  );
}

function Watchlist({ symbols, activeSymbol, onSelect, activeVerdicts }) {
  const verdictMap = {};
  activeVerdicts.forEach(v => {
    if (v.verdict === "GO") verdictMap[v.symbol] = v.direction;
  });

  return (
    <div className="watchlist">
      <div className="watchlist-header">WATCHLIST</div>
      {symbols.map((s) => {
        const v = verdictMap[s];
        const isActive = s === activeSymbol;
        return (
          <button
            key={s}
            className={`watchlist-item ${isActive ? "wl-active" : ""}`}
            onClick={() => onSelect(s)}
          >
            <span className="wl-symbol">{s}</span>
            {v && (
              <span className={`wl-verdict ${v === "CALL" ? "wl-call" : "wl-put"}`}>
                {v === "CALL" ? "▲" : "▼"}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

function DailyStats({ stats }) {
  return (
    <div className="daily-stats">
      <div className="stat-box">
        <span className="stat-num">{stats.verdicts_today}</span>
        <span className="stat-label">Verdicts</span>
      </div>
      <div className="stat-box">
        <span className="stat-num stat-go">{stats.go_count}</span>
        <span className="stat-label">GO</span>
      </div>
      <div className="stat-box">
        <span className="stat-num stat-skip">{stats.skip_count}</span>
        <span className="stat-label">Skip</span>
      </div>
      <div className="stat-box">
        <span className="stat-num stat-win">{stats.wins}</span>
        <span className="stat-label">Wins</span>
      </div>
      <div className="stat-box">
        <span className="stat-num stat-loss">{stats.losses}</span>
        <span className="stat-label">Losses</span>
      </div>
      <div className="stat-box">
        <span className="stat-num stat-wr">{stats.win_rate}%</span>
        <span className="stat-label">Win Rate</span>
      </div>
      <div className="stat-box">
        <span className={`stat-num ${stats.pnl >= 0 ? "stat-win" : "stat-loss"}`}>
          {stats.pnl >= 0 ? "+" : ""}${stats.pnl}
        </span>
        <span className="stat-label">P&L</span>
      </div>
    </div>
  );
}

// ============================================================================
// MAIN DASHBOARD
// ============================================================================

export default function YelenaDashboard() {
  const [activeSymbol, setActiveSymbol] = useState(
    () => localStorage.getItem("yelena_symbol") || "SPY"
  );
  const [tab, setTab] = useState("verdicts"); // verdicts or history
  const [time, setTime] = useState(new Date());
  const { active, history, stats, connected } = useVerdicts();

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const handleSymbolSelect = (s) => {
    localStorage.setItem("yelena_symbol", s);
    setActiveSymbol(s);
  };

  const marketOpen = time.getUTCHours() >= 14 && time.getUTCHours() < 21;

  return (
    <div className="yelena-root">
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Instrument+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <header className="yelena-header">
        <div className="header-left">
          <span className="logo">KENNY</span>
          {/*<span className="logo-sub">v2</span>*/}
          <span className="header-divider" />
          <span className="market-status">
            <span className={`status-dot ${marketOpen ? "dot-open" : "dot-closed"}`} />
            {marketOpen ? "MARKET OPEN" : "MARKET CLOSED"}
          </span>
          <span className="header-tag">15m DELAYED</span>
        </div>
        <div className="header-right">
          <span className="header-clock">
            {time.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </span>
          <span className={`conn-status ${connected ? "conn-on" : "conn-off"}`}>
            <span className={`status-dot ${connected ? "dot-on" : "dot-off"}`} />
            {connected ? "LIVE" : "OFFLINE"}
          </span>
        </div>
      </header>

      {/* Body */}
      <div className="yelena-body">
        {/* Sidebar */}
        <aside className="yelena-sidebar">
          <Watchlist
            symbols={SYMBOLS}
            activeSymbol={activeSymbol}
            onSelect={handleSymbolSelect}
            activeVerdicts={active}
          />
        </aside>

        {/* Main */}
        <main className="yelena-main">
          <DailyStats stats={stats} />

          {/* Tabs */}
          <div className="tab-bar">
            <button
              className={`tab-btn ${tab === "verdicts" ? "tab-active" : ""}`}
              onClick={() => setTab("verdicts")}
            >
              <span className="pulse-dot" />
              Active Verdicts
              {active.length > 0 && <span className="tab-count">{active.length}</span>}
            </button>
            <button
              className={`tab-btn ${tab === "history" ? "tab-active" : ""}`}
              onClick={() => setTab("history")}
            >
              History
              {history.length > 0 && <span className="tab-count-muted">{history.length}</span>}
            </button>
          </div>

          {/* Verdicts Tab */}
          {tab === "verdicts" && (() => {
            const filtered = active
              .filter(v => v.symbol === activeSymbol)
              .sort((a, b) => {
                const tA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
                const tB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
                return tB - tA;
              });
            return (
              <section className="verdicts-section">
                {filtered.length === 0 ? (
                  <div className="no-verdicts">
                    <div className="no-v-icon">⏳</div>
                    <p className="no-v-text">No active verdicts for {activeSymbol}</p>
                    <p className="no-v-sub">
                      YELENA is monitoring {activeSymbol} across 4 timeframes.
                      A GO verdict appears when the 15min anchor + confirming timeframe agree
                      with 70%+ confidence and acceptable risk/reward.
                    </p>
                  </div>
                ) : (
                  <div className="verdicts-grid">
                    {filtered.map((v, i) => (
                      <VerdictCard key={v.id || v.symbol || i} verdict={v} />
                    ))}
                  </div>
                )}
              </section>
            );
          })()}

          {/* History Tab */}
          {tab === "history" && (
            <section className="history-section">
              {history.length === 0 ? (
                <div className="no-verdicts">
                  <div className="no-v-icon">📋</div>
                  <p className="no-v-text">No verdict history yet today</p>
                  <p className="no-v-sub">
                    Closed and expired verdicts will appear here.
                  </p>
                </div>
              ) : (
                <div className="history-table">
                  <div className="history-header-row">
                    <span>Time</span>
                    <span>Symbol</span>
                    <span>Direction</span>
                    <span>Confidence</span>
                    <span>Verdict</span>
                    <span>Result</span>
                    <span>P&L</span>
                    <span>Reason</span>
                  </div>
                  {history.map((h, i) => (
                    <HistoryRow key={h.id || i} item={h} />
                  ))}
                </div>
              )}
            </section>
          )}
        </main>
      </div>

      <style>{`
        * { margin: 0; padding: 0; box-sizing: border-box; }
        .yelena-root {
          font-family: 'Instrument Sans', -apple-system, sans-serif;
          background: #0a0e17; color: #c8cdd6;
          min-height: 100vh; display: flex; flex-direction: column;
          overflow: hidden; height: 100vh;
        }
        .yelena-header {
          display: flex; align-items: center; justify-content: space-between;
          padding: 0 20px; height: 48px; background: #0d1220;
          border-bottom: 1px solid #1a2035; flex-shrink: 0;
        }
        .header-left, .header-right { display: flex; align-items: center; gap: 12px; }
        .logo { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 16px; color: #fff; letter-spacing: 2px; }
        .logo-sub { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #4a6cf7; font-weight: 500; margin-left: -8px; }
        .header-divider { width: 1px; height: 20px; background: #1e2a42; }
        .market-status { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #8892a6; display: flex; align-items: center; gap: 6px; letter-spacing: 0.5px; }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
        .dot-open { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
        .dot-closed { background: #6b7280; }
        .dot-on { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
        .dot-off { background: #ef4444; box-shadow: 0 0 6px #ef444488; }
        .header-tag { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #f59e0b; background: #f59e0b15; padding: 2px 6px; border-radius: 3px; border: 1px solid #f59e0b30; letter-spacing: 0.5px; }
        .header-clock { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #8892a6; }
        .conn-status { font-family: 'JetBrains Mono', monospace; font-size: 10px; display: flex; align-items: center; gap: 5px; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.5px; }
        .conn-on { color: #22c55e; background: #22c55e10; border: 1px solid #22c55e25; }
        .conn-off { color: #ef4444; background: #ef444410; border: 1px solid #ef444425; }

        .yelena-body { display: flex; flex: 1; overflow: hidden; }

        .yelena-sidebar { width: 72px; background: #0d1220; border-right: 1px solid #1a2035; flex-shrink: 0; overflow-y: auto; }
        .watchlist { padding: 8px 0; }
        .watchlist-header { font-family: 'JetBrains Mono', monospace; font-size: 8px; color: #4a5568; text-align: center; padding: 6px 0 8px; letter-spacing: 1.5px; }
        .watchlist-item { display: flex; align-items: center; justify-content: center; gap: 2px; width: 100%; padding: 8px 4px; background: none; border: none; color: #6b7a90; font-family: 'JetBrains Mono', monospace; font-size: 10px; cursor: pointer; transition: all 0.15s; border-left: 2px solid transparent; }
        .watchlist-item:hover { background: #131b2e; color: #c8cdd6; }
        .wl-active { background: #131b2e; color: #fff; border-left-color: #4a6cf7; }
        .wl-symbol { font-weight: 500; }
        .wl-verdict { font-size: 8px; }
        .wl-call { color: #22c55e; }
        .wl-put { color: #ef4444; }
        .yelena-sidebar::-webkit-scrollbar { width: 0; }

        .yelena-main { flex: 1; padding: 20px 28px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }

        .daily-stats { display: flex; gap: 2px; background: #111827; border-radius: 8px; overflow: hidden; border: 1px solid #1a2035; }
        .stat-box { flex: 1; padding: 12px 10px; text-align: center; background: #0f1626; }
        .stat-num { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 600; color: #e2e8f0; display: block; }
        .stat-label { font-size: 9px; color: #4a5568; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; display: block; }
        .stat-win { color: #22c55e; }
        .stat-loss { color: #ef4444; }
        .stat-wr { color: #4a6cf7; }
        .stat-go { color: #22c55e; }
        .stat-skip { color: #6b7280; }

        .tab-bar { display: flex; gap: 4px; }
        .tab-btn { font-family: 'JetBrains Mono', monospace; font-size: 12px; padding: 8px 16px; background: #0f1626; border: 1px solid #1a2035; border-radius: 6px 6px 0 0; color: #6b7a90; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.15s; border-bottom: none; }
        .tab-btn:hover { color: #c8cdd6; }
        .tab-active { color: #fff; background: #131b2e; border-color: #2a3a5c; }
        .tab-count { font-size: 10px; background: #22c55e20; color: #22c55e; padding: 1px 6px; border-radius: 8px; }
        .tab-count-muted { font-size: 10px; background: #4a556820; color: #6b7a90; padding: 1px 6px; border-radius: 8px; }

        .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: #4a6cf7; display: inline-block; animation: pulse 2s ease-in-out infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; box-shadow: 0 0 0 0 #4a6cf744; } 50% { opacity: 0.7; box-shadow: 0 0 0 6px #4a6cf700; } }

        .verdicts-grid { display: flex; flex-direction: column; gap: 12px; }

        .verdict-card { background: #0f1626; border-radius: 10px; padding: 18px 20px; border: 1px solid #1a2035; transition: all 0.2s; }
        .verdict-call { border-left: 3px solid #22c55e; }
        .verdict-put { border-left: 3px solid #ef4444; }
        .verdict-skip { border-left: 3px solid #4a5568; opacity: 0.7; }

        .verdict-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
        .verdict-left { display: flex; align-items: center; gap: 10px; }
        .verdict-badge { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 4px; letter-spacing: 1px; }
        .badge-go { background: #22c55e18; color: #22c55e; border: 1px solid #22c55e40; }
        .badge-skip { background: #6b728018; color: #9ca3af; border: 1px solid #6b728040; }
        .verdict-symbol { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #fff; }
        .verdict-direction { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; }
        .dir-call { color: #22c55e; }
        .dir-put { color: #ef4444; }
        .verdict-right { display: flex; align-items: baseline; gap: 10px; }
        .verdict-confidence { font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: #fff; }
        .verdict-age { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #4a5568; }

        .tf-bar { display: flex; gap: 3px; margin-bottom: 14px; }
        .tf-cell { flex: 1; text-align: center; padding: 6px 0; border-radius: 4px; display: flex; flex-direction: column; gap: 1px; }
        .tf-call { background: #22c55e12; border: 1px solid #22c55e25; }
        .tf-put { background: #ef444412; border: 1px solid #ef444425; }
        .tf-hold { background: #1a202c; border: 1px solid #1e2a42; }
        .tf-label { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #4a5568; letter-spacing: 0.5px; }
        .tf-dir { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; }
        .tf-call .tf-dir { color: #22c55e; }
        .tf-put .tf-dir { color: #ef4444; }
        .tf-hold .tf-dir { color: #374151; }
        .tf-conf { font-family: 'JetBrains Mono', monospace; font-size: 8px; color: #6b7a90; }

        .trade-plan { margin-bottom: 12px; }
        .plan-row { display: flex; gap: 2px; background: #0a0e17; border-radius: 6px; overflow: hidden; }
        .plan-item { flex: 1; padding: 10px 12px; text-align: center; }
        .plan-label { font-size: 9px; color: #4a5568; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px; }
        .plan-value { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: #e2e8f0; display: block; }
        .plan-target { color: #22c55e; }
        .plan-stop { color: #ef4444; }
        .plan-rr { color: #4a6cf7; }

        .verdict-reason { font-size: 12px; color: #6b7a90; line-height: 1.5; }

        .no-verdicts { text-align: center; padding: 60px 20px; background: #0f1626; border-radius: 10px; border: 1px dashed #1e2a42; }
        .no-v-icon { font-size: 32px; margin-bottom: 12px; }
        .no-v-text { font-size: 15px; color: #8892a6; font-weight: 500; margin-bottom: 6px; }
        .no-v-sub { font-size: 12px; color: #4a5568; max-width: 400px; margin: 0 auto; line-height: 1.5; }

        .history-table { background: #0f1626; border-radius: 8px; border: 1px solid #1a2035; overflow: hidden; }
        .history-header-row { display: grid; grid-template-columns: 70px 60px 80px 85px 60px 60px 60px 1fr; padding: 8px 16px; background: #111827; font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #4a5568; text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid #1a2035; }
        .history-row { display: grid; grid-template-columns: 70px 60px 80px 85px 60px 60px 60px 1fr; padding: 10px 16px; font-family: 'JetBrains Mono', monospace; font-size: 12px; border-bottom: 1px solid #0e1525; transition: background 0.1s; align-items: center; }
        .history-row:last-child { border-bottom: none; }
        .history-row:hover { background: #131b2e; }
        .hist-time { color: #4a5568; font-size: 11px; }
        .hist-symbol { color: #e2e8f0; font-weight: 600; }
        .hist-dir { font-size: 11px; font-weight: 500; }
        .hist-confidence { color: #8892a6; }
        .hist-verdict { font-size: 10px; font-weight: 600; }
        .badge-go-sm { color: #22c55e; }
        .badge-skip-sm { color: #6b7280; }
        .hist-result { font-weight: 600; }
        .result-win { color: #22c55e; }
        .result-loss { color: #ef4444; }
        .result-skip { color: #4a5568; }
        .hist-pnl { font-weight: 600; }
        .pnl-pos { color: #22c55e; }
        .pnl-neg { color: #ef4444; }
        .pnl-none { color: #4a5568; }
        .hist-reason { color: #4a5568; font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

        .yelena-main::-webkit-scrollbar { width: 4px; }
        .yelena-main::-webkit-scrollbar-track { background: transparent; }
        .yelena-main::-webkit-scrollbar-thumb { background: #1e2a42; border-radius: 2px; }
      `}</style>
    </div>
  );
}
