import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================================
// KENNY — Trade Verdict Dashboard
// Connects to api.luckykenny.cloud (REST) and ws.luckykenny.cloud (WebSocket)
// ============================================================================

const API_BASE = "https://api.luckykenny.cloud";
const WS_URL = "wss://ws.luckykenny.cloud";
const SYMBOLS = ["SPY","QQQ","AAPL","AMZN","GOOGL","META","MSFT","NFLX","NVDA","AMD","TSLA"];

// ============================================================================
// VERDICT HOOKS
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
      const [activeRes, histRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/api/verdicts/latest`),
        fetch(`${API_BASE}/api/verdicts?limit=20`),
        fetch(`${API_BASE}/api/verdicts/stats`),
      ]);
      if (activeRes.ok) setActive(await activeRes.json());
      if (histRes.ok) setHistory(await histRes.json());
      if (statsRes.ok) setStats(await statsRes.json());
    } catch (e) {
      console.error("Fetch verdicts error:", e);
    }
  }, []);

  useEffect(() => {
    fetchVerdicts();
    const poll = setInterval(fetchVerdicts, 15000);

    // WebSocket for real-time verdicts
    function connectWs() {
      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          reconnectRef.current = 0;
          console.log("WebSocket connected");
        };

        ws.onmessage = (e) => {
          try {
            const msg = JSON.parse(e.data);
            if (msg.type === "verdict") {
              fetchVerdicts();
            } else if (msg.type === "health") {
              // Could update health state
            }
          } catch {}
        };

        ws.onclose = () => {
          setConnected(false);
          const delay = Math.min(1000 * 2 ** reconnectRef.current, 30000);
          reconnectRef.current += 1;
          setTimeout(connectWs, delay);
        };

        ws.onerror = () => ws.close();
      } catch {
        setTimeout(connectWs, 5000);
      }
    }

    connectWs();
    return () => {
      clearInterval(poll);
      if (wsRef.current) wsRef.current.close();
    };
  }, [fetchVerdicts]);

  return { active, history, stats, connected };
}

function useHealth() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    async function check() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) setHealth(await res.json());
      } catch {
        setHealth(null);
      }
    }
    check();
    const poll = setInterval(check, 10000);
    return () => clearInterval(poll);
  }, []);

  return health;
}

// ============================================================================
// COMPONENTS
// ============================================================================

function StatusBar({ health, connected }) {
  const dbOk = health?.components?.database?.status === "healthy";
  const mlOk = health?.components?.ml_models?.status !== "not_loaded";

  return (
    <div className="bg-gray-900 border-b border-gray-700 px-4 py-2 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-bold text-white tracking-wide">
          🎰 KENNY
        </h1>
        <span className="text-xs text-gray-400">AI Options Trading</span>
      </div>
      <div className="flex items-center gap-3 text-xs">
        <Dot color={health ? "green" : "red"} label="API" />
        <Dot color={dbOk ? "green" : "red"} label="DB" />
        <Dot color={mlOk ? "green" : "yellow"} label="ML" />
        <Dot color={connected ? "green" : "yellow"} label="WS" />
        <span className="text-gray-500">
          {new Date().toLocaleTimeString("en-US", { hour12: true, hour: "numeric", minute: "2-digit", second: "2-digit" })}
        </span>
      </div>
    </div>
  );
}

function Dot({ color, label }) {
  const colors = {
    green: "bg-green-500",
    yellow: "bg-yellow-500",
    red: "bg-red-500",
    gray: "bg-gray-500",
  };
  return (
    <div className="flex items-center gap-1">
      <div className={`w-2 h-2 rounded-full ${colors[color] || colors.gray}`} />
      <span className="text-gray-400">{label}</span>
    </div>
  );
}

function StatsRow({ stats }) {
  return (
    <div className="grid grid-cols-4 md:grid-cols-8 gap-2 p-3">
      <StatCard label="Today" value={stats.verdicts_today} />
      <StatCard label="GO" value={stats.go_count} color="text-green-400" />
      <StatCard label="SKIP" value={stats.skip_count} color="text-gray-400" />
      <StatCard label="Active" value={stats.active_count} color="text-blue-400" />
      <StatCard label="Wins" value={stats.wins} color="text-green-400" />
      <StatCard label="Losses" value={stats.losses} color="text-red-400" />
      <StatCard
        label="Win Rate"
        value={stats.win_rate ? `${stats.win_rate.toFixed(0)}%` : "—"}
        color={stats.win_rate >= 60 ? "text-green-400" : "text-gray-400"}
      />
      <StatCard
        label="P&L"
        value={stats.pnl ? `$${stats.pnl.toFixed(0)}` : "$0"}
        color={stats.pnl > 0 ? "text-green-400" : stats.pnl < 0 ? "text-red-400" : "text-gray-400"}
      />
    </div>
  );
}

function StatCard({ label, value, color = "text-white" }) {
  return (
    <div className="bg-gray-800 rounded px-3 py-2 text-center">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</div>
      <div className={`text-lg font-bold ${color}`}>{value}</div>
    </div>
  );
}

function ActiveVerdicts({ verdicts }) {
  if (!verdicts.length) {
    return (
      <div className="p-6 text-center text-gray-500">
        <div className="text-4xl mb-2">🎰</div>
        <div className="text-sm">No active verdicts</div>
        <div className="text-xs text-gray-600 mt-1">
          KENNY is watching the market — GO signals appear here
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2 p-3">
      {verdicts.map((v, i) => (
        <VerdictCard key={v.id || i} verdict={v} active />
      ))}
    </div>
  );
}

function VerdictCard({ verdict: v, active }) {
  const isGo = v.verdict === "GO";
  const isCall = v.direction === "CALL";

  const borderColor = isGo
    ? isCall ? "border-green-500" : "border-red-500"
    : "border-gray-600";

  const bgColor = isGo
    ? isCall ? "bg-green-500/5" : "bg-red-500/5"
    : "bg-gray-800/50";

  const dirColor = isCall ? "text-green-400" : "text-red-400";
  const verdictBadge = isGo
    ? "bg-green-500/20 text-green-400 border border-green-500/30"
    : "bg-gray-700/50 text-gray-400 border border-gray-600";

  const age = v.timestamp
    ? Math.floor((Date.now() - new Date(v.timestamp).getTime()) / 60000)
    : 0;

  return (
    <div className={`rounded-lg border-l-4 ${borderColor} ${bgColor} p-3`}>
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-white font-bold text-lg">{v.symbol}</span>
          <span className={`font-bold ${dirColor}`}>{v.direction}</span>
          <span className={`text-xs px-2 py-0.5 rounded-full ${verdictBadge}`}>
            {v.verdict}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          {active && age > 0 && <span>{age}m ago</span>}
          <span className="text-gray-400">{v.confidence?.toFixed?.(1) || v.confidence}%</span>
        </div>
      </div>

      {/* Trade plan */}
      {isGo && (
        <div className="grid grid-cols-4 gap-2 text-xs mb-2">
          <PlanItem label="Entry" value={`$${v.entry}`} />
          <PlanItem label="Target" value={`$${v.target}`} color="text-green-400" />
          <PlanItem label="Stop" value={`$${v.stop}`} color="text-red-400" />
          <PlanItem label="R:R" value={v.riskReward || v.risk_reward} color="text-blue-400" />
        </div>
      )}

      {/* TF alignment */}
      <div className="flex items-center gap-1 text-xs">
        {Object.entries(v.timeframeAlignment || v.timeframe_alignment || {}).map(([tf, dir]) => (
          <span
            key={tf}
            className={`px-1.5 py-0.5 rounded text-[10px] ${
              dir === "CALL" ? "bg-green-500/20 text-green-400"
              : dir === "PUT" ? "bg-red-500/20 text-red-400"
              : "bg-gray-700 text-gray-500"
            }`}
          >
            {tf}: {dir}
          </span>
        ))}
        {v.modelsAgreeing && (
          <span className="text-gray-500 ml-1">
            {v.modelsAgreeing}/4 agree
          </span>
        )}
      </div>

      {/* Reason */}
      {v.reason && (
        <div className="text-[11px] text-gray-500 mt-1 italic">{v.reason}</div>
      )}

      {/* Result (for closed verdicts) */}
      {v.result && (
        <div className={`text-xs mt-1 font-bold ${
          v.result === "WIN" ? "text-green-400" : "text-red-400"
        }`}>
          {v.result} {v.pnl && `($${v.pnl > 0 ? "+" : ""}${v.pnl.toFixed(2)})`}
          {v.closeReason && <span className="text-gray-500 font-normal ml-1">— {v.closeReason}</span>}
        </div>
      )}
    </div>
  );
}

function PlanItem({ label, value, color = "text-white" }) {
  return (
    <div className="bg-gray-800/50 rounded px-2 py-1">
      <div className="text-gray-500 text-[10px]">{label}</div>
      <div className={`font-mono font-bold ${color}`}>{value}</div>
    </div>
  );
}

function VerdictHistory({ history }) {
  if (!history.length) {
    return (
      <div className="p-4 text-center text-gray-600 text-xs">
        No verdict history yet
      </div>
    );
  }

  return (
    <div className="space-y-1 p-3 max-h-96 overflow-y-auto">
      {history.map((v, i) => (
        <VerdictCard key={v.id || i} verdict={v} />
      ))}
    </div>
  );
}

function SymbolGrid({ health }) {
  const symbols = health?.symbols || SYMBOLS;
  const [predictions, setPredictions] = useState({});

  useEffect(() => {
    async function fetchSignals() {
      try {
        const res = await fetch(`${API_BASE}/api/signals/latest`);
        if (res.ok) {
          const data = await res.json();
          // Group by symbol
          const bySymbol = {};
          (Array.isArray(data) ? data : []).forEach(s => {
            if (!bySymbol[s.symbol]) bySymbol[s.symbol] = {};
            bySymbol[s.symbol][s.timeframe] = s;
          });
          setPredictions(bySymbol);
        }
      } catch {}
    }
    fetchSignals();
    const poll = setInterval(fetchSignals, 30000);
    return () => clearInterval(poll);
  }, []);

  return (
    <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2 p-3">
      {symbols.map(sym => {
        const preds = predictions[sym] || {};
        const primary = preds["15min"] || preds["5min"] || {};
        const dir = primary.direction;
        const conf = primary.confidence;

        return (
          <div key={sym} className="bg-gray-800 rounded-lg p-2 text-center">
            <div className="text-white font-bold text-sm">{sym}</div>
            {dir && dir !== "HOLD" ? (
              <>
                <div className={`text-xs font-bold ${
                  dir === "CALL" ? "text-green-400" : "text-red-400"
                }`}>
                  {dir}
                </div>
                <div className="text-[10px] text-gray-500">
                  {conf?.toFixed?.(0) || conf}%
                </div>
              </>
            ) : (
              <div className="text-xs text-gray-600">—</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ============================================================================
// MAIN DASHBOARD
// ============================================================================

export default function KennyDashboard() {
  const { active, history, stats, connected } = useVerdicts();
  const health = useHealth();
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const tick = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(tick);
  }, []);

  const isMarketOpen = (() => {
    const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
    const h = et.getHours(), m = et.getMinutes();
    const day = et.getDay();
    if (day === 0 || day === 6) return false;
    const mins = h * 60 + m;
    return mins >= 570 && mins < 960; // 9:30 AM - 4:00 PM ET
  })();

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <StatusBar health={health} connected={connected} />

      {/* Market status */}
      <div className="px-4 py-1 text-xs flex justify-between text-gray-500">
        <span>
          Market: {" "}
          <span className={isMarketOpen ? "text-green-400" : "text-red-400"}>
            {isMarketOpen ? "OPEN" : "CLOSED"}
          </span>
        </span>
        <span>
          {now.toLocaleString("en-US", {
            timeZone: "America/New_York",
            weekday: "short", month: "short", day: "numeric",
            hour: "numeric", minute: "2-digit", second: "2-digit",
            hour12: true,
          })} ET
        </span>
      </div>

      {/* Stats */}
      <StatsRow stats={stats} />

      {/* Active verdicts */}
      <div className="px-3 pt-2">
        <SectionHeader title="Active Verdicts" count={active.length} color="green" />
      </div>
      <ActiveVerdicts verdicts={active} />

      {/* Symbol grid */}
      <div className="px-3 pt-2">
        <SectionHeader title="Watchlist" />
      </div>
      <SymbolGrid health={health} />

      {/* History */}
      <div className="px-3 pt-2">
        <SectionHeader title="Recent Verdicts" count={history.length} />
      </div>
      <VerdictHistory history={history} />

      {/* Footer */}
      <div className="p-4 text-center text-xs text-gray-700">
        KENNY v1.0 — 32 ML Models • 4 Timeframes • 11 Symbols
        {health?.version && ` • API ${health.version}`}
      </div>
    </div>
  );
}

function SectionHeader({ title, count, color = "gray" }) {
  const dotColor = color === "green" ? "bg-green-500" : "bg-gray-600";
  return (
    <div className="flex items-center gap-2 text-xs text-gray-400 uppercase tracking-wider">
      {count !== undefined && <div className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />}
      {title}
      {count !== undefined && <span className="text-gray-600">({count})</span>}
    </div>
  );
}