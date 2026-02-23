"""
YELENA v2 — Signals + Verdicts API Router
REST endpoints + WebSocket for real-time verdict delivery to dashboard.

Endpoints:
    POST   /api/signals          — Orchestrator pushes raw predictions (stored for analysis)
    POST   /api/verdicts         — Orchestrator pushes trade verdicts (shown on dashboard)
    GET    /api/verdicts          — Dashboard fetches active + recent verdicts
    GET    /api/verdicts/stats    — Dashboard fetches daily verdict stats
    GET    /api/verdicts/history  — Dashboard fetches closed verdicts
    WS     /ws/signals            — Real-time verdict stream to dashboard
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.database import get_db

logger = logging.getLogger("yelena.signals")

router = APIRouter()


# ============================================================
# WebSocket Connection Manager
# ============================================================

class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info(f"Dashboard client connected ({len(self.active)} total)")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info(f"Dashboard client disconnected ({len(self.active)} total)")

    async def broadcast(self, data: dict):
        if not self.active:
            return 0
        payload = json.dumps(data)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self.active -= dead
        return len(self.active)


manager = ConnectionManager()


# ============================================================
# Pydantic Models
# ============================================================

class SignalInput(BaseModel):
    symbol: str
    timeframe: str
    direction: str
    probability: float
    confidence: float
    grade: str
    models_agreeing: int
    unanimous: bool
    individual: dict
    feature_ms: float = 0
    predict_ms: float = 0
    total_ms: float = 0
    timestamp: str = ""


class VerdictInput(BaseModel):
    """Trade verdict from the Verdict Engine."""
    id: int = 0
    symbol: str
    direction: str
    verdict: str            # GO or SKIP
    confidence: float
    entry: float = 0
    target: float = 0
    stop: float = 0
    riskReward: str = ""
    timeframeAlignment: dict = {}
    tfConfidences: dict = {}
    modelsAgreeing: int = 0
    anchorTf: str = "15min"
    confirmingTfs: list = []
    reason: str = ""
    status: str = "active"
    timestamp: str = ""
    closedAt: str | None = None
    result: str | None = None
    pnl: float | None = None
    closeReason: str | None = None


# ============================================================
# In-Memory Verdict Store (backed by DB for persistence)
# ============================================================

class VerdictStore:
    """
    In-memory store for fast dashboard access.
    Also persists to PostgreSQL for history.
    """
    def __init__(self):
        self.active: dict = {}      # {symbol: verdict_dict}
        self.history: list = []     # Recent closed verdicts
        self.stats = {
            "verdicts_today": 0,
            "go_count": 0,
            "skip_count": 0,
            "active_count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
        }
    
    def upsert_verdict(self, v: dict):
        """Add or update a verdict."""
        symbol = v["symbol"]
        status = v.get("status", "active")
        
        if status == "active" and v.get("verdict") == "GO":
            self.active[symbol] = v
        elif status == "closed":
            # Remove from active, add to history
            self.active.pop(symbol, None)
            self.history.insert(0, v)
            self.history = self.history[:50]  # Keep last 50
        
        # Update stats
        self.stats["verdicts_today"] += 1
        if v.get("verdict") == "GO":
            self.stats["go_count"] += 1
        else:
            self.stats["skip_count"] += 1
        self.stats["active_count"] = len(self.active)
        
        if v.get("result") == "WIN":
            self.stats["wins"] += 1
        elif v.get("result") == "LOSS":
            self.stats["losses"] += 1
        
        total = self.stats["wins"] + self.stats["losses"]
        self.stats["win_rate"] = round(
            self.stats["wins"] / total * 100, 1
        ) if total > 0 else 0.0
        
        if v.get("pnl"):
            self.stats["pnl"] = round(self.stats["pnl"] + v["pnl"], 2)


verdict_store = VerdictStore()


# ============================================================
# VERDICT Endpoints
# ============================================================

@router.post("/api/verdicts")
async def create_verdict(verdict: VerdictInput, db: AsyncSession = Depends(get_db)):
    """
    Receive a trade verdict from the orchestrator.
    Stores in DB, updates in-memory store, broadcasts to dashboard.
    """
    now = datetime.now(timezone.utc)
    
    v_dict = verdict.dict()
    v_dict["receivedAt"] = now.isoformat()
    
    # Compute age string
    if verdict.timestamp:
        try:
            opened = datetime.fromisoformat(verdict.timestamp)
            age_sec = (now - opened).total_seconds()
            v_dict["age"] = f"{age_sec:.0f}s ago" if age_sec < 60 else f"{age_sec / 60:.0f}m ago"
        except Exception:
            v_dict["age"] = "just now"
    else:
        v_dict["age"] = "just now"
    
    # Save to DB
    try:
        result = await db.execute(
            text("""
                INSERT INTO verdicts (time, symbol, direction, verdict, confidence,
                    entry_price, target_price, stop_price, risk_reward,
                    timeframe_alignment, reason, status, result, pnl)
                VALUES (:time, :symbol, :direction, :verdict, :confidence,
                    :entry, :target, :stop, :rr,
                    :alignment, :reason, :status, :result, :pnl)
                RETURNING id
            """),
            {
                "time": now,
                "symbol": verdict.symbol,
                "direction": verdict.direction,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "entry": verdict.entry,
                "target": verdict.target,
                "stop": verdict.stop,
                "rr": verdict.riskReward,
                "alignment": json.dumps(verdict.timeframeAlignment),
                "reason": verdict.reason,
                "status": verdict.status,
                "result": verdict.result,
                "pnl": verdict.pnl,
            }
        )
        await db.commit()
        row = result.fetchone()
        v_dict["dbId"] = row[0]
    except Exception as e:
        logger.error(f"Failed to save verdict to DB: {e}")
        # Continue even if DB fails — in-memory store still works

    # Update in-memory store
    verdict_store.upsert_verdict(v_dict)
    
    # Broadcast to dashboard
    broadcast_data = {"type": "verdict", **v_dict}
    count = await manager.broadcast(broadcast_data)
    
    logger.info(
        f"Verdict: {verdict.symbol} {verdict.verdict} {verdict.direction} "
        f"{verdict.confidence:.1f}% → {count} clients"
    )
    
    return {"status": "ok", "broadcast_to": count}


@router.get("/api/verdicts")
async def get_verdicts():
    """Get active verdicts + recent closed ones for the dashboard."""
    now = datetime.now(timezone.utc)
    
    # Update ages on active verdicts
    active = []
    for symbol, v in verdict_store.active.items():
        if v.get("timestamp"):
            try:
                opened = datetime.fromisoformat(v["timestamp"])
                age_sec = (now - opened).total_seconds()
                v["age"] = f"{age_sec:.0f}s ago" if age_sec < 60 else f"{age_sec / 60:.0f}m ago"
            except Exception:
                pass
        active.append(v)
    
    return {
        "active": active,
        "history": verdict_store.history[:20],
    }


@router.get("/api/verdicts/stats")
async def get_verdict_stats():
    """Get daily verdict statistics."""
    return verdict_store.stats


@router.get("/api/verdicts/history")
async def get_verdict_history(limit: int = Query(20, ge=1, le=100)):
    """Get closed verdict history."""
    return verdict_store.history[:limit]


# ============================================================
# RAW SIGNALS Endpoints (kept for analysis/debugging)
# ============================================================

@router.post("/api/signals")
async def create_signal(signal: SignalInput, db: AsyncSession = Depends(get_db)):
    """Store raw prediction (for analysis, not shown on dashboard)."""
    if signal.timestamp:
        from datetime import datetime as dt
        try:
            now = dt.fromisoformat(signal.timestamp)
        except Exception:
            now = datetime.now(timezone.utc)
    else:
        now = datetime.now(timezone.utc)

    probabilities = {
        "direction": signal.direction,
        "probability": signal.probability,
        "confidence": signal.confidence,
        "grade": signal.grade,
        "models_agreeing": signal.models_agreeing,
        "unanimous": signal.unanimous,
        "individual": signal.individual,
        "latency": {
            "feature_ms": signal.feature_ms,
            "predict_ms": signal.predict_ms,
            "total_ms": signal.total_ms
        }
    }

    result = await db.execute(
        text("""
            INSERT INTO predictions (time, symbol, timeframe, model_name, signal, confidence, probabilities, model_version)
            VALUES (:time, :symbol, :timeframe, :model_name, :signal, :confidence, :probabilities, :model_version)
            RETURNING id
        """),
        {
            "time": now,
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "model_name": "ensemble_v2",
            "signal": signal.direction,
            "confidence": signal.confidence,
            "probabilities": json.dumps(probabilities),
            "model_version": "2.0.0"
        }
    )
    await db.commit()
    row = result.fetchone()

    return {"id": row[0], "status": "saved", "broadcast_to": 0}


@router.get("/api/signals")
async def get_signals(
    symbol: str = Query(None),
    timeframe: str = Query(None),
    direction: str = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    query = """
        SELECT id, time, symbol, timeframe, signal, confidence, probabilities
        FROM predictions
        WHERE model_name = 'ensemble_v2'
    """
    params = {}

    if symbol:
        query += " AND symbol = :symbol"
        params["symbol"] = symbol
    if timeframe:
        query += " AND timeframe = :timeframe"
        params["timeframe"] = timeframe
    if direction:
        query += " AND signal = :direction"
        params["direction"] = direction

    query += " ORDER BY time DESC LIMIT :limit"
    params["limit"] = limit

    result = await db.execute(text(query), params)
    rows = result.fetchall()

    signals = []
    for r in rows:
        probs = r[6] if isinstance(r[6], dict) else json.loads(r[6])
        signals.append({
            "id": r[0],
            "time": r[1].isoformat() if hasattr(r[1], 'isoformat') else str(r[1]),
            "symbol": r[2],
            "timeframe": r[3],
            "direction": r[4],
            "confidence": r[5],
            "probability": probs.get("probability", 0),
            "grade": probs.get("grade", ""),
            "models_agreeing": probs.get("models_agreeing", 0),
            "unanimous": probs.get("unanimous", False),
            "individual": probs.get("individual", {})
        })

    return signals


@router.get("/api/signals/stats")
async def get_signal_stats(db: AsyncSession = Depends(get_db)):
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    result = await db.execute(
        text("""
            SELECT COUNT(*) as total,
                COUNT(*) FILTER (WHERE signal = 'CALL') as calls,
                COUNT(*) FILTER (WHERE signal = 'PUT') as puts,
                COUNT(*) FILTER (WHERE signal = 'HOLD') as holds,
                COALESCE(AVG(confidence), 0) as avg_confidence,
                COUNT(*) FILTER (WHERE probabilities->>'unanimous' = 'true') as unanimous
            FROM predictions
            WHERE model_name = 'ensemble_v2' AND time >= :today
        """),
        {"today": today_start}
    )
    row = result.fetchone()

    return {
        "signals_today": row[0],
        "calls_today": row[1],
        "puts_today": row[2],
        "holds_today": row[3],
        "avg_confidence": round(float(row[4]), 1),
        "unanimous_count": row[5],
    }


# ============================================================
# WebSocket
# ============================================================

@router.websocket("/ws/signals")
async def websocket_signals(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send current active verdicts on connect
        active_verdicts = verdict_store.active
        await ws.send_text(json.dumps({
            "type": "connected",
            "message": "YELENA verdict stream connected",
            "clients": len(manager.active),
            "activeVerdicts": list(active_verdicts.values()),
            "stats": verdict_store.stats,
        }))

        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                if data == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                try:
                    await ws.send_text(json.dumps({"type": "heartbeat"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)
