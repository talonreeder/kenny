"""
KENNY Verdicts Router — Trade verdict storage, retrieval, and stats.
"""

import json
from app.websocket.manager import ws_manager
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.database import db

logger = logging.getLogger(__name__)
router = APIRouter()


class VerdictCreate(BaseModel):
    symbol: str
    verdict: str
    direction: Optional[str] = None
    confidence: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None      # maps to stop_price
    stop_price: Optional[float] = None
    tp1: Optional[float] = None            # maps to target_price
    target_price: Optional[float] = None
    risk_reward: Optional[str] = None
    tf_alignment: Optional[str] = None     # JSON string for timeframe_alignment
    ml_agreement: Optional[str] = None
    reasoning: Optional[dict] = None       # extracted to reason text
    quality_signals: Optional[dict] = None
    metadata: Optional[dict] = None


def _transform_verdict(row):
    """Transform DB row to dashboard-friendly format."""
    d = dict(row)
    reasoning = d.get("reasoning") or {}
    metadata = d.get("metadata") or {}
    quality = d.get("quality_signals") or {}
    tf_raw = d.get("timeframe_alignment") or d.get("tf_alignment") or "{}"
    try:
        tf_align = json.loads(tf_raw) if isinstance(tf_raw, str) else tf_raw
    except:
        tf_align = {}
    return {
        "id": d.get("id"),
        "symbol": d.get("symbol"),
        "verdict": d.get("verdict"),
        "direction": d.get("direction"),
        "confidence": d.get("confidence"),
        "entry": round(d["entry_price"], 2) if d.get("entry_price") else None,
        "target": round(d["target_price"], 2) if d.get("target_price") else None,
        "stop": round(d["stop_price"], 2) if d.get("stop_price") else None,
        "riskReward": d.get("risk_reward", ""),
        "timeframeAlignment": tf_align,
        "timeframe_alignment": tf_align,
        "modelsAgreeing": None,
        "models_agreeing": None,
        "reason": d.get("reason", ""),
        "timestamp": d.get("time").isoformat() if d.get("time") else None,
        "status": d.get("status", "active"),
        "result": d.get("result"),
        "pnl": d.get("pnl"),
        "closeReason": d.get("close_reason"),
    }

@router.get("/verdicts")
async def list_verdicts(
    symbol: Optional[str] = Query(None),
    verdict: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    hours: float = Query(24, ge=0.1, le=168)
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = "SELECT * FROM verdicts WHERE time > $1"
    params = [cutoff]
    idx = 2
    if symbol:
        query += f" AND symbol = ${idx}"
        params.append(symbol.upper())
        idx += 1
    if verdict:
        query += f" AND verdict = ${idx}"
        params.append(verdict.upper())
        idx += 1
    if direction:
        query += f" AND direction = ${idx}"
        params.append(direction.upper())
        idx += 1
    query += f" ORDER BY time DESC LIMIT ${idx}"
    params.append(limit)
    try:
        rows = await db.fetch(query, *params)
        return {"verdicts": [_transform_verdict(row) for row in rows], "count": len(rows)}
    except Exception as e:
        return {"verdicts": [], "count": 0, "error": str(e)}


@router.get("/verdicts/latest")
async def latest_verdicts(symbol: Optional[str] = Query(None)):
    query = "SELECT DISTINCT ON (symbol) * FROM verdicts WHERE time > NOW() - INTERVAL '4 hours'"
    params = []
    if symbol:
        query += " AND symbol = $1"
        params.append(symbol.upper())
    query += " ORDER BY symbol, time DESC"
    try:
        rows = await db.fetch(query, *params)
        return {"verdicts": [_transform_verdict(row) for row in rows], "count": len(rows)}
    except Exception as e:
        return {"verdicts": [], "count": 0, "error": str(e)}


@router.get("/verdicts/stats")
async def verdict_stats(
    hours: float = Query(168, ge=1, le=720),
    symbol: Optional[str] = Query(None)
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    where = "WHERE time > $1"
    params = [cutoff]
    idx = 2
    if symbol:
        where += f" AND symbol = ${idx}"
        params.append(symbol.upper())
    try:
        totals = await db.fetch(
            f"SELECT verdict, direction, COUNT(*) as count FROM verdicts {where} GROUP BY verdict, direction",
            *params
        )
        go_stats = await db.fetchrow(
            f"SELECT COUNT(*) as total_go, AVG(confidence) as avg_confidence FROM verdicts {where} AND verdict = 'GO'",
            *params
        )
        return {
            "period_hours": hours,
            "totals": [dict(row) for row in totals],
            "go_verdicts": dict(go_stats) if go_stats else {},
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/verdicts")
async def store_verdict(v: VerdictCreate):
    try:
        stop = v.stop_price or v.stop_loss
        target = v.target_price or v.tp1
        rr = v.risk_reward or (v.reasoning or {}).get("risk_reward", "")
        reason = (v.reasoning or {}).get("reason", "") if v.reasoning else ""
        # Parse tf_alignment: could be JSON string or come from reasoning
        tf_align = None
        if v.tf_alignment:
            try:
                tf_align = json.loads(v.tf_alignment) if isinstance(v.tf_alignment, str) else v.tf_alignment
            except:
                tf_align = None

        await db.execute(
            """INSERT INTO verdicts (
                time, symbol, verdict, direction, confidence,
                entry_price, target_price, stop_price, risk_reward,
                timeframe_alignment, reason, status
            ) VALUES (
                NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11
            )""",
            v.symbol.upper(), v.verdict, v.direction, v.confidence,
            v.entry_price, target, stop, rr,
            json.dumps(tf_align) if tf_align else None,
            reason, "active"
        )
        logger.info(f"Verdict stored: {v.symbol} {v.verdict} {v.direction or ''} {v.confidence}%")
        # Broadcast to dashboard via WebSocket
        try:
            await ws_manager.broadcast_verdict({
                "symbol": v.symbol.upper(),
                "verdict": v.verdict,
                "direction": v.direction,
                "confidence": v.confidence,
                "entry": v.entry_price,
                "target": target,
                "stop": stop,
                "riskReward": rr,
                "timeframeAlignment": tf_align or {},
                "reason": reason,
            })
        except Exception as ws_err:
            logger.debug(f"WS broadcast failed: {ws_err}")
        return {"status": "stored"}
    except Exception as e:
        logger.error(f"Verdict store error: {e}")
        return {"status": "error", "error": str(e)}