"""
KENNY Verdicts Router — Trade verdict storage, retrieval, and stats.
"""

import json
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
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    ml_agreement: Optional[str] = None
    tf_alignment: Optional[str] = None
    tv_agreement: Optional[str] = None
    tv_score: Optional[float] = None
    tv_grade: Optional[str] = None
    tv_confidence_delta: Optional[float] = None
    reasoning: Optional[dict] = None
    quality_signals: Optional[dict] = None
    metadata: Optional[dict] = None


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
        return {"verdicts": [dict(row) for row in rows], "count": len(rows)}
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
        return {"verdicts": [dict(row) for row in rows], "count": len(rows)}
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
        await db.execute(
            """INSERT INTO verdicts (
                time, symbol, verdict, direction, confidence,
                entry_price, stop_loss, tp1, tp2, tp3,
                ml_agreement, tf_alignment, tv_agreement, tv_score, tv_grade, tv_confidence_delta,
                reasoning, quality_signals, metadata
            ) VALUES (
                NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9,
                $10, $11, $12, $13, $14, $15, $16::jsonb, $17::jsonb, $18::jsonb
            )""",
            v.symbol.upper(), v.verdict, v.direction, v.confidence,
            v.entry_price, v.stop_loss, v.tp1, v.tp2, v.tp3,
            v.ml_agreement, v.tf_alignment, v.tv_agreement, v.tv_score, v.tv_grade, v.tv_confidence_delta,
            json.dumps(v.reasoning) if v.reasoning else None,
            json.dumps(v.quality_signals) if v.quality_signals else None,
            json.dumps(v.metadata) if v.metadata else None
        )
        logger.info(f"Verdict: {v.symbol} {v.verdict} {v.direction or ''}")
        return {"status": "stored"}
    except Exception as e:
        return {"status": "error", "error": str(e)}