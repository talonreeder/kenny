"""
KENNY Signals Router — ML prediction signal storage and retrieval.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.database import db

logger = logging.getLogger(__name__)
router = APIRouter()


class SignalCreate(BaseModel):
    symbol: str
    timeframe: str
    model_name: str
    signal: str
    confidence: float
    probabilities: dict
    model_version: Optional[str] = None


@router.get("/signals")
async def list_signals(
    symbol: Optional[str] = Query(None),
    timeframe: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    hours: float = Query(24, ge=0.1, le=168)
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = "SELECT * FROM predictions WHERE time > $1"
    params = [cutoff]
    idx = 2
    if symbol:
        query += f" AND symbol = ${idx}"
        params.append(symbol.upper())
        idx += 1
    if timeframe:
        query += f" AND timeframe = ${idx}"
        params.append(timeframe)
        idx += 1
    query += f" ORDER BY time DESC LIMIT ${idx}"
    params.append(limit)
    try:
        rows = await db.fetch(query, *params)
        return {"signals": [dict(row) for row in rows], "count": len(rows)}
    except Exception as e:
        return {"signals": [], "count": 0, "error": str(e)}


@router.get("/signals/latest")
async def latest_signals(symbol: Optional[str] = Query(None)):
    query = """
        SELECT DISTINCT ON (symbol, timeframe, model_name)
            time, symbol, timeframe, model_name, signal, confidence, probabilities
        FROM predictions WHERE time > NOW() - INTERVAL '1 hour'
    """
    params = []
    if symbol:
        query += " AND symbol = $1"
        params.append(symbol.upper())
    query += " ORDER BY symbol, timeframe, model_name, time DESC"
    try:
        rows = await db.fetch(query, *params)
        return {"signals": [dict(row) for row in rows], "count": len(rows)}
    except Exception as e:
        return {"signals": [], "count": 0, "error": str(e)}


@router.post("/signals")
async def store_signal(signal: SignalCreate):
    try:
        import json
        await db.execute(
            """INSERT INTO predictions (time, symbol, timeframe, model_name, signal, confidence, probabilities, model_version)
            VALUES (NOW(), $1, $2, $3, $4, $5, $6::jsonb, $7)""",
            signal.symbol.upper(), signal.timeframe, signal.model_name,
            signal.signal, signal.confidence,
            json.dumps(signal.probabilities), signal.model_version
        )
        return {"status": "stored"}
    except Exception as e:
        return {"status": "error", "error": str(e)}