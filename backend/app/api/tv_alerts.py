"""
KENNY TV Alerts Router — TradingView Master Confluence webhook handler.
This is tv_alerts_v2 from day one. No legacy v1.
"""

import json
import logging
import time as time_module
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.database import db

logger = logging.getLogger(__name__)
router = APIRouter()


class TVAlertPayload(BaseModel):
    passphrase: str
    symbol: str
    timeframe: str = "15"
    direction: str
    score: float = 0.0
    grade: str = "B"
    confidence: float = 0.0
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    qcloud_score: Optional[float] = None
    qline_score: Optional[float] = None
    qwave_score: Optional[float] = None
    qbands_score: Optional[float] = None
    moneyball_score: Optional[float] = None
    qmomentum_score: Optional[float] = None
    qcvd_score: Optional[float] = None
    qsmc_score: Optional[float] = None
    qgrid_score: Optional[float] = None
    squeeze_active: Optional[bool] = None
    bos_detected: Optional[bool] = None
    volume_spike: Optional[bool] = None
    trend_strength: Optional[float] = None


class TVSignalCached(BaseModel):
    symbol: str
    timeframe: str
    direction: str
    score: float
    grade: str
    confidence: float
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    components: Dict[str, Optional[float]]
    quality_signals: Dict[str, Any]
    timestamp: float
    is_rich: bool = True


class TVAlertStore:
    """In-memory cache for TV signals. Verdict engine connects via set_tv_store()."""

    def __init__(self):
        self._signals: Dict[str, TVSignalCached] = {}
        self._signal_count = 0
        self._last_signal_time: Optional[float] = None
        self._settings = get_settings()

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}:{timeframe}"

    def store(self, signal: TVSignalCached):
        key = self._key(signal.symbol, signal.timeframe)
        self._signals[key] = signal
        self._signal_count += 1
        self._last_signal_time = time_module.time()

    def get(self, symbol: str, timeframe: str) -> Optional[TVSignalCached]:
        key = self._key(symbol, timeframe)
        signal = self._signals.get(key)
        if signal is None:
            return None
        age = time_module.time() - signal.timestamp
        if age > self._settings.tv_signal_ttl_seconds:
            return None
        return signal

    def get_freshness(self, symbol: str, timeframe: str) -> Optional[float]:
        signal = self.get(symbol, timeframe)
        if signal is None:
            return None
        age = time_module.time() - signal.timestamp
        half_life = self._settings.tv_signal_half_life_seconds
        ttl = self._settings.tv_signal_ttl_seconds
        if age <= half_life:
            return 1.0
        elif age <= ttl:
            return 0.5 + 0.5 * (ttl - age) / (ttl - half_life)
        return 0.0

    def get_multi_tf_confluence(self, symbol: str) -> Dict[str, Optional[TVSignalCached]]:
        return {tf: self.get(symbol, tf) for tf in self._settings.timeframes}

    def get_all_active(self) -> list:
        now = time_module.time()
        return [
            {"key": k, "direction": s.direction, "grade": s.grade, "score": s.score,
             "age_seconds": round(now - s.timestamp, 1)}
            for k, s in self._signals.items()
            if now - s.timestamp <= self._settings.tv_signal_ttl_seconds
        ]

    @property
    def stats(self) -> dict:
        return {
            "total_received": self._signal_count,
            "cached": len(self._signals),
            "active": len(self.get_all_active()),
            "last_signal_ago": round(time_module.time() - self._last_signal_time, 1) if self._last_signal_time else None
        }


tv_alert_store = TVAlertStore()


@router.post("/tv-alert")
async def receive_tv_alert(request: Request):
    settings = get_settings()
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        try:
            body = json.loads(raw.decode())
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

    if body.get("passphrase", "") != settings.webhook_passphrase:
        raise HTTPException(status_code=401, detail="Invalid passphrase")

    try:
        payload = TVAlertPayload(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

    tf_map = {"1": "1min", "5": "5min", "15": "15min", "60": "1hr", "D": "daily"}
    timeframe = tf_map.get(payload.timeframe, payload.timeframe)

    components = {
        "qcloud": payload.qcloud_score, "qline": payload.qline_score,
        "qwave": payload.qwave_score, "qbands": payload.qbands_score,
        "moneyball": payload.moneyball_score, "qmomentum": payload.qmomentum_score,
        "qcvd": payload.qcvd_score, "qsmc": payload.qsmc_score,
        "qgrid": payload.qgrid_score,
    }
    quality_signals = {
        "squeeze_active": payload.squeeze_active, "bos_detected": payload.bos_detected,
        "volume_spike": payload.volume_spike, "trend_strength": payload.trend_strength,
    }
    is_rich = any(v is not None for v in components.values())

    cached_signal = TVSignalCached(
        symbol=payload.symbol.upper(), timeframe=timeframe,
        direction=payload.direction.upper(), score=payload.score,
        grade=payload.grade, confidence=payload.confidence,
        entry=payload.entry, stop_loss=payload.stop_loss,
        tp1=payload.tp1, tp2=payload.tp2, tp3=payload.tp3,
        components=components, quality_signals=quality_signals,
        timestamp=time_module.time(), is_rich=is_rich
    )
    tv_alert_store.store(cached_signal)

    try:
        await db.execute(
            """INSERT INTO tv_signals (time, symbol, timeframe, direction, score, grade, confidence,
                entry_price, stop_loss, tp1, tp2, tp3, components, raw_signal, is_rich)
            VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13::jsonb, $14)""",
            payload.symbol.upper(), timeframe, payload.direction.upper(),
            payload.score, payload.grade, payload.confidence,
            payload.entry, payload.stop_loss, payload.tp1, payload.tp2, payload.tp3,
            json.dumps(components), json.dumps(body), is_rich
        )
    except Exception as e:
        logger.error(f"TV Alert DB error: {e}")

    logger.info(f"TV Alert: {payload.symbol} {timeframe} {payload.direction} grade={payload.grade}")
    return {"status": "received", "symbol": payload.symbol, "timeframe": timeframe, "direction": payload.direction}


@router.get("/tv-alert/latest")
async def latest_tv_signals(symbol: Optional[str] = None):
    if symbol:
        signals = tv_alert_store.get_multi_tf_confluence(symbol.upper())
        return {"symbol": symbol.upper(), "signals": {
            tf: s.model_dump() if s else None for tf, s in signals.items()
        }}
    return {"active_signals": tv_alert_store.get_all_active(), "store_stats": tv_alert_store.stats}


@router.get("/tv-alert/health")
async def tv_alert_health():
    return {"status": "ready", "store": tv_alert_store.stats}

@router.get("/tv-alerts/confluence/{symbol}")
async def tv_confluence(symbol: str):
    """Aggregate TV signals into a single confluence verdict for the verdict engine."""
    symbol = symbol.upper()
    signals = tv_alert_store.get_multi_tf_confluence(symbol)
    
    active = {tf: s for tf, s in signals.items() if s is not None}
    if not active:
        return {"symbol": symbol, "direction": "HOLD", "score": 0, "signals": 0}
    
    # Aggregate: count CALL vs PUT, weighted by score
    call_score = 0.0
    put_score = 0.0
    call_count = 0
    put_count = 0
    
    for tf, s in active.items():
        freshness = tv_alert_store.get_freshness(s.symbol, tf) or 0.5
        weighted = s.score * freshness
        if s.direction == "CALL":
            call_score += weighted
            call_count += 1
        elif s.direction == "PUT":
            put_score += weighted
            put_count += 1
    
    total_signals = call_count + put_count
    if call_score > put_score:
        direction = "CALL"
        score = call_score / max(call_count, 1)
    elif put_score > call_score:
        direction = "PUT"
        score = put_score / max(put_count, 1)
    else:
        direction = "HOLD"
        score = 0
    
    return {
        "symbol": symbol,
        "direction": direction,
        "score": round(score, 1),
        "signals": total_signals,
        "breakdown": {tf: {"direction": s.direction, "score": s.score, "grade": s.grade} for tf, s in active.items()}
    }
