"""
YELENA v2 — TradingView Webhook Integration
Receives webhook alerts from TradingView Pine Script indicators
and feeds them into the Verdict Engine as confluence signals.

TradingView Alert Setup:
    1. In TradingView, set alert Webhook URL to:
       http://YOUR_EC2_IP:8000/api/tv-alerts
    2. Set alert message body to JSON:
       {
           "passphrase": "YOUR_PASSPHRASE",
           "symbol": "{{ticker}}",
           "action": "{{strategy.order.action}}",
           "indicator": "QLine",
           "signal": "BEAR",
           "price": {{close}},
           "timeframe": "15",
           "time": "{{time}}"
       }

Supported indicators:
    - QLine: BULL/BEAR signals
    - BounceScore: BOUNCE_UP/BOUNCE_DOWN
    - QWave: WAVE_UP/WAVE_DOWN
    - Adaptive Trend Ribbon: TREND_UP/TREND_DOWN
    - SuperTrend: ST_BUY/ST_SELL
    - Master Confluence: STRONG_BULL/STRONG_BEAR/BULL/BEAR

The webhook passphrase prevents unauthorized alerts.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("yelena.tv_alerts")

router = APIRouter()


# ============================================================
# Configuration
# ============================================================

# Passphrase from SSM or environment
WEBHOOK_PASSPHRASE = os.getenv(
    "YELENA_WEBHOOK_PASSPHRASE",
    # Will be loaded from SSM at startup
    ""
)


# ============================================================
# Models
# ============================================================

class TVAlertInput(BaseModel):
    """Incoming TradingView webhook payload."""
    passphrase: str
    symbol: str
    indicator: str = "unknown"      # QLine, BounceScore, QWave, etc.
    signal: str = ""                # BULL, BEAR, BOUNCE_UP, etc.
    action: str = ""                # buy, sell (from strategy alerts)
    price: float = 0.0
    timeframe: str = "15"           # TradingView timeframe (minutes)
    time: str = ""                  # TradingView time string
    # Optional extra fields from alert message
    confidence: float = 0.0
    details: str = ""


class TVAlertResponse(BaseModel):
    status: str
    message: str
    direction: str = ""
    confluence_score: float = 0.0


# ============================================================
# Signal Normalization
# ============================================================

# Map TradingView signals to normalized CALL/PUT/HOLD
SIGNAL_MAP = {
    # QLine
    "BULL": "CALL",
    "BEAR": "PUT",
    # BounceScore
    "BOUNCE_UP": "CALL",
    "BOUNCE_DOWN": "PUT",
    # QWave
    "WAVE_UP": "CALL",
    "WAVE_DOWN": "PUT",
    # Adaptive Trend Ribbon
    "TREND_UP": "CALL",
    "TREND_DOWN": "PUT",
    # SuperTrend
    "ST_BUY": "CALL",
    "ST_SELL": "PUT",
    # Master Confluence
    "STRONG_BULL": "CALL",
    "STRONG_BEAR": "PUT",
    # Strategy actions
    "buy": "CALL",
    "sell": "PUT",
}

# Weight per indicator (how much it matters for confluence)
INDICATOR_WEIGHTS = {
    "MasterConfluence": 3.0,    # Highest — it's already a composite
    "QLine": 2.0,               # Strong trend indicator
    "QWave": 1.5,               # Momentum
    "BounceScore": 1.5,         # Support/resistance
    "AdaptiveTrend": 1.5,       # Trend ribbon
    "SuperTrend": 1.0,          # Basic trend
    "unknown": 1.0,             # Default
}


# ============================================================
# TV Alert Store (in-memory, per symbol)
# ============================================================

@dataclass
class TVSignal:
    """Normalized TradingView signal."""
    symbol: str
    indicator: str
    direction: str          # CALL, PUT, HOLD
    raw_signal: str         # Original signal name
    price: float
    weight: float
    timestamp: float        # Unix time
    timeframe: str


class TVAlertStore:
    """
    Stores recent TradingView alerts per symbol.
    Computes confluence score for the Verdict Engine.
    """

    def __init__(self, max_age_sec: float = 300):  # 5 min default
        self.signals: Dict[str, Dict[str, TVSignal]] = {}  # {symbol: {indicator: signal}}
        self.max_age_sec = max_age_sec
        self.total_received = 0
        self.history: List[dict] = []

    def add_signal(self, sig: TVSignal):
        """Add/update a signal for a symbol + indicator."""
        if sig.symbol not in self.signals:
            self.signals[sig.symbol] = {}
        self.signals[sig.symbol][sig.indicator] = sig
        self.total_received += 1

        # Keep history
        self.history.insert(0, {
            "symbol": sig.symbol,
            "indicator": sig.indicator,
            "direction": sig.direction,
            "raw_signal": sig.raw_signal,
            "price": sig.price,
            "time": datetime.fromtimestamp(sig.timestamp, tz=timezone.utc).isoformat(),
        })
        self.history = self.history[:100]  # Keep last 100

    def get_confluence(self, symbol: str) -> dict:
        """
        Calculate confluence score for a symbol.
        Returns direction and score based on recent TV signals.
        """
        self._prune_stale()

        sigs = self.signals.get(symbol, {})
        if not sigs:
            return {"direction": "HOLD", "score": 0.0, "signals": 0, "details": []}

        call_weight = 0.0
        put_weight = 0.0
        details = []

        for indicator, sig in sigs.items():
            if sig.direction == "CALL":
                call_weight += sig.weight
            elif sig.direction == "PUT":
                put_weight += sig.weight

            details.append({
                "indicator": indicator,
                "direction": sig.direction,
                "weight": sig.weight,
                "raw": sig.raw_signal,
                "age_sec": round(time.time() - sig.timestamp),
            })

        total_weight = call_weight + put_weight
        if total_weight == 0:
            return {"direction": "HOLD", "score": 0.0, "signals": len(sigs), "details": details}

        if call_weight > put_weight:
            direction = "CALL"
            score = (call_weight / total_weight) * 100
        elif put_weight > call_weight:
            direction = "PUT"
            score = (put_weight / total_weight) * 100
        else:
            direction = "HOLD"
            score = 50.0

        return {
            "direction": direction,
            "score": round(score, 1),
            "signals": len(sigs),
            "call_weight": round(call_weight, 1),
            "put_weight": round(put_weight, 1),
            "details": details,
        }

    def _prune_stale(self):
        """Remove signals older than max_age_sec."""
        now = time.time()
        for symbol in list(self.signals.keys()):
            stale = [
                ind for ind, sig in self.signals[symbol].items()
                if now - sig.timestamp > self.max_age_sec
            ]
            for ind in stale:
                del self.signals[symbol][ind]
            if not self.signals[symbol]:
                del self.signals[symbol]

    def get_stats(self) -> dict:
        self._prune_stale()
        active_symbols = list(self.signals.keys())
        total_active = sum(len(v) for v in self.signals.values())
        return {
            "total_received": self.total_received,
            "active_signals": total_active,
            "active_symbols": active_symbols,
        }


# Global store instance
tv_store = TVAlertStore(max_age_sec=300)


# ============================================================
# API Endpoints
# ============================================================

@router.post("/api/tv-alerts")
async def receive_tv_alert(alert: TVAlertInput):
    """
    Receive a webhook alert from TradingView.
    Validates passphrase, normalizes signal, stores for confluence.
    """
    # Validate passphrase
    passphrase = WEBHOOK_PASSPHRASE or os.getenv("YELENA_WEBHOOK_PASSPHRASE", "")
    if not passphrase:
        # Try loading from SSM (cached after first call)
        try:
            import boto3
            ssm = boto3.client("ssm", region_name="us-east-1")
            resp = ssm.get_parameter(Name="/yelena/webhook-passphrase", WithDecryption=True)
            passphrase = resp["Parameter"]["Value"]
            # Cache it
            os.environ["YELENA_WEBHOOK_PASSPHRASE"] = passphrase
        except Exception as e:
            logger.error(f"Failed to get webhook passphrase from SSM: {e}")
            raise HTTPException(status_code=500, detail="Passphrase not configured")

    if alert.passphrase != passphrase:
        logger.warning(f"Invalid passphrase from TV alert: {alert.symbol} {alert.indicator}")
        raise HTTPException(status_code=401, detail="Invalid passphrase")

    # Normalize symbol (TV might send "AMEX:SPY" — strip exchange prefix)
    symbol = alert.symbol.split(":")[-1].upper().strip()

    # Normalize signal direction
    raw_signal = alert.signal or alert.action
    direction = SIGNAL_MAP.get(raw_signal, "HOLD")

    if direction == "HOLD":
        # Try action field as fallback
        direction = SIGNAL_MAP.get(alert.action, "HOLD")

    # Get indicator weight
    weight = INDICATOR_WEIGHTS.get(alert.indicator, 1.0)

    # Strong signals get bonus weight
    if "STRONG" in raw_signal.upper():
        weight *= 1.5

    # Normalize timeframe
    tf_map = {"1": "1min", "5": "5min", "15": "15min", "60": "1hr", "D": "1D"}
    timeframe = tf_map.get(str(alert.timeframe), f"{alert.timeframe}min")

    # Create normalized signal
    sig = TVSignal(
        symbol=symbol,
        indicator=alert.indicator,
        direction=direction,
        raw_signal=raw_signal,
        price=alert.price,
        weight=weight,
        timestamp=time.time(),
        timeframe=timeframe,
    )

    # Store
    tv_store.add_signal(sig)

    # Get current confluence
    confluence = tv_store.get_confluence(symbol)

    logger.info(
        f"📺 TV Alert: {symbol} {alert.indicator} {raw_signal} → {direction} "
        f"(weight {weight}) | Confluence: {confluence['direction']} {confluence['score']:.0f}%"
    )

    return TVAlertResponse(
        status="received",
        message=f"{symbol} {alert.indicator} {raw_signal} processed",
        direction=direction,
        confluence_score=confluence["score"],
    )


@router.get("/api/tv-alerts/confluence/{symbol}")
async def get_symbol_confluence(symbol: str):
    """Get current TV confluence for a symbol."""
    return tv_store.get_confluence(symbol.upper())


@router.get("/api/tv-alerts/stats")
async def get_tv_stats():
    """Get TV alert statistics."""
    return tv_store.get_stats()


@router.get("/api/tv-alerts/history")
async def get_tv_history(limit: int = 20):
    """Get recent TV alert history."""
    return tv_store.history[:limit]
