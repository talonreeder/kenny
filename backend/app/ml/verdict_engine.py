"""
YELENA v2 — Trade Verdict Engine
The brain between raw ML predictions and actionable trade decisions.

Consolidates multi-timeframe signals into binary GO/SKIP verdicts
with pre-computed trade plans (entry, target, stop-loss, R:R).

Rules:
    1. 15min anchor must fire (CALL or PUT) with >= 65% confidence
    2. At least 1 confirming timeframe (1min or 5min) in same direction
    3. If 1hr agrees, boost confidence
    4. Calculate entry/target/stop from ATR
    5. Only surface GO if R:R >= 1.5
    6. One active verdict per symbol at a time
    7. Close verdict when: direction flips, confidence drops, or max duration hit
"""

import time
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger("yelena.verdict")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Prediction:
    """A single ML prediction for one symbol + timeframe."""
    symbol: str
    timeframe: str
    direction: str          # CALL, PUT, HOLD
    probability: float
    confidence: float
    grade: str
    models_agreeing: int
    unanimous: bool
    individual: dict
    timestamp: float        # Unix timestamp
    feature_ms: float = 0
    predict_ms: float = 0
    total_ms: float = 0


@dataclass
class TradePlan:
    """Pre-computed trade plan with entry/target/stop."""
    entry: float
    target: float
    stop: float
    risk_reward: float
    atr: float              # Current ATR used for calculation
    direction: str          # CALL or PUT


@dataclass
class Verdict:
    """A trade verdict — the final decision."""
    id: Optional[int] = None
    symbol: str = ""
    direction: str = ""     # CALL or PUT
    verdict: str = ""       # GO or SKIP
    confidence: float = 0.0
    
    # Trade plan
    entry: float = 0.0
    target: float = 0.0
    stop: float = 0.0
    risk_reward: str = ""
    
    # Alignment detail
    timeframe_alignment: Dict[str, str] = field(default_factory=dict)  # {tf: direction}
    tf_confidences: Dict[str, float] = field(default_factory=dict)     # {tf: confidence}
    models_agreeing: int = 0
    anchor_tf: str = "15min"
    confirming_tfs: List[str] = field(default_factory=list)
    
    # Reasoning
    reason: str = ""
    
    # Lifecycle
    status: str = "active"  # active, closed, expired
    opened_at: str = ""
    closed_at: Optional[str] = None
    result: Optional[str] = None   # WIN, LOSS, null
    pnl: Optional[float] = None
    close_reason: Optional[str] = None
    
    # Timing
    max_duration_min: int = 15  # Auto-close after 15 min
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 1),
            "entry": round(self.entry, 2),
            "target": round(self.target, 2),
            "stop": round(self.stop, 2),
            "riskReward": self.risk_reward,
            "timeframeAlignment": self.timeframe_alignment,
            "tfConfidences": {k: round(v, 1) for k, v in self.tf_confidences.items()},
            "modelsAgreeing": self.models_agreeing,
            "anchorTf": self.anchor_tf,
            "confirmingTfs": self.confirming_tfs,
            "reason": self.reason,
            "status": self.status,
            "timestamp": self.opened_at,
            "closedAt": self.closed_at,
            "result": self.result,
            "pnl": self.pnl,
            "closeReason": self.close_reason,
        }


# ============================================================================
# VERDICT ENGINE CONFIGURATION
# ============================================================================

@dataclass
class VerdictConfig:
    """Tunable parameters for verdict generation."""
    # Anchor requirements
    anchor_tf: str = "15min"                # Primary timeframe
    anchor_min_confidence: float = 75.0     # Minimum confidence on anchor
    
    # Confirmation requirements
    confirming_tfs: List[str] = field(default_factory=lambda: ["1min", "5min"])
    min_confirming: int = 2                 # At least 1 confirming TF
    confirming_min_confidence: float = 65.0 # Min confidence for confirming TFs
    
    # Boost
    boost_tf: str = "1hr"                   # If this agrees, boost confidence
    boost_amount: float = 5.0               # Add this to confidence
    
    # TradingView confluence
    tv_confluence_boost: float = 7.0        # Boost when TV agrees with ML
    tv_confluence_penalty: float = -5.0     # Penalty when TV disagrees
    tv_min_signals: int = 1                 # Minimum TV signals needed to factor in
    
    # Trade plan
    atr_target_mult: float = 1.5            # Target = entry ± ATR * mult
    atr_stop_mult: float = 0.6              # Stop = entry ∓ ATR * mult
    min_risk_reward: float = 1.5            # Minimum R:R to GO
    
    # Lifecycle
    max_active_per_symbol: int = 1          # Only 1 active verdict per symbol
    max_duration_min: int = 15              # Auto-expire after N minutes
    cooldown_min: int = 3                   # Wait N min after closing before new verdict
    
    # Prediction staleness
    prediction_max_age_sec: float = 120.0   # Ignore predictions older than 2min
    
    # Overall GO threshold
    min_go_confidence: float = 78.0         # Final confidence must be >= this


# ============================================================================
# VERDICT ENGINE
# ============================================================================

class VerdictEngine:
    """
    Evaluates incoming ML predictions and produces Trade Verdicts.
    
    Maintains state per symbol:
    - Latest prediction per timeframe
    - Current active verdict (if any)
    - Recent price data for ATR calculation
    - Cooldown timers
    """
    
    def __init__(self, config: VerdictConfig = None):
        self.config = config or VerdictConfig()
        
        # TradingView confluence source
        # Can be: tv_store reference (same process) or backend URL (cross-process)
        self.tv_store = None        # Direct reference if in same process
        self.tv_api_url = None      # e.g. "http://localhost:8000" for cross-process
        
        # Per-symbol state
        # {symbol: {timeframe: Prediction}}
        self.latest_predictions: Dict[str, Dict[str, Prediction]] = {}
        
        # Active verdicts: {symbol: Verdict}
        self.active_verdicts: Dict[str, Verdict] = {}
        
        # Verdict history (today)
        self.history: List[Verdict] = []
        
        # Cooldown: {symbol: expiry_timestamp}
        self.cooldowns: Dict[str, float] = {}
        
        # Price data for ATR: {symbol: list of recent close prices}
        self.recent_closes: Dict[str, List[float]] = {}
        self.recent_highs: Dict[str, List[float]] = {}
        self.recent_lows: Dict[str, List[float]] = {}
        
        # Stats
        self.verdicts_today = 0
        self.go_count = 0
        self.skip_count = 0
        
        # Verdict ID counter (will be replaced by DB IDs in production)
        self._next_id = 1
    
    # ----------------------------------------------------------------
    # PRICE DATA — for ATR calculation
    # ----------------------------------------------------------------
    
    def update_price(self, symbol: str, close: float, high: float, low: float):
        """Update recent price data for a symbol (called with each 1min bar)."""
        max_bars = 20  # Keep last 20 bars for ATR
        
        if symbol not in self.recent_closes:
            self.recent_closes[symbol] = []
            self.recent_highs[symbol] = []
            self.recent_lows[symbol] = []
        
        self.recent_closes[symbol].append(close)
        self.recent_highs[symbol].append(high)
        self.recent_lows[symbol].append(low)
        
        # Trim
        self.recent_closes[symbol] = self.recent_closes[symbol][-max_bars:]
        self.recent_highs[symbol] = self.recent_highs[symbol][-max_bars:]
        self.recent_lows[symbol] = self.recent_lows[symbol][-max_bars:]
    
    def compute_atr(self, symbol: str, period: int = 14) -> float:
        """Compute Average True Range from recent bars."""
        closes = self.recent_closes.get(symbol, [])
        highs = self.recent_highs.get(symbol, [])
        lows = self.recent_lows.get(symbol, [])
        
        if len(closes) < 2:
            # Fallback: estimate ATR as 0.1% of price
            return closes[-1] * 0.001 if closes else 1.0
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            true_ranges.append(tr)
        
        # Use last `period` true ranges (or all available)
        recent_tr = true_ranges[-period:]
        return sum(recent_tr) / len(recent_tr) if recent_tr else 1.0
    
    # ----------------------------------------------------------------
    # TRADINGVIEW CONFLUENCE
    # ----------------------------------------------------------------
    
    def _get_tv_confluence(self, symbol: str) -> Optional[dict]:
        """Get TV confluence — from direct store or HTTP API."""
        # Method 1: Direct store reference (same process)
        if self.tv_store:
            return self.tv_store.get_confluence(symbol)
        
        # Method 2: HTTP API (cross-process — orchestrator → backend)
        if self.tv_api_url:
            try:
                import requests
                resp = requests.get(
                    f"{self.tv_api_url}/api/tv-alerts/confluence/{symbol}",
                    timeout=0.5  # Fast timeout — don't block predictions
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass  # TV confluence is optional — fail silently
        
        return None
    
    # ----------------------------------------------------------------
    # PREDICTION INTAKE
    # ----------------------------------------------------------------
    
    def ingest_prediction(self, pred: Prediction) -> Optional[Verdict]:
        """
        Process a new prediction. Returns a new Verdict if one is generated,
        or None if no verdict change.
        """
        symbol = pred.symbol
        tf = pred.timeframe
        
        # Store latest prediction for this symbol + timeframe
        if symbol not in self.latest_predictions:
            self.latest_predictions[symbol] = {}
        self.latest_predictions[symbol][tf] = pred
        
        # Check if we should evaluate a verdict
        # Only evaluate when the anchor TF fires
        if tf == self.config.anchor_tf:
            return self._evaluate_verdict(symbol)
        
        # Also evaluate when a confirming TF fires AND we don't have an active verdict
        if tf in self.config.confirming_tfs and symbol not in self.active_verdicts:
            return self._evaluate_verdict(symbol)
        
        # Check if existing active verdict should be updated/closed
        if symbol in self.active_verdicts:
            return self._check_active_verdict(symbol, pred)
        
        return None
    
    # ----------------------------------------------------------------
    # VERDICT EVALUATION
    # ----------------------------------------------------------------
    
    def _evaluate_verdict(self, symbol: str) -> Optional[Verdict]:
        """
        Evaluate whether to generate a GO or SKIP verdict for a symbol.
        This is the core decision engine.
        """
        now = time.time()
        preds = self.latest_predictions.get(symbol, {})
        
        # Check cooldown
        if symbol in self.cooldowns and now < self.cooldowns[symbol]:
            remaining = self.cooldowns[symbol] - now
            logger.debug(f"{symbol}: In cooldown ({remaining:.0f}s remaining)")
            return None
        
        # Check if we already have an active verdict
        if symbol in self.active_verdicts:
            return None
        
        # ── Step 1: Check anchor TF ──
        anchor = preds.get(self.config.anchor_tf)
        if not anchor:
            return None
        
        # Is anchor prediction fresh?
        if now - anchor.timestamp > self.config.prediction_max_age_sec:
            logger.debug(f"{symbol}: Anchor prediction stale ({now - anchor.timestamp:.0f}s old)")
            return None
        
        # Is anchor directional with sufficient confidence?
        if anchor.direction == "HOLD":
            return None
        
        if anchor.confidence < self.config.anchor_min_confidence:
            logger.debug(
                f"{symbol}: Anchor confidence too low "
                f"({anchor.confidence:.1f}% < {self.config.anchor_min_confidence}%)"
            )
            return None
        
        anchor_direction = anchor.direction
        
        # ── Step 2: Check confirming TFs ──
        confirming = []
        tf_alignment = {self.config.anchor_tf: anchor_direction}
        tf_confidences = {self.config.anchor_tf: anchor.confidence}
        
        for ctf in self.config.confirming_tfs:
            cp = preds.get(ctf)
            if not cp:
                tf_alignment[ctf] = "HOLD"
                tf_confidences[ctf] = 0.0
                continue
            
            # Check staleness
            if now - cp.timestamp > self.config.prediction_max_age_sec:
                tf_alignment[ctf] = "STALE"
                tf_confidences[ctf] = 0.0
                continue
            
            tf_alignment[ctf] = cp.direction
            tf_confidences[ctf] = cp.confidence
            
            if cp.direction == anchor_direction and cp.confidence >= self.config.confirming_min_confidence:
                confirming.append(ctf)
        
        # Check 1hr for boost
        boost = 0.0
        hr_pred = preds.get(self.config.boost_tf)
        if hr_pred:
            tf_alignment[self.config.boost_tf] = hr_pred.direction
            tf_confidences[self.config.boost_tf] = hr_pred.confidence
            if hr_pred.direction == anchor_direction:
                boost = self.config.boost_amount
        else:
            tf_alignment[self.config.boost_tf] = "HOLD"
            tf_confidences[self.config.boost_tf] = 0.0
        
        # ── Step 3: Enough confirmation? ──
        if len(confirming) < self.config.min_confirming:
            logger.debug(
                f"{symbol}: Not enough confirming TFs "
                f"({len(confirming)}/{self.config.min_confirming})"
            )
            return None  # Not enough — stay silent (don't even SKIP)
        
        # ── Step 4: Calculate composite confidence ──
        # Weighted: anchor 50%, confirming TFs split remaining 50%
        anchor_weight = 0.50
        confirm_weight = 0.50 / max(len(confirming), 1)
        
        composite = anchor.confidence * anchor_weight
        for ctf in confirming:
            cp = preds[ctf]
            composite += cp.confidence * confirm_weight
        composite += boost
        composite = min(composite, 99.0)
        
        # ── Step 4b: TradingView Confluence ──
        tv_confluence = None
        tv_boost = 0.0
        tv_confluence = self._get_tv_confluence(symbol)
        if tv_confluence and tv_confluence.get("signals", 0) >= self.config.tv_min_signals:
            tv_dir = tv_confluence["direction"]
            tv_score = tv_confluence["score"]
            
            if tv_dir == anchor_direction:
                # TV agrees with ML — boost confidence
                tv_boost = self.config.tv_confluence_boost * (tv_score / 100)
                composite += tv_boost
                logger.info(
                    f"  📺 TV confluence AGREES: {tv_dir} {tv_score:.0f}% "
                    f"(+{tv_boost:.1f}% boost)"
                )
            elif tv_dir != "HOLD" and tv_dir != anchor_direction:
                # TV disagrees — penalize
                tv_boost = self.config.tv_confluence_penalty * (tv_score / 100)
                composite += tv_boost
                logger.info(
                    f"  📺 TV confluence DISAGREES: {tv_dir} {tv_score:.0f}% "
                    f"({tv_boost:.1f}% penalty)"
                )
            
            composite = max(0, min(composite, 99.0))
        
        # ── Step 5: Compute trade plan ──
        atr = self.compute_atr(symbol)
        current_price = self.recent_closes.get(symbol, [0])[-1] if self.recent_closes.get(symbol) else 0
        
        if current_price == 0:
            logger.warning(f"{symbol}: No price data for trade plan")
            return None
        
        plan = self._compute_trade_plan(anchor_direction, current_price, atr)
        
        # ── Step 6: Final GO/SKIP decision ──
        is_go = (
            composite >= self.config.min_go_confidence
            and plan.risk_reward >= self.config.min_risk_reward
        )
        
        # ── Step 7: Build verdict ──
        verdict = Verdict(
            id=self._next_id,
            symbol=symbol,
            direction=anchor_direction,
            verdict="GO" if is_go else "SKIP",
            confidence=composite,
            entry=plan.entry,
            target=plan.target,
            stop=plan.stop,
            risk_reward=f"1:{plan.risk_reward:.1f}",
            timeframe_alignment=tf_alignment,
            tf_confidences=tf_confidences,
            models_agreeing=anchor.models_agreeing,
            anchor_tf=self.config.anchor_tf,
            confirming_tfs=confirming,
            reason=self._build_reason(
                symbol, anchor_direction, anchor, confirming,
                tf_alignment, composite, boost > 0, is_go, plan, tv_confluence, tv_boost
            ),
            status="active" if is_go else "closed",
            opened_at=datetime.now(timezone.utc).isoformat(),
            max_duration_min=self.config.max_duration_min,
        )
        self._next_id += 1
        
        # Track
        self.verdicts_today += 1
        if is_go:
            self.go_count += 1
            self.active_verdicts[symbol] = verdict
            logger.info(
                f"🎯 VERDICT {symbol}: GO {anchor_direction} "
                f"{composite:.1f}% | Entry ${plan.entry:.2f} "
                f"Target ${plan.target:.2f} Stop ${plan.stop:.2f} "
                f"R:R {plan.risk_reward:.1f} | "
                f"TFs: {'+'.join([self.config.anchor_tf] + confirming)}"
            )
        else:
            self.skip_count += 1
            self.history.append(verdict)
            logger.info(
                f"⏭️ VERDICT {symbol}: SKIP {anchor_direction} "
                f"{composite:.1f}% (min {self.config.min_go_confidence}%) | "
                f"R:R {plan.risk_reward:.1f} (min {self.config.min_risk_reward})"
            )
        
        return verdict
    
    def _compute_trade_plan(self, direction: str, price: float, atr: float) -> TradePlan:
        """Calculate entry, target, stop from current price and ATR."""
        entry = price
        
        if direction == "CALL":
            target = entry + atr * self.config.atr_target_mult
            stop = entry - atr * self.config.atr_stop_mult
        else:  # PUT
            target = entry - atr * self.config.atr_target_mult
            stop = entry + atr * self.config.atr_stop_mult
        
        target_dist = abs(target - entry)
        stop_dist = abs(stop - entry)
        rr = target_dist / stop_dist if stop_dist > 0 else 0.0
        
        return TradePlan(
            entry=round(entry, 2),
            target=round(target, 2),
            stop=round(stop, 2),
            risk_reward=round(rr, 1),
            atr=atr,
            direction=direction
        )
    
    def _build_reason(
        self, symbol, direction, anchor, confirming, alignment,
        composite, has_boost, is_go, plan, tv_confluence=None, tv_boost=0.0
    ) -> str:
        """Build human-readable reason string."""
        parts = []
        
        # Anchor
        parts.append(
            f"{self.config.anchor_tf} anchor {direction} {anchor.confidence:.0f}%"
        )
        
        # Confirming
        if confirming:
            parts.append(f"{'+'.join(confirming)} confirm{'s' if len(confirming) == 1 else ''}")
        
        # Boost
        if has_boost:
            parts.append("1hr agrees (+boost)")
        
        # TV Confluence
        if tv_confluence and tv_confluence.get("signals", 0) > 0:
            tv_dir = tv_confluence["direction"]
            if tv_dir == direction:
                parts.append(f"TV confluence agrees ({tv_confluence['signals']} indicators, +{tv_boost:.0f}%)")
            elif tv_dir != "HOLD":
                parts.append(f"TV confluence disagrees ({tv_confluence['signals']} indicators, {tv_boost:.0f}%)")
        
        # Opposition
        opposing = [
            tf for tf, d in alignment.items()
            if d not in (direction, "HOLD", "STALE", "") and tf != self.config.anchor_tf
        ]
        if opposing:
            parts.append(f"{'+'.join(opposing)} opposing")
        
        reason = ". ".join(parts) + "."
        
        if not is_go:
            if composite < self.config.min_go_confidence:
                reason += f" Confidence {composite:.0f}% below {self.config.min_go_confidence:.0f}% threshold."
            if plan.risk_reward < self.config.min_risk_reward:
                reason += f" R:R {plan.risk_reward:.1f} below {self.config.min_risk_reward:.1f} minimum."
        
        return reason
    
    # ----------------------------------------------------------------
    # ACTIVE VERDICT MANAGEMENT
    # ----------------------------------------------------------------
    
    def _check_active_verdict(self, symbol: str, new_pred: Prediction) -> Optional[Verdict]:
        """Check if an active verdict should be closed based on new data."""
        verdict = self.active_verdicts.get(symbol)
        if not verdict:
            return None
        
        now = time.time()
        
        # Check expiry
        opened_time = datetime.fromisoformat(verdict.opened_at).timestamp()
        age_min = (now - opened_time) / 60
        
        if age_min >= verdict.max_duration_min:
            return self._close_verdict(symbol, "expired", f"Max duration {verdict.max_duration_min}m reached")
        
        # Check if anchor flipped direction
        anchor = self.latest_predictions.get(symbol, {}).get(self.config.anchor_tf)
        if anchor and anchor.direction != "HOLD" and anchor.direction != verdict.direction:
            return self._close_verdict(
                symbol, "flipped",
                f"Anchor {self.config.anchor_tf} flipped to {anchor.direction}"
            )
        
        # Check if anchor confidence dropped significantly
        if anchor and anchor.direction == verdict.direction:
            if anchor.confidence < self.config.anchor_min_confidence - 10:
                return self._close_verdict(
                    symbol, "weakened",
                    f"Anchor confidence dropped to {anchor.confidence:.0f}%"
                )
        
        # Update confidence on the active verdict (live tracking)
        if new_pred.timeframe in [self.config.anchor_tf] + self.config.confirming_tfs:
            self._update_active_confidence(symbol)
        
        return None
    
    def _update_active_confidence(self, symbol: str):
        """Recalculate confidence for an active verdict."""
        verdict = self.active_verdicts.get(symbol)
        if not verdict:
            return
        
        preds = self.latest_predictions.get(symbol, {})
        anchor = preds.get(self.config.anchor_tf)
        
        if not anchor or anchor.direction != verdict.direction:
            return
        
        # Recalculate alignment
        for tf in ["1min", "5min", "15min", "1hr"]:
            p = preds.get(tf)
            if p:
                verdict.timeframe_alignment[tf] = p.direction
                verdict.tf_confidences[tf] = p.confidence
    
    def _close_verdict(self, symbol: str, reason: str, detail: str) -> Optional[Verdict]:
        """Close an active verdict."""
        verdict = self.active_verdicts.pop(symbol, None)
        if not verdict:
            return None
        
        verdict.status = "closed"
        verdict.closed_at = datetime.now(timezone.utc).isoformat()
        verdict.close_reason = detail
        
        # Set cooldown
        self.cooldowns[symbol] = time.time() + self.config.cooldown_min * 60
        
        # Move to history
        self.history.append(verdict)
        
        logger.info(f"📕 CLOSED {symbol} {verdict.direction}: {detail}")
        
        return verdict
    
    def close_verdict_with_result(
        self, symbol: str, result: str, pnl: float = 0.0
    ) -> Optional[Verdict]:
        """Close a verdict with a WIN/LOSS result (called by execution layer)."""
        verdict = self.active_verdicts.pop(symbol, None)
        if not verdict:
            return None
        
        verdict.status = "closed"
        verdict.closed_at = datetime.now(timezone.utc).isoformat()
        verdict.result = result
        verdict.pnl = pnl
        verdict.close_reason = f"{result}: ${pnl:+.2f}"
        
        self.cooldowns[symbol] = time.time() + self.config.cooldown_min * 60
        self.history.append(verdict)
        
        emoji = "✅" if result == "WIN" else "❌"
        logger.info(f"{emoji} {symbol} {verdict.direction}: {result} ${pnl:+.2f}")
        
        return verdict
    
    # ----------------------------------------------------------------
    # PERIODIC MAINTENANCE
    # ----------------------------------------------------------------
    
    def check_expirations(self) -> List[Verdict]:
        """Check all active verdicts for expiration. Call periodically."""
        closed = []
        now = time.time()
        
        for symbol in list(self.active_verdicts.keys()):
            verdict = self.active_verdicts[symbol]
            opened_time = datetime.fromisoformat(verdict.opened_at).timestamp()
            age_min = (now - opened_time) / 60
            
            if age_min >= verdict.max_duration_min:
                v = self._close_verdict(symbol, "expired", f"Max duration {verdict.max_duration_min}m reached")
                if v:
                    closed.append(v)
        
        return closed
    
    # ----------------------------------------------------------------
    # QUERIES
    # ----------------------------------------------------------------
    
    def get_active_verdicts(self) -> List[dict]:
        """Get all active GO verdicts."""
        now = time.time()
        results = []
        for symbol, v in self.active_verdicts.items():
            d = v.to_dict()
            # Add computed age
            opened_time = datetime.fromisoformat(v.opened_at).timestamp()
            age_sec = now - opened_time
            if age_sec < 60:
                d["age"] = f"{age_sec:.0f}s ago"
            else:
                d["age"] = f"{age_sec / 60:.0f}m ago"
            results.append(d)
        return results
    
    def get_history(self, limit: int = 20) -> List[dict]:
        """Get recent verdict history."""
        return [v.to_dict() for v in reversed(self.history[-limit:])]
    
    def get_stats(self) -> dict:
        """Get daily verdict statistics."""
        wins = sum(1 for v in self.history if v.result == "WIN")
        losses = sum(1 for v in self.history if v.result == "LOSS")
        total_pnl = sum(v.pnl or 0 for v in self.history)
        total_decided = wins + losses
        
        return {
            "verdicts_today": self.verdicts_today,
            "go_count": self.go_count,
            "skip_count": self.skip_count,
            "active_count": len(self.active_verdicts),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total_decided * 100, 1) if total_decided > 0 else 0.0,
            "pnl": round(total_pnl, 2),
        }
    
    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        self.history = []
        self.active_verdicts = {}
        self.cooldowns = {}
        self.verdicts_today = 0
        self.go_count = 0
        self.skip_count = 0
        logger.info("Daily reset complete")
