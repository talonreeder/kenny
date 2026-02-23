"""
YELENA v2 — TV-ML Synergy Integration
══════════════════════════════════════════
Tier 1 Integration: TradingView Confluence as Verdict Engine Input

This module bridges two completely different analytical structures:
    - ML Ensemble: Statistical pattern recognition (XGBoost, Transformer, CNN, RL)
    - TV Indicators: Technical analysis (9 sub-indicators via Master Confluence)

They work synergistically because they see the market through different lenses:
    - ML sees patterns in raw numerical features (163 engineered features)
    - TV sees chart structure (trend, momentum, volume, S/R, smart money)

Integration Points:
    1. Confidence Adjustment — TV agreement/divergence modifies ML confidence
    2. Entry/Exit Enhancement — TV's ATR-based SL/TP compared with ML targets
    3. Component Signals     — Individual TV indicators as quality filters
    4. Multi-TF Alignment    — TV signals across timeframes boost conviction

Usage in Verdict Engine:
    from app.ml.tv_confluence import TVConfluenceIntegrator

    integrator = TVConfluenceIntegrator(tv_store)
    adjustment = integrator.compute_adjustment(
        symbol="SPY",
        timeframe="15min",
        ml_direction="CALL",
        ml_confidence=72.0,
    )
    # adjustment.confidence_delta = +12.0 (TV agrees strongly)
    # adjustment.should_use_tv_levels = True
    # adjustment.tv_entry = 603.50, tv_sl = 601.20, tv_tp1 = 605.80
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger("yelena.tv_confluence")


# ============================================================
# Adjustment Result
# ============================================================

@dataclass
class TVAdjustment:
    """Result of TV-ML synergy computation."""

    # ── Core Adjustment ──
    confidence_delta: float = 0.0       # How much to adjust ML confidence (+/-)
    adjusted_confidence: float = 0.0    # Final confidence after adjustment
    agreement: str = "none"             # "strong_agree", "agree", "neutral", "diverge", "strong_diverge"

    # ── TV Signal Data ──
    tv_direction: str = "HOLD"
    tv_score: float = 0.0              # -10 to +10
    tv_grade: str = ""
    tv_confidence: float = 0.0
    tv_age_sec: int = 0

    # ── Entry/Exit Enhancement ──
    should_use_tv_levels: bool = False
    tv_entry: float = 0.0
    tv_stop_loss: float = 0.0
    tv_tp1: float = 0.0
    tv_tp2: float = 0.0
    tv_tp3: float = 0.0

    # ── Component Quality Signals ──
    squeeze_active: bool = False        # QCloud or QBands squeeze = volatility expansion coming
    structure_break: bool = False       # QSMC BOS/CHoCH = structural confirmation
    volume_spike: bool = False          # QCVD spike = institutional activity
    bounce_signal: bool = False         # QLine bounce = S/R confirmation
    multi_tf_aligned: bool = False      # Multiple TFs agree

    # ── Explanation ──
    reasons: list = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


# ============================================================
# TV Confluence Integrator
# ============================================================

class TVConfluenceIntegrator:
    """
    Computes synergistic adjustments between ML predictions and TV signals.

    The integrator never OVERRIDES ML — it MODULATES confidence:
    - Strong TV agreement: ML confidence boosted up to +15%
    - TV divergence: ML confidence penalized up to -20%
    - No TV signal: No adjustment (ML stands alone)
    """

    # ── Tunable Parameters ──
    MAX_BOOST = 15.0            # Max confidence increase when TV agrees
    MAX_PENALTY = -20.0         # Max confidence decrease when TV diverges
    STRONG_AGREE_THRESHOLD = 7.0  # TV score magnitude for "strong" agreement
    AGREE_THRESHOLD = 4.0       # TV score magnitude for basic agreement
    STALE_PENALTY_FACTOR = 0.5  # Reduce adjustment when TV signal is aging
    GRADE_MULTIPLIER = {        # TV grade quality multiplier
        "A+": 1.0,
        "A": 0.8,
        "B+": 0.5,
        "B": 0.3,
        "": 0.0,
    }

    def __init__(self, tv_store):
        """
        Args:
            tv_store: TVAlertStore instance from tv_alerts.py
        """
        self.tv_store = tv_store

    def compute_adjustment(
        self,
        symbol: str,
        timeframe: str,
        ml_direction: str,
        ml_confidence: float,
    ) -> TVAdjustment:
        """
        Compute the synergistic adjustment for a verdict.

        Args:
            symbol: Trading symbol (e.g., "SPY")
            timeframe: Timeframe (e.g., "15min")
            ml_direction: ML predicted direction ("CALL" or "PUT")
            ml_confidence: ML confidence (0-100)

        Returns:
            TVAdjustment with confidence delta and enhanced data
        """
        # Get TV confluence
        tv = self.tv_store.get_confluence(symbol, timeframe)

        adj = TVAdjustment()
        adj.tv_direction = tv["direction"]
        adj.tv_score = tv["score"]
        adj.tv_grade = tv["grade"]
        adj.tv_confidence = tv["confidence"]
        adj.tv_age_sec = tv["age_sec"]

        # ── No TV signal → ML stands alone ──
        if tv["signals"] == 0 or tv["direction"] == "HOLD":
            adj.agreement = "none"
            adj.confidence_delta = 0.0
            adj.adjusted_confidence = ml_confidence
            adj.reasons.append("No active TV signal — ML prediction stands alone")
            return adj

        # ── Compute directional agreement ──
        same_direction = (ml_direction == tv["direction"])
        score_magnitude = abs(tv["score"])
        grade_mult = self.GRADE_MULTIPLIER.get(tv["grade"], 0.3)

        # ── Freshness decay: older signals get less weight ──
        if tv.get("is_stale", False):
            freshness = self.STALE_PENALTY_FACTOR
            adj.reasons.append(f"TV signal aging ({tv['age_sec']}s) — reduced weight")
        else:
            freshness = 1.0

        # ── Calculate confidence delta ──
        if same_direction:
            # TV AGREES with ML
            if score_magnitude >= self.STRONG_AGREE_THRESHOLD:
                adj.agreement = "strong_agree"
                raw_boost = self.MAX_BOOST
                adj.reasons.append(
                    f"Strong TV agreement: TV {tv['direction']} {tv['grade']} "
                    f"(score {tv['score']:.1f}) aligns with ML {ml_direction}"
                )
            elif score_magnitude >= self.AGREE_THRESHOLD:
                adj.agreement = "agree"
                raw_boost = self.MAX_BOOST * 0.6
                adj.reasons.append(
                    f"TV agreement: TV {tv['direction']} (score {tv['score']:.1f}) "
                    f"supports ML {ml_direction}"
                )
            else:
                adj.agreement = "weak_agree"
                raw_boost = self.MAX_BOOST * 0.3
                adj.reasons.append(
                    f"Weak TV agreement: TV {tv['direction']} (score {tv['score']:.1f})"
                )

            adj.confidence_delta = raw_boost * grade_mult * freshness
        else:
            # TV DIVERGES from ML
            if score_magnitude >= self.STRONG_AGREE_THRESHOLD:
                adj.agreement = "strong_diverge"
                raw_penalty = self.MAX_PENALTY
                adj.reasons.append(
                    f"⚠️ Strong TV divergence: TV {tv['direction']} {tv['grade']} "
                    f"(score {tv['score']:.1f}) OPPOSES ML {ml_direction}"
                )
            elif score_magnitude >= self.AGREE_THRESHOLD:
                adj.agreement = "diverge"
                raw_penalty = self.MAX_PENALTY * 0.6
                adj.reasons.append(
                    f"⚠️ TV divergence: TV {tv['direction']} (score {tv['score']:.1f}) "
                    f"opposes ML {ml_direction}"
                )
            else:
                adj.agreement = "weak_diverge"
                raw_penalty = self.MAX_PENALTY * 0.3
                adj.reasons.append(
                    f"Mild TV divergence: TV {tv['direction']} (score {tv['score']:.1f})"
                )

            adj.confidence_delta = raw_penalty * grade_mult * freshness

        # ── Apply adjustment (clamp 0-100) ──
        adj.adjusted_confidence = max(0.0, min(100.0,
            ml_confidence + adj.confidence_delta
        ))

        # ── Entry/Exit Enhancement ──
        if tv["entry"] > 0 and tv["stop_loss"] > 0 and same_direction:
            adj.should_use_tv_levels = True
            adj.tv_entry = tv["entry"]
            adj.tv_stop_loss = tv["stop_loss"]
            adj.tv_tp1 = tv["tp1"]
            adj.tv_tp2 = tv["tp2"]
            adj.tv_tp3 = tv["tp3"]
            adj.reasons.append(
                f"TV entry/SL/TP available: E={tv['entry']:.2f} "
                f"SL={tv['stop_loss']:.2f} TP1={tv['tp1']:.2f}"
            )

        # ── Component Quality Signals ──
        self._extract_component_signals(tv, adj)

        # ── Multi-TF Check ──
        multi_tf = self.tv_store.get_multi_tf_confluence(symbol)
        if multi_tf["direction"] == ml_direction and abs(multi_tf["alignment"]) > 0.5:
            adj.multi_tf_aligned = True
            bonus = 3.0 * abs(multi_tf["alignment"])
            adj.confidence_delta += bonus
            adj.adjusted_confidence = max(0.0, min(100.0,
                adj.adjusted_confidence + bonus
            ))
            adj.reasons.append(
                f"Multi-TF alignment: {len(multi_tf['timeframes'])} TFs "
                f"align {multi_tf['direction']} ({multi_tf['alignment']:.2f})"
            )

        return adj

    def _extract_component_signals(self, tv_data: dict, adj: TVAdjustment):
        """Extract quality signals from individual TV components."""
        components = tv_data.get("components")
        if not components:
            return

        # QCloud squeeze → volatility expansion imminent
        qcloud = components.get("qcloud", {})
        if qcloud and qcloud.get("squeeze") == 1:
            adj.squeeze_active = True
            adj.reasons.append("QCloud SQUEEZE active — volatility expansion imminent")

        # QBands squeeze fire → directional breakout
        qbands = components.get("qbands", {})
        if qbands and qbands.get("squeeze_fire") != 0:
            adj.squeeze_active = True
            fire_dir = "BULL" if qbands["squeeze_fire"] == 1 else "BEAR"
            adj.reasons.append(f"QBands squeeze FIRED {fire_dir}")

        # QSMC structure break → institutional confirmation
        qsmc = components.get("qsmc", {})
        if qsmc:
            if qsmc.get("bos") != 0 or qsmc.get("choch") != 0:
                adj.structure_break = True
                event = "BOS" if qsmc.get("bos") != 0 else "CHoCH"
                adj.reasons.append(f"QSMC {event} — structural break confirmed")

        # QCVD spike → big institutional volume
        qcvd = components.get("qcvd", {})
        if qcvd and qcvd.get("spike") != 0:
            adj.volume_spike = True
            adj.reasons.append("QCVD volume SPIKE — institutional activity detected")

        # QLine bounce → support/resistance confirmation
        qline = components.get("qline", {})
        if qline and qline.get("bounce", 0) >= 3:
            adj.bounce_signal = True
            adj.reasons.append(f"QLine bounce score {qline['bounce']} — strong S/R confirmation")


# ============================================================
# Historical TV Signal Simulator (for Backtesting)
# ============================================================

class TVConfluenceSimulator:
    """
    Simulates TV confluence signals from historical bar data.
    Used when backtesting — since we don't have actual historical TV alerts,
    we reconstruct approximate signals using the same logic as the Pine indicators.

    This recreates a simplified version of the Master Confluence scoring:
    - Trend from MA crossovers (QCloud/QLine analog)
    - Momentum from RSI/ADX (QWave/QMomentum analog)
    - Volume from volume delta (QCVD analog)
    - Structure from pivot highs/lows (QSMC analog)
    - Bands from Bollinger position (QBands analog)

    NOTE: This is an APPROXIMATION — actual TV signals may differ.
    The simulator provides a reasonable baseline for backtesting
    the TV-ML synergy model.
    """

    def __init__(self):
        self.name = "tv_sim"

    def simulate_confluence(
        self,
        bars: list,
        current_idx: int,
        features: dict = None,
    ) -> dict:
        """
        Simulate TV confluence from historical bars and features.

        Uses ML feature engine's already-computed indicators where available,
        falling back to simple calculations from raw bars.

        Args:
            bars: List of bar dicts with OHLCV
            current_idx: Current bar index
            features: Optional pre-computed features from feature engine

        Returns:
            Simulated confluence dict matching TVAlertStore.get_confluence() format
        """
        if features is None or len(bars) < 20:
            return self._empty_result()

        score = 0.0
        components = {}

        # ── Trend Score (QCloud analog, max ±1.5) ──
        # Use SMA crossover features if available
        sma_fast = features.get("sma_8", features.get("ema_8", 0))
        sma_mid = features.get("sma_21", features.get("ema_21", 0))
        sma_slow = features.get("sma_50", features.get("ema_50", 0))
        close = features.get("close", bars[current_idx].get("close", 0)) if current_idx < len(bars) else 0

        if sma_fast and sma_mid and sma_slow and close:
            bull_signals = sum([
                1 if close > sma_fast else 0,
                1 if sma_fast > sma_mid else 0,
                1 if sma_mid > sma_slow else 0,
                1 if close > sma_slow else 0,
            ])
            trend_score = 1.5 if bull_signals >= 4 else 1.0 if bull_signals == 3 else \
                         -1.5 if bull_signals == 0 else -1.0 if bull_signals == 1 else 0.0
        else:
            trend_score = 0.0
        score += trend_score
        components["trend"] = {"score": trend_score}

        # ── Momentum Score (QWave/QMomentum analog, max ±1.0 each) ──
        rsi = features.get("rsi_14", 50.0)
        adx = features.get("adx_14", 0.0)

        # RSI momentum
        if rsi > 60:
            mom_score = 1.0 if rsi > 70 else 0.5
        elif rsi < 40:
            mom_score = -1.0 if rsi < 30 else -0.5
        else:
            mom_score = 0.0
        score += mom_score
        components["momentum"] = {"score": mom_score, "rsi": rsi}

        # ADX trend strength
        if adx > 25:
            wave_score = 0.5 if rsi > 50 else -0.5
        else:
            wave_score = 0.0
        score += wave_score
        components["wave"] = {"score": wave_score, "adx": adx}

        # ── Bands Score (QBands analog, max ±1.0) ──
        bb_upper = features.get("bb_upper_20", 0)
        bb_lower = features.get("bb_lower_20", 0)
        if bb_upper and bb_lower and close:
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                bb_pos = (close - bb_lower) / bb_range
                if bb_pos < 0.2:
                    bands_score = 1.0   # Near lower band = bullish
                elif bb_pos > 0.8:
                    bands_score = -1.0  # Near upper band = bearish
                else:
                    bands_score = 0.0
            else:
                bands_score = 0.0
        else:
            bands_score = 0.0
        score += bands_score
        components["bands"] = {"score": bands_score}

        # ── Volume Score (QCVD analog, max ±1.0) ──
        vol = features.get("volume", 0)
        vol_sma = features.get("volume_sma_20", 1)
        price_change = features.get("returns_1", 0)
        if vol and vol_sma and vol_sma > 0:
            vol_ratio = vol / vol_sma
            if vol_ratio > 1.5 and price_change > 0:
                vol_score = 1.0
            elif vol_ratio > 1.5 and price_change < 0:
                vol_score = -1.0
            elif price_change > 0:
                vol_score = 0.3
            elif price_change < 0:
                vol_score = -0.3
            else:
                vol_score = 0.0
        else:
            vol_score = 0.0
        score += vol_score
        components["volume"] = {"score": vol_score}

        # ── VWAP Score (QGrid analog, max ±0.5) ──
        vwap = features.get("vwap", 0)
        if vwap and close:
            vwap_score = 0.5 if close > vwap else -0.5
        else:
            vwap_score = 0.0
        score += vwap_score
        components["vwap"] = {"score": vwap_score}

        # ── Total ──
        # Simulated max: ±6.0 (vs real MC's ±10.0), normalize proportionally
        normalized = score / 6.0 * 10.0  # Scale to MC's range
        normalized = max(-10.0, min(10.0, normalized))

        direction = "CALL" if normalized >= 6.0 else "PUT" if normalized <= -6.0 else "HOLD"
        grade = "A+" if abs(normalized) >= 8.0 else "A" if abs(normalized) >= 6.0 else \
                "B+" if abs(normalized) >= 4.0 else "B"

        return {
            "direction": direction,
            "score": round(normalized, 2),
            "normalized_score": round(normalized / 10.0, 3),
            "grade": grade,
            "confidence": round(min(abs(normalized) / 10.0 * 100, 100), 1),
            "components": components,
            "is_simulated": True,
            "age_sec": 0,
            "is_stale": False,
            "signals": 1,
            "entry": close,
            "stop_loss": 0.0,
            "tp1": 0.0, "tp2": 0.0, "tp3": 0.0,
            "timeframe": "",
        }

    def _empty_result(self):
        return {
            "direction": "HOLD", "score": 0.0, "normalized_score": 0.0,
            "grade": "", "confidence": 0.0, "components": None,
            "is_simulated": True, "age_sec": 0, "is_stale": True,
            "signals": 0, "entry": 0.0, "stop_loss": 0.0,
            "tp1": 0.0, "tp2": 0.0, "tp3": 0.0, "timeframe": "",
        }
