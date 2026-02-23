"""
YELENA v2 Feature Engine (v2.0)
================================
Computes 170+ technical features from raw OHLCV bars.

Categories:
1. Moving Averages (22 features)
2. Momentum (18 features)
3. Volatility (12 features)
4. Volume (14 features)
5. Price Action (20 features)
6. Time-Based (8 features)
--- v2 additions below ---
7. Smart Money Concepts (22 features)
8. VWAP & Level Features (8 features)
9. Divergence Detection (4 features)
10. Additional Derived (12 features)
11. Multi-Timeframe Alignment (up to 30 features)

Usage:
    engine = FeatureEngine()
    features = engine.compute_all(df)  # df = OHLCV DataFrame

    # For multi-timeframe features (optional):
    features = engine.compute_all(df, htf_data={
        '5min': df_5min,   # Higher timeframe DataFrames
        '15min': df_15min,
        '1hr': df_1hr,
    })

Changelog:
    v1.0 (Feb 12, 2026): 94 features across 6 categories
    v2.0 (Feb 18, 2026): 170+ features. Added SMC, VWAP, Divergence, Derived, MTF.
                          Unified feature naming for training pipeline consistency.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib

logger = logging.getLogger(__name__)

FEATURE_VERSION = 2


class FeatureEngine:
    """Computes technical features from OHLCV data."""

    def __init__(self):
        self.version = FEATURE_VERSION

    # ================================================================
    # CORE: Compute all features for a DataFrame of OHLCV bars
    # ================================================================

    def compute_all(
        self,
        df: pd.DataFrame,
        htf_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Compute all features for a DataFrame of OHLCV bars.

        Args:
            df: DataFrame with columns [time, symbol, open, high, low, close, volume]
                Must be sorted by time ascending.
            htf_data: Optional dict of higher-timeframe DataFrames for MTF features.
                      Keys are timeframe strings ('5min', '15min', '1hr').
                      Each DataFrame must have [time, open, high, low, close, volume].

        Returns:
            DataFrame with a 'features' column containing dicts of all computed features.
        """
        if len(df) < 200:
            logger.warning(
                f"DataFrame has {len(df)} rows, need at least 200 for all features. "
                f"Some will be NaN."
            )

        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        v = df["volume"].values.astype(float)

        all_features = {}

        # Category 1: Moving Averages (22 features)
        all_features.update(self._moving_averages(o, h, l, c, v))

        # Category 2: Momentum (18 features)
        all_features.update(self._momentum(o, h, l, c, v))

        # Category 3: Volatility (12 features)
        all_features.update(self._volatility(o, h, l, c, v))

        # Category 4: Volume (14 features)
        all_features.update(self._volume(o, h, l, c, v))

        # Category 5: Price Action (20 features)
        all_features.update(self._price_action(o, h, l, c, v))

        # Category 6: Time-Based (8 features)
        all_features.update(self._time_based(df))

        # ── v2 additions ──

        # Category 7: Smart Money Concepts (22 features)
        all_features.update(self._smart_money_concepts(o, h, l, c, v))

        # Category 8: VWAP & Level Features (8 features)
        all_features.update(self._vwap_and_levels(o, h, l, c, v, df))

        # Category 9: Divergence Detection (4 features)
        all_features.update(self._divergence(o, h, l, c, v))

        # Category 10: Additional Derived (12 features)
        all_features.update(self._additional_derived(o, h, l, c, v, all_features))

        # Category 11: Multi-Timeframe Alignment (up to 30 features)
        if htf_data:
            all_features.update(self._multi_timeframe(df, htf_data))
        else:
            # Generate empty MTF features so column count is consistent
            all_features.update(self._multi_timeframe_empty(len(df)))

        # Convert dict of arrays into per-row feature dicts
        feature_names = list(all_features.keys())
        n_rows = len(df)
        rows = []
        for i in range(n_rows):
            row_features = {}
            for name in feature_names:
                val = all_features[name][i]
                if pd.isna(val) or np.isinf(val):
                    row_features[name] = None
                else:
                    row_features[name] = round(float(val), 6)
            rows.append(row_features)

        df = df.copy()
        df["features"] = rows
        return df

    # ================================================================
    # CATEGORY 1: MOVING AVERAGES (22 features)
    # ================================================================

    def _moving_averages(self, o, h, l, c, v):
        """
        Moving averages and derived signals.
        Features:
            sma_5, sma_8, sma_13, sma_21, sma_50, sma_100, sma_200
            ema_5, ema_8, ema_13, ema_21, ema_50, ema_100, ema_200
            vwma_20
            price_vs_sma_50, price_vs_sma_200
            sma_50_vs_200 (golden/death cross position)
            ema_8_vs_21 (fast trend)
            ema_slope_8, ema_slope_21
            ma_ribbon_bullish (count of EMAs price is above)
        """
        features = {}

        # Simple Moving Averages
        for period in [5, 8, 13, 21, 50, 100, 200]:
            features[f"sma_{period}"] = talib.SMA(c, timeperiod=period)

        # Exponential Moving Averages
        for period in [5, 8, 13, 21, 50, 100, 200]:
            features[f"ema_{period}"] = talib.EMA(c, timeperiod=period)

        # VWMA (Volume Weighted Moving Average) — manual since TA-Lib doesn't have it
        vwma_period = 20
        vwma = np.full_like(c, np.nan)
        for i in range(vwma_period - 1, len(c)):
            window_c = c[i - vwma_period + 1:i + 1]
            window_v = v[i - vwma_period + 1:i + 1]
            total_vol = np.sum(window_v)
            if total_vol > 0:
                vwma[i] = np.sum(window_c * window_v) / total_vol
        features["vwma_20"] = vwma

        # Price relative to key MAs (percentage)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["price_vs_sma_50"] = np.where(
                features["sma_50"] > 0,
                (c - features["sma_50"]) / features["sma_50"] * 100,
                np.nan
            )
            features["price_vs_sma_200"] = np.where(
                features["sma_200"] > 0,
                (c - features["sma_200"]) / features["sma_200"] * 100,
                np.nan
            )

            # SMA 50 vs 200 (golden cross = positive, death cross = negative)
            features["sma_50_vs_200"] = np.where(
                features["sma_200"] > 0,
                (features["sma_50"] - features["sma_200"]) / features["sma_200"] * 100,
                np.nan
            )

            # EMA 8 vs 21 (fast trend signal)
            features["ema_8_vs_21"] = np.where(
                features["ema_21"] > 0,
                (features["ema_8"] - features["ema_21"]) / features["ema_21"] * 100,
                np.nan
            )

        # EMA slopes (rate of change over 3 bars)
        ema_8 = features["ema_8"]
        ema_21 = features["ema_21"]
        features["ema_slope_8"] = np.concatenate([[np.nan] * 3, (ema_8[3:] - ema_8[:-3]) / 3])
        features["ema_slope_21"] = np.concatenate([[np.nan] * 3, (ema_21[3:] - ema_21[:-3]) / 3])

        # MA Ribbon: count of EMAs that price is above (0-7 scale)
        ema_keys = ["ema_5", "ema_8", "ema_13", "ema_21", "ema_50", "ema_100", "ema_200"]
        ribbon = np.zeros(len(c))
        for key in ema_keys:
            ribbon += np.where(~np.isnan(features[key]), np.where(c > features[key], 1, 0), 0)
        features["ma_ribbon_bullish"] = ribbon

        return features

    # ================================================================
    # CATEGORY 2: MOMENTUM (18 features)
    # ================================================================

    def _momentum(self, o, h, l, c, v):
        """
        Momentum indicators.
        Features:
            rsi_14, rsi_7
            macd, macd_signal, macd_hist
            stoch_k, stoch_d
            stoch_rsi_k, stoch_rsi_d
            cci_14, cci_50
            williams_r
            roc_10, roc_20
            mfi_14
            adx_14, plus_di, minus_di
        """
        features = {}

        # RSI
        features["rsi_14"] = talib.RSI(c, timeperiod=14)
        features["rsi_7"] = talib.RSI(c, timeperiod=7)

        # MACD (12, 26, 9)
        macd, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        features["macd"] = macd
        features["macd_signal"] = macd_signal
        features["macd_hist"] = macd_hist

        # Stochastic (14, 3, 3)
        stoch_k, stoch_d = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d

        # Stochastic RSI
        stoch_rsi_k, stoch_rsi_d = talib.STOCHRSI(c, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        features["stoch_rsi_k"] = stoch_rsi_k
        features["stoch_rsi_d"] = stoch_rsi_d

        # CCI
        features["cci_14"] = talib.CCI(h, l, c, timeperiod=14)
        features["cci_50"] = talib.CCI(h, l, c, timeperiod=50)

        # Williams %R
        features["williams_r"] = talib.WILLR(h, l, c, timeperiod=14)

        # Rate of Change
        features["roc_10"] = talib.ROC(c, timeperiod=10)
        features["roc_20"] = talib.ROC(c, timeperiod=20)

        # Money Flow Index
        features["mfi_14"] = talib.MFI(h, l, c, v, timeperiod=14)

        # ADX + Directional Indicators
        features["adx_14"] = talib.ADX(h, l, c, timeperiod=14)
        features["plus_di"] = talib.PLUS_DI(h, l, c, timeperiod=14)
        features["minus_di"] = talib.MINUS_DI(h, l, c, timeperiod=14)

        return features

    # ================================================================
    # CATEGORY 3: VOLATILITY (12 features)
    # ================================================================

    def _volatility(self, o, h, l, c, v):
        """
        Volatility indicators.
        Features:
            atr_14, atr_7
            atr_percent (ATR as % of price)
            bb_upper, bb_middle, bb_lower, bb_width, bb_percent
            keltner_upper, keltner_lower
            squeeze (BB inside Keltner = True)
            true_range
        """
        features = {}

        # ATR
        features["atr_14"] = talib.ATR(h, l, c, timeperiod=14)
        features["atr_7"] = talib.ATR(h, l, c, timeperiod=7)

        # ATR as percentage of price
        with np.errstate(divide='ignore', invalid='ignore'):
            features["atr_percent"] = np.where(c > 0, features["atr_14"] / c * 100, np.nan)

        # Bollinger Bands (20, 2)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        features["bb_upper"] = bb_upper
        features["bb_middle"] = bb_middle
        features["bb_lower"] = bb_lower

        # BB Width (volatility measure)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["bb_width"] = np.where(
                bb_middle > 0,
                (bb_upper - bb_lower) / bb_middle * 100,
                np.nan
            )
            # BB %B (where price is within bands: 0 = lower, 1 = upper)
            features["bb_percent"] = np.where(
                (bb_upper - bb_lower) > 0,
                (c - bb_lower) / (bb_upper - bb_lower),
                np.nan
            )

        # Keltner Channels (20, 1.5 ATR)
        keltner_mid = talib.EMA(c, timeperiod=20)
        keltner_atr = talib.ATR(h, l, c, timeperiod=20)
        features["keltner_upper"] = keltner_mid + 1.5 * keltner_atr
        features["keltner_lower"] = keltner_mid - 1.5 * keltner_atr

        # Squeeze: BB inside Keltner (1 = squeeze on, 0 = squeeze off)
        features["squeeze"] = np.where(
            ~np.isnan(bb_lower) & ~np.isnan(features["keltner_lower"]),
            np.where(
                (bb_lower > features["keltner_lower"]) & (bb_upper < features["keltner_upper"]),
                1.0, 0.0
            ),
            np.nan
        )

        # True Range
        features["true_range"] = talib.TRANGE(h, l, c)

        return features

    # ================================================================
    # CATEGORY 4: VOLUME (14 features)
    # ================================================================

    def _volume(self, o, h, l, c, v):
        """
        Volume indicators.
        Features:
            volume_sma_20, volume_ratio (current / sma)
            obv, obv_slope
            ad_line (Accumulation/Distribution)
            cmf_20 (Chaikin Money Flow)
            vpt (Volume Price Trend)
            volume_delta (estimated buy - sell volume)
            volume_delta_cum (cumulative delta)
            high_volume_bar (volume > 2x average)
            volume_trend (rising or falling over 10 bars)
            relative_volume (vs same time yesterday — placeholder, needs time context)
            force_index_13
            eom_14 (Ease of Movement)
        """
        features = {}

        # Volume SMA
        features["volume_sma_20"] = talib.SMA(v, timeperiod=20)

        # Volume Ratio (current bar volume / 20-bar average)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["volume_ratio"] = np.where(
                features["volume_sma_20"] > 0,
                v / features["volume_sma_20"],
                np.nan
            )

        # On-Balance Volume
        features["obv"] = talib.OBV(c, v)

        # OBV slope (change over 5 bars)
        obv = features["obv"]
        features["obv_slope"] = np.concatenate([[np.nan] * 5, obv[5:] - obv[:-5]])

        # Accumulation/Distribution Line
        features["ad_line"] = talib.AD(h, l, c, v)

        # Chaikin Money Flow (20-period)
        bar_range = h - l
        mf_multiplier = np.where(
            bar_range > 0,
            ((c - l) - (h - c)) / bar_range,
            0.0
        )
        mf_volume = mf_multiplier * v
        cmf = np.full_like(c, np.nan)
        for i in range(19, len(c)):
            vol_sum = np.sum(v[i - 19:i + 1])
            if vol_sum > 0:
                cmf[i] = np.sum(mf_volume[i - 19:i + 1]) / vol_sum
        features["cmf_20"] = cmf

        # Volume Price Trend
        vpt = np.zeros_like(c)
        for i in range(1, len(c)):
            if c[i - 1] > 0:
                vpt[i] = vpt[i - 1] + v[i] * ((c[i] - c[i - 1]) / c[i - 1])
        features["vpt"] = vpt

        # Estimated Volume Delta (buy vs sell estimation using bar position)
        close_position = np.where(bar_range > 0, (c - l) / bar_range, 0.5)
        buy_vol = v * close_position
        sell_vol = v * (1 - close_position)
        features["volume_delta"] = buy_vol - sell_vol

        # Cumulative Volume Delta
        features["volume_delta_cum"] = np.cumsum(features["volume_delta"])

        # High Volume Bar (volume > 2x 20-bar average)
        features["high_volume_bar"] = np.where(
            ~np.isnan(features["volume_sma_20"]),
            np.where(v > 2 * features["volume_sma_20"], 1.0, 0.0),
            np.nan
        )

        # Volume Trend (slope of volume over 10 bars)
        vol_sma_short = talib.SMA(v, timeperiod=5)
        vol_sma_long = talib.SMA(v, timeperiod=20)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["volume_trend"] = np.where(
                vol_sma_long > 0,
                vol_sma_short / vol_sma_long,
                np.nan
            )

        # Force Index (13-period EMA of price change * volume)
        force = np.zeros_like(c)
        for i in range(1, len(c)):
            force[i] = (c[i] - c[i - 1]) * v[i]
        features["force_index_13"] = talib.EMA(force, timeperiod=13)

        # Ease of Movement (14-period)
        distance = ((h + l) / 2) - np.concatenate([[np.nan], (h[:-1] + l[:-1]) / 2])
        with np.errstate(divide='ignore', invalid='ignore'):
            box_ratio = np.where(bar_range > 0, (v / 1e6) / bar_range, 0)
            eom = np.where(box_ratio > 0, distance / box_ratio, 0)
        features["eom_14"] = talib.SMA(eom, timeperiod=14)

        return features

    # ================================================================
    # CATEGORY 5: PRICE ACTION (20 features)
    # ================================================================

    def _price_action(self, o, h, l, c, v):
        """
        Price action features.
        Features:
            candle_body, candle_body_pct, candle_upper_wick, candle_lower_wick
            candle_direction (1=bull, -1=bear)
            consecutive_bull, consecutive_bear
            higher_high, lower_low, higher_low, lower_high
            bar_range, bar_range_vs_atr
            gap_up, gap_down
            close_vs_range (where close sits in bar: 0-1)
            engulfing (1=bull engulf, -1=bear engulf, 0=none)
            doji (1=doji, 0=not)
            hammer (1=hammer/hanging man pattern)
            inside_bar (1=inside bar, 0=not)
        """
        features = {}

        # Basic candle metrics
        features["candle_body"] = c - o
        bar_range = h - l
        features["bar_range"] = bar_range

        with np.errstate(divide='ignore', invalid='ignore'):
            features["candle_body_pct"] = np.where(bar_range > 0, np.abs(c - o) / bar_range, 0)

        features["candle_upper_wick"] = h - np.maximum(o, c)
        features["candle_lower_wick"] = np.minimum(o, c) - l

        # Direction
        features["candle_direction"] = np.where(c > o, 1.0, np.where(c < o, -1.0, 0.0))

        # Consecutive bullish/bearish bars
        direction = features["candle_direction"]
        consec_bull = np.zeros_like(c)
        consec_bear = np.zeros_like(c)
        for i in range(1, len(c)):
            if direction[i] > 0:
                consec_bull[i] = consec_bull[i - 1] + 1
                consec_bear[i] = 0
            elif direction[i] < 0:
                consec_bear[i] = consec_bear[i - 1] + 1
                consec_bull[i] = 0
        features["consecutive_bull"] = consec_bull
        features["consecutive_bear"] = consec_bear

        # Higher High, Lower Low, Higher Low, Lower High
        features["higher_high"] = np.concatenate([[np.nan], np.where(h[1:] > h[:-1], 1.0, 0.0)])
        features["lower_low"] = np.concatenate([[np.nan], np.where(l[1:] < l[:-1], 1.0, 0.0)])
        features["higher_low"] = np.concatenate([[np.nan], np.where(l[1:] > l[:-1], 1.0, 0.0)])
        features["lower_high"] = np.concatenate([[np.nan], np.where(h[1:] < h[:-1], 1.0, 0.0)])

        # Bar range vs ATR
        atr_14 = talib.ATR(h, l, c, timeperiod=14)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["bar_range_vs_atr"] = np.where(atr_14 > 0, bar_range / atr_14, np.nan)

        # Gaps
        features["gap_up"] = np.concatenate([[np.nan], np.where(l[1:] > h[:-1], 1.0, 0.0)])
        features["gap_down"] = np.concatenate([[np.nan], np.where(h[1:] < l[:-1], 1.0, 0.0)])

        # Close vs Range (0 = closed at low, 1 = closed at high)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["close_vs_range"] = np.where(bar_range > 0, (c - l) / bar_range, 0.5)

        # Engulfing pattern
        engulfing = np.zeros(len(c))
        for i in range(1, len(c)):
            # Bullish engulfing: prev bearish, current bullish, current body engulfs prev body
            if o[i - 1] > c[i - 1] and c[i] > o[i]:
                if c[i] > o[i - 1] and o[i] < c[i - 1]:
                    engulfing[i] = 1.0
            # Bearish engulfing: prev bullish, current bearish, current body engulfs prev body
            elif c[i - 1] > o[i - 1] and o[i] > c[i]:
                if o[i] > c[i - 1] and c[i] < o[i - 1]:
                    engulfing[i] = -1.0
        features["engulfing"] = engulfing

        # Doji (body < 10% of range)
        with np.errstate(divide='ignore', invalid='ignore'):
            body_ratio = np.where(bar_range > 0, np.abs(c - o) / bar_range, 0)
        features["doji"] = np.where(body_ratio < 0.1, 1.0, 0.0)

        # Hammer/Hanging Man (small body at top, long lower wick)
        hammer = np.zeros(len(c))
        for i in range(len(c)):
            if bar_range[i] > 0:
                lower_wick_ratio = (min(o[i], c[i]) - l[i]) / bar_range[i]
                upper_wick_ratio = (h[i] - max(o[i], c[i])) / bar_range[i]
                body_pct = abs(c[i] - o[i]) / bar_range[i]
                if lower_wick_ratio > 0.6 and upper_wick_ratio < 0.1 and body_pct < 0.3:
                    hammer[i] = 1.0
        features["hammer"] = hammer

        # Inside Bar (current high < prev high AND current low > prev low)
        features["inside_bar"] = np.concatenate([[np.nan], np.where(
            (h[1:] < h[:-1]) & (l[1:] > l[:-1]), 1.0, 0.0
        )])

        return features

    # ================================================================
    # CATEGORY 6: TIME-BASED (8 features)
    # ================================================================

    def _time_based(self, df: pd.DataFrame):
        """
        Time-based features.
        Features:
            minute_of_day, hour_of_day
            is_market_open (first 30 min), is_market_close (last 30 min)
            is_power_hour (3-4pm ET)
            day_of_week (0=Mon, 4=Fri)
            is_monday, is_friday
            minutes_since_open
        """
        features = {}

        times = pd.to_datetime(df["time"])
        # Convert to US/Eastern for market time features
        if times.dt.tz is None:
            times = times.dt.tz_localize("UTC")
        times_et = times.dt.tz_convert("US/Eastern")

        hours = times_et.dt.hour.values.astype(float)
        minutes = times_et.dt.minute.values.astype(float)

        features["hour_of_day"] = hours
        features["minute_of_day"] = hours * 60 + minutes

        # Minutes since market open (9:30 AM ET = 570 minutes)
        features["minutes_since_open"] = features["minute_of_day"] - 570.0

        # Market open period (9:30-10:00 AM ET)
        features["is_market_open"] = np.where(
            (hours == 9) & (minutes >= 30) | (hours == 10) & (minutes == 0),
            1.0, 0.0
        )

        # Market close period (3:30-4:00 PM ET)
        features["is_market_close"] = np.where(
            (hours == 15) & (minutes >= 30),
            1.0, 0.0
        )

        # Power hour (3:00-4:00 PM ET)
        features["is_power_hour"] = np.where(hours == 15, 1.0, 0.0)

        # Day of week
        features["day_of_week"] = times_et.dt.dayofweek.values.astype(float)
        features["is_monday"] = np.where(features["day_of_week"] == 0, 1.0, 0.0)
        features["is_friday"] = np.where(features["day_of_week"] == 4, 1.0, 0.0)

        return features

    # ================================================================
    # CATEGORY 7: SMART MONEY CONCEPTS (22 features)  [v2 NEW]
    # ================================================================

    def _smart_money_concepts(self, o, h, l, c, v):
        """
        Smart Money Concepts — structural market features.

        Features:
            swing_high, swing_low                   — 5-bar swing point detection
            bos_bullish, bos_bearish                — Break of Structure
            bars_since_bos                          — Recency of last BOS
            choch_bullish, choch_bearish            — Change of Character
            bars_since_choch                        — Recency of last CHoCH
            fvg_bullish, fvg_bearish                — Fair Value Gap (3-candle imbalance)
            fvg_nearest_distance_pct                — Distance to nearest unfilled FVG
            ob_bullish, ob_bearish                  — Order Block zones
            ob_nearest_distance_pct                 — Distance to nearest untested OB
            displacement_up, displacement_down      — Institutional momentum candles
            premium_discount                        — Position in swing range (-1 to +1)
            liquidity_sweep_high, liquidity_sweep_low — Stop hunts
            market_structure                        — Current structure state
            structure_shift                         — Structure just changed
        """
        n = len(c)
        features = {}
        swing_lookback = 5  # Bars on each side to confirm swing

        # ── Swing Point Detection ──
        swing_high = np.zeros(n)
        swing_low = np.zeros(n)
        # Track most recent swing highs/lows for BOS/CHoCH
        recent_swing_high_price = np.full(n, np.nan)
        recent_swing_low_price = np.full(n, np.nan)

        last_sh_price = np.nan
        last_sl_price = np.nan

        for i in range(swing_lookback, n - swing_lookback):
            # Swing high: h[i] is highest in window
            if h[i] == np.max(h[i - swing_lookback:i + swing_lookback + 1]):
                swing_high[i] = 1.0
                last_sh_price = h[i]
            # Swing low: l[i] is lowest in window
            if l[i] == np.min(l[i - swing_lookback:i + swing_lookback + 1]):
                swing_low[i] = 1.0
                last_sl_price = l[i]

            recent_swing_high_price[i] = last_sh_price
            recent_swing_low_price[i] = last_sl_price

        # Fill forward the recent swing prices for remaining bars
        for i in range(n - swing_lookback, n):
            recent_swing_high_price[i] = last_sh_price
            recent_swing_low_price[i] = last_sl_price

        features["swing_high"] = swing_high
        features["swing_low"] = swing_low

        # ── Break of Structure (BOS) & Change of Character (CHoCH) ──
        bos_bullish = np.zeros(n)
        bos_bearish = np.zeros(n)
        choch_bullish = np.zeros(n)
        choch_bearish = np.zeros(n)
        bars_since_bos = np.full(n, np.nan)
        bars_since_choch = np.full(n, np.nan)
        market_structure = np.zeros(n)  # 1=bullish, -1=bearish, 0=neutral

        # Track last two swing highs and lows for structure analysis
        prev_sh = np.nan
        prev_sl = np.nan
        last_bos_bar = -999
        last_choch_bar = -999
        current_structure = 0  # 0=neutral, 1=bullish, -1=bearish

        for i in range(1, n):
            # Update structure tracking when new swing points form
            # (Use confirmed swing points from `swing_lookback` bars ago)
            check_idx = i - swing_lookback
            if check_idx >= swing_lookback:
                if swing_high[check_idx] == 1.0:
                    prev_sh = h[check_idx]
                if swing_low[check_idx] == 1.0:
                    prev_sl = l[check_idx]

            # BOS Bullish: close breaks above previous swing high
            if not np.isnan(prev_sh) and c[i] > prev_sh and c[i - 1] <= prev_sh:
                if current_structure >= 0:
                    bos_bullish[i] = 1.0  # Continuation BOS
                else:
                    choch_bullish[i] = 1.0  # Reversal = CHoCH
                    last_choch_bar = i
                current_structure = 1
                last_bos_bar = i

            # BOS Bearish: close breaks below previous swing low
            if not np.isnan(prev_sl) and c[i] < prev_sl and c[i - 1] >= prev_sl:
                if current_structure <= 0:
                    bos_bearish[i] = 1.0  # Continuation BOS
                else:
                    choch_bearish[i] = 1.0  # Reversal = CHoCH
                    last_choch_bar = i
                current_structure = -1
                last_bos_bar = i

            bars_since_bos[i] = i - last_bos_bar if last_bos_bar >= 0 else np.nan
            bars_since_choch[i] = i - last_choch_bar if last_choch_bar >= 0 else np.nan
            market_structure[i] = current_structure

        features["bos_bullish"] = bos_bullish
        features["bos_bearish"] = bos_bearish
        features["bars_since_bos"] = bars_since_bos
        features["choch_bullish"] = choch_bullish
        features["choch_bearish"] = choch_bearish
        features["bars_since_choch"] = bars_since_choch
        features["market_structure"] = market_structure

        # Structure shift: did structure just change this bar?
        structure_shift = np.zeros(n)
        for i in range(1, n):
            if market_structure[i] != market_structure[i - 1] and market_structure[i] != 0:
                structure_shift[i] = 1.0
        features["structure_shift"] = structure_shift

        # ── Fair Value Gaps (FVG) ──
        fvg_bullish = np.zeros(n)
        fvg_bearish = np.zeros(n)
        # Track unfilled FVGs for distance calculation
        active_bull_fvgs = []  # List of (top, bottom) tuples
        active_bear_fvgs = []
        fvg_nearest_dist = np.full(n, np.nan)

        for i in range(2, n):
            # Bullish FVG: bar[i-2] high < bar[i] low (gap up, bar[i-1] didn't fill)
            if l[i] > h[i - 2]:
                fvg_bullish[i] = 1.0
                active_bull_fvgs.append((l[i], h[i - 2]))  # top, bottom

            # Bearish FVG: bar[i-2] low > bar[i] high (gap down)
            if h[i] < l[i - 2]:
                fvg_bearish[i] = 1.0
                active_bear_fvgs.append((l[i - 2], h[i]))  # top, bottom

            # Remove filled FVGs
            active_bull_fvgs = [(t, b) for t, b in active_bull_fvgs if l[i] > b]
            active_bear_fvgs = [(t, b) for t, b in active_bear_fvgs if h[i] < t]

            # Distance to nearest unfilled FVG
            min_dist = np.inf
            for top, bottom in active_bull_fvgs:
                mid = (top + bottom) / 2
                dist = abs(c[i] - mid) / c[i] * 100 if c[i] > 0 else np.inf
                min_dist = min(min_dist, dist)
            for top, bottom in active_bear_fvgs:
                mid = (top + bottom) / 2
                dist = abs(c[i] - mid) / c[i] * 100 if c[i] > 0 else np.inf
                min_dist = min(min_dist, dist)
            fvg_nearest_dist[i] = min_dist if min_dist < np.inf else np.nan

        features["fvg_bullish"] = fvg_bullish
        features["fvg_bearish"] = fvg_bearish
        features["fvg_nearest_distance_pct"] = fvg_nearest_dist

        # ── Order Blocks (OB) ──
        # OB = last opposing candle before an impulsive move (>1.5 ATR body)
        atr_14 = talib.ATR(h, l, c, timeperiod=14)
        ob_bullish = np.zeros(n)
        ob_bearish = np.zeros(n)
        active_bull_obs = []  # List of (high, low) zones
        active_bear_obs = []
        ob_nearest_dist = np.full(n, np.nan)

        for i in range(2, n):
            if np.isnan(atr_14[i]):
                continue

            body = abs(c[i] - o[i])
            # Bullish OB: current bar is strong bullish, previous bar was bearish
            if c[i] > o[i] and body > 1.5 * atr_14[i] and c[i - 1] < o[i - 1]:
                ob_bullish[i] = 1.0
                active_bull_obs.append((o[i - 1], c[i - 1]))  # OB zone = prev bar body

            # Bearish OB: current bar is strong bearish, previous bar was bullish
            if c[i] < o[i] and body > 1.5 * atr_14[i] and c[i - 1] > o[i - 1]:
                ob_bearish[i] = 1.0
                active_bear_obs.append((c[i - 1], o[i - 1]))  # OB zone = prev bar body

            # Remove tested OBs (price passed through)
            active_bull_obs = [(hi, lo) for hi, lo in active_bull_obs if l[i] > lo]
            active_bear_obs = [(hi, lo) for hi, lo in active_bear_obs if h[i] < hi]

            # Keep only last 10 OBs to prevent memory growth
            active_bull_obs = active_bull_obs[-10:]
            active_bear_obs = active_bear_obs[-10:]

            # Distance to nearest untested OB
            min_dist = np.inf
            for hi, lo in active_bull_obs:
                mid = (hi + lo) / 2
                dist = abs(c[i] - mid) / c[i] * 100 if c[i] > 0 else np.inf
                min_dist = min(min_dist, dist)
            for hi, lo in active_bear_obs:
                mid = (hi + lo) / 2
                dist = abs(c[i] - mid) / c[i] * 100 if c[i] > 0 else np.inf
                min_dist = min(min_dist, dist)
            ob_nearest_dist[i] = min_dist if min_dist < np.inf else np.nan

        features["ob_bullish"] = ob_bullish
        features["ob_bearish"] = ob_bearish
        features["ob_nearest_distance_pct"] = ob_nearest_dist

        # ── Displacement Candles ──
        displacement_up = np.zeros(n)
        displacement_down = np.zeros(n)
        for i in range(14, n):
            if np.isnan(atr_14[i]):
                continue
            body = c[i] - o[i]
            if body > 1.5 * atr_14[i]:
                displacement_up[i] = 1.0
            elif body < -1.5 * atr_14[i]:
                displacement_down[i] = 1.0
        features["displacement_up"] = displacement_up
        features["displacement_down"] = displacement_down

        # ── Premium / Discount Zone ──
        # Position within recent swing range: -1 (deep discount) to +1 (deep premium)
        premium_discount = np.full(n, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            swing_range = recent_swing_high_price - recent_swing_low_price
            premium_discount = np.where(
                swing_range > 0,
                2 * (c - recent_swing_low_price) / swing_range - 1,
                np.nan
            )
        features["premium_discount"] = premium_discount

        # ── Liquidity Sweeps ──
        liq_sweep_high = np.zeros(n)
        liq_sweep_low = np.zeros(n)
        for i in range(1, n):
            # Sweep high: wick above recent swing high but close below it
            if not np.isnan(recent_swing_high_price[i - 1]):
                if h[i] > recent_swing_high_price[i - 1] and c[i] < recent_swing_high_price[i - 1]:
                    liq_sweep_high[i] = 1.0
            # Sweep low: wick below recent swing low but close above it
            if not np.isnan(recent_swing_low_price[i - 1]):
                if l[i] < recent_swing_low_price[i - 1] and c[i] > recent_swing_low_price[i - 1]:
                    liq_sweep_low[i] = 1.0
        features["liquidity_sweep_high"] = liq_sweep_high
        features["liquidity_sweep_low"] = liq_sweep_low

        return features

    # ================================================================
    # CATEGORY 8: VWAP & LEVEL FEATURES (8 features)  [v2 NEW]
    # ================================================================

    def _vwap_and_levels(self, o, h, l, c, v, df):
        """
        VWAP (session-resetting) and key level features.

        Features:
            vwap                    — Session VWAP (resets at 9:30 AM ET)
            vwap_distance_pct       — Price distance from VWAP as %
            vwap_upper_1std         — VWAP + 1 standard deviation
            vwap_lower_1std         — VWAP - 1 standard deviation
            vwap_position           — Position within VWAP bands (0-1)
            pct_from_20_high        — Distance from 20-bar rolling high
            pct_from_20_low         — Distance from 20-bar rolling low
            range_20_position       — Position within 20-bar range (0-1)
        """
        n = len(c)
        features = {}

        # ── Session VWAP (resets daily at 9:30 AM ET) ──
        vwap = np.full(n, np.nan)
        vwap_upper = np.full(n, np.nan)
        vwap_lower = np.full(n, np.nan)

        times = pd.to_datetime(df["time"])
        if times.dt.tz is None:
            times = times.dt.tz_localize("UTC")
        times_et = times.dt.tz_convert("US/Eastern")

        # Detect session boundaries (new day = new session)
        dates = times_et.dt.date.values
        typical_price = (h + l + c) / 3
        cum_tp_vol = 0.0
        cum_vol = 0.0
        cum_tp_vol_sq = 0.0
        prev_date = None

        for i in range(n):
            current_date = dates[i]
            if current_date != prev_date:
                # New session — reset
                cum_tp_vol = 0.0
                cum_vol = 0.0
                cum_tp_vol_sq = 0.0
                prev_date = current_date

            cum_tp_vol += typical_price[i] * v[i]
            cum_vol += v[i]
            cum_tp_vol_sq += (typical_price[i] ** 2) * v[i]

            if cum_vol > 0:
                vwap_val = cum_tp_vol / cum_vol
                vwap[i] = vwap_val
                # Standard deviation of VWAP
                variance = (cum_tp_vol_sq / cum_vol) - (vwap_val ** 2)
                std = np.sqrt(max(variance, 0))
                vwap_upper[i] = vwap_val + std
                vwap_lower[i] = vwap_val - std

        features["vwap"] = vwap

        # VWAP distance (percentage)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["vwap_distance_pct"] = np.where(
                vwap > 0, (c - vwap) / vwap * 100, np.nan
            )

        features["vwap_upper_1std"] = vwap_upper
        features["vwap_lower_1std"] = vwap_lower

        # VWAP position (0 = at lower band, 1 = at upper band)
        with np.errstate(divide='ignore', invalid='ignore'):
            band_width = vwap_upper - vwap_lower
            features["vwap_position"] = np.where(
                band_width > 0, (c - vwap_lower) / band_width, np.nan
            )

        # ── 20-bar rolling high/low levels ──
        high_20 = np.full(n, np.nan)
        low_20 = np.full(n, np.nan)
        for i in range(19, n):
            high_20[i] = np.max(h[i - 19:i + 1])
            low_20[i] = np.min(l[i - 19:i + 1])

        with np.errstate(divide='ignore', invalid='ignore'):
            features["pct_from_20_high"] = np.where(
                high_20 > 0, (c - high_20) / high_20 * 100, np.nan
            )
            features["pct_from_20_low"] = np.where(
                low_20 > 0, (c - low_20) / low_20 * 100, np.nan
            )
            range_20 = high_20 - low_20
            features["range_20_position"] = np.where(
                range_20 > 0, (c - low_20) / range_20, np.nan
            )

        return features

    # ================================================================
    # CATEGORY 9: DIVERGENCE DETECTION (4 features)  [v2 NEW]
    # ================================================================

    def _divergence(self, o, h, l, c, v):
        """
        Divergence detection between price and oscillators.

        Uses a lookback window to find swing points in both price and
        oscillator, then checks for divergence patterns.

        Features:
            rsi_divergence_bull   — Price lower low, RSI higher low
            rsi_divergence_bear   — Price higher high, RSI lower high
            macd_divergence_bull  — Price lower low, MACD hist higher low
            macd_divergence_bear  — Price higher high, MACD hist lower high
        """
        n = len(c)
        features = {}
        lookback = 20  # Window to search for divergence

        rsi = talib.RSI(c, timeperiod=14)
        _, _, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)

        rsi_div_bull = np.zeros(n)
        rsi_div_bear = np.zeros(n)
        macd_div_bull = np.zeros(n)
        macd_div_bear = np.zeros(n)

        for i in range(lookback + 5, n):
            window_start = i - lookback

            # Find most recent local low in price (within lookback window)
            price_lows = []
            for j in range(window_start + 2, i - 1):
                if l[j] <= l[j - 1] and l[j] <= l[j - 2] and l[j] <= l[j + 1] and l[j] <= l[j + 2]:
                    price_lows.append(j)

            # Find most recent local high in price
            price_highs = []
            for j in range(window_start + 2, i - 1):
                if h[j] >= h[j - 1] and h[j] >= h[j - 2] and h[j] >= h[j + 1] and h[j] >= h[j + 2]:
                    price_highs.append(j)

            # ── RSI Bullish Divergence ──
            # Current bar near a low + price made lower low but RSI made higher low
            if len(price_lows) >= 2:
                prev_low_idx = price_lows[-2]
                curr_low_idx = price_lows[-1]
                if (l[curr_low_idx] < l[prev_low_idx] and
                        not np.isnan(rsi[curr_low_idx]) and
                        not np.isnan(rsi[prev_low_idx]) and
                        rsi[curr_low_idx] > rsi[prev_low_idx]):
                    # Mark the current bar if we're near the divergence point
                    if i - curr_low_idx <= 3:
                        rsi_div_bull[i] = 1.0

            # ── RSI Bearish Divergence ──
            if len(price_highs) >= 2:
                prev_high_idx = price_highs[-2]
                curr_high_idx = price_highs[-1]
                if (h[curr_high_idx] > h[prev_high_idx] and
                        not np.isnan(rsi[curr_high_idx]) and
                        not np.isnan(rsi[prev_high_idx]) and
                        rsi[curr_high_idx] < rsi[prev_high_idx]):
                    if i - curr_high_idx <= 3:
                        rsi_div_bear[i] = 1.0

            # ── MACD Bullish Divergence ──
            if len(price_lows) >= 2:
                prev_low_idx = price_lows[-2]
                curr_low_idx = price_lows[-1]
                if (l[curr_low_idx] < l[prev_low_idx] and
                        not np.isnan(macd_hist[curr_low_idx]) and
                        not np.isnan(macd_hist[prev_low_idx]) and
                        macd_hist[curr_low_idx] > macd_hist[prev_low_idx]):
                    if i - curr_low_idx <= 3:
                        macd_div_bull[i] = 1.0

            # ── MACD Bearish Divergence ──
            if len(price_highs) >= 2:
                prev_high_idx = price_highs[-2]
                curr_high_idx = price_highs[-1]
                if (h[curr_high_idx] > h[prev_high_idx] and
                        not np.isnan(macd_hist[curr_high_idx]) and
                        not np.isnan(macd_hist[prev_high_idx]) and
                        macd_hist[curr_high_idx] < macd_hist[prev_high_idx]):
                    if i - curr_high_idx <= 3:
                        macd_div_bear[i] = 1.0

        features["rsi_divergence_bull"] = rsi_div_bull
        features["rsi_divergence_bear"] = rsi_div_bear
        features["macd_divergence_bull"] = macd_div_bull
        features["macd_divergence_bear"] = macd_div_bear

        return features

    # ================================================================
    # CATEGORY 10: ADDITIONAL DERIVED FEATURES (12 features)  [v2 NEW]
    # ================================================================

    def _additional_derived(self, o, h, l, c, v, existing_features):
        """
        Additional derived and interaction features that bridge gaps
        between Feature Engine v1 and the training pipeline's inline features.

        These ensure the unified Feature Engine produces everything the
        training pipeline needs — one source of truth.

        Features:
            return_1, return_5, return_10, return_20   — Multi-period returns (%)
            macd_hist_slope                            — MACD histogram rate of change
            std_20                                     — 20-bar price standard deviation
            ema_cross_9_21                             — EMA 9/21 crossover state
            ema_cross_50_200                           — Golden/death cross state
            pct_from_ema_9, pct_from_ema_21, pct_from_ema_50 — EMA distance %
            rsi_bb_combo                               — RSI × BB position interaction
        """
        n = len(c)
        features = {}

        # Multi-period percentage returns
        for period in [1, 5, 10, 20]:
            ret = np.full(n, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                ret[period:] = (c[period:] - c[:-period]) / c[:-period] * 100
            features[f"return_{period}"] = ret

        # MACD histogram slope (1-bar change)
        macd_hist = existing_features.get("macd_hist", np.full(n, np.nan))
        features["macd_hist_slope"] = np.concatenate([[np.nan], macd_hist[1:] - macd_hist[:-1]])

        # 20-bar standard deviation
        std_20 = np.full(n, np.nan)
        for i in range(19, n):
            std_20[i] = np.std(c[i - 19:i + 1])
        features["std_20"] = std_20

        # EMA crossover states
        ema_8 = existing_features.get("ema_8", talib.EMA(c, timeperiod=8))
        ema_13 = existing_features.get("ema_13", talib.EMA(c, timeperiod=13))
        ema_21 = existing_features.get("ema_21", talib.EMA(c, timeperiod=21))
        ema_50 = existing_features.get("ema_50", talib.EMA(c, timeperiod=50))
        ema_200 = existing_features.get("ema_200", talib.EMA(c, timeperiod=200))

        # We use ema_8 and ema_21 (matching what v1 computes) for the fast cross
        # and add ema_9 for training pipeline compatibility
        ema_9 = talib.EMA(c, timeperiod=9)
        features["ema_cross_9_21"] = np.where(
            ~np.isnan(ema_9) & ~np.isnan(ema_21),
            np.where(ema_9 > ema_21, 1.0, -1.0),
            np.nan
        )
        features["ema_cross_50_200"] = np.where(
            ~np.isnan(ema_50) & ~np.isnan(ema_200),
            np.where(ema_50 > ema_200, 1.0, -1.0),
            np.nan
        )

        # Price distance from key EMAs (percentage)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["pct_from_ema_9"] = np.where(
                ema_9 > 0, (c - ema_9) / ema_9 * 100, np.nan
            )
            features["pct_from_ema_21"] = np.where(
                ema_21 > 0, (c - ema_21) / ema_21 * 100, np.nan
            )
            features["pct_from_ema_50"] = np.where(
                ema_50 > 0, (c - ema_50) / ema_50 * 100, np.nan
            )

        # Interaction feature: RSI × BB position (momentum × volatility context)
        rsi_14 = existing_features.get("rsi_14", talib.RSI(c, timeperiod=14))
        bb_percent = existing_features.get("bb_percent", np.full(n, np.nan))
        with np.errstate(invalid='ignore'):
            features["rsi_bb_combo"] = np.where(
                ~np.isnan(rsi_14) & ~np.isnan(bb_percent),
                rsi_14 * bb_percent / 100,  # Normalize to reasonable range
                np.nan
            )

        return features

    # ================================================================
    # CATEGORY 11: MULTI-TIMEFRAME ALIGNMENT (up to 30 features) [v2 NEW]
    # ================================================================

    def _multi_timeframe(self, df: pd.DataFrame, htf_data: Dict[str, pd.DataFrame]):
        """
        Multi-timeframe alignment features.

        For each higher timeframe, computes key indicators and aligns them
        to the base timeframe using forward-fill (most recent completed bar).

        Per higher TF (up to 3):
            htf_{tf}_trend        — EMA-based trend direction (+1/-1)
            htf_{tf}_rsi          — RSI value
            htf_{tf}_adx          — ADX trend strength
            htf_{tf}_squeeze      — Squeeze status
            htf_{tf}_macd_hist    — MACD histogram value
            htf_{tf}_volume_ratio — Volume ratio
            htf_{tf}_candle_dir   — Candle direction

        Plus alignment scores:
            mtf_trend_alignment     — How many TFs agree on trend (0-3)
            mtf_momentum_alignment  — How many TFs agree on momentum (0-3)
            mtf_confluence_score    — Weighted alignment score (0-100)
        """
        n = len(df)
        features = {}

        base_times = pd.to_datetime(df["time"])
        if base_times.dt.tz is None:
            base_times = base_times.dt.tz_localize("UTC")

        tf_order = ["5min", "15min", "1hr"]
        trend_agreements = np.zeros(n)
        momentum_agreements = np.zeros(n)
        num_htfs = 0

        for tf_label in tf_order:
            if tf_label not in htf_data:
                # Generate NaN placeholders for missing HTFs
                for suffix in ["trend", "rsi", "adx", "squeeze", "macd_hist", "volume_ratio", "candle_dir"]:
                    features[f"htf_{tf_label}_{suffix}"] = np.full(n, np.nan)
                continue

            htf_df = htf_data[tf_label]
            if len(htf_df) < 50:
                for suffix in ["trend", "rsi", "adx", "squeeze", "macd_hist", "volume_ratio", "candle_dir"]:
                    features[f"htf_{tf_label}_{suffix}"] = np.full(n, np.nan)
                continue

            num_htfs += 1

            # Compute HTF indicators
            htf_o = htf_df["open"].values.astype(float)
            htf_h = htf_df["high"].values.astype(float)
            htf_l = htf_df["low"].values.astype(float)
            htf_c = htf_df["close"].values.astype(float)
            htf_v = htf_df["volume"].values.astype(float)

            htf_ema_9 = talib.EMA(htf_c, timeperiod=9)
            htf_ema_21 = talib.EMA(htf_c, timeperiod=21)
            htf_rsi = talib.RSI(htf_c, timeperiod=14)
            htf_adx = talib.ADX(htf_h, htf_l, htf_c, timeperiod=14)
            _, _, htf_macd_hist = talib.MACD(htf_c, fastperiod=12, slowperiod=26, signalperiod=9)
            htf_vol_sma = talib.SMA(htf_v, timeperiod=20)

            # BB + Keltner for squeeze
            htf_bb_upper, _, htf_bb_lower = talib.BBANDS(htf_c, timeperiod=20, nbdevup=2, nbdevdn=2)
            htf_kc_mid = talib.EMA(htf_c, timeperiod=20)
            htf_kc_atr = talib.ATR(htf_h, htf_l, htf_c, timeperiod=20)
            htf_kc_upper = htf_kc_mid + 1.5 * htf_kc_atr
            htf_kc_lower = htf_kc_mid - 1.5 * htf_kc_atr

            # Build HTF feature series
            htf_times = pd.to_datetime(htf_df["time"])
            if htf_times.dt.tz is None:
                htf_times = htf_times.dt.tz_localize("UTC")

            htf_trend = np.where(
                ~np.isnan(htf_ema_9) & ~np.isnan(htf_ema_21),
                np.where(htf_ema_9 > htf_ema_21, 1.0, -1.0),
                np.nan
            )
            htf_squeeze = np.where(
                ~np.isnan(htf_bb_lower) & ~np.isnan(htf_kc_lower),
                np.where(
                    (htf_bb_lower > htf_kc_lower) & (htf_bb_upper < htf_kc_upper),
                    1.0, 0.0
                ),
                np.nan
            )
            htf_candle_dir = np.where(htf_c > htf_o, 1.0, np.where(htf_c < htf_o, -1.0, 0.0))

            with np.errstate(divide='ignore', invalid='ignore'):
                htf_vol_ratio = np.where(htf_vol_sma > 0, htf_v / htf_vol_sma, np.nan)

            # Create HTF series indexed by time for alignment
            htf_series = {
                "trend": pd.Series(htf_trend, index=htf_times),
                "rsi": pd.Series(htf_rsi, index=htf_times),
                "adx": pd.Series(htf_adx, index=htf_times),
                "squeeze": pd.Series(htf_squeeze, index=htf_times),
                "macd_hist": pd.Series(htf_macd_hist, index=htf_times),
                "volume_ratio": pd.Series(htf_vol_ratio, index=htf_times),
                "candle_dir": pd.Series(htf_candle_dir, index=htf_times),
            }

            # Align HTF to base timeframe via asof merge (forward fill)
            for suffix, series in htf_series.items():
                aligned = np.full(n, np.nan)
                series_sorted = series.sort_index().dropna()

                if len(series_sorted) > 0:
                    # For each base bar, find most recent HTF value
                    htf_idx = series_sorted.index
                    htf_vals = series_sorted.values

                    for i in range(n):
                        base_t = base_times.iloc[i]
                        # Find last HTF bar at or before this base time
                        mask = htf_idx <= base_t
                        if mask.any():
                            pos = mask.nonzero()[0][-1]
                            aligned[i] = htf_vals[pos]

                features[f"htf_{tf_label}_{suffix}"] = aligned

            # Track alignment
            htf_trend_aligned = features[f"htf_{tf_label}_trend"]
            htf_rsi_aligned = features[f"htf_{tf_label}_rsi"]

            # Compute base timeframe trend for alignment comparison
            base_c = df["close"].values.astype(float)
            base_ema_9 = talib.EMA(base_c, timeperiod=9)
            base_ema_21 = talib.EMA(base_c, timeperiod=21)
            base_trend = np.where(
                ~np.isnan(base_ema_9) & ~np.isnan(base_ema_21),
                np.where(base_ema_9 > base_ema_21, 1.0, -1.0),
                np.nan
            )

            trend_match = np.where(
                ~np.isnan(htf_trend_aligned) & ~np.isnan(base_trend),
                np.where(htf_trend_aligned == base_trend, 1.0, 0.0),
                0.0
            )
            trend_agreements += trend_match

            # Momentum agreement: +1 if HTF RSI agrees with direction
            mom_match = np.where(
                ~np.isnan(htf_rsi_aligned) & ~np.isnan(base_trend),
                np.where(
                    ((base_trend > 0) & (htf_rsi_aligned > 50)) |
                    ((base_trend < 0) & (htf_rsi_aligned < 50)),
                    1.0, 0.0
                ),
                0.0
            )
            momentum_agreements += mom_match

        features["mtf_trend_alignment"] = trend_agreements
        features["mtf_momentum_alignment"] = momentum_agreements

        # Confluence score: weighted combination (0-100)
        max_htfs = max(num_htfs, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["mtf_confluence_score"] = (
                (trend_agreements / max_htfs) * 60 +
                (momentum_agreements / max_htfs) * 40
            )

        return features

    def _multi_timeframe_empty(self, n: int):
        """Generate empty MTF features when no HTF data is provided."""
        features = {}
        tf_order = ["5min", "15min", "1hr"]
        for tf_label in tf_order:
            for suffix in ["trend", "rsi", "adx", "squeeze", "macd_hist", "volume_ratio", "candle_dir"]:
                features[f"htf_{tf_label}_{suffix}"] = np.full(n, np.nan)
        features["mtf_trend_alignment"] = np.full(n, np.nan)
        features["mtf_momentum_alignment"] = np.full(n, np.nan)
        features["mtf_confluence_score"] = np.full(n, np.nan)
        return features

    # ================================================================
    # FEATURE COUNT
    # ================================================================

    def get_feature_count(self):
        """Return total number of features computed."""
        return {
            "moving_averages": 22,
            "momentum": 18,
            "volatility": 12,
            "volume": 14,
            "price_action": 20,
            "time_based": 8,
            "smart_money_concepts": 22,
            "vwap_and_levels": 8,
            "divergence": 4,
            "additional_derived": 12,
            "multi_timeframe": 24,  # 7 per HTF × 3 + 3 alignment = 24
            "total": 164,
        }
