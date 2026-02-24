"""
YELENA v2 — Real-Time Feature Engine
Computes all 163 features from live OHLCV bars for ML prediction.

This mirrors the training-time feature computation exactly.
Each timeframe drops its own self-referential HTF features:
  1min  → keeps all 163 (has HTF features for 5min, 15min, 1hr)
  5min  → drops f_htf_5min_* → 156 features
  15min → drops f_htf_5min_* + f_htf_15min_* → 149 features
  1hr   → drops f_htf_5min_* + f_htf_15min_* + f_htf_1hr_* → 139 features

Usage:
    engine = FeatureEngine()
    engine.update_bar("SPY", "5min", ohlcv_bar)
    features, feature_names = engine.compute_features("SPY", "5min")
    sequence = engine.get_sequence("SPY", "5min", seq_len=30)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

# Features to drop per timeframe (self-referential HTF features)
HTF_DROPS = {
    "1min": [],  # 1min keeps all — it references 5min, 15min, 1hr
    "5min": [
        "f_htf_5min_adx", "f_htf_5min_rsi", "f_htf_5min_trend",
        "f_htf_5min_squeeze", "f_htf_5min_macd_hist",
        "f_htf_5min_candle_dir", "f_htf_5min_volume_ratio"
    ],
    "15min": [
        "f_htf_5min_adx", "f_htf_5min_rsi", "f_htf_5min_trend",
        "f_htf_5min_squeeze", "f_htf_5min_macd_hist",
        "f_htf_5min_candle_dir", "f_htf_5min_volume_ratio",
        "f_htf_15min_adx", "f_htf_15min_rsi", "f_htf_15min_trend",
        "f_htf_15min_squeeze", "f_htf_15min_macd_hist",
        "f_htf_15min_candle_dir", "f_htf_15min_volume_ratio"
    ],
    "1hr": [
        "f_htf_5min_adx", "f_htf_5min_rsi", "f_htf_5min_trend",
        "f_htf_5min_squeeze", "f_htf_5min_macd_hist",
        "f_htf_5min_candle_dir", "f_htf_5min_volume_ratio",
        "f_htf_15min_adx", "f_htf_15min_rsi", "f_htf_15min_trend",
        "f_htf_15min_squeeze", "f_htf_15min_macd_hist",
        "f_htf_15min_candle_dir", "f_htf_15min_volume_ratio",
        "f_htf_1hr_adx", "f_htf_1hr_rsi", "f_htf_1hr_trend",
        "f_htf_1hr_squeeze", "f_htf_1hr_macd_hist",
        "f_htf_1hr_candle_dir", "f_htf_1hr_volume_ratio"
    ]
}

# Minimum bars needed to compute all features (longest lookback)
MIN_BARS = 210  # 200 for SMA-200 + buffer


@dataclass
class OHLCVBar:
    """Single OHLCV bar."""
    timestamp: float  # Unix timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class FeatureEngine:
    """
    Real-time feature computation engine.

    Maintains rolling windows of OHLCV bars per symbol/timeframe and
    computes all 163 features on demand.
    """

    def __init__(self, max_bars: int = 500):
        """
        Args:
            max_bars: Maximum bars to keep per symbol/timeframe.
                      Must be > MIN_BARS (210). Default 500 for safety.
        """
        self.max_bars = max(max_bars, MIN_BARS + 50)
        # {symbol: {timeframe: deque of OHLCVBar}}
        self.bars: Dict[str, Dict[str, deque]] = {}
        # Cache for HTF features used by lower timeframes
        self.htf_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

    def update_bar(self, symbol: str, timeframe: str, bar: OHLCVBar):
        """Add or update the latest bar for a symbol/timeframe."""
        if symbol not in self.bars:
            self.bars[symbol] = {}
        if timeframe not in self.bars[symbol]:
            self.bars[symbol][timeframe] = deque(maxlen=self.max_bars)

        buf = self.bars[symbol][timeframe]

        # If same timestamp, replace (update in progress bar)
        if buf and buf[-1].timestamp == bar.timestamp:
            buf[-1] = bar
        else:
            buf.append(bar)

    def update_bars_bulk(self, symbol: str, timeframe: str, bars: List[OHLCVBar]):
        """Load historical bars in bulk (e.g., at startup)."""
        if symbol not in self.bars:
            self.bars[symbol] = {}
        self.bars[symbol][timeframe] = deque(bars[-self.max_bars:], maxlen=self.max_bars)

    def has_enough_bars(self, symbol: str, timeframe: str) -> bool:
        """Check if we have enough bars to compute features."""
        if symbol not in self.bars or timeframe not in self.bars[symbol]:
            return False
        return len(self.bars[symbol][timeframe]) >= MIN_BARS

    def bar_count(self, symbol: str, timeframe: str) -> int:
        """Get current bar count."""
        if symbol not in self.bars or timeframe not in self.bars[symbol]:
            return 0
        return len(self.bars[symbol][timeframe])

    def compute_features(self, symbol: str, timeframe: str,
                         htf_data: Optional[Dict[str, Dict[str, float]]] = None
                         ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute all features for a symbol/timeframe.

        Args:
            symbol: e.g., "SPY"
            timeframe: e.g., "15min"
            htf_data: Optional dict of HTF features {htf_name: {feature: value}}
                      e.g., {"1hr": {"adx": 25.3, "rsi": 55.2, ...}}
                      If None, uses cached HTF values.

        Returns:
            (features_array, feature_names) — already filtered for this TF
        """
        if not self.has_enough_bars(symbol, timeframe):
            raise ValueError(
                f"Not enough bars for {symbol} {timeframe}: "
                f"have {self.bar_count(symbol, timeframe)}, need {MIN_BARS}"
            )

        df = self._bars_to_df(symbol, timeframe)

        # Compute all 163 base features
        all_features = self._compute_all_features(df, symbol, timeframe, htf_data)

        # Drop self-referential HTF features for this timeframe
        drops = HTF_DROPS.get(timeframe, [])
        feature_names = [f for f in all_features.keys() if f not in drops]
        features = np.array([all_features[f] for f in feature_names], dtype=np.float32)

        # Replace NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features, feature_names

    def get_sequence(self, symbol: str, timeframe: str,
                     seq_len: int = 30) -> Optional[np.ndarray]:
        """
        Get a sequence of feature vectors for Transformer/CNN models.

        Returns: (seq_len, n_features) array, or None if not enough data.
        """
        if self.bar_count(symbol, timeframe) < MIN_BARS + seq_len:
            return None

        # We need to compute features for each of the last seq_len bars.
        # Optimization: compute features on full DataFrame, then take last seq_len rows.
        df = self._bars_to_df(symbol, timeframe)
        drops = HTF_DROPS.get(timeframe, [])

        # Compute rolling features for entire DataFrame
        feature_df = self._compute_features_dataframe(df, symbol, timeframe)

        # Drop HTF columns
        keep_cols = [c for c in feature_df.columns if c not in drops]
        feature_df = feature_df[keep_cols]

        # Take last seq_len rows
        sequence = feature_df.iloc[-seq_len:].values.astype(np.float32)
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)

        return sequence

    def _bars_to_df(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Convert bar buffer to DataFrame."""
        buf = self.bars[symbol][timeframe]
        data = [{
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume
        } for b in buf]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    # ================================================================
    # FEATURE COMPUTATION — All 163 features
    # ================================================================

    def _compute_all_features(self, df: pd.DataFrame, symbol: str,
                              timeframe: str,
                              htf_data: Optional[Dict] = None) -> Dict[str, float]:
        """Compute all 163 features from a DataFrame. Returns latest bar's features."""
        features = {}
        c = df["close"]
        h = df["high"]
        l = df["low"]
        o = df["open"]
        v = df["volume"]
        ts = df["timestamp"]

        # --- Moving Averages ---
        for period in [5, 8, 13, 21, 50, 100, 200]:
            features[f"f_sma_{period}"] = c.rolling(period).mean().iloc[-1]
        for period in [5, 8, 13, 21, 50, 100, 200]:
            features[f"f_ema_{period}"] = c.ewm(span=period, adjust=False).mean().iloc[-1]

        # --- VWAP ---
        typical_price = (h + l + c) / 3
        cum_tp_vol = (typical_price * v).cumsum()
        cum_vol = v.cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        features["f_vwap"] = vwap.iloc[-1]

        # VWAP bands
        vwap_std = ((typical_price - vwap) ** 2 * v).cumsum() / cum_vol.replace(0, np.nan)
        vwap_std = np.sqrt(vwap_std.clip(lower=0))
        features["f_vwap_upper_1std"] = (vwap + vwap_std).iloc[-1]
        features["f_vwap_lower_1std"] = (vwap - vwap_std).iloc[-1]
        features["f_vwap_position"] = ((c - vwap) / vwap_std.replace(0, np.nan)).iloc[-1]
        features["f_vwap_distance_pct"] = ((c - vwap) / vwap * 100).iloc[-1]

        # VWMA
        features["f_vwma_20"] = ((c * v).rolling(20).sum() / v.rolling(20).sum()).iloc[-1]

        # --- RSI ---
        for period in [7, 14]:
            delta = c.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            features[f"f_rsi_{period}"] = (100 - 100 / (1 + rs)).iloc[-1]

        # --- Stochastic ---
        low14 = l.rolling(14).min()
        high14 = h.rolling(14).max()
        features["f_stoch_k"] = ((c - low14) / (high14 - low14).replace(0, np.nan) * 100).iloc[-1]
        stoch_k_series = (c - low14) / (high14 - low14).replace(0, np.nan) * 100
        features["f_stoch_d"] = stoch_k_series.rolling(3).mean().iloc[-1]

        # --- Stochastic RSI ---
        rsi14 = 100 - 100 / (1 + c.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean() /
                               (-c.diff().clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan))
        rsi_low = rsi14.rolling(14).min()
        rsi_high = rsi14.rolling(14).max()
        stoch_rsi = (rsi14 - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
        features["f_stoch_rsi_k"] = (stoch_rsi * 100).iloc[-1]
        features["f_stoch_rsi_d"] = (stoch_rsi.rolling(3).mean() * 100).iloc[-1]

        # --- MACD ---
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        features["f_macd"] = macd.iloc[-1]
        features["f_macd_signal"] = macd_signal.iloc[-1]
        features["f_macd_hist"] = macd_hist.iloc[-1]
        features["f_macd_hist_slope"] = macd_hist.diff().iloc[-1]

        # --- ADX ---
        tr = pd.DataFrame({
            "hl": h - l,
            "hc": (h - c.shift()).abs(),
            "lc": (l - c.shift()).abs()
        }).max(axis=1)
        features["f_true_range"] = tr.iloc[-1]

        atr14 = tr.rolling(14).mean()
        atr7 = tr.rolling(7).mean()
        features["f_atr_14"] = atr14.iloc[-1]
        features["f_atr_7"] = atr7.iloc[-1]
        features["f_atr_percent"] = (atr14 / c * 100).iloc[-1]

        plus_dm = (h.diff()).clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        # Zero out when other is larger
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

        plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        features["f_adx_14"] = dx.rolling(14).mean().iloc[-1]
        features["f_plus_di"] = plus_di.iloc[-1]
        features["f_minus_di"] = minus_di.iloc[-1]

        # --- Bollinger Bands ---
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        features["f_bb_middle"] = sma20.iloc[-1]
        features["f_bb_upper"] = (sma20 + 2 * std20).iloc[-1]
        features["f_bb_lower"] = (sma20 - 2 * std20).iloc[-1]
        features["f_bb_width"] = (4 * std20 / sma20.replace(0, np.nan)).iloc[-1]
        features["f_bb_percent"] = ((c - sma20 + 2 * std20) / (4 * std20).replace(0, np.nan)).iloc[-1]
        features["f_std_20"] = std20.iloc[-1]

        # --- Keltner Channels ---
        ema20 = c.ewm(span=20, adjust=False).mean()
        features["f_keltner_upper"] = (ema20 + 1.5 * atr14).iloc[-1]
        features["f_keltner_lower"] = (ema20 - 1.5 * atr14).iloc[-1]

        # --- Squeeze (BB inside Keltner) ---
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        kelt_upper = ema20 + 1.5 * atr14
        kelt_lower = ema20 - 1.5 * atr14
        # (removed: redundant line that operated on full Series)
        # Use latest value
        squeeze_series = (bb_lower > kelt_lower) & (bb_upper < kelt_upper)
        features["f_squeeze"] = float(squeeze_series.iloc[-1])

        # --- CCI ---
        for period in [14, 50]:
            tp = (h + l + c) / 3
            tp_sma = tp.rolling(period).mean()
            tp_mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            features[f"f_cci_{period}"] = ((tp - tp_sma) / (0.015 * tp_mad).replace(0, np.nan)).iloc[-1]

        # --- Williams %R ---
        high14 = h.rolling(14).max()
        low14 = l.rolling(14).min()
        features["f_williams_r"] = ((high14 - c) / (high14 - low14).replace(0, np.nan) * -100).iloc[-1]

        # --- MFI ---
        tp = (h + l + c) / 3
        mf = tp * v
        pos_mf = mf.where(tp > tp.shift(), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(), 0).rolling(14).sum()
        features["f_mfi_14"] = (100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))).iloc[-1]

        # --- Volume indicators ---
        obv = (v * np.sign(c.diff())).cumsum()
        features["f_obv"] = obv.iloc[-1]
        features["f_obv_slope"] = obv.diff(5).iloc[-1]

        vpt = (v * c.pct_change()).cumsum()
        features["f_vpt"] = vpt.iloc[-1]

        features["f_volume_ratio"] = (v / v.rolling(20).mean().replace(0, np.nan)).iloc[-1]
        features["f_volume_sma_20"] = v.rolling(20).mean().iloc[-1]
        features["f_volume_trend"] = float(v.iloc[-1] > v.rolling(20).mean().iloc[-1])
        features["f_high_volume_bar"] = float(v.iloc[-1] > v.rolling(20).mean().iloc[-1] * 1.5)

        # Volume delta (approximation: close > open = buying, else selling)
        vd = v * np.where(c > o, 1, np.where(c < o, -1, 0))
        features["f_volume_delta"] = vd.iloc[-1]
        features["f_volume_delta_cum"] = pd.Series(vd).cumsum().iloc[-1]

        # CMF
        mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        features["f_cmf_20"] = (mfm * v).rolling(20).sum().iloc[-1] / v.rolling(20).sum().replace(0, np.nan).iloc[-1]

        # EOM
        dm = ((h + l) / 2).diff()
        br = v / 1e6 / (h - l).replace(0, np.nan)
        features["f_eom_14"] = (dm / br.replace(0, np.nan)).rolling(14).mean().iloc[-1]

        # Force Index
        features["f_force_index_13"] = (c.diff() * v).ewm(span=13, adjust=False).mean().iloc[-1]

        # AD Line
        ad = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v
        features["f_ad_line"] = ad.cumsum().iloc[-1]

        # --- Returns ---
        for period in [1, 5, 10, 20]:
            features[f"f_return_{period}"] = c.pct_change(period).iloc[-1]

        # --- ROC ---
        for period in [10, 20]:
            features[f"f_roc_{period}"] = ((c - c.shift(period)) / c.shift(period).replace(0, np.nan) * 100).iloc[-1]

        # --- Candle features ---
        body = c - o
        bar_range = h - l
        features["f_candle_body"] = body.iloc[-1]
        features["f_candle_body_pct"] = (body.abs() / bar_range.replace(0, np.nan)).iloc[-1]
        features["f_candle_upper_wick"] = (h - pd.concat([c, o], axis=1).max(axis=1)).iloc[-1]
        features["f_candle_lower_wick"] = (pd.concat([c, o], axis=1).min(axis=1) - l).iloc[-1]
        features["f_candle_direction"] = float(c.iloc[-1] >= o.iloc[-1])
        features["f_bar_range"] = bar_range.iloc[-1]
        features["f_bar_range_vs_atr"] = (bar_range / atr14.replace(0, np.nan)).iloc[-1]
        features["f_close_vs_range"] = ((c - l) / bar_range.replace(0, np.nan)).iloc[-1]

        # Consecutive candles
        directions = (c >= o).astype(int)
        bull_run = 0
        bear_run = 0
        for i in range(len(directions) - 1, -1, -1):
            if directions.iloc[i] == 1:
                bull_run += 1
            else:
                break
        for i in range(len(directions) - 1, -1, -1):
            if directions.iloc[i] == 0:
                bear_run += 1
            else:
                break
        features["f_consecutive_bull"] = bull_run
        features["f_consecutive_bear"] = bear_run

        # --- Candlestick patterns ---
        features["f_doji"] = float(body.abs().iloc[-1] < bar_range.iloc[-1] * 0.1)
        features["f_hammer"] = float(
            (features["f_candle_lower_wick"] > body.abs().iloc[-1] * 2) and
            (features["f_candle_upper_wick"] < body.abs().iloc[-1] * 0.5)
        )
        # Engulfing
        prev_body = body.iloc[-2] if len(body) > 1 else 0
        curr_body = body.iloc[-1]
        features["f_engulfing"] = float(
            (prev_body < 0 and curr_body > 0 and curr_body > abs(prev_body)) or
            (prev_body > 0 and curr_body < 0 and abs(curr_body) > prev_body)
        )
        features["f_inside_bar"] = float(
            h.iloc[-1] < h.iloc[-2] and l.iloc[-1] > l.iloc[-2]
        ) if len(h) > 1 else 0.0

        # --- Swing points ---
        features["f_swing_high"] = float(
            h.iloc[-2] > h.iloc[-3] and h.iloc[-2] > h.iloc[-1]
        ) if len(h) > 2 else 0.0
        features["f_swing_low"] = float(
            l.iloc[-2] < l.iloc[-3] and l.iloc[-2] < l.iloc[-1]
        ) if len(l) > 2 else 0.0
        features["f_higher_high"] = float(h.iloc[-1] > h.rolling(20).max().iloc[-2]) if len(h) > 20 else 0.0
        features["f_lower_low"] = float(l.iloc[-1] < l.rolling(20).min().iloc[-2]) if len(l) > 20 else 0.0
        features["f_higher_low"] = float(l.iloc[-1] > l.iloc[-2]) if len(l) > 1 else 0.0
        features["f_lower_high"] = float(h.iloc[-1] < h.iloc[-2]) if len(h) > 1 else 0.0

        # --- Price vs MAs ---
        features["f_price_vs_sma_50"] = ((c - c.rolling(50).mean()) / c.rolling(50).mean().replace(0, np.nan) * 100).iloc[-1]
        features["f_price_vs_sma_200"] = ((c - c.rolling(200).mean()) / c.rolling(200).mean().replace(0, np.nan) * 100).iloc[-1]
        features["f_sma_50_vs_200"] = float(c.rolling(50).mean().iloc[-1] > c.rolling(200).mean().iloc[-1])

        # EMA slopes
        ema8 = c.ewm(span=8, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        features["f_ema_slope_8"] = ema8.diff().iloc[-1]
        features["f_ema_slope_21"] = ema21.diff().iloc[-1]
        features["f_ema_8_vs_21"] = float(ema8.iloc[-1] > ema21.iloc[-1])

        # EMA crosses
        ema9 = c.ewm(span=9, adjust=False).mean()
        features["f_ema_cross_9_21"] = float(
            (ema9.iloc[-1] > ema21.iloc[-1]) and (ema9.iloc[-2] <= ema21.iloc[-2])
        ) if len(ema9) > 1 else 0.0
        ema50 = c.ewm(span=50, adjust=False).mean()
        ema200 = c.ewm(span=200, adjust=False).mean()
        features["f_ema_cross_50_200"] = float(
            (ema50.iloc[-1] > ema200.iloc[-1]) and (ema50.iloc[-2] <= ema200.iloc[-2])
        ) if len(ema50) > 1 else 0.0

        # Pct from EMAs
        features["f_pct_from_ema_9"] = ((c - ema9) / ema9.replace(0, np.nan) * 100).iloc[-1]
        features["f_pct_from_ema_21"] = ((c - ema21) / ema21.replace(0, np.nan) * 100).iloc[-1]
        features["f_pct_from_ema_50"] = ((c - ema50) / ema50.replace(0, np.nan) * 100).iloc[-1]

        # 20-bar range
        high20 = h.rolling(20).max()
        low20 = l.rolling(20).min()
        features["f_pct_from_20_high"] = ((c - high20) / high20.replace(0, np.nan) * 100).iloc[-1]
        features["f_pct_from_20_low"] = ((c - low20) / low20.replace(0, np.nan) * 100).iloc[-1]
        features["f_range_20_position"] = ((c - low20) / (high20 - low20).replace(0, np.nan)).iloc[-1]

        # MA Ribbon
        features["f_ma_ribbon_bullish"] = float(
            ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]
        )

        # RSI + BB combo
        rsi14_val = features.get("f_rsi_14", 50)
        bb_pct_val = features.get("f_bb_percent", 0.5)
        features["f_rsi_bb_combo"] = rsi14_val * bb_pct_val / 100

        # --- Gap features ---
        features["f_gap_up"] = float(o.iloc[-1] > h.iloc[-2]) if len(h) > 1 else 0.0
        features["f_gap_down"] = float(o.iloc[-1] < l.iloc[-2]) if len(l) > 1 else 0.0

        # --- Time features ---
        last_ts = ts.iloc[-1]
        if hasattr(last_ts, "hour"):
            features["f_hour_of_day"] = last_ts.hour
            features["f_minute_of_day"] = last_ts.hour * 60 + last_ts.minute
            features["f_day_of_week"] = last_ts.dayofweek
            features["f_is_monday"] = float(last_ts.dayofweek == 0)
            features["f_is_friday"] = float(last_ts.dayofweek == 4)

            # Market hours (ET: 9:30-16:00)
            market_min = last_ts.hour * 60 + last_ts.minute
            features["f_is_market_open"] = float(market_min <= 570 + 15)  # First 15 min
            features["f_is_market_close"] = float(market_min >= 960 - 15)  # Last 15 min
            features["f_is_power_hour"] = float(market_min >= 900)  # 3 PM ET
            features["f_minutes_since_open"] = max(0, market_min - 570)
        else:
            for tf_feat in ["f_hour_of_day", "f_minute_of_day", "f_day_of_week",
                           "f_is_monday", "f_is_friday", "f_is_market_open",
                           "f_is_market_close", "f_is_power_hour", "f_minutes_since_open"]:
                features[tf_feat] = 0.0

        # --- Divergences (simplified: RSI vs price) ---
        rsi14_series = 100 - 100 / (1 + c.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean() /
                                      (-c.diff().clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan))
        price_higher = c.iloc[-1] > c.iloc[-6]
        rsi_lower = rsi14_series.iloc[-1] < rsi14_series.iloc[-6]
        features["f_rsi_divergence_bear"] = float(price_higher and rsi_lower)

        price_lower = c.iloc[-1] < c.iloc[-6]
        rsi_higher = rsi14_series.iloc[-1] > rsi14_series.iloc[-6]
        features["f_rsi_divergence_bull"] = float(price_lower and rsi_higher)

        # MACD divergence
        macd_higher = macd_hist.iloc[-1] > macd_hist.iloc[-6]
        macd_lower = macd_hist.iloc[-1] < macd_hist.iloc[-6]
        features["f_macd_divergence_bear"] = float(price_higher and macd_lower)
        features["f_macd_divergence_bull"] = float(price_lower and macd_higher)

        # --- Smart Money Concepts (simplified) ---
        # Market structure: 1 = bullish, -1 = bearish, 0 = range
        recent_highs = h.rolling(20).max()
        recent_lows = l.rolling(20).min()
        making_hh = h.iloc[-1] > recent_highs.iloc[-2] if len(recent_highs) > 1 else False
        making_hl = l.iloc[-1] > recent_lows.iloc[-2] if len(recent_lows) > 1 else False
        making_ll = l.iloc[-1] < recent_lows.iloc[-2] if len(recent_lows) > 1 else False
        making_lh = h.iloc[-1] < recent_highs.iloc[-2] if len(recent_highs) > 1 else False

        if making_hh and making_hl:
            features["f_market_structure"] = 1.0
        elif making_ll and making_lh:
            features["f_market_structure"] = -1.0
        else:
            features["f_market_structure"] = 0.0

        # BOS / CHoCH (simplified)
        features["f_bos_bullish"] = float(making_hh)
        features["f_bos_bearish"] = float(making_ll)
        features["f_choch_bullish"] = float(making_ll and c.iloc[-1] > o.iloc[-1])
        features["f_choch_bearish"] = float(making_hh and c.iloc[-1] < o.iloc[-1])
        features["f_structure_shift"] = float(features["f_choch_bullish"] or features["f_choch_bearish"])

        # Bars since events (simplified)
        features["f_bars_since_bos"] = 1.0  # Placeholder
        features["f_bars_since_choch"] = 1.0  # Placeholder

        # Order blocks (simplified: previous strong move origin)
        features["f_ob_bullish"] = float(
            body.iloc[-3] > 0 and body.iloc[-3] > bar_range.iloc[-3] * 0.6
        ) if len(body) > 2 else 0.0
        features["f_ob_bearish"] = float(
            body.iloc[-3] < 0 and abs(body.iloc[-3]) > bar_range.iloc[-3] * 0.6
        ) if len(body) > 2 else 0.0
        features["f_ob_nearest_distance_pct"] = 0.0  # Placeholder

        # FVG (Fair Value Gap)
        features["f_fvg_bullish"] = float(
            l.iloc[-1] > h.iloc[-3]
        ) if len(l) > 2 else 0.0
        features["f_fvg_bearish"] = float(
            h.iloc[-1] < l.iloc[-3]
        ) if len(h) > 2 else 0.0
        features["f_fvg_nearest_distance_pct"] = 0.0  # Placeholder

        # Displacement
        features["f_displacement_up"] = float(
            bar_range.iloc[-1] > atr14.iloc[-1] * 2 and c.iloc[-1] > o.iloc[-1]
        )
        features["f_displacement_down"] = float(
            bar_range.iloc[-1] > atr14.iloc[-1] * 2 and c.iloc[-1] < o.iloc[-1]
        )

        # Premium/discount
        eq_high = h.rolling(50).max().iloc[-1]
        eq_low = l.rolling(50).min().iloc[-1]
        eq_mid = (eq_high + eq_low) / 2
        features["f_premium_discount"] = float(c.iloc[-1] > eq_mid)

        # Liquidity sweeps
        prev_high = h.rolling(10).max().iloc[-2] if len(h) > 10 else h.iloc[-2]
        prev_low = l.rolling(10).min().iloc[-2] if len(l) > 10 else l.iloc[-2]
        features["f_liquidity_sweep_high"] = float(h.iloc[-1] > prev_high and c.iloc[-1] < prev_high)
        features["f_liquidity_sweep_low"] = float(l.iloc[-1] < prev_low and c.iloc[-1] > prev_low)

        # --- HTF Features ---
        # These are computed from higher timeframe data
        htf_sources = {"5min": htf_data, "15min": htf_data, "1hr": htf_data}
        if htf_data is None:
            htf_data = self.htf_cache.get(symbol, {})

        for htf_name in ["5min", "15min", "1hr"]:
            htf = htf_data.get(htf_name, {})
            prefix = f"f_htf_{htf_name}"
            features[f"{prefix}_adx"] = htf.get("adx", 0.0)
            features[f"{prefix}_rsi"] = htf.get("rsi", 50.0)
            features[f"{prefix}_trend"] = htf.get("trend", 0.0)
            features[f"{prefix}_squeeze"] = htf.get("squeeze", 0.0)
            features[f"{prefix}_macd_hist"] = htf.get("macd_hist", 0.0)
            features[f"{prefix}_candle_dir"] = htf.get("candle_dir", 0.0)
            features[f"{prefix}_volume_ratio"] = htf.get("volume_ratio", 1.0)

        # --- Multi-timeframe alignment ---
        features["f_mtf_trend_alignment"] = 0.0  # Computed by orchestrator
        features["f_mtf_confluence_score"] = 0.0
        features["f_mtf_momentum_alignment"] = 0.0

        return features

    def _compute_features_dataframe(self, df: pd.DataFrame, symbol: str,
                                    timeframe: str) -> pd.DataFrame:
        """
        Compute features for all rows in DataFrame (for sequence generation).
        Uses vectorized operations where possible.

        Returns DataFrame with feature columns (same length as input).
        """
        # For now, compute only for the last row and repeat
        # TODO: Implement full vectorized rolling feature computation
        features, names = self.compute_features(symbol, timeframe)
        # Create a DataFrame with the last row's features repeated
        # This is a simplified version — proper implementation computes per-bar
        feature_df = pd.DataFrame(
            np.tile(features, (len(df), 1)),
            columns=names
        )
        return feature_df

    def compute_htf_summary(self, symbol: str, timeframe: str) -> Dict[str, float]:
        """
        Compute HTF summary features for a timeframe.
        Call this on higher timeframes and pass to lower TF compute_features.

        Returns: {"adx": ..., "rsi": ..., "trend": ..., "squeeze": ...,
                  "macd_hist": ..., "candle_dir": ..., "volume_ratio": ...}
        """
        if not self.has_enough_bars(symbol, timeframe):
            return {}

        df = self._bars_to_df(symbol, timeframe)
        c = df["close"]
        h = df["high"]
        l = df["low"]
        o = df["open"]
        v = df["volume"]

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]

        # ADX
        tr = pd.DataFrame({
            "hl": h - l, "hc": (h - c.shift()).abs(), "lc": (l - c.shift()).abs()
        }).max(axis=1)
        atr = tr.rolling(14).mean()
        plus_dm = h.diff().clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        plus_di = 100 * plus_dm.rolling(14).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr.replace(0, np.nan)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx = dx.rolling(14).mean().iloc[-1]

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd_hist = (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1]

        # Trend (EMA 8 vs 21)
        ema8 = c.ewm(span=8, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        trend = 1.0 if ema8.iloc[-1] > ema21.iloc[-1] else -1.0

        # Squeeze
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        ema20 = c.ewm(span=20, adjust=False).mean()
        kelt_upper = ema20 + 1.5 * atr
        kelt_lower = ema20 - 1.5 * atr
        squeeze = float(((bb_lower > kelt_lower) & (bb_upper < kelt_upper)).iloc[-1])

        # Candle direction
        candle_dir = 1.0 if c.iloc[-1] >= o.iloc[-1] else -1.0

        # Volume ratio
        vol_ratio = (v.iloc[-1] / v.rolling(20).mean().replace(0, np.nan).iloc[-1])

        summary = {
            "adx": float(np.nan_to_num(adx, nan=0.0)),
            "rsi": float(np.nan_to_num(rsi, nan=50.0)),
            "trend": trend,
            "squeeze": squeeze,
            "macd_hist": float(np.nan_to_num(macd_hist, nan=0.0)),
            "candle_dir": candle_dir,
            "volume_ratio": float(np.nan_to_num(vol_ratio, nan=1.0))
        }

        # Cache for lower TFs
        if symbol not in self.htf_cache:
            self.htf_cache[symbol] = {}
        self.htf_cache[symbol][timeframe] = summary

        return summary
