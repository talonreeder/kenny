"""
YELENA v2 — Parameter Optimizer
================================
Connects to PostgreSQL on RDS, pulls real OHLCV data, simulates QCloud, QLine,
QWave, and QBands indicators with different parameter combinations, scores signal
quality, and outputs optimal params per symbol/timeframe with ready-to-paste Pine
Script lookup tables.

Usage:
    cd ~/yelena/backend
    source ../venv/bin/activate
    python -m scripts.optimize_params                    # Full optimization (all symbols, all timeframes)
    python -m scripts.optimize_params --symbol SPY       # Single symbol
    python -m scripts.optimize_params --timeframe 5min   # Single timeframe
    python -m scripts.optimize_params --indicator qbands # Single indicator
    python -m scripts.optimize_params --quick            # Quick mode (fewer combos)

Requirements:
    pip install psycopg2-binary pandas numpy tabulate
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary --break-system-packages")
    sys.exit(1)

try:
    from tabulate import tabulate
except ImportError:
    print("WARNING: tabulate not installed. Run: pip install tabulate --break-system-packages")
    tabulate = None


# ============================================================================
# CONFIGURATION
# ============================================================================

# Database connection — reads from environment or SSM
DB_HOST = os.environ.get("YELENA_DB_HOST", "yelena-db.ci3ykeq8mckh.us-east-1.rds.amazonaws.com")
DB_NAME = os.environ.get("YELENA_DB_NAME", "yelena")
DB_USER = os.environ.get("YELENA_DB_USER", "postgres")
DB_PASS = os.environ.get("YELENA_DB_PASS", "")  # Set via env var or SSM

# If no password in env, try reading from SSM
if not DB_PASS:
    try:
        import boto3
        ssm = boto3.client("ssm", region_name="us-east-1")
        resp = ssm.get_parameter(Name="/yelena/database-url", WithDecryption=True)
        db_url = resp["Parameter"]["Value"]
        # Parse password from URL: postgresql://user:pass@host:port/db
        if "@" in db_url and ":" in db_url:
            DB_PASS = db_url.split("://")[1].split(":")[1].split("@")[0]
    except Exception as e:
        print(f"WARNING: Could not read DB password from SSM: {e}")
        print("Set YELENA_DB_PASS environment variable or configure SSM access.")

# Symbols to optimize (SPX excluded — no minute bars)
ALL_SYMBOLS = ["SPY", "QQQ", "TSLA", "NVDA", "META", "NFLX", "AAPL", "GOOGL", "MSFT", "AMZN", "AMD"]

# Timeframe mapping: label -> (table_name, tf_group)
TIMEFRAMES = {
    "1min":  ("bars_1min",  "scalp"),
    "5min":  ("bars_5min",  "intraday"),
    "15min": ("bars_15min", "intraday"),
    "1hr":   ("bars_1hr",   "swing"),
}

# Output directory
RESULTS_DIR = Path(__file__).parent.parent.parent / "optimization_results"

# ============================================================================
# PARAMETER SEARCH SPACES
# ============================================================================

# QCloud parameters to test
QCLOUD_PARAMS = {
    "base_length": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "step_mult": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    "smoothing": [2, 3],
}

# QCloud quick mode (fewer combos for testing)
QCLOUD_PARAMS_QUICK = {
    "base_length": [8, 10, 12, 14],
    "step_mult": [1.4, 1.5, 1.6],
    "smoothing": [2],
}

# QLine parameters to test
QLINE_PARAMS = {
    "atr_length": [8, 10, 12, 14, 16, 18, 20],
    "factor": [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5],
}

# QLine quick mode
QLINE_PARAMS_QUICK = {
    "atr_length": [10, 12, 14, 16],
    "factor": [1.8, 2.0, 2.5, 3.0],
}

# QWave parameters to test (ADX-based directional movement)
QWAVE_PARAMS = {
    "adx_length": [8, 10, 12, 14, 16, 18, 20, 22, 25],
    "smoothing": [1, 2, 3, 4, 5],
}

# QWave quick mode
QWAVE_PARAMS_QUICK = {
    "adx_length": [10, 14, 18, 22],
    "smoothing": [1, 3],
}

# QBands: Bollinger + Keltner squeeze detection
QBANDS_PARAMS = {
    "bb_length": [14, 16, 18, 20, 22, 25, 30],
    "bb_mult": [1.5, 1.8, 2.0, 2.2, 2.5],
    "kc_mult": [1.0, 1.2, 1.5, 1.8, 2.0],
}

# QBands quick mode
QBANDS_PARAMS_QUICK = {
    "bb_length": [16, 20, 25],
    "bb_mult": [1.8, 2.0, 2.2],
    "kc_mult": [1.2, 1.5, 1.8],
}

# Moneyball: Volume-weighted momentum oscillator
MONEYBALL_PARAMS = {
    "roc_length": [5, 6, 8, 10, 12, 14, 16, 20],
    "smooth_length": [2, 3, 4, 5, 6, 8],
}

# Moneyball quick mode
MONEYBALL_PARAMS_QUICK = {
    "roc_length": [6, 10, 14, 20],
    "smooth_length": [2, 4],
}

# QMomentum: RSI + StochRSI divergence oscillator
QMOMENTUM_PARAMS = {
    "rsi_length": [8, 10, 12, 14, 16, 18, 20],
    "stoch_length": [8, 10, 12, 14, 16, 18, 20],
}

# QMomentum quick mode
QMOMENTUM_PARAMS_QUICK = {
    "rsi_length": [10, 14, 18, 20],
    "stoch_length": [10, 14, 18],
}

# QCVD: Cumulative Volume Delta
QCVD_PARAMS = {
    "smooth_length": [1, 2, 3, 4, 5, 8],
    "trend_length": [8, 10, 12, 14, 16, 20, 25],
}

# QCVD quick mode
QCVD_PARAMS_QUICK = {
    "smooth_length": [1, 3, 5],
    "trend_length": [10, 14, 20],
}

# QSMC: Smart Money Concepts
QSMC_PARAMS = {
    "swing_length": [3, 4, 5, 6, 7, 8, 9, 10],
    "ob_strength": [1.0, 1.5, 2.0, 2.5, 3.0],
}

# QSMC quick mode
QSMC_PARAMS_QUICK = {
    "swing_length": [3, 5, 7, 10],
    "ob_strength": [1.0, 1.5, 2.5],
}

# QGrid: Dynamic S/R Grid
QGRID_PARAMS = {
    "left_bars": [3, 4, 5, 6, 7, 8, 9, 10],
    "right_bars": [2, 3, 4, 5, 6, 7, 8],
}

# QGrid quick mode
QGRID_PARAMS_QUICK = {
    "left_bars": [3, 5, 7, 10],
    "right_bars": [2, 4, 6],
}

# Signal quality measurement
LOOKAHEAD_BARS = [5, 10, 15]  # Check N bars forward for flip accuracy
BOUNCE_LOOKAHEAD = [5, 10, 15]  # Check N bars forward for bounce win rate
WHIPSAW_THRESHOLD = 5  # Flip that reverses within N bars = whipsaw


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_db_connection():
    """Create a database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            connect_timeout=10,
        )
        return conn
    except psycopg2.Error as e:
        print(f"ERROR: Cannot connect to database: {e}")
        print(f"  Host: {DB_HOST}")
        print(f"  Database: {DB_NAME}")
        print(f"  User: {DB_USER}")
        print(f"  Password set: {'Yes' if DB_PASS else 'NO — set YELENA_DB_PASS'}")
        sys.exit(1)


def load_bars(conn, table: str, symbol: str) -> pd.DataFrame:
    """
    Load OHLCV bars from a specific table for a symbol.
    Filters out weekend bars and sorts by time ascending.
    """
    query = f"""
        SELECT time, open, high, low, close, volume
        FROM {table}
        WHERE symbol = %s
          AND EXTRACT(DOW FROM time) NOT IN (0, 6)
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ============================================================================
# INDICATOR SIMULATIONS
# ============================================================================

def compute_ema(series: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA using numpy for speed."""
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(series, dtype=np.float64)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = alpha * series[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Compute Average True Range."""
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    # ATR = SMA of TR over period (matching Pine Script default behavior)
    atr = np.empty(n, dtype=np.float64)
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])
    # Then RMA (Wilder's smoothing) from period onward
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def simulate_qcloud(
    close: np.ndarray,
    base_length: int,
    step_mult: float,
    smoothing: int,
) -> Dict:
    """
    Simulate QCloud indicator on historical price data.

    Computes 5 EMAs with stepped lengths, counts bullish layers,
    applies smoothing, and tracks state transitions (flips).

    Returns dict with arrays: bullish_count, flipped_bullish, flipped_bearish,
    is_squeeze, and computed metrics.
    """
    n = len(close)

    # Compute 5 EMA lengths
    len1 = base_length
    len2 = max(2, round(base_length * step_mult ** 1))
    len3 = max(2, round(base_length * step_mult ** 2))
    len4 = max(2, round(base_length * step_mult ** 3))
    len5 = max(2, round(base_length * step_mult ** 4))

    # Compute EMAs
    ma1 = compute_ema(close, len1)
    ma2 = compute_ema(close, len2)
    ma3 = compute_ema(close, len3)
    ma4 = compute_ema(close, len4)
    ma5 = compute_ema(close, len5)

    # Layer scoring: bullish when price > MA AND MA > next slower MA
    layer1 = (close > ma1) & (ma1 > ma2)
    layer2 = (close > ma2) & (ma2 > ma3)
    layer3 = (close > ma3) & (ma3 > ma4)
    layer4 = (close > ma4) & (ma4 > ma5)
    layer5 = close > ma5

    bull_count_raw = (
        layer1.astype(int) + layer2.astype(int) + layer3.astype(int)
        + layer4.astype(int) + layer5.astype(int)
    )

    # Apply smoothing (same logic as Pine Script)
    confirmed = np.empty(n, dtype=int)
    confirmed[0] = 3  # Start neutral
    pending = 3
    streak = 0

    for i in range(1, n):
        raw = int(bull_count_raw[i])
        if raw != pending:
            pending = raw
            streak = 1
        elif raw == pending and pending != confirmed[i - 1]:
            streak += 1
            if streak >= smoothing:
                confirmed[i] = pending
                streak = 0
                continue
        else:
            streak = 0
        confirmed[i] = confirmed[i - 1]

    # Detect flips: bullish = crosses from <=2 to >=3, bearish = crosses from >=3 to <=2
    flipped_bullish = np.zeros(n, dtype=bool)
    flipped_bearish = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if confirmed[i] >= 3 and confirmed[i - 1] < 3:
            flipped_bullish[i] = True
        if confirmed[i] <= 2 and confirmed[i - 1] > 2:
            flipped_bearish[i] = True

    # Cloud width for squeeze detection
    cloud_width = np.abs(ma1 - ma5)
    avg_width_20 = pd.Series(cloud_width).rolling(20, min_periods=1).mean().values
    is_squeeze = cloud_width < (avg_width_20 * 0.6)
    squeeze_started = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if is_squeeze[i] and not is_squeeze[i - 1]:
            squeeze_started[i] = True

    return {
        "bullish_count": confirmed,
        "flipped_bullish": flipped_bullish,
        "flipped_bearish": flipped_bearish,
        "is_squeeze": is_squeeze,
        "squeeze_started": squeeze_started,
        "ma1": ma1,
        "ma5": ma5,
    }


def simulate_qline(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_length: int,
    factor: float,
    touch_atr_pct: float = 0.15,
) -> Dict:
    """
    Simulate QLine (SuperTrend) indicator on historical price data.

    Implements proper SuperTrend ratcheting logic matching Pine Script's
    built-in ta.supertrend(factor, atr_length).

    Returns dict with arrays: qline, direction, flipped_bullish, flipped_bearish,
    is_touching, bounce_bullish, bounce_bearish, trend_duration.
    """
    n = len(close)
    atr = compute_atr(high, low, close, atr_length)
    hl2 = (high + low) / 2.0

    # Raw bands
    upper_raw = hl2 + factor * atr
    lower_raw = hl2 - factor * atr

    # SuperTrend with proper ratcheting
    upper_band = np.empty(n, dtype=np.float64)
    lower_band = np.empty(n, dtype=np.float64)
    direction = np.ones(n, dtype=int)  # 1 = downtrend, -1 = uptrend
    qline = np.empty(n, dtype=np.float64)

    upper_band[0] = upper_raw[0]
    lower_band[0] = lower_raw[0]
    qline[0] = upper_raw[0]

    for i in range(1, n):
        if np.isnan(atr[i]):
            upper_band[i] = upper_raw[i] if not np.isnan(upper_raw[i]) else upper_band[i - 1]
            lower_band[i] = lower_raw[i] if not np.isnan(lower_raw[i]) else lower_band[i - 1]
            direction[i] = direction[i - 1]
            qline[i] = qline[i - 1]
            continue

        # ============================================================
        # RATCHETING — exact match to Pine Script ta.supertrend()
        # Pine names: up = lower_band (support), dn = upper_band (resistance)
        # ============================================================

        # Lower band (Pine "up"): ratchet UP when prev close was above it
        # Pine: up := close[1] > up[1] ? math.max(up, up[1]) : up
        if close[i - 1] > lower_band[i - 1]:
            lower_band[i] = max(lower_raw[i], lower_band[i - 1])
        else:
            lower_band[i] = lower_raw[i]

        # Upper band (Pine "dn"): ratchet DOWN when prev close was below it
        # Pine: dn := close[1] < dn[1] ? math.min(dn, dn[1]) : dn
        if close[i - 1] < upper_band[i - 1]:
            upper_band[i] = min(upper_raw[i], upper_band[i - 1])
        else:
            upper_band[i] = upper_raw[i]

        # ============================================================
        # DIRECTION — exact match to Pine Script
        # Pine: trend := close > dn[1] ? 1 : close < up[1] ? -1 : nz(trend[1], 1)
        # My convention: -1 = uptrend, 1 = downtrend (inverted from Pine)
        # ============================================================
        if close[i] > upper_band[i - 1]:
            direction[i] = -1  # Uptrend (Pine trend = 1)
        elif close[i] < lower_band[i - 1]:
            direction[i] = 1   # Downtrend (Pine trend = -1)
        else:
            direction[i] = direction[i - 1]  # No change

        # QLine value
        if direction[i] == -1:
            qline[i] = lower_band[i]
        else:
            qline[i] = upper_band[i]

    # Flip detection
    is_uptrend = direction == -1
    flipped_bullish = np.zeros(n, dtype=bool)
    flipped_bearish = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if is_uptrend[i] and not is_uptrend[i - 1]:
            flipped_bullish[i] = True
        if not is_uptrend[i] and is_uptrend[i - 1]:
            flipped_bearish[i] = True

    # Trend duration
    trend_duration = np.ones(n, dtype=int)
    for i in range(1, n):
        if is_uptrend[i] == is_uptrend[i - 1]:
            trend_duration[i] = trend_duration[i - 1] + 1
        else:
            trend_duration[i] = 1

    # Touch detection
    touch_threshold = atr * touch_atr_pct
    dist_to_qline = np.abs(close - qline)
    is_touching = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(touch_threshold[i]):
            is_touching[i] = dist_to_qline[i] <= touch_threshold[i]

    # Bounce detection with 3-bar gap
    touch_count = np.zeros(n, dtype=int)
    bars_since_touch = np.full(n, 100, dtype=int)
    bounce_bullish = np.zeros(n, dtype=bool)
    bounce_bearish = np.zeros(n, dtype=bool)

    for i in range(1, n):
        # Reset on flip
        if is_uptrend[i] != is_uptrend[i - 1]:
            touch_count[i] = 0
            bars_since_touch[i] = 100
        else:
            touch_count[i] = touch_count[i - 1]
            bars_since_touch[i] = bars_since_touch[i - 1] + 1

        if is_touching[i] and bars_since_touch[i] >= 3:
            touch_count[i] = touch_count[i - 1] + 1
            bars_since_touch[i] = 0

            # Bullish bounce: touching in uptrend + bullish candle
            if is_uptrend[i]:
                if close[i] > (high[i] + low[i]) / 2.0 and close[i] > close[i - 1]:
                    bounce_bullish[i] = True
            else:
                if close[i] < (high[i] + low[i]) / 2.0 and close[i] < close[i - 1]:
                    bounce_bearish[i] = True

    return {
        "qline": qline,
        "direction": direction,
        "is_uptrend": is_uptrend,
        "flipped_bullish": flipped_bullish,
        "flipped_bearish": flipped_bearish,
        "trend_duration": trend_duration,
        "is_touching": is_touching,
        "touch_count": touch_count,
        "bounce_bullish": bounce_bullish,
        "bounce_bearish": bounce_bearish,
        "atr": atr,
    }


def simulate_qwave(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    adx_length: int,
    smoothing: int,
) -> Dict:
    """
    Simulate QWave (ADX-based directional movement) indicator.

    Implements: direction × pow(adx_norm, 1.5) × 100
    Where direction = (DI+ - DI-) / (DI+ + DI-)
    And adx_norm = min(ADX / 50, 1.0)

    This matches Pine Script's ta.dmi(adx_length, adx_length).
    """
    n = len(close)

    # ---- Step 1: Compute True Range ----
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # ---- Step 2: Compute +DM and -DM ----
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # ---- Step 3: Wilder smoothing (RMA) for TR, +DM, -DM ----
    # Pine's ta.rma is equivalent to Wilder's smoothing
    def wilder_smooth(arr, period):
        result = np.empty(n, dtype=np.float64)
        result[:] = np.nan
        # Seed with SMA of first 'period' values
        if n >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, n):
                result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
        return result

    atr_smooth = wilder_smooth(tr, adx_length)
    plus_dm_smooth = wilder_smooth(plus_dm, adx_length)
    minus_dm_smooth = wilder_smooth(minus_dm, adx_length)

    # ---- Step 4: DI+ and DI- ----
    di_plus = np.zeros(n, dtype=np.float64)
    di_minus = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(atr_smooth[i]) and atr_smooth[i] > 0:
            di_plus[i] = 100.0 * plus_dm_smooth[i] / atr_smooth[i]
            di_minus[i] = 100.0 * minus_dm_smooth[i] / atr_smooth[i]

    # ---- Step 5: DX and ADX ----
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum

    adx = wilder_smooth(dx, adx_length)
    # Replace NaNs with 0 for safety
    adx = np.where(np.isnan(adx), 0.0, adx)

    # ---- Step 6: QWave formula ----
    qwave_raw = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0.001:
            direction = (di_plus[i] - di_minus[i]) / di_sum
        else:
            direction = 0.0
        adx_norm = min(adx[i] / 50.0, 1.0)
        amplifier = adx_norm ** 1.5
        qwave_raw[i] = direction * amplifier * 100.0

    # ---- Step 7: Optional EMA smoothing ----
    if smoothing > 1:
        qwave = compute_ema(qwave_raw, smoothing)
    else:
        qwave = qwave_raw.copy()

    # ---- Step 8: Zone classification ----
    # 6=Strong Bull(>=60), 5=Bull(>=30), 4=Weak Bull(>=0),
    # 3=Weak Bear(>=-30), 2=Bear(>=-60), 1=Strong Bear(<-60)
    zone = np.zeros(n, dtype=int)
    for i in range(n):
        if qwave[i] >= 60:
            zone[i] = 6
        elif qwave[i] >= 30:
            zone[i] = 5
        elif qwave[i] >= 0:
            zone[i] = 4
        elif qwave[i] >= -30:
            zone[i] = 3
        elif qwave[i] >= -60:
            zone[i] = 2
        else:
            zone[i] = 1

    # ---- Step 9: Signal detection ----
    # Zero-line flips
    flipped_bullish = np.zeros(n, dtype=bool)
    flipped_bearish = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if qwave[i] >= 0 and qwave[i - 1] < 0:
            flipped_bullish[i] = True
        elif qwave[i] < 0 and qwave[i - 1] >= 0:
            flipped_bearish[i] = True

    # Zone crossings into Bull (>=30) and Bear (<=-30)
    entered_bull = np.zeros(n, dtype=bool)
    entered_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if zone[i] >= 5 and zone[i - 1] < 5:  # Entered Bull or Strong Bull
            entered_bull[i] = True
        if zone[i] <= 2 and zone[i - 1] > 2:  # Entered Bear or Strong Bear
            entered_bear[i] = True

    return {
        "qwave": qwave,
        "qwave_raw": qwave_raw,
        "zone": zone,
        "adx": adx,
        "di_plus": di_plus,
        "di_minus": di_minus,
        "flipped_bullish": flipped_bullish,
        "flipped_bearish": flipped_bearish,
        "entered_bull": entered_bull,
        "entered_bear": entered_bear,
    }


def simulate_qbands(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
) -> Dict:
    """
    Simulate QBands indicator: Bollinger Bands + Keltner Channels with squeeze detection.

    Returns dict with:
        - basis: EMA center line
        - bb_upper/bb_lower: 2σ Bollinger Bands
        - kc_upper/kc_lower: Keltner Channel bands
        - squeeze_on: bool array (BB inside KC = volatility compressed)
        - squeeze_started: bool array (squeeze just began)
        - squeeze_fired: bool array (squeeze just released)
        - squeeze_fire_bullish: bool array (fired with upward momentum)
        - squeeze_fire_bearish: bool array (fired with downward momentum)
        - squeeze_bars: int array (consecutive bars in squeeze)
        - bandwidth: float array (BB width as % of price)
        - band_position: float array (0=lower band, 1=upper band)
        - upper_touch: bool array (price touched upper 2σ)
        - lower_touch: bool array (price touched lower 2σ)
        - upper_touch_quality: int array (3=wick, 2=body, 1=close beyond, 0=none)
        - lower_touch_quality: int array (same scoring)
    """
    n = len(close)

    # --- EMA basis ---
    basis = compute_ema(close, bb_length)

    # --- Standard deviation for Bollinger Bands ---
    stdev = np.zeros(n)
    for i in range(bb_length - 1, n):
        window = close[i - bb_length + 1: i + 1]
        stdev[i] = np.std(window, ddof=0)  # population stdev (matches Pine)

    bb_upper = basis + bb_mult * stdev
    bb_lower = basis - bb_mult * stdev

    # --- ATR for Keltner Channels ---
    atr = compute_atr(high, low, close, bb_length)
    kc_upper = basis + kc_mult * atr
    kc_lower = basis - kc_mult * atr

    # --- Squeeze detection ---
    squeeze_on = np.zeros(n, dtype=bool)
    squeeze_started = np.zeros(n, dtype=bool)
    squeeze_fired = np.zeros(n, dtype=bool)
    squeeze_fire_bullish = np.zeros(n, dtype=bool)
    squeeze_fire_bearish = np.zeros(n, dtype=bool)
    squeeze_bars_arr = np.zeros(n, dtype=int)

    for i in range(bb_length, n):
        # BB inside KC = squeeze ON
        if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
            squeeze_on[i] = True

        # State transitions
        if i > 0:
            if squeeze_on[i] and not squeeze_on[i - 1]:
                squeeze_started[i] = True
            if not squeeze_on[i] and squeeze_on[i - 1]:
                squeeze_fired[i] = True
                # Fire direction based on momentum
                momentum = close[i] - basis[i]
                if momentum > 0:
                    squeeze_fire_bullish[i] = True
                else:
                    squeeze_fire_bearish[i] = True

        # Squeeze duration
        if squeeze_on[i]:
            squeeze_bars_arr[i] = squeeze_bars_arr[i - 1] + 1 if i > 0 else 1

    # --- Bandwidth (BB width as % of price) ---
    bandwidth = np.zeros(n)
    for i in range(bb_length, n):
        if basis[i] > 0:
            bandwidth[i] = (bb_upper[i] - bb_lower[i]) / basis[i] * 100.0

    # --- Band position (0 = lower, 0.5 = basis, 1 = upper) ---
    band_position = np.full(n, 0.5)
    for i in range(bb_length, n):
        band_range = bb_upper[i] - bb_lower[i]
        if band_range > 0:
            band_position[i] = (close[i] - bb_lower[i]) / band_range
            band_position[i] = max(0.0, min(1.0, band_position[i]))

    # --- Band touch detection with quality scoring ---
    upper_touch = np.zeros(n, dtype=bool)
    lower_touch = np.zeros(n, dtype=bool)
    upper_touch_quality = np.zeros(n, dtype=int)
    lower_touch_quality = np.zeros(n, dtype=int)

    open_prices = close.copy()  # Approximate open with close[i-1] where available
    # We don't have open in our data model, so we approximate:
    # wick-only = high touched band but close stayed inside
    # body = close crossed band
    # close beyond = close ended beyond band

    for i in range(bb_length, n):
        # Upper band touches
        if high[i] >= bb_upper[i]:
            upper_touch[i] = True
            if close[i] < bb_upper[i]:
                upper_touch_quality[i] = 3  # Wick-only (best)
            else:
                upper_touch_quality[i] = 1  # Close beyond (worst)
        elif close[i] >= bb_upper[i] * 0.998:  # Within 0.2% of band
            upper_touch[i] = True
            upper_touch_quality[i] = 2  # Near touch

        # Lower band touches
        if low[i] <= bb_lower[i]:
            lower_touch[i] = True
            if close[i] > bb_lower[i]:
                lower_touch_quality[i] = 3  # Wick-only (best)
            else:
                lower_touch_quality[i] = 1  # Close beyond (worst)
        elif close[i] <= bb_lower[i] * 1.002:  # Within 0.2% of band
            lower_touch[i] = True
            lower_touch_quality[i] = 2  # Near touch

    return {
        "basis": basis,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "kc_upper": kc_upper,
        "kc_lower": kc_lower,
        "squeeze_on": squeeze_on,
        "squeeze_started": squeeze_started,
        "squeeze_fired": squeeze_fired,
        "squeeze_fire_bullish": squeeze_fire_bullish,
        "squeeze_fire_bearish": squeeze_fire_bearish,
        "squeeze_bars": squeeze_bars_arr,
        "bandwidth": bandwidth,
        "band_position": band_position,
        "upper_touch": upper_touch,
        "lower_touch": lower_touch,
        "upper_touch_quality": upper_touch_quality,
        "lower_touch_quality": lower_touch_quality,
    }


def simulate_moneyball(
    close: np.ndarray,
    volume: np.ndarray,
    roc_length: int = 10,
    smooth_length: int = 3,
    norm_length: int = 50,
    extreme_threshold: float = 70.0,
) -> Dict:
    """
    Simulate Moneyball indicator: Volume-weighted momentum oscillator (-100 to +100).

    Pipeline: ROC → Volume Weighting → EMA Smoothing → Normalization

    Returns dict with:
        - moneyball: float array (-100 to +100)
        - zone: int array (1=Strong Bear, 2=Bear, 3=Neutral, 4=Bull, 5=Strong Bull)
        - flipped_bullish: bool array (crossed above zero)
        - flipped_bearish: bool array (crossed below zero)
        - entered_strong_bull: bool array (entered ±70 zone)
        - entered_strong_bear: bool array
        - vol_weight: float array (volume weight factor)
    """
    n = len(close)

    # --- Step 1: Rate of Change ---
    roc = np.zeros(n)
    for i in range(roc_length, n):
        if close[i - roc_length] > 0:
            roc[i] = (close[i] - close[i - roc_length]) / close[i - roc_length] * 100.0

    # --- Step 2: Volume Weighting ---
    vol_sma = np.zeros(n)
    # Simple moving average of volume
    for i in range(roc_length - 1, n):
        vol_sma[i] = np.mean(volume[i - roc_length + 1: i + 1])

    vol_weight = np.ones(n)
    weighted_roc = np.zeros(n)
    for i in range(roc_length, n):
        if vol_sma[i] > 0:
            ratio = volume[i] / vol_sma[i]
            vol_weight[i] = max(0.5, min(2.0, ratio))
        weighted_roc[i] = roc[i] * vol_weight[i]

    # --- Step 3: EMA Smoothing ---
    smoothed = compute_ema(weighted_roc, smooth_length)

    # --- Step 4: Normalization to -100/+100 ---
    moneyball = np.zeros(n)
    for i in range(norm_length, n):
        window = smoothed[i - norm_length + 1: i + 1]
        highest_abs = max(abs(np.max(window)), abs(np.min(window)))
        if highest_abs > 0:
            moneyball[i] = (smoothed[i] / highest_abs) * 100.0
        moneyball[i] = max(-100.0, min(100.0, moneyball[i]))

    # --- Zone classification ---
    zone = np.full(n, 3, dtype=int)  # Default neutral
    for i in range(n):
        if moneyball[i] > extreme_threshold:
            zone[i] = 5
        elif moneyball[i] > 30:
            zone[i] = 4
        elif moneyball[i] > -30:
            zone[i] = 3
        elif moneyball[i] > -extreme_threshold:
            zone[i] = 2
        else:
            zone[i] = 1

    # --- Zero-line flips ---
    flipped_bullish = np.zeros(n, dtype=bool)
    flipped_bearish = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if moneyball[i] > 0 and moneyball[i - 1] <= 0:
            flipped_bullish[i] = True
        if moneyball[i] < 0 and moneyball[i - 1] >= 0:
            flipped_bearish[i] = True

    # --- Extreme zone entries ---
    entered_strong_bull = np.zeros(n, dtype=bool)
    entered_strong_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if zone[i] == 5 and zone[i - 1] != 5:
            entered_strong_bull[i] = True
        if zone[i] == 1 and zone[i - 1] != 1:
            entered_strong_bear[i] = True

    return {
        "moneyball": moneyball,
        "zone": zone,
        "flipped_bullish": flipped_bullish,
        "flipped_bearish": flipped_bearish,
        "entered_strong_bull": entered_strong_bull,
        "entered_strong_bear": entered_strong_bear,
        "vol_weight": vol_weight,
    }


def simulate_qmomentum(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    rsi_length: int = 14,
    stoch_length: int = 14,
    ob_level: float = 70.0,
    os_level: float = 30.0,
) -> Dict:
    """
    Simulate QMomentum indicator: RSI with zone tracking.

    Returns dict with:
        - rsi: float array (0-100)
        - zone: int array (1=Extreme OS, 2=OS, 3=Neutral, 4=OB, 5=Extreme OB)
        - crossed_above_50: bool array (RSI crossed above midline)
        - crossed_below_50: bool array (RSI crossed below midline)
        - entered_ob: bool array (entered overbought)
        - entered_os: bool array (entered oversold)
        - left_ob: bool array (left overbought — potential short)
        - left_os: bool array (left oversold — potential long)
    """
    n = len(close)

    # --- RSI calculation using Wilder smoothing ---
    rsi = np.full(n, 50.0)
    gains = np.zeros(n)
    losses = np.zeros(n)

    for i in range(1, n):
        change = close[i] - close[i - 1]
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)

    # Initial average (SMA for first rsi_length bars)
    if n > rsi_length:
        avg_gain = np.mean(gains[1: rsi_length + 1])
        avg_loss = np.mean(losses[1: rsi_length + 1])

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[rsi_length] = 100.0 - (100.0 / (1.0 + rs))

        # Wilder smoothing for remaining bars
        for i in range(rsi_length + 1, n):
            avg_gain = (avg_gain * (rsi_length - 1) + gains[i]) / rsi_length
            avg_loss = (avg_loss * (rsi_length - 1) + losses[i]) / rsi_length
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                rsi[i] = 100.0

    # --- Zone classification ---
    zone = np.full(n, 3, dtype=int)
    for i in range(n):
        if rsi[i] > 80:
            zone[i] = 5
        elif rsi[i] > ob_level:
            zone[i] = 4
        elif rsi[i] > os_level:
            zone[i] = 3
        elif rsi[i] > 20:
            zone[i] = 2
        else:
            zone[i] = 1

    # --- Midline crossings (used as flip proxy) ---
    crossed_above_50 = np.zeros(n, dtype=bool)
    crossed_below_50 = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if rsi[i] > 50 and rsi[i - 1] <= 50:
            crossed_above_50[i] = True
        if rsi[i] < 50 and rsi[i - 1] >= 50:
            crossed_below_50[i] = True

    # --- OB/OS events ---
    entered_ob = np.zeros(n, dtype=bool)
    entered_os = np.zeros(n, dtype=bool)
    left_ob = np.zeros(n, dtype=bool)
    left_os = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if rsi[i] >= ob_level and rsi[i - 1] < ob_level:
            entered_ob[i] = True
        if rsi[i] <= os_level and rsi[i - 1] > os_level:
            entered_os[i] = True
        if rsi[i] < ob_level and rsi[i - 1] >= ob_level:
            left_ob[i] = True
        if rsi[i] > os_level and rsi[i - 1] <= os_level:
            left_os[i] = True

    return {
        "rsi": rsi,
        "zone": zone,
        "crossed_above_50": crossed_above_50,
        "crossed_below_50": crossed_below_50,
        "entered_ob": entered_ob,
        "entered_os": entered_os,
        "left_ob": left_ob,
        "left_os": left_os,
    }


def simulate_qcvd(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    smooth_length: int = 3,
    trend_length: int = 14,
) -> Dict:
    """
    Simulate QCVD indicator: Cumulative Volume Delta.

    Returns dict with:
        - cvd: float array (cumulative volume delta)
        - smoothed_delta: float array (per-bar smoothed delta)
        - flipped_bullish: bool array (CVD crossed above SMA)
        - flipped_bearish: bool array (CVD crossed below SMA)
    """
    n = len(close)

    # --- Per-bar delta ---
    raw_delta = np.zeros(n)
    for i in range(n):
        bar_range = high[i] - low[i]
        if bar_range > 0:
            raw_delta[i] = volume[i] * (2.0 * close[i] - high[i] - low[i]) / bar_range

    # --- Smoothed delta ---
    smoothed_delta = compute_ema(raw_delta, smooth_length) if smooth_length > 1 else raw_delta.copy()

    # --- CVD ---
    cvd = np.cumsum(smoothed_delta)

    # --- CVD trend (SMA) ---
    cvd_sma = np.zeros(n)
    for i in range(trend_length - 1, n):
        cvd_sma[i] = np.mean(cvd[i - trend_length + 1: i + 1])

    cvd_bullish = cvd > cvd_sma

    # --- Trend flips ---
    flipped_bullish = np.zeros(n, dtype=bool)
    flipped_bearish = np.zeros(n, dtype=bool)
    for i in range(trend_length, n):
        if cvd_bullish[i] and not cvd_bullish[i - 1]:
            flipped_bullish[i] = True
        if not cvd_bullish[i] and cvd_bullish[i - 1]:
            flipped_bearish[i] = True

    return {
        "cvd": cvd,
        "smoothed_delta": smoothed_delta,
        "cvd_sma": cvd_sma,
        "cvd_bullish": cvd_bullish,
        "flipped_bullish": flipped_bullish,
        "flipped_bearish": flipped_bearish,
    }


def simulate_qsmc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_length: int = 5,
    ob_strength: float = 1.5,
    atr_length: int = 14,
) -> Dict:
    """
    Simulate QSMC: BOS/CHoCH detection based on swing breaks.

    Returns dict with:
        - structure: int array (1=uptrend, -1=downtrend, 0=ranging)
        - bos_bull/bos_bear: bool arrays for BOS signals
        - choch_bull/choch_bear: bool arrays for CHoCH signals
    """
    n = len(close)

    # --- Swing detection using simple pivot ---
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in range(swing_length, n - swing_length):
        is_ph = True
        is_pl = True
        for j in range(1, swing_length + 1):
            if high[i] <= high[i - j] or high[i] <= high[i + j]:
                is_ph = False
            if low[i] >= low[i - j] or low[i] >= low[i + j]:
                is_pl = False
        if is_ph:
            swing_highs[i] = high[i]
        if is_pl:
            swing_lows[i] = low[i]

    # --- Structure tracking ---
    structure = np.zeros(n, dtype=int)
    bos_bull = np.zeros(n, dtype=bool)
    bos_bear = np.zeros(n, dtype=bool)
    choch_bull = np.zeros(n, dtype=bool)
    choch_bear = np.zeros(n, dtype=bool)

    last_sh = np.nan
    last_sl = np.nan
    current_structure = 0

    for i in range(swing_length * 2, n):
        # Update swing levels
        if not np.isnan(swing_highs[i]):
            last_sh = swing_highs[i]
        if not np.isnan(swing_lows[i]):
            last_sl = swing_lows[i]

        # BOS/CHoCH detection
        if not np.isnan(last_sh) and high[i] > last_sh and high[i - 1] <= last_sh:
            if current_structure >= 0:
                bos_bull[i] = True
            else:
                choch_bull[i] = True
            current_structure = 1

        if not np.isnan(last_sl) and low[i] < last_sl and low[i - 1] >= last_sl:
            if current_structure <= 0:
                bos_bear[i] = True
            else:
                choch_bear[i] = True
            current_structure = -1

        structure[i] = current_structure

    return {
        "structure": structure,
        "bos_bull": bos_bull,
        "bos_bear": bos_bear,
        "choch_bull": choch_bull,
        "choch_bear": choch_bear,
    }


def simulate_qgrid(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    left_bars: int = 5,
    right_bars: int = 3,
) -> Dict:
    """
    Simulate QGrid: Detect swing-based S/R levels and measure bounces/breaks.

    Returns dict with:
        - level_bounced: bool array (price tested and bounced from a level)
        - level_broken: bool array (price broke through a level)
        - total_levels_detected: int
    """
    n = len(close)

    # Detect pivots
    levels = []  # list of (price, bar_index, type) where type=1=resistance, -1=support

    for i in range(left_bars, n - right_bars):
        is_ph = True
        is_pl = True
        for j in range(1, left_bars + 1):
            if high[i] <= high[i - j]:
                is_ph = False
                break
        if is_ph:
            for j in range(1, right_bars + 1):
                if i + j < n and high[i] <= high[i + j]:
                    is_ph = False
                    break
        for j in range(1, left_bars + 1):
            if low[i] >= low[i - j]:
                is_pl = False
                break
        if is_pl:
            for j in range(1, right_bars + 1):
                if i + j < n and low[i] >= low[i + j]:
                    is_pl = False
                    break

        if is_ph:
            levels.append((high[i], i, 1))
        if is_pl:
            levels.append((low[i], i, -1))

    # Measure bounces and breaks at levels
    level_bounced = np.zeros(n, dtype=bool)
    level_broken = np.zeros(n, dtype=bool)

    # Simple ATR for proximity
    atr = np.zeros(n)
    for i in range(1, n):
        atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n > 14:
        for i in range(14, n):
            atr[i] = np.mean(atr[i-13:i+1])

    for lvl_price, lvl_bar, lvl_type in levels:
        # Check bars after the level was created
        for i in range(lvl_bar + right_bars + 1, min(lvl_bar + 100, n)):
            proximity = 0.3 * atr[i] if atr[i] > 0 else 0.01
            if lvl_type == 1:  # Resistance
                if high[i] >= lvl_price - proximity and close[i] < lvl_price:
                    level_bounced[i] = True
                    break
                elif close[i] > lvl_price and close[i-1] <= lvl_price:
                    level_broken[i] = True
                    break
            else:  # Support
                if low[i] <= lvl_price + proximity and close[i] > lvl_price:
                    level_bounced[i] = True
                    break
                elif close[i] < lvl_price and close[i-1] >= lvl_price:
                    level_broken[i] = True
                    break

    return {
        "level_bounced": level_bounced,
        "level_broken": level_broken,
        "total_levels_detected": len(levels),
    }


# ============================================================================
# SIGNAL QUALITY METRICS
# ============================================================================

def measure_flip_accuracy(
    close: np.ndarray,
    flipped_bullish: np.ndarray,
    flipped_bearish: np.ndarray,
    lookahead: int,
) -> float:
    """
    Measure flip accuracy: when a flip occurs, does price move in that direction
    within the next N bars?

    Returns accuracy as float 0.0-1.0.
    """
    n = len(close)
    correct = 0
    total = 0

    for i in range(n):
        if i + lookahead >= n:
            break

        if flipped_bullish[i]:
            total += 1
            # Check if price went up within lookahead
            future_high = np.max(close[i + 1: i + 1 + lookahead])
            if future_high > close[i]:
                correct += 1

        elif flipped_bearish[i]:
            total += 1
            # Check if price went down within lookahead
            future_low = np.min(close[i + 1: i + 1 + lookahead])
            if future_low < close[i]:
                correct += 1

    if total == 0:
        return 0.0
    return correct / total


def measure_whipsaw_rate(
    flipped_bullish: np.ndarray,
    flipped_bearish: np.ndarray,
    threshold: int = 5,
) -> float:
    """
    Measure whipsaw rate: percentage of flips that reverse within N bars.

    Returns rate as float 0.0-1.0 (lower is better).
    """
    n = len(flipped_bullish)
    whipsaws = 0
    total_flips = 0

    for i in range(n):
        if flipped_bullish[i] or flipped_bearish[i]:
            total_flips += 1
            # Check if opposite flip occurs within threshold bars
            for j in range(i + 1, min(i + 1 + threshold, n)):
                if flipped_bullish[i] and flipped_bearish[j]:
                    whipsaws += 1
                    break
                if flipped_bearish[i] and flipped_bullish[j]:
                    whipsaws += 1
                    break

    if total_flips == 0:
        return 1.0  # No flips = worst case
    return whipsaws / total_flips


def measure_avg_trend_duration(
    flipped_bullish: np.ndarray,
    flipped_bearish: np.ndarray,
) -> float:
    """
    Average number of bars between flips (longer trends = less noise).
    """
    flip_indices = []
    for i in range(len(flipped_bullish)):
        if flipped_bullish[i] or flipped_bearish[i]:
            flip_indices.append(i)

    if len(flip_indices) < 2:
        return 0.0

    durations = []
    for i in range(1, len(flip_indices)):
        durations.append(flip_indices[i] - flip_indices[i - 1])

    return float(np.mean(durations)) if durations else 0.0


def measure_bounce_win_rate(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    bounce_bullish: np.ndarray,
    bounce_bearish: np.ndarray,
    atr: np.ndarray,
    lookahead: int,
) -> float:
    """
    When a bounce occurs, does price continue in trend direction within N bars?
    Uses 1x ATR as the "win" threshold.

    Returns win rate as float 0.0-1.0.
    """
    n = len(close)
    wins = 0
    total = 0

    for i in range(n):
        if i + lookahead >= n:
            break
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        target = atr[i] * 1.0  # 1x ATR = "win"

        if bounce_bullish[i]:
            total += 1
            future_high = np.max(high[i + 1: i + 1 + lookahead])
            if future_high >= close[i] + target:
                wins += 1

        elif bounce_bearish[i]:
            total += 1
            future_low = np.min(low[i + 1: i + 1 + lookahead])
            if future_low <= close[i] - target:
                wins += 1

    if total == 0:
        return 0.0
    return wins / total


def measure_squeeze_accuracy(
    close: np.ndarray,
    squeeze_started: np.ndarray,
    lookahead: int = 20,
    move_threshold_pct: float = 0.5,
) -> float:
    """
    When squeeze fires, does a significant directional move follow?
    'Significant' = price moves > move_threshold_pct% within lookahead bars.
    """
    n = len(close)
    correct = 0
    total = 0

    for i in range(n):
        if i + lookahead >= n:
            break
        if squeeze_started[i]:
            total += 1
            future_max = np.max(close[i + 1: i + 1 + lookahead])
            future_min = np.min(close[i + 1: i + 1 + lookahead])
            up_move = (future_max - close[i]) / close[i] * 100
            down_move = (close[i] - future_min) / close[i] * 100
            if up_move >= move_threshold_pct or down_move >= move_threshold_pct:
                correct += 1

    if total == 0:
        return 0.0
    return correct / total


def measure_zone_crossing_accuracy(
    close: np.ndarray,
    entered_bull: np.ndarray,
    entered_bear: np.ndarray,
    lookahead: int,
) -> float:
    """
    When QWave enters Bull zone (>+30), does price continue up?
    When QWave enters Bear zone (<-30), does price continue down?

    Returns accuracy as float 0.0-1.0.
    """
    n = len(close)
    correct = 0
    total = 0

    for i in range(n):
        if i + lookahead >= n:
            break

        if entered_bull[i]:
            total += 1
            future_high = np.max(close[i + 1: i + 1 + lookahead])
            if future_high > close[i]:
                correct += 1

        elif entered_bear[i]:
            total += 1
            future_low = np.min(close[i + 1: i + 1 + lookahead])
            if future_low < close[i]:
                correct += 1

    if total == 0:
        return 0.0
    return correct / total


def measure_band_touch_reversal(
    close: np.ndarray,
    upper_touch: np.ndarray,
    lower_touch: np.ndarray,
    upper_quality: np.ndarray,
    lower_quality: np.ndarray,
    basis: np.ndarray,
    lookahead: int = 10,
    min_quality: int = 2,
) -> Tuple[float, int]:
    """
    When price touches the 2σ band (quality >= min_quality), does it revert toward basis?

    Upper touch → price should drop toward basis
    Lower touch → price should rise toward basis

    Returns (reversal_rate, total_touches).
    """
    n = len(close)
    correct = 0
    total = 0

    for i in range(n):
        if i + lookahead >= n:
            break

        if upper_touch[i] and upper_quality[i] >= min_quality:
            total += 1
            # Check if price moves toward basis within lookahead
            future_min = np.min(close[i + 1: i + 1 + lookahead])
            distance_to_basis = close[i] - basis[i]
            if distance_to_basis > 0:
                reversion = (close[i] - future_min) / distance_to_basis
                if reversion >= 0.5:  # Reverted at least 50% toward basis
                    correct += 1

        elif lower_touch[i] and lower_quality[i] >= min_quality:
            total += 1
            # Check if price moves toward basis within lookahead
            future_max = np.max(close[i + 1: i + 1 + lookahead])
            distance_to_basis = basis[i] - close[i]
            if distance_to_basis > 0:
                reversion = (future_max - close[i]) / distance_to_basis
                if reversion >= 0.5:
                    correct += 1

    rate = correct / total if total > 0 else 0.0
    return rate, total


def measure_squeeze_fire_accuracy(
    close: np.ndarray,
    squeeze_fire_bullish: np.ndarray,
    squeeze_fire_bearish: np.ndarray,
    lookahead: int = 15,
) -> Tuple[float, int]:
    """
    When squeeze fires, does price continue in the fire direction?

    Bullish fire → price should go up within lookahead
    Bearish fire → price should go down within lookahead

    Returns (accuracy, total_fires).
    """
    n = len(close)
    correct = 0
    total = 0

    for i in range(n):
        if i + lookahead >= n:
            break

        if squeeze_fire_bullish[i]:
            total += 1
            future_max = np.max(close[i + 1: i + 1 + lookahead])
            if future_max > close[i]:
                correct += 1

        elif squeeze_fire_bearish[i]:
            total += 1
            future_min = np.min(close[i + 1: i + 1 + lookahead])
            if future_min < close[i]:
                correct += 1

    rate = correct / total if total > 0 else 0.0
    return rate, total


def measure_squeeze_whipsaw_rate(
    squeeze_fired: np.ndarray,
    squeeze_fire_bullish: np.ndarray,
    squeeze_fire_bearish: np.ndarray,
    close: np.ndarray,
    threshold_bars: int = 5,
) -> float:
    """
    How often does a squeeze fire lead to a quick reversal (whipsaw)?
    A squeeze fire that reverses direction within threshold_bars = whipsaw.
    """
    n = len(close)
    total = 0
    whipsaws = 0

    for i in range(n):
        if not squeeze_fired[i]:
            continue
        if i + threshold_bars >= n:
            break

        total += 1
        is_bull = squeeze_fire_bullish[i]

        # Check if the move reverses within threshold
        if is_bull:
            # Bull fire whipsaw = price drops below entry within threshold
            entry_price = close[i]
            future_min = np.min(close[i + 1: i + 1 + threshold_bars])
            if future_min < entry_price * 0.998:  # 0.2% below entry
                whipsaws += 1
        else:
            # Bear fire whipsaw = price rises above entry within threshold
            entry_price = close[i]
            future_max = np.max(close[i + 1: i + 1 + threshold_bars])
            if future_max > entry_price * 1.002:
                whipsaws += 1

    if total == 0:
        return 0.0
    return whipsaws / total


# ============================================================================
# COMPOSITE SCORING
# ============================================================================

def compute_composite_score(
    flip_accuracy: float,
    whipsaw_rate: float,
    avg_duration: float,
    max_duration: float,
    extra_metric: float = 0.0,
    extra_weight: float = 0.0,
) -> float:
    """
    Composite score for ranking parameter combinations.

    Score = (flip_accuracy × 0.35) + ((1 - whipsaw_rate) × 0.25)
          + (normalized_duration × 0.25) + (extra_metric × 0.15)

    extra_metric: squeeze accuracy for QCloud, bounce win rate for QLine
    """
    # Normalize duration: 0.0 (worst) to 1.0 (best)
    norm_duration = min(avg_duration / max_duration, 1.0) if max_duration > 0 else 0.0

    base_weight = 0.40 if extra_weight == 0.0 else 0.35
    anti_whipsaw_weight = 0.30 if extra_weight == 0.0 else 0.25
    duration_weight = 0.30 if extra_weight == 0.0 else 0.25
    extra_w = extra_weight if extra_weight > 0 else 0.0

    score = (
        flip_accuracy * base_weight
        + (1.0 - whipsaw_rate) * anti_whipsaw_weight
        + norm_duration * duration_weight
        + extra_metric * extra_w
    )
    return score


# ============================================================================
# OPTIMIZATION RUNNERS
# ============================================================================

def optimize_qcloud(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    params: Dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run QCloud optimization for a single symbol/timeframe.
    Tests all parameter combinations and returns ranked results.
    """
    close = df["close"].values
    n = len(close)

    if n < 200:
        print(f"    SKIP: Only {n} bars (need >=200)")
        return pd.DataFrame()

    combos = list(product(
        params["base_length"],
        params["step_mult"],
        params["smoothing"],
    ))
    total = len(combos)

    if verbose:
        print(f"    Testing {total} QCloud parameter combinations on {n:,} bars...")

    results = []
    all_durations = []  # Collect all durations for normalization

    for idx, (bl, sm, smooth) in enumerate(combos):
        if verbose and (idx + 1) % 20 == 0:
            print(f"      Progress: {idx + 1}/{total}")

        try:
            sim = simulate_qcloud(close, bl, sm, smooth)

            # Measure metrics across multiple lookahead values
            flip_accs = []
            for la in LOOKAHEAD_BARS:
                acc = measure_flip_accuracy(
                    close, sim["flipped_bullish"], sim["flipped_bearish"], la
                )
                flip_accs.append(acc)
            avg_flip_acc = float(np.mean(flip_accs))

            whipsaw = measure_whipsaw_rate(
                sim["flipped_bullish"], sim["flipped_bearish"], WHIPSAW_THRESHOLD
            )

            avg_dur = measure_avg_trend_duration(
                sim["flipped_bullish"], sim["flipped_bearish"]
            )
            all_durations.append(avg_dur)

            squeeze_acc = measure_squeeze_accuracy(close, sim["squeeze_started"])

            total_flips = int(np.sum(sim["flipped_bullish"]) + np.sum(sim["flipped_bearish"]))
            total_squeezes = int(np.sum(sim["squeeze_started"]))

            results.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "base_length": bl,
                "step_mult": sm,
                "smoothing": smooth,
                "flip_accuracy": avg_flip_acc,
                "whipsaw_rate": whipsaw,
                "avg_trend_duration": avg_dur,
                "squeeze_accuracy": squeeze_acc,
                "total_flips": total_flips,
                "total_squeezes": total_squeezes,
                "bars_tested": n,
            })
        except Exception as e:
            if verbose:
                print(f"      ERROR with {bl}/{sm}/{smooth}: {e}")

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Filter: need at least 20 flips for statistical meaning
    results_df = results_df[results_df["total_flips"] >= 20].copy()

    if results_df.empty:
        print(f"    WARNING: No combos produced >= 20 flips")
        return pd.DataFrame()

    # Compute composite score
    max_dur = results_df["avg_trend_duration"].max()
    results_df["composite_score"] = results_df.apply(
        lambda row: compute_composite_score(
            row["flip_accuracy"],
            row["whipsaw_rate"],
            row["avg_trend_duration"],
            max_dur,
            row["squeeze_accuracy"],
            0.15,
        ),
        axis=1,
    )

    results_df = results_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    if verbose and len(results_df) > 0:
        top = results_df.iloc[0]
        print(f"    BEST: base={int(top['base_length'])}, step={top['step_mult']:.1f}, "
              f"smooth={int(top['smoothing'])} | "
              f"FlipAcc={top['flip_accuracy']:.1%}, Whipsaw={top['whipsaw_rate']:.1%}, "
              f"AvgDur={top['avg_trend_duration']:.0f}, Score={top['composite_score']:.3f}")

    return results_df


def optimize_qline(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    params: Dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run QLine optimization for a single symbol/timeframe.
    Tests all parameter combinations and returns ranked results.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    if n < 200:
        print(f"    SKIP: Only {n} bars (need >=200)")
        return pd.DataFrame()

    combos = list(product(
        params["atr_length"],
        params["factor"],
    ))
    total = len(combos)

    if verbose:
        print(f"    Testing {total} QLine parameter combinations on {n:,} bars...")

    results = []
    all_durations = []

    for idx, (al, fac) in enumerate(combos):
        if verbose and (idx + 1) % 10 == 0:
            print(f"      Progress: {idx + 1}/{total}")

        try:
            sim = simulate_qline(high, low, close, al, fac)

            # Flip accuracy across multiple lookaheads
            flip_accs = []
            for la in LOOKAHEAD_BARS:
                acc = measure_flip_accuracy(
                    close, sim["flipped_bullish"], sim["flipped_bearish"], la
                )
                flip_accs.append(acc)
            avg_flip_acc = float(np.mean(flip_accs))

            whipsaw = measure_whipsaw_rate(
                sim["flipped_bullish"], sim["flipped_bearish"], WHIPSAW_THRESHOLD
            )

            avg_dur = measure_avg_trend_duration(
                sim["flipped_bullish"], sim["flipped_bearish"]
            )
            all_durations.append(avg_dur)

            # Bounce win rate across multiple lookaheads
            bounce_wrs = []
            for la in BOUNCE_LOOKAHEAD:
                bwr = measure_bounce_win_rate(
                    close, high, low,
                    sim["bounce_bullish"], sim["bounce_bearish"],
                    sim["atr"], la,
                )
                bounce_wrs.append(bwr)
            avg_bounce_wr = float(np.mean(bounce_wrs))

            total_flips = int(np.sum(sim["flipped_bullish"]) + np.sum(sim["flipped_bearish"]))
            total_bounces = int(np.sum(sim["bounce_bullish"]) + np.sum(sim["bounce_bearish"]))

            results.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "atr_length": al,
                "factor": fac,
                "flip_accuracy": avg_flip_acc,
                "whipsaw_rate": whipsaw,
                "avg_trend_duration": avg_dur,
                "bounce_win_rate": avg_bounce_wr,
                "total_flips": total_flips,
                "total_bounces": total_bounces,
                "bars_tested": n,
            })
        except Exception as e:
            if verbose:
                print(f"      ERROR with {al}/{fac}: {e}")

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Filter: need at least 20 flips
    results_df = results_df[results_df["total_flips"] >= 20].copy()

    if results_df.empty:
        print(f"    WARNING: No combos produced >= 20 flips")
        return pd.DataFrame()

    # Composite score with bounce win rate as extra metric
    max_dur = results_df["avg_trend_duration"].max()
    results_df["composite_score"] = results_df.apply(
        lambda row: compute_composite_score(
            row["flip_accuracy"],
            row["whipsaw_rate"],
            row["avg_trend_duration"],
            max_dur,
            row["bounce_win_rate"],
            0.15,
        ),
        axis=1,
    )

    results_df = results_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    if verbose and len(results_df) > 0:
        top = results_df.iloc[0]
        print(f"    BEST: atr={int(top['atr_length'])}, factor={top['factor']:.1f} | "
              f"FlipAcc={top['flip_accuracy']:.1%}, Whipsaw={top['whipsaw_rate']:.1%}, "
              f"BounceWR={top['bounce_win_rate']:.1%}, Score={top['composite_score']:.3f}")

    return results_df


def optimize_qwave(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    params: Dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run QWave optimization for a single symbol/timeframe.
    Tests all parameter combinations and returns ranked results.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    if n < 200:
        print(f"    SKIP: Only {n} bars (need >=200)")
        return pd.DataFrame()

    combos = list(product(
        params["adx_length"],
        params["smoothing"],
    ))
    total = len(combos)

    if verbose:
        print(f"    Testing {total} QWave parameter combinations on {n:,} bars...")

    results = []
    all_durations = []

    for idx, (al, sm) in enumerate(combos):
        if verbose and (idx + 1) % 10 == 0:
            print(f"      Progress: {idx + 1}/{total}")

        try:
            sim = simulate_qwave(high, low, close, al, sm)

            # Flip accuracy across multiple lookaheads
            flip_accs = []
            for la in LOOKAHEAD_BARS:
                acc = measure_flip_accuracy(
                    close, sim["flipped_bullish"], sim["flipped_bearish"], la
                )
                flip_accs.append(acc)
            avg_flip_acc = float(np.mean(flip_accs))

            whipsaw = measure_whipsaw_rate(
                sim["flipped_bullish"], sim["flipped_bearish"], WHIPSAW_THRESHOLD
            )

            avg_dur = measure_avg_trend_duration(
                sim["flipped_bullish"], sim["flipped_bearish"]
            )
            all_durations.append(avg_dur)

            # Zone crossing accuracy across multiple lookaheads
            zone_accs = []
            for la in LOOKAHEAD_BARS:
                zacc = measure_zone_crossing_accuracy(
                    close, sim["entered_bull"], sim["entered_bear"], la
                )
                zone_accs.append(zacc)
            avg_zone_acc = float(np.mean(zone_accs))

            total_flips = int(np.sum(sim["flipped_bullish"]) + np.sum(sim["flipped_bearish"]))
            total_zone_entries = int(np.sum(sim["entered_bull"]) + np.sum(sim["entered_bear"]))

            results.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "adx_length": al,
                "smoothing": sm,
                "flip_accuracy": avg_flip_acc,
                "whipsaw_rate": whipsaw,
                "avg_trend_duration": avg_dur,
                "zone_crossing_accuracy": avg_zone_acc,
                "total_flips": total_flips,
                "total_zone_entries": total_zone_entries,
                "bars_tested": n,
            })
        except Exception as e:
            if verbose:
                print(f"      ERROR with adx={al}/smooth={sm}: {e}")

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Filter: need at least 20 flips
    results_df = results_df[results_df["total_flips"] >= 20].copy()

    if results_df.empty:
        print(f"    WARNING: No combos produced >= 20 flips")
        return pd.DataFrame()

    # Composite score with zone crossing accuracy as extra metric
    max_dur = results_df["avg_trend_duration"].max()
    results_df["composite_score"] = results_df.apply(
        lambda row: compute_composite_score(
            row["flip_accuracy"],
            row["whipsaw_rate"],
            row["avg_trend_duration"],
            max_dur,
            row["zone_crossing_accuracy"],
            0.15,
        ),
        axis=1,
    )

    results_df = results_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    if verbose and len(results_df) > 0:
        top = results_df.iloc[0]
        print(f"    BEST: adx={int(top['adx_length'])}, smooth={int(top['smoothing'])} | "
              f"FlipAcc={top['flip_accuracy']:.1%}, Whipsaw={top['whipsaw_rate']:.1%}, "
              f"ZoneAcc={top['zone_crossing_accuracy']:.1%}, Score={top['composite_score']:.3f}")

    return results_df


def optimize_qbands(
    df: pd.DataFrame,
    symbol: str,
    tf_label: str,
    param_grid: Dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Optimize QBands parameters (bb_length, bb_mult, kc_mult) for a single symbol/timeframe.

    Quality metrics:
      1. Squeeze fire accuracy (does directional move follow?) — weight 0.30
      2. Band touch reversal rate (do 2σ touches revert?) — weight 0.25
      3. Squeeze whipsaw rate (false fires) — weight 0.25
      4. Avg squeeze duration (longer = better compression) — weight 0.10
      5. Squeeze frequency (too many = noisy, too few = useless) — weight 0.10
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    bb_lengths = param_grid["bb_length"]
    bb_mults = param_grid["bb_mult"]
    kc_mults = param_grid["kc_mult"]

    total_combos = len(bb_lengths) * len(bb_mults) * len(kc_mults)
    if verbose:
        print(f"    Testing {total_combos} parameter combinations on {n:,} bars...")

    results = []
    combo_count = 0

    for bb_length in bb_lengths:
        for bb_mult in bb_mults:
            for kc_mult_val in kc_mults:
                combo_count += 1

                sim = simulate_qbands(high, low, close, bb_length, bb_mult, kc_mult_val)

                # --- Squeeze fire accuracy ---
                fire_acc, total_fires = measure_squeeze_fire_accuracy(
                    close, sim["squeeze_fire_bullish"], sim["squeeze_fire_bearish"],
                    lookahead=15,
                )

                # --- Band touch reversal rate ---
                touch_rev, total_touches = measure_band_touch_reversal(
                    close, sim["upper_touch"], sim["lower_touch"],
                    sim["upper_touch_quality"], sim["lower_touch_quality"],
                    sim["basis"], lookahead=10, min_quality=2,
                )

                # --- Squeeze whipsaw rate ---
                sqz_whipsaw = measure_squeeze_whipsaw_rate(
                    sim["squeeze_fired"], sim["squeeze_fire_bullish"],
                    sim["squeeze_fire_bearish"], close, threshold_bars=5,
                )

                # --- Avg squeeze duration ---
                squeeze_durations = []
                current_dur = 0
                for i in range(n):
                    if sim["squeeze_on"][i]:
                        current_dur += 1
                    else:
                        if current_dur > 0:
                            squeeze_durations.append(current_dur)
                        current_dur = 0
                if current_dur > 0:
                    squeeze_durations.append(current_dur)
                avg_squeeze_dur = np.mean(squeeze_durations) if squeeze_durations else 0.0
                total_squeezes = len(squeeze_durations)

                # --- Squeeze frequency (squeezes per 1000 bars) ---
                squeeze_freq = total_squeezes / (n / 1000.0) if n > 0 else 0.0

                # --- Composite scoring ---
                # Fire accuracy: higher is better (0-1)
                fire_score = fire_acc

                # Touch reversal: higher is better (0-1)
                touch_score = touch_rev

                # Whipsaw: lower is better (invert)
                whipsaw_score = max(0.0, 1.0 - sqz_whipsaw)

                # Squeeze duration: moderate is ideal (8-25 bars)
                # Too short (<5) = noise, too long (>40) = useless
                if avg_squeeze_dur < 3:
                    dur_score = 0.2
                elif avg_squeeze_dur <= 8:
                    dur_score = 0.5 + 0.5 * ((avg_squeeze_dur - 3) / 5)
                elif avg_squeeze_dur <= 25:
                    dur_score = 1.0
                elif avg_squeeze_dur <= 40:
                    dur_score = 1.0 - 0.5 * ((avg_squeeze_dur - 25) / 15)
                else:
                    dur_score = 0.3

                # Frequency: moderate is ideal (5-30 per 1000 bars)
                if squeeze_freq < 2:
                    freq_score = 0.3
                elif squeeze_freq <= 5:
                    freq_score = 0.5 + 0.5 * ((squeeze_freq - 2) / 3)
                elif squeeze_freq <= 30:
                    freq_score = 1.0
                elif squeeze_freq <= 60:
                    freq_score = 1.0 - 0.5 * ((squeeze_freq - 30) / 30)
                else:
                    freq_score = 0.3

                # Weighted composite
                composite = (
                    fire_score * 0.30
                    + touch_score * 0.25
                    + whipsaw_score * 0.25
                    + dur_score * 0.10
                    + freq_score * 0.10
                )

                results.append({
                    "bb_length": bb_length,
                    "bb_mult": bb_mult,
                    "kc_mult": kc_mult_val,
                    "fire_accuracy": fire_acc,
                    "touch_reversal": touch_rev,
                    "squeeze_whipsaw": sqz_whipsaw,
                    "avg_squeeze_dur": avg_squeeze_dur,
                    "squeeze_freq": squeeze_freq,
                    "total_fires": total_fires,
                    "total_touches": total_touches,
                    "total_squeezes": total_squeezes,
                    "composite_score": composite,
                    "bars_tested": n,
                })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values("composite_score", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    if verbose and len(results_df) > 0:
        top = results_df.iloc[0]
        print(f"    BEST: bb_len={int(top['bb_length'])}, bb_mult={top['bb_mult']:.1f}, "
              f"kc_mult={top['kc_mult']:.1f} | "
              f"FireAcc={top['fire_accuracy']:.1%}, TouchRev={top['touch_reversal']:.1%}, "
              f"SqzWhip={top['squeeze_whipsaw']:.1%}, Score={top['composite_score']:.3f}")

    return results_df


def optimize_moneyball(
    df: pd.DataFrame,
    symbol: str,
    tf_label: str,
    param_grid: Dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Optimize Moneyball parameters (roc_length, smooth_length) for a single symbol/timeframe.

    Quality metrics:
      1. Flip accuracy (does price follow flip direction?) — weight 0.35
      2. Whipsaw rate (false flips) — weight 0.30
      3. Avg trend duration (bars same side of zero) — weight 0.20
      4. Extreme zone accuracy (does ±70 zone predict continuation?) — weight 0.15
    """
    close = df["close"].values
    volume = df["volume"].values
    n = len(close)

    roc_lengths = param_grid["roc_length"]
    smooth_lengths = param_grid["smooth_length"]

    total_combos = len(roc_lengths) * len(smooth_lengths)
    if verbose:
        print(f"    Testing {total_combos} parameter combinations on {n:,} bars...")

    results = []

    for roc_len in roc_lengths:
        for smooth_len in smooth_lengths:
            sim = simulate_moneyball(close, volume, roc_len, smooth_len)

            # --- Flip accuracy ---
            flip_acc_vals = []
            for lookahead in LOOKAHEAD_BARS:
                acc = measure_flip_accuracy(
                    close, sim["flipped_bullish"], sim["flipped_bearish"], lookahead
                )
                flip_acc_vals.append(acc)
            flip_acc = np.mean(flip_acc_vals) if flip_acc_vals else 0.0

            # --- Whipsaw rate ---
            whipsaw = measure_whipsaw_rate(
                sim["flipped_bullish"], sim["flipped_bearish"], WHIPSAW_THRESHOLD
            )

            # --- Avg trend duration ---
            # Using zero-line side as "trend"
            max_dur = max(n / 20, 50)
            avg_dur = measure_avg_trend_duration(
                sim["flipped_bullish"], sim["flipped_bearish"]
            )

            # --- Extreme zone accuracy ---
            zone_acc = measure_zone_crossing_accuracy(
                close, sim["entered_strong_bull"], sim["entered_strong_bear"],
                lookahead=10,
            )

            # --- Composite ---
            total_flips = int(np.sum(sim["flipped_bullish"]) + np.sum(sim["flipped_bearish"]))
            total_extreme = int(np.sum(sim["entered_strong_bull"]) + np.sum(sim["entered_strong_bear"]))

            composite = compute_composite_score(
                flip_accuracy=flip_acc,
                whipsaw_rate=whipsaw,
                avg_duration=avg_dur,
                max_duration=max_dur,
                extra_metric=zone_acc,
                extra_weight=0.15,
            )

            results.append({
                "roc_length": roc_len,
                "smooth_length": smooth_len,
                "flip_accuracy": flip_acc,
                "whipsaw_rate": whipsaw,
                "avg_trend_duration": avg_dur,
                "zone_accuracy": zone_acc,
                "composite_score": composite,
                "total_flips": total_flips,
                "total_extreme_entries": total_extreme,
                "bars_tested": n,
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values("composite_score", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    if verbose and len(results_df) > 0:
        top = results_df.iloc[0]
        print(f"    BEST: roc={int(top['roc_length'])}, smooth={int(top['smooth_length'])} | "
              f"FlipAcc={top['flip_accuracy']:.1%}, Whipsaw={top['whipsaw_rate']:.1%}, "
              f"ZoneAcc={top['zone_accuracy']:.1%}, Score={top['composite_score']:.3f}")

    return results_df


def optimize_qmomentum(
    df: pd.DataFrame,
    symbol: str,
    tf_label: str,
    param_grid: Dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Optimize QMomentum parameters (rsi_length, stoch_length) for a single symbol/timeframe.

    Quality metrics:
      1. Midline cross accuracy (RSI crossing 50 predicts direction) — weight 0.35
      2. Whipsaw rate (false midline crosses) — weight 0.25
      3. Avg trend duration (bars same side of 50) — weight 0.25
      4. Zone exit accuracy (leaving OB/OS predicts reversal) — weight 0.15
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    rsi_lengths = param_grid["rsi_length"]
    stoch_lengths = param_grid["stoch_length"]

    total_combos = len(rsi_lengths) * len(stoch_lengths)
    if verbose:
        print(f"    Testing {total_combos} parameter combinations on {n:,} bars...")

    results = []

    for rsi_len in rsi_lengths:
        for stoch_len in stoch_lengths:
            sim = simulate_qmomentum(high, low, close, rsi_len, stoch_len)

            # --- Midline cross accuracy (used as flip proxy) ---
            flip_acc_vals = []
            for lookahead in LOOKAHEAD_BARS:
                acc = measure_flip_accuracy(
                    close, sim["crossed_above_50"], sim["crossed_below_50"], lookahead
                )
                flip_acc_vals.append(acc)
            flip_acc = np.mean(flip_acc_vals) if flip_acc_vals else 0.0

            # --- Whipsaw rate ---
            whipsaw = measure_whipsaw_rate(
                sim["crossed_above_50"], sim["crossed_below_50"], WHIPSAW_THRESHOLD
            )

            # --- Avg trend duration (same side of 50) ---
            max_dur = max(n / 20, 50)
            avg_dur = measure_avg_trend_duration(
                sim["crossed_above_50"], sim["crossed_below_50"]
            )

            # --- Zone exit accuracy (left_os = bullish, left_ob = bearish) ---
            zone_exit_acc = measure_zone_crossing_accuracy(
                close, sim["left_os"], sim["left_ob"], lookahead=10,
            )

            total_crosses = int(np.sum(sim["crossed_above_50"]) + np.sum(sim["crossed_below_50"]))
            total_zone_exits = int(np.sum(sim["left_ob"]) + np.sum(sim["left_os"]))

            composite = compute_composite_score(
                flip_accuracy=flip_acc,
                whipsaw_rate=whipsaw,
                avg_duration=avg_dur,
                max_duration=max_dur,
                extra_metric=zone_exit_acc,
                extra_weight=0.15,
            )

            results.append({
                "rsi_length": rsi_len,
                "stoch_length": stoch_len,
                "cross_accuracy": flip_acc,
                "whipsaw_rate": whipsaw,
                "avg_trend_duration": avg_dur,
                "zone_exit_accuracy": zone_exit_acc,
                "composite_score": composite,
                "total_crosses": total_crosses,
                "total_zone_exits": total_zone_exits,
                "bars_tested": n,
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values("composite_score", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    if verbose and len(results_df) > 0:
        top = results_df.iloc[0]
        print(f"    BEST: rsi={int(top['rsi_length'])}, stoch={int(top['stoch_length'])} | "
              f"CrossAcc={top['cross_accuracy']:.1%}, Whipsaw={top['whipsaw_rate']:.1%}, "
              f"ZoneExitAcc={top['zone_exit_accuracy']:.1%}, Score={top['composite_score']:.3f}")

    return results_df


def optimize_qcvd(
    df: pd.DataFrame,
    symbol: str,
    tf_label: str,
    param_grid: Dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Optimize QCVD parameters (smooth_length, trend_length).

    Quality metrics: CVD trend flip accuracy, whipsaw rate, avg trend duration.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    volume = df["volume"].values
    n = len(close)

    smooth_lengths = param_grid["smooth_length"]
    trend_lengths = param_grid["trend_length"]

    total_combos = len(smooth_lengths) * len(trend_lengths)
    if verbose:
        print(f"    Testing {total_combos} parameter combinations on {n:,} bars...")

    results = []

    for smooth_len in smooth_lengths:
        for trend_len in trend_lengths:
            sim = simulate_qcvd(high, low, close, volume, smooth_len, trend_len)

            # Flip accuracy
            flip_acc_vals = []
            for lookahead in LOOKAHEAD_BARS:
                acc = measure_flip_accuracy(
                    close, sim["flipped_bullish"], sim["flipped_bearish"], lookahead
                )
                flip_acc_vals.append(acc)
            flip_acc = np.mean(flip_acc_vals) if flip_acc_vals else 0.0

            # Whipsaw rate
            whipsaw = measure_whipsaw_rate(
                sim["flipped_bullish"], sim["flipped_bearish"], WHIPSAW_THRESHOLD
            )

            # Avg trend duration
            max_dur = max(n / 20, 50)
            avg_dur = measure_avg_trend_duration(
                sim["flipped_bullish"], sim["flipped_bearish"]
            )

            total_flips = int(np.sum(sim["flipped_bullish"]) + np.sum(sim["flipped_bearish"]))

            composite = compute_composite_score(
                flip_accuracy=flip_acc,
                whipsaw_rate=whipsaw,
                avg_duration=avg_dur,
                max_duration=max_dur,
            )

            results.append({
                "smooth_length": smooth_len,
                "trend_length": trend_len,
                "flip_accuracy": flip_acc,
                "whipsaw_rate": whipsaw,
                "avg_trend_duration": avg_dur,
                "composite_score": composite,
                "total_flips": total_flips,
                "bars_tested": n,
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values("composite_score", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    if verbose and len(results_df) > 0:
        top = results_df.iloc[0]
        print(f"    BEST: smooth={int(top['smooth_length'])}, trend={int(top['trend_length'])} | "
              f"FlipAcc={top['flip_accuracy']:.1%}, Whipsaw={top['whipsaw_rate']:.1%}, "
              f"Score={top['composite_score']:.3f}")

    return results_df


def optimize_qsmc(
    df: pd.DataFrame,
    symbol: str,
    tf_label: str,
    param_grid: Dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Optimize QSMC parameters (swing_length, ob_strength).

    Quality metrics: BOS/CHoCH accuracy (structure break → price follows), whipsaw, duration.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    swing_lengths = param_grid["swing_length"]
    ob_strengths = param_grid["ob_strength"]

    total_combos = len(swing_lengths) * len(ob_strengths)
    if verbose:
        print(f"    Testing {total_combos} parameter combinations on {n:,} bars...")

    results = []

    for swing_len in swing_lengths:
        for ob_str in ob_strengths:
            sim = simulate_qsmc(high, low, close, swing_len, ob_str)

            # BOS + CHoCH combined as "structure flips"
            bull_flips = sim["bos_bull"] | sim["choch_bull"]
            bear_flips = sim["bos_bear"] | sim["choch_bear"]

            # Flip accuracy
            flip_acc_vals = []
            for lookahead in LOOKAHEAD_BARS:
                acc = measure_flip_accuracy(close, bull_flips, bear_flips, lookahead)
                flip_acc_vals.append(acc)
            flip_acc = np.mean(flip_acc_vals) if flip_acc_vals else 0.0

            # Whipsaw
            whipsaw = measure_whipsaw_rate(bull_flips, bear_flips, WHIPSAW_THRESHOLD)

            # Avg duration
            max_dur = max(n / 20, 50)
            avg_dur = measure_avg_trend_duration(bull_flips, bear_flips)

            total_signals = int(np.sum(bull_flips) + np.sum(bear_flips))
            total_choch = int(np.sum(sim["choch_bull"]) + np.sum(sim["choch_bear"]))

            composite = compute_composite_score(
                flip_accuracy=flip_acc,
                whipsaw_rate=whipsaw,
                avg_duration=avg_dur,
                max_duration=max_dur,
            )

            results.append({
                "swing_length": swing_len,
                "ob_strength": ob_str,
                "flip_accuracy": flip_acc,
                "whipsaw_rate": whipsaw,
                "avg_trend_duration": avg_dur,
                "composite_score": composite,
                "total_signals": total_signals,
                "total_choch": total_choch,
                "bars_tested": n,
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values("composite_score", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    if verbose and len(results_df) > 0:
        top = results_df.iloc[0]
        print(f"    BEST: swing={int(top['swing_length'])}, ob_str={top['ob_strength']:.1f} | "
              f"FlipAcc={top['flip_accuracy']:.1%}, Whipsaw={top['whipsaw_rate']:.1%}, "
              f"Score={top['composite_score']:.3f}")

    return results_df


def optimize_qgrid(
    df: pd.DataFrame,
    symbol: str,
    tf_label: str,
    param_grid: Dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Optimize QGrid parameters (left_bars, right_bars).

    Quality metrics: Level bounce rate (how often price reverses at detected levels).
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    left_bars_list = param_grid["left_bars"]
    right_bars_list = param_grid["right_bars"]

    total_combos = len(left_bars_list) * len(right_bars_list)
    if verbose:
        print(f"    Testing {total_combos} parameter combinations on {n:,} bars...")

    results = []

    for lb in left_bars_list:
        for rb in right_bars_list:
            sim = simulate_qgrid(high, low, close, lb, rb)

            total_bounces = int(np.sum(sim["level_bounced"]))
            total_breaks = int(np.sum(sim["level_broken"]))
            total_interactions = total_bounces + total_breaks
            total_levels = sim["total_levels_detected"]

            # Bounce rate: what % of level interactions are bounces
            bounce_rate = total_bounces / total_interactions if total_interactions > 0 else 0.0

            # Level density: levels per 1000 bars (not too few, not too many)
            density = total_levels / (n / 1000) if n > 0 else 0.0
            # Ideal density is 10-50 per 1000 bars. Penalize extremes.
            density_score = 1.0 - abs(density - 30) / 30 if density > 0 else 0.0
            density_score = max(0.0, min(1.0, density_score))

            # Interaction rate: what % of levels got tested
            interaction_rate = total_interactions / total_levels if total_levels > 0 else 0.0

            # Composite: bounce rate 40%, interaction rate 30%, density 30%
            composite = bounce_rate * 0.40 + min(interaction_rate, 1.0) * 0.30 + density_score * 0.30

            results.append({
                "left_bars": lb,
                "right_bars": rb,
                "bounce_rate": bounce_rate,
                "interaction_rate": interaction_rate,
                "density_score": density_score,
                "composite_score": composite,
                "total_levels": total_levels,
                "total_bounces": total_bounces,
                "total_breaks": total_breaks,
                "bars_tested": n,
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values("composite_score", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    if verbose and len(results_df) > 0:
        top = results_df.iloc[0]
        print(f"    BEST: left={int(top['left_bars'])}, right={int(top['right_bars'])} | "
              f"Bounce={top['bounce_rate']:.1%}, IntRate={top['interaction_rate']:.1%}, "
              f"Score={top['composite_score']:.3f}")

    return results_df


# ============================================================================
# PINE SCRIPT CODE GENERATOR
# ============================================================================

def generate_pine_qcloud_lookup(best_params: Dict) -> str:
    """
    Generate Pine Script v6 auto-optimization lookup function for QCloud.
    Uses real optimized values from the optimizer.

    best_params structure:
    {
        "SPY": {"1min": {"base_length": 8, "step_mult": 1.5, "smoothing": 2}, "5min": {...}, ...},
        "TSLA": {...},
        ...
    }
    """
    lines = []
    lines.append("// ==================== AUTO-OPTIMIZATION LOOKUP TABLES ====================")
    lines.append("// Generated by optimize_params.py on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("// These values are REAL optimized results from backtesting against historical data.")
    lines.append("// DO NOT manually edit — re-run the optimizer to update.")
    lines.append("")

    # --- base_length function ---
    lines.append("f_auto_base_length() =>")
    lines.append("    string t = syminfo.ticker")

    first_symbol = True
    for symbol in ALL_SYMBOLS:
        if symbol not in best_params:
            continue
        sym_data = best_params[symbol]
        prefix = "if" if first_symbol else "else if"
        first_symbol = False

        # Get values per timeframe group
        scalp_val = _get_param(sym_data, "scalp", "base_length", 10)
        intra_val = _get_param(sym_data, "intraday", "base_length", 10)
        swing_val = _get_param(sym_data, "swing", "base_length", 14)

        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    # Default fallback
    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 10 : tf_group == "intraday" ? 10 : 14')
    lines.append("")

    # --- step_mult function ---
    lines.append("f_auto_step_mult() =>")
    lines.append("    string t = syminfo.ticker")

    first_symbol = True
    for symbol in ALL_SYMBOLS:
        if symbol not in best_params:
            continue
        sym_data = best_params[symbol]
        prefix = "if" if first_symbol else "else if"
        first_symbol = False

        scalp_val = _get_param(sym_data, "scalp", "step_mult", 1.5)
        intra_val = _get_param(sym_data, "intraday", "step_mult", 1.5)
        swing_val = _get_param(sym_data, "swing", "step_mult", 1.5)

        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 1.5 : tf_group == "intraday" ? 1.5 : 1.5')
    lines.append("")

    # --- smoothing function ---
    lines.append("f_auto_smoothing() =>")
    lines.append("    string t = syminfo.ticker")

    first_symbol = True
    for symbol in ALL_SYMBOLS:
        if symbol not in best_params:
            continue
        sym_data = best_params[symbol]
        prefix = "if" if first_symbol else "else if"
        first_symbol = False

        scalp_val = _get_param(sym_data, "scalp", "smoothing", 2)
        intra_val = _get_param(sym_data, "intraday", "smoothing", 2)
        swing_val = _get_param(sym_data, "swing", "smoothing", 3)

        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 2 : tf_group == "intraday" ? 2 : 3')

    return "\n".join(lines)


def generate_pine_qline_lookup(best_params: Dict) -> str:
    """
    Generate Pine Script v6 auto-optimization lookup function for QLine.
    """
    lines = []
    lines.append("// ==================== AUTO-OPTIMIZATION LOOKUP TABLES ====================")
    lines.append("// Generated by optimize_params.py on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("// These values are REAL optimized results from backtesting against historical data.")
    lines.append("// DO NOT manually edit — re-run the optimizer to update.")
    lines.append("")

    # --- atr_length function ---
    lines.append("f_auto_atr_length() =>")
    lines.append("    string t = syminfo.ticker")

    first_symbol = True
    for symbol in ALL_SYMBOLS:
        if symbol not in best_params:
            continue
        sym_data = best_params[symbol]
        prefix = "if" if first_symbol else "else if"
        first_symbol = False

        scalp_val = _get_param(sym_data, "scalp", "atr_length", 14)
        intra_val = _get_param(sym_data, "intraday", "atr_length", 14)
        swing_val = _get_param(sym_data, "swing", "atr_length", 16)

        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 14 : tf_group == "intraday" ? 14 : 16')
    lines.append("")

    # --- factor function ---
    lines.append("f_auto_factor() =>")
    lines.append("    string t = syminfo.ticker")

    first_symbol = True
    for symbol in ALL_SYMBOLS:
        if symbol not in best_params:
            continue
        sym_data = best_params[symbol]
        prefix = "if" if first_symbol else "else if"
        first_symbol = False

        scalp_val = _get_param(sym_data, "scalp", "factor", 2.0)
        intra_val = _get_param(sym_data, "intraday", "factor", 2.0)
        swing_val = _get_param(sym_data, "swing", "factor", 2.5)

        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 2.0 : tf_group == "intraday" ? 2.0 : 2.5')

    return "\n".join(lines)


def generate_pine_qwave_lookup(best_params: Dict) -> str:
    """
    Generate Pine Script v6 auto-optimization lookup function for QWave.
    """
    lines = []
    lines.append("// ==================== AUTO-OPTIMIZATION LOOKUP TABLES ====================")
    lines.append("// Generated by optimize_params.py on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("// These values are REAL optimized results from backtesting against historical data.")
    lines.append("// DO NOT manually edit — re-run the optimizer to update.")
    lines.append("")

    # --- adx_length function ---
    lines.append("f_auto_adx_length() =>")
    lines.append("    string t = syminfo.ticker")

    first_symbol = True
    for symbol in ALL_SYMBOLS:
        if symbol not in best_params:
            continue
        sym_data = best_params[symbol]
        prefix = "if" if first_symbol else "else if"
        first_symbol = False

        scalp_val = _get_param(sym_data, "scalp", "adx_length", 14)
        intra_val = _get_param(sym_data, "intraday", "adx_length", 14)
        swing_val = _get_param(sym_data, "swing", "adx_length", 16)

        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 14 : tf_group == "intraday" ? 14 : 16')
    lines.append("")

    # --- smoothing function ---
    lines.append("f_auto_smoothing() =>")
    lines.append("    string t = syminfo.ticker")

    first_symbol = True
    for symbol in ALL_SYMBOLS:
        if symbol not in best_params:
            continue
        sym_data = best_params[symbol]
        prefix = "if" if first_symbol else "else if"
        first_symbol = False

        scalp_val = _get_param(sym_data, "scalp", "smoothing", 2)
        intra_val = _get_param(sym_data, "intraday", "smoothing", 2)
        swing_val = _get_param(sym_data, "swing", "smoothing", 3)

        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 2 : tf_group == "intraday" ? 2 : 3')

    return "\n".join(lines)


def generate_pine_qbands_lookup(best_params: Dict) -> str:
    """
    Generate Pine Script v6 auto-optimization lookup functions for QBands.

    best_params structure:
    {
        "SPY": {"1min": {"bb_length": 20, "bb_mult": 2.0, "kc_mult": 1.5, ...}, ...},
        "TSLA": {...},
    }
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append("// ==================== AUTO-OPTIMIZATION LOOKUP TABLES ====================")
    lines.append(f"// Generated by optimize_params.py on {timestamp}")
    lines.append("// These values are REAL optimized results from backtesting against historical data.")
    lines.append("// DO NOT manually edit — re-run the optimizer to update.")

    # --- bb_length lookup ---
    lines.append("")
    lines.append("f_auto_bb_length() =>")
    lines.append('    string t = syminfo.ticker')

    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "bb_length", 20)
        intra_val = _get_param(sym_data, "intraday", "bb_length", 20)
        swing_val = _get_param(sym_data, "swing", "bb_length", 22)

        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 20 : tf_group == "intraday" ? 20 : 22')

    # --- bb_mult lookup ---
    lines.append("")
    lines.append("f_auto_bb_mult() =>")
    lines.append('    string t = syminfo.ticker')

    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "bb_mult", 2.0)
        intra_val = _get_param(sym_data, "intraday", "bb_mult", 2.0)
        swing_val = _get_param(sym_data, "swing", "bb_mult", 2.0)

        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val:.1f} : tf_group == "intraday" ? {intra_val:.1f} : {swing_val:.1f}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 2.0 : tf_group == "intraday" ? 2.0 : 2.0')

    # --- kc_mult lookup ---
    lines.append("")
    lines.append("f_auto_kc_mult() =>")
    lines.append('    string t = syminfo.ticker')

    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "kc_mult", 1.5)
        intra_val = _get_param(sym_data, "intraday", "kc_mult", 1.5)
        swing_val = _get_param(sym_data, "swing", "kc_mult", 1.5)

        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val:.1f} : tf_group == "intraday" ? {intra_val:.1f} : {swing_val:.1f}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 1.5 : tf_group == "intraday" ? 1.5 : 1.5')

    return "\n".join(lines)


def generate_pine_moneyball_lookup(best_params: Dict) -> str:
    """
    Generate Pine Script v6 auto-optimization lookup functions for Moneyball.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append("// ==================== AUTO-OPTIMIZATION LOOKUP TABLES ====================")
    lines.append(f"// Generated by optimize_params.py on {timestamp}")
    lines.append("// These values are REAL optimized results from backtesting against historical data.")
    lines.append("// DO NOT manually edit — re-run the optimizer to update.")

    # --- roc_length lookup ---
    lines.append("")
    lines.append("f_auto_roc_length() =>")
    lines.append('    string t = syminfo.ticker')

    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "roc_length", 10)
        intra_val = _get_param(sym_data, "intraday", "roc_length", 10)
        swing_val = _get_param(sym_data, "swing", "roc_length", 12)

        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 10 : tf_group == "intraday" ? 10 : 12')

    # --- smooth_length lookup ---
    lines.append("")
    lines.append("f_auto_smooth_length() =>")
    lines.append('    string t = syminfo.ticker')

    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "smooth_length", 3)
        intra_val = _get_param(sym_data, "intraday", "smooth_length", 3)
        swing_val = _get_param(sym_data, "swing", "smooth_length", 3)

        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')

    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 3 : tf_group == "intraday" ? 3 : 3')

    return "\n".join(lines)


def generate_pine_qmomentum_lookup(best_params: Dict) -> str:
    """Generate Pine Script v6 auto-optimization lookup functions for QMomentum."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append("// ==================== AUTO-OPTIMIZATION LOOKUP TABLES ====================")
    lines.append(f"// Generated by optimize_params.py on {timestamp}")
    lines.append("// These values are REAL optimized results from backtesting against historical data.")
    lines.append("// DO NOT manually edit — re-run the optimizer to update.")

    # --- rsi_length lookup ---
    lines.append("")
    lines.append("f_auto_rsi_length() =>")
    lines.append('    string t = syminfo.ticker')
    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "rsi_length", 14)
        intra_val = _get_param(sym_data, "intraday", "rsi_length", 14)
        swing_val = _get_param(sym_data, "swing", "rsi_length", 16)
        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')
    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 14 : tf_group == "intraday" ? 14 : 16')

    # --- stoch_length lookup ---
    lines.append("")
    lines.append("f_auto_stoch_length() =>")
    lines.append('    string t = syminfo.ticker')
    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "stoch_length", 14)
        intra_val = _get_param(sym_data, "intraday", "stoch_length", 14)
        swing_val = _get_param(sym_data, "swing", "stoch_length", 14)
        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')
    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 14 : tf_group == "intraday" ? 14 : 14')

    return "\n".join(lines)


def generate_pine_qcvd_lookup(best_params: Dict) -> str:
    """Generate Pine Script v6 auto-optimization lookup functions for QCVD."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append("// ==================== AUTO-OPTIMIZATION LOOKUP TABLES ====================")
    lines.append(f"// Generated by optimize_params.py on {timestamp}")
    lines.append("// These values are REAL optimized results from backtesting against historical data.")
    lines.append("// DO NOT manually edit — re-run the optimizer to update.")

    # --- smooth_length lookup ---
    lines.append("")
    lines.append("f_auto_smooth_length() =>")
    lines.append('    string t = syminfo.ticker')
    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "smooth_length", 3)
        intra_val = _get_param(sym_data, "intraday", "smooth_length", 3)
        swing_val = _get_param(sym_data, "swing", "smooth_length", 4)
        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')
    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 3 : tf_group == "intraday" ? 3 : 4')

    # --- trend_length lookup ---
    lines.append("")
    lines.append("f_auto_trend_length() =>")
    lines.append('    string t = syminfo.ticker')
    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "trend_length", 14)
        intra_val = _get_param(sym_data, "intraday", "trend_length", 14)
        swing_val = _get_param(sym_data, "swing", "trend_length", 16)
        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')
    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 14 : tf_group == "intraday" ? 14 : 16')

    return "\n".join(lines)


def generate_pine_qsmc_lookup(best_params: Dict) -> str:
    """Generate Pine Script v6 auto-optimization lookup functions for QSMC."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append("// ==================== AUTO-OPTIMIZATION LOOKUP TABLES ====================")
    lines.append(f"// Generated by optimize_params.py on {timestamp}")

    # --- swing_length ---
    lines.append("")
    lines.append("f_auto_swing_length() =>")
    lines.append('    string t = syminfo.ticker')
    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "swing_length", 4)
        intra_val = _get_param(sym_data, "intraday", "swing_length", 5)
        swing_val = _get_param(sym_data, "swing", "swing_length", 6)
        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')
    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 4 : tf_group == "intraday" ? 5 : 6')

    # --- ob_strength ---
    lines.append("")
    lines.append("f_auto_ob_strength() =>")
    lines.append('    string t = syminfo.ticker')
    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "ob_strength", 1.5)
        intra_val = _get_param(sym_data, "intraday", "ob_strength", 1.5)
        swing_val = _get_param(sym_data, "swing", "ob_strength", 2.0)
        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')
    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 1.5 : tf_group == "intraday" ? 1.5 : 2.0')

    return "\n".join(lines)


def generate_pine_qgrid_lookup(best_params: Dict) -> str:
    """Generate Pine Script v6 auto-optimization lookup functions for QGrid."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append("// ==================== AUTO-OPTIMIZATION LOOKUP TABLES ====================")
    lines.append(f"// Generated by optimize_params.py on {timestamp}")

    # --- left_bars ---
    lines.append("")
    lines.append("f_auto_left_bars() =>")
    lines.append('    string t = syminfo.ticker')
    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "left_bars", 4)
        intra_val = _get_param(sym_data, "intraday", "left_bars", 5)
        swing_val = _get_param(sym_data, "swing", "left_bars", 7)
        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')
    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 4 : tf_group == "intraday" ? 5 : 7')

    # --- right_bars ---
    lines.append("")
    lines.append("f_auto_right_bars() =>")
    lines.append('    string t = syminfo.ticker')
    for idx, symbol in enumerate(ALL_SYMBOLS):
        sym_data = best_params.get(symbol, {})
        scalp_val = _get_param(sym_data, "scalp", "right_bars", 3)
        intra_val = _get_param(sym_data, "intraday", "right_bars", 3)
        swing_val = _get_param(sym_data, "swing", "right_bars", 4)
        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} t == "{symbol}"')
        lines.append(f'        tf_group == "scalp" ? {scalp_val} : tf_group == "intraday" ? {intra_val} : {swing_val}')
    lines.append("    else")
    lines.append('        tf_group == "scalp" ? 3 : tf_group == "intraday" ? 3 : 4')

    return "\n".join(lines)


def _get_param(sym_data: Dict, tf_group: str, param_name: str, default):
    """
    Get optimized parameter value for a symbol's timeframe group.

    For 'intraday' group, we check both 5min and 15min results and pick
    the one with the higher composite score. If both exist, use the 5min
    result (primary intraday timeframe).
    """
    # Direct tf_group match
    tf_map = {
        "scalp": ["1min"],
        "intraday": ["5min", "15min"],
        "swing": ["1hr"],
    }

    best_val = default
    best_score = -1.0

    for tf_label in tf_map.get(tf_group, []):
        if tf_label in sym_data:
            entry = sym_data[tf_label]
            if param_name in entry:
                score = entry.get("composite_score", 0.0)
                if score > best_score:
                    best_score = score
                    best_val = entry[param_name]

    return best_val


# ============================================================================
# MAIN OPTIMIZATION ORCHESTRATOR
# ============================================================================

def run_full_optimization(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    indicators: Optional[List[str]] = None,
    quick: bool = False,
) -> Dict:
    """
    Run full optimization across all symbols and timeframes.
    """
    if symbols is None:
        symbols = ALL_SYMBOLS
    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())
    if indicators is None:
        indicators = ["qcloud", "qline", "qwave", "qbands", "moneyball", "qmomentum", "qcvd", "qsmc", "qgrid"]

    qcloud_params = QCLOUD_PARAMS_QUICK if quick else QCLOUD_PARAMS
    qline_params = QLINE_PARAMS_QUICK if quick else QLINE_PARAMS
    qwave_params = QWAVE_PARAMS_QUICK if quick else QWAVE_PARAMS
    qbands_params = QBANDS_PARAMS_QUICK if quick else QBANDS_PARAMS
    moneyball_params = MONEYBALL_PARAMS_QUICK if quick else MONEYBALL_PARAMS
    qmomentum_params = QMOMENTUM_PARAMS_QUICK if quick else QMOMENTUM_PARAMS
    qcvd_params = QCVD_PARAMS_QUICK if quick else QCVD_PARAMS
    qsmc_params = QSMC_PARAMS_QUICK if quick else QSMC_PARAMS
    qgrid_params = QGRID_PARAMS_QUICK if quick else QGRID_PARAMS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("YELENA v2 — PARAMETER OPTIMIZER")
    print("=" * 80)
    print(f"Symbols:    {symbols}")
    print(f"Timeframes: {timeframes}")
    print(f"Indicators: {indicators}")
    print(f"Mode:       {'QUICK' if quick else 'FULL'}")
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:     {RESULTS_DIR}")
    print("=" * 80)

    # Connect to database
    print("\nConnecting to PostgreSQL...")
    conn = get_db_connection()
    print("  Connected successfully.")

    # Results storage
    all_qcloud_results = {}
    all_qline_results = {}
    qcloud_best = {}  # {symbol: {timeframe: {params + metrics}}}
    qline_best = {}
    qwave_best = {}
    qbands_best = {}
    moneyball_best = {}
    qmomentum_best = {}
    qcvd_best = {}
    qsmc_best = {}
    qgrid_best = {}

    start_time = time.time()

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print(f"{'='*60}")

        for tf_label in timeframes:
            table_name, tf_group = TIMEFRAMES[tf_label]
            print(f"\n  Timeframe: {tf_label} (table={table_name}, group={tf_group})")

            # Load data
            print(f"  Loading data...")
            df = load_bars(conn, table_name, symbol)

            if df.empty:
                print(f"  SKIP: No data for {symbol} in {table_name}")
                continue

            print(f"  Loaded {len(df):,} bars ({df['time'].min()} to {df['time'].max()})")

            # --- QCloud ---
            if "qcloud" in indicators:
                print(f"\n  --- QCloud Optimization ---")
                qc_results = optimize_qcloud(df, symbol, tf_label, qcloud_params)
                if not qc_results.empty:
                    # Save CSV
                    csv_path = RESULTS_DIR / f"qcloud_{symbol}_{tf_label}.csv"
                    qc_results.to_csv(csv_path, index=False)

                    # Store best
                    best = qc_results.iloc[0]
                    if symbol not in qcloud_best:
                        qcloud_best[symbol] = {}
                    qcloud_best[symbol][tf_label] = {
                        "base_length": int(best["base_length"]),
                        "step_mult": float(best["step_mult"]),
                        "smoothing": int(best["smoothing"]),
                        "flip_accuracy": float(best["flip_accuracy"]),
                        "whipsaw_rate": float(best["whipsaw_rate"]),
                        "avg_trend_duration": float(best["avg_trend_duration"]),
                        "squeeze_accuracy": float(best["squeeze_accuracy"]),
                        "composite_score": float(best["composite_score"]),
                        "total_flips": int(best["total_flips"]),
                        "bars_tested": int(best["bars_tested"]),
                    }

            # --- QLine ---
            if "qline" in indicators:
                print(f"\n  --- QLine Optimization ---")
                ql_results = optimize_qline(df, symbol, tf_label, qline_params)
                if not ql_results.empty:
                    csv_path = RESULTS_DIR / f"qline_{symbol}_{tf_label}.csv"
                    ql_results.to_csv(csv_path, index=False)

                    best = ql_results.iloc[0]
                    if symbol not in qline_best:
                        qline_best[symbol] = {}
                    qline_best[symbol][tf_label] = {
                        "atr_length": int(best["atr_length"]),
                        "factor": float(best["factor"]),
                        "flip_accuracy": float(best["flip_accuracy"]),
                        "whipsaw_rate": float(best["whipsaw_rate"]),
                        "avg_trend_duration": float(best["avg_trend_duration"]),
                        "bounce_win_rate": float(best["bounce_win_rate"]),
                        "composite_score": float(best["composite_score"]),
                        "total_flips": int(best["total_flips"]),
                        "total_bounces": int(best["total_bounces"]),
                        "bars_tested": int(best["bars_tested"]),
                    }

            # --- QWave ---
            if "qwave" in indicators:
                print(f"\n  --- QWave Optimization ---")
                qw_results = optimize_qwave(df, symbol, tf_label, qwave_params)
                if not qw_results.empty:
                    csv_path = RESULTS_DIR / f"qwave_{symbol}_{tf_label}.csv"
                    qw_results.to_csv(csv_path, index=False)

                    best = qw_results.iloc[0]
                    if symbol not in qwave_best:
                        qwave_best[symbol] = {}
                    qwave_best[symbol][tf_label] = {
                        "adx_length": int(best["adx_length"]),
                        "smoothing": int(best["smoothing"]),
                        "flip_accuracy": float(best["flip_accuracy"]),
                        "whipsaw_rate": float(best["whipsaw_rate"]),
                        "avg_trend_duration": float(best["avg_trend_duration"]),
                        "zone_crossing_accuracy": float(best["zone_crossing_accuracy"]),
                        "composite_score": float(best["composite_score"]),
                        "total_flips": int(best["total_flips"]),
                        "total_zone_entries": int(best["total_zone_entries"]),
                        "bars_tested": int(best["bars_tested"]),
                    }

            # --- QBands ---
            if "qbands" in indicators:
                print(f"\n  --- QBands Optimization ---")
                qb_results = optimize_qbands(df, symbol, tf_label, qbands_params)
                if not qb_results.empty:
                    csv_path = RESULTS_DIR / f"qbands_{symbol}_{tf_label}.csv"
                    qb_results.to_csv(csv_path, index=False)

                    best = qb_results.iloc[0]
                    if symbol not in qbands_best:
                        qbands_best[symbol] = {}
                    qbands_best[symbol][tf_label] = {
                        "bb_length": int(best["bb_length"]),
                        "bb_mult": float(best["bb_mult"]),
                        "kc_mult": float(best["kc_mult"]),
                        "fire_accuracy": float(best["fire_accuracy"]),
                        "touch_reversal": float(best["touch_reversal"]),
                        "squeeze_whipsaw": float(best["squeeze_whipsaw"]),
                        "avg_squeeze_dur": float(best["avg_squeeze_dur"]),
                        "squeeze_freq": float(best["squeeze_freq"]),
                        "total_fires": int(best["total_fires"]),
                        "total_touches": int(best["total_touches"]),
                        "total_squeezes": int(best["total_squeezes"]),
                        "composite_score": float(best["composite_score"]),
                        "bars_tested": int(best["bars_tested"]),
                    }

            # --- Moneyball ---
            if "moneyball" in indicators:
                print(f"\n  --- Moneyball Optimization ---")
                mb_results = optimize_moneyball(df, symbol, tf_label, moneyball_params)
                if not mb_results.empty:
                    csv_path = RESULTS_DIR / f"moneyball_{symbol}_{tf_label}.csv"
                    mb_results.to_csv(csv_path, index=False)

                    best = mb_results.iloc[0]
                    if symbol not in moneyball_best:
                        moneyball_best[symbol] = {}
                    moneyball_best[symbol][tf_label] = {
                        "roc_length": int(best["roc_length"]),
                        "smooth_length": int(best["smooth_length"]),
                        "flip_accuracy": float(best["flip_accuracy"]),
                        "whipsaw_rate": float(best["whipsaw_rate"]),
                        "avg_trend_duration": float(best["avg_trend_duration"]),
                        "zone_accuracy": float(best["zone_accuracy"]),
                        "composite_score": float(best["composite_score"]),
                        "total_flips": int(best["total_flips"]),
                        "total_extreme_entries": int(best["total_extreme_entries"]),
                        "bars_tested": int(best["bars_tested"]),
                    }

            # --- QMomentum ---
            if "qmomentum" in indicators:
                print(f"\n  --- QMomentum Optimization ---")
                qm_results = optimize_qmomentum(df, symbol, tf_label, qmomentum_params)
                if not qm_results.empty:
                    csv_path = RESULTS_DIR / f"qmomentum_{symbol}_{tf_label}.csv"
                    qm_results.to_csv(csv_path, index=False)

                    best = qm_results.iloc[0]
                    if symbol not in qmomentum_best:
                        qmomentum_best[symbol] = {}
                    qmomentum_best[symbol][tf_label] = {
                        "rsi_length": int(best["rsi_length"]),
                        "stoch_length": int(best["stoch_length"]),
                        "cross_accuracy": float(best["cross_accuracy"]),
                        "whipsaw_rate": float(best["whipsaw_rate"]),
                        "avg_trend_duration": float(best["avg_trend_duration"]),
                        "zone_exit_accuracy": float(best["zone_exit_accuracy"]),
                        "composite_score": float(best["composite_score"]),
                        "total_crosses": int(best["total_crosses"]),
                        "total_zone_exits": int(best["total_zone_exits"]),
                        "bars_tested": int(best["bars_tested"]),
                    }

            # --- QCVD ---
            if "qcvd" in indicators:
                print(f"\n  --- QCVD Optimization ---")
                cvd_results = optimize_qcvd(df, symbol, tf_label, qcvd_params)
                if not cvd_results.empty:
                    csv_path = RESULTS_DIR / f"qcvd_{symbol}_{tf_label}.csv"
                    cvd_results.to_csv(csv_path, index=False)

                    best = cvd_results.iloc[0]
                    if symbol not in qcvd_best:
                        qcvd_best[symbol] = {}
                    qcvd_best[symbol][tf_label] = {
                        "smooth_length": int(best["smooth_length"]),
                        "trend_length": int(best["trend_length"]),
                        "flip_accuracy": float(best["flip_accuracy"]),
                        "whipsaw_rate": float(best["whipsaw_rate"]),
                        "avg_trend_duration": float(best["avg_trend_duration"]),
                        "composite_score": float(best["composite_score"]),
                        "total_flips": int(best["total_flips"]),
                        "bars_tested": int(best["bars_tested"]),
                    }

            # --- QSMC ---
            if "qsmc" in indicators:
                print(f"\n  --- QSMC Optimization ---")
                smc_results = optimize_qsmc(df, symbol, tf_label, qsmc_params)
                if not smc_results.empty:
                    csv_path = RESULTS_DIR / f"qsmc_{symbol}_{tf_label}.csv"
                    smc_results.to_csv(csv_path, index=False)

                    best = smc_results.iloc[0]
                    if symbol not in qsmc_best:
                        qsmc_best[symbol] = {}
                    qsmc_best[symbol][tf_label] = {
                        "swing_length": int(best["swing_length"]),
                        "ob_strength": float(best["ob_strength"]),
                        "flip_accuracy": float(best["flip_accuracy"]),
                        "whipsaw_rate": float(best["whipsaw_rate"]),
                        "avg_trend_duration": float(best["avg_trend_duration"]),
                        "composite_score": float(best["composite_score"]),
                        "total_signals": int(best["total_signals"]),
                        "total_choch": int(best["total_choch"]),
                        "bars_tested": int(best["bars_tested"]),
                    }

            # --- QGrid ---
            if "qgrid" in indicators:
                print(f"\n  --- QGrid Optimization ---")
                grid_results = optimize_qgrid(df, symbol, tf_label, qgrid_params)
                if not grid_results.empty:
                    csv_path = RESULTS_DIR / f"qgrid_{symbol}_{tf_label}.csv"
                    grid_results.to_csv(csv_path, index=False)

                    best = grid_results.iloc[0]
                    if symbol not in qgrid_best:
                        qgrid_best[symbol] = {}
                    qgrid_best[symbol][tf_label] = {
                        "left_bars": int(best["left_bars"]),
                        "right_bars": int(best["right_bars"]),
                        "bounce_rate": float(best["bounce_rate"]),
                        "interaction_rate": float(best["interaction_rate"]),
                        "density_score": float(best["density_score"]),
                        "composite_score": float(best["composite_score"]),
                        "total_levels": int(best["total_levels"]),
                        "total_bounces": int(best["total_bounces"]),
                        "total_breaks": int(best["total_breaks"]),
                        "bars_tested": int(best["bars_tested"]),
                    }

    conn.close()
    elapsed = time.time() - start_time

    # ==================== SAVE RESULTS ====================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save best params JSON
    if qcloud_best:
        with open(RESULTS_DIR / "qcloud_best_params.json", "w") as f:
            json.dump(qcloud_best, f, indent=2)
        print(f"  Saved: qcloud_best_params.json")

    if qline_best:
        with open(RESULTS_DIR / "qline_best_params.json", "w") as f:
            json.dump(qline_best, f, indent=2)
        print(f"  Saved: qline_best_params.json")

    # Generate Pine Script lookup tables
    if qcloud_best:
        pine_qcloud = generate_pine_qcloud_lookup(qcloud_best)
        with open(RESULTS_DIR / "pine_qcloud_lookup.pine", "w") as f:
            f.write(pine_qcloud)
        print(f"  Saved: pine_qcloud_lookup.pine")

    if qline_best:
        pine_qline = generate_pine_qline_lookup(qline_best)
        with open(RESULTS_DIR / "pine_qline_lookup.pine", "w") as f:
            f.write(pine_qline)
        print(f"  Saved: pine_qline_lookup.pine")

    if qwave_best:
        with open(RESULTS_DIR / "qwave_best_params.json", "w") as f:
            json.dump(qwave_best, f, indent=2)
        print(f"  Saved: qwave_best_params.json")

    if qwave_best:
        pine_qwave = generate_pine_qwave_lookup(qwave_best)
        with open(RESULTS_DIR / "pine_qwave_lookup.pine", "w") as f:
            f.write(pine_qwave)
        print(f"  Saved: pine_qwave_lookup.pine")

    if qbands_best:
        with open(RESULTS_DIR / "qbands_best_params.json", "w") as f:
            json.dump(qbands_best, f, indent=2)
        print(f"  Saved: qbands_best_params.json")

    if qbands_best:
        pine_qbands = generate_pine_qbands_lookup(qbands_best)
        with open(RESULTS_DIR / "pine_qbands_lookup.pine", "w") as f:
            f.write(pine_qbands)
        print(f"  Saved: pine_qbands_lookup.pine")

    if moneyball_best:
        with open(RESULTS_DIR / "moneyball_best_params.json", "w") as f:
            json.dump(moneyball_best, f, indent=2)
        print(f"  Saved: moneyball_best_params.json")

    if moneyball_best:
        pine_moneyball = generate_pine_moneyball_lookup(moneyball_best)
        with open(RESULTS_DIR / "pine_moneyball_lookup.pine", "w") as f:
            f.write(pine_moneyball)
        print(f"  Saved: pine_moneyball_lookup.pine")

    if qmomentum_best:
        with open(RESULTS_DIR / "qmomentum_best_params.json", "w") as f:
            json.dump(qmomentum_best, f, indent=2)
        print(f"  Saved: qmomentum_best_params.json")

    if qmomentum_best:
        pine_qmomentum = generate_pine_qmomentum_lookup(qmomentum_best)
        with open(RESULTS_DIR / "pine_qmomentum_lookup.pine", "w") as f:
            f.write(pine_qmomentum)
        print(f"  Saved: pine_qmomentum_lookup.pine")

    if qcvd_best:
        with open(RESULTS_DIR / "qcvd_best_params.json", "w") as f:
            json.dump(qcvd_best, f, indent=2)
        print(f"  Saved: qcvd_best_params.json")

    if qcvd_best:
        pine_qcvd = generate_pine_qcvd_lookup(qcvd_best)
        with open(RESULTS_DIR / "pine_qcvd_lookup.pine", "w") as f:
            f.write(pine_qcvd)
        print(f"  Saved: pine_qcvd_lookup.pine")

    if qsmc_best:
        with open(RESULTS_DIR / "qsmc_best_params.json", "w") as f:
            json.dump(qsmc_best, f, indent=2)
        print(f"  Saved: qsmc_best_params.json")

    if qsmc_best:
        pine_qsmc = generate_pine_qsmc_lookup(qsmc_best)
        with open(RESULTS_DIR / "pine_qsmc_lookup.pine", "w") as f:
            f.write(pine_qsmc)
        print(f"  Saved: pine_qsmc_lookup.pine")

    if qgrid_best:
        with open(RESULTS_DIR / "qgrid_best_params.json", "w") as f:
            json.dump(qgrid_best, f, indent=2)
        print(f"  Saved: qgrid_best_params.json")

    if qgrid_best:
        pine_qgrid = generate_pine_qgrid_lookup(qgrid_best)
        with open(RESULTS_DIR / "pine_qgrid_lookup.pine", "w") as f:
            f.write(pine_qgrid)
        print(f"  Saved: pine_qgrid_lookup.pine")

    # ==================== PRINT SUMMARY REPORT ====================
    print("\n" + "=" * 80)
    print("OPTIMIZATION REPORT")
    print("=" * 80)
    print(f"Completed in {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Results saved to: {RESULTS_DIR}")

    if qcloud_best:
        print(f"\n{'─'*70}")
        print("QCLOUD — BEST PARAMETERS PER SYMBOL/TIMEFRAME")
        print(f"{'─'*70}")
        rows = []
        for sym in ALL_SYMBOLS:
            if sym not in qcloud_best:
                continue
            for tf in ["1min", "5min", "15min", "1hr"]:
                if tf not in qcloud_best[sym]:
                    continue
                p = qcloud_best[sym][tf]
                rows.append([
                    sym, tf,
                    p["base_length"], f"{p['step_mult']:.1f}", p["smoothing"],
                    f"{p['flip_accuracy']:.1%}", f"{p['whipsaw_rate']:.1%}",
                    f"{p['avg_trend_duration']:.0f}",
                    f"{p['composite_score']:.3f}",
                    p["total_flips"], p["bars_tested"],
                ])
        headers = ["Symbol", "TF", "Base", "Step", "Smooth",
                    "FlipAcc", "Whipsaw", "AvgDur", "Score", "Flips", "Bars"]
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print(f"{'Symbol':<8} {'TF':<6} {'Base':<5} {'Step':<5} {'Smooth':<7} "
                  f"{'FlipAcc':<8} {'Whipsaw':<8} {'AvgDur':<7} {'Score':<7} {'Flips':<6} {'Bars':<8}")
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<6} {r[2]:<5} {r[3]:<5} {r[4]:<7} "
                      f"{r[5]:<8} {r[6]:<8} {r[7]:<7} {r[8]:<7} {r[9]:<6} {r[10]:<8}")

    if qline_best:
        print(f"\n{'─'*70}")
        print("QLINE — BEST PARAMETERS PER SYMBOL/TIMEFRAME")
        print(f"{'─'*70}")
        rows = []
        for sym in ALL_SYMBOLS:
            if sym not in qline_best:
                continue
            for tf in ["1min", "5min", "15min", "1hr"]:
                if tf not in qline_best[sym]:
                    continue
                p = qline_best[sym][tf]
                rows.append([
                    sym, tf,
                    p["atr_length"], f"{p['factor']:.1f}",
                    f"{p['flip_accuracy']:.1%}", f"{p['whipsaw_rate']:.1%}",
                    f"{p['bounce_win_rate']:.1%}",
                    f"{p['avg_trend_duration']:.0f}",
                    f"{p['composite_score']:.3f}",
                    p["total_flips"], p["bars_tested"],
                ])
        headers = ["Symbol", "TF", "ATR", "Factor",
                    "FlipAcc", "Whipsaw", "BounceWR", "AvgDur", "Score", "Flips", "Bars"]
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print(f"{'Symbol':<8} {'TF':<6} {'ATR':<5} {'Factor':<7} "
                  f"{'FlipAcc':<8} {'Whipsaw':<8} {'BounceWR':<9} {'AvgDur':<7} "
                  f"{'Score':<7} {'Flips':<6} {'Bars':<8}")
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<6} {r[2]:<5} {r[3]:<7} "
                      f"{r[4]:<8} {r[5]:<8} {r[6]:<9} {r[7]:<7} "
                      f"{r[8]:<7} {r[9]:<6} {r[10]:<8}")

    if qwave_best:
        print(f"\n{'─'*70}")
        print("QWAVE — BEST PARAMETERS PER SYMBOL/TIMEFRAME")
        print(f"{'─'*70}")
        rows = []
        for sym in ALL_SYMBOLS:
            if sym not in qwave_best:
                continue
            for tf in ["1min", "5min", "15min", "1hr"]:
                if tf not in qwave_best[sym]:
                    continue
                p = qwave_best[sym][tf]
                rows.append([
                    sym, tf,
                    p["adx_length"], p["smoothing"],
                    f"{p['flip_accuracy']:.1%}", f"{p['whipsaw_rate']:.1%}",
                    f"{p['zone_crossing_accuracy']:.1%}",
                    f"{p['avg_trend_duration']:.0f}",
                    f"{p['composite_score']:.3f}",
                    p["total_flips"], p["bars_tested"],
                ])
        headers = ["Symbol", "TF", "ADX", "Smooth",
                    "FlipAcc", "Whipsaw", "ZoneAcc", "AvgDur", "Score", "Flips", "Bars"]
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print(f"{'Symbol':<8} {'TF':<6} {'ADX':<5} {'Smooth':<7} "
                  f"{'FlipAcc':<8} {'Whipsaw':<8} {'ZoneAcc':<8} {'AvgDur':<7} "
                  f"{'Score':<7} {'Flips':<6} {'Bars':<8}")
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<6} {r[2]:<5} {r[3]:<7} "
                      f"{r[4]:<8} {r[5]:<8} {r[6]:<8} {r[7]:<7} "
                      f"{r[8]:<7} {r[9]:<6} {r[10]:<8}")

    if qbands_best:
        print(f"\n{'─'*70}")
        print("QBANDS — BEST PARAMETERS PER SYMBOL/TIMEFRAME")
        print(f"{'─'*70}")
        rows = []
        for sym in ALL_SYMBOLS:
            if sym not in qbands_best:
                continue
            for tf in ["1min", "5min", "15min", "1hr"]:
                if tf not in qbands_best[sym]:
                    continue
                p = qbands_best[sym][tf]
                rows.append([
                    sym, tf,
                    p["bb_length"], f"{p['bb_mult']:.1f}", f"{p['kc_mult']:.1f}",
                    f"{p['fire_accuracy']:.1%}", f"{p['touch_reversal']:.1%}",
                    f"{p['squeeze_whipsaw']:.1%}",
                    f"{p['avg_squeeze_dur']:.0f}",
                    f"{p['composite_score']:.3f}",
                    p["total_fires"], p["total_touches"], p["bars_tested"],
                ])
        headers = ["Symbol", "TF", "BBLen", "BBMult", "KCMult",
                    "FireAcc", "TouchRev", "SqzWhip", "AvgDur", "Score",
                    "Fires", "Touches", "Bars"]
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print(f"{'Symbol':<8} {'TF':<6} {'BBLen':<6} {'BBMult':<7} {'KCMult':<7} "
                  f"{'FireAcc':<8} {'TouchRev':<9} {'SqzWhip':<8} {'AvgDur':<7} "
                  f"{'Score':<7} {'Fires':<7} {'Touch':<7} {'Bars':<8}")
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<6} {r[2]:<6} {r[3]:<7} {r[4]:<7} "
                      f"{r[5]:<8} {r[6]:<9} {r[7]:<8} {r[8]:<7} "
                      f"{r[9]:<7} {r[10]:<7} {r[11]:<7} {r[12]:<8}")

    if moneyball_best:
        print(f"\n{'─'*70}")
        print("MONEYBALL — BEST PARAMETERS PER SYMBOL/TIMEFRAME")
        print(f"{'─'*70}")
        rows = []
        for sym in ALL_SYMBOLS:
            if sym not in moneyball_best:
                continue
            for tf in ["1min", "5min", "15min", "1hr"]:
                if tf not in moneyball_best[sym]:
                    continue
                p = moneyball_best[sym][tf]
                rows.append([
                    sym, tf,
                    p["roc_length"], p["smooth_length"],
                    f"{p['flip_accuracy']:.1%}", f"{p['whipsaw_rate']:.1%}",
                    f"{p['zone_accuracy']:.1%}",
                    f"{p['avg_trend_duration']:.0f}",
                    f"{p['composite_score']:.3f}",
                    p["total_flips"], p["bars_tested"],
                ])
        headers = ["Symbol", "TF", "ROC", "Smooth",
                    "FlipAcc", "Whipsaw", "ZoneAcc", "AvgDur", "Score", "Flips", "Bars"]
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print(f"{'Symbol':<8} {'TF':<6} {'ROC':<5} {'Smooth':<7} "
                  f"{'FlipAcc':<8} {'Whipsaw':<8} {'ZoneAcc':<8} {'AvgDur':<7} "
                  f"{'Score':<7} {'Flips':<6} {'Bars':<8}")
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<6} {r[2]:<5} {r[3]:<7} "
                      f"{r[4]:<8} {r[5]:<8} {r[6]:<8} {r[7]:<7} "
                      f"{r[8]:<7} {r[9]:<6} {r[10]:<8}")

    if qmomentum_best:
        print(f"\n{'─'*70}")
        print("QMOMENTUM — BEST PARAMETERS PER SYMBOL/TIMEFRAME")
        print(f"{'─'*70}")
        rows = []
        for sym in ALL_SYMBOLS:
            if sym not in qmomentum_best:
                continue
            for tf in ["1min", "5min", "15min", "1hr"]:
                if tf not in qmomentum_best[sym]:
                    continue
                p = qmomentum_best[sym][tf]
                rows.append([
                    sym, tf,
                    p["rsi_length"], p["stoch_length"],
                    f"{p['cross_accuracy']:.1%}", f"{p['whipsaw_rate']:.1%}",
                    f"{p['zone_exit_accuracy']:.1%}",
                    f"{p['avg_trend_duration']:.0f}",
                    f"{p['composite_score']:.3f}",
                    p["total_crosses"], p["bars_tested"],
                ])
        headers = ["Symbol", "TF", "RSI", "Stoch",
                    "CrossAcc", "Whipsaw", "ZoneExit", "AvgDur", "Score", "Cross", "Bars"]
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print(f"{'Symbol':<8} {'TF':<6} {'RSI':<5} {'Stoch':<6} "
                  f"{'CrossAcc':<9} {'Whipsaw':<8} {'ZoneExit':<9} {'AvgDur':<7} "
                  f"{'Score':<7} {'Cross':<6} {'Bars':<8}")
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<6} {r[2]:<5} {r[3]:<6} "
                      f"{r[4]:<9} {r[5]:<8} {r[6]:<9} {r[7]:<7} "
                      f"{r[8]:<7} {r[9]:<6} {r[10]:<8}")

    # Print Pine Script code for easy copy-paste
    if qcloud_best:
        print(f"\n{'='*80}")
        print("PINE SCRIPT — QCLOUD AUTO-OPTIMIZATION LOOKUP (copy-paste ready)")
        print(f"{'='*80}")
        print(generate_pine_qcloud_lookup(qcloud_best))

    if qline_best:
        print(f"\n{'='*80}")
        print("PINE SCRIPT — QLINE AUTO-OPTIMIZATION LOOKUP (copy-paste ready)")
        print(f"{'='*80}")
        print(generate_pine_qline_lookup(qline_best))

    if qwave_best:
        print(f"\n{'='*80}")
        print("PINE SCRIPT — QWAVE AUTO-OPTIMIZATION LOOKUP (copy-paste ready)")
        print(f"{'='*80}")
        print(generate_pine_qwave_lookup(qwave_best))

    if qbands_best:
        print(f"\n{'='*80}")
        print("PINE SCRIPT — QBANDS AUTO-OPTIMIZATION LOOKUP (copy-paste ready)")
        print(f"{'='*80}")
        print(generate_pine_qbands_lookup(qbands_best))

    if moneyball_best:
        print(f"\n{'='*80}")
        print("PINE SCRIPT — MONEYBALL AUTO-OPTIMIZATION LOOKUP (copy-paste ready)")
        print(f"{'='*80}")
        print(generate_pine_moneyball_lookup(moneyball_best))

    if qmomentum_best:
        print(f"\n{'='*80}")
        print("PINE SCRIPT — QMOMENTUM AUTO-OPTIMIZATION LOOKUP (copy-paste ready)")
        print(f"{'='*80}")
        print(generate_pine_qmomentum_lookup(qmomentum_best))

    if qcvd_best:
        print(f"\n{'─'*70}")
        print("QCVD — BEST PARAMETERS PER SYMBOL/TIMEFRAME")
        print(f"{'─'*70}")
        rows = []
        for sym in ALL_SYMBOLS:
            if sym not in qcvd_best:
                continue
            for tf in ["1min", "5min", "15min", "1hr"]:
                if tf not in qcvd_best[sym]:
                    continue
                p = qcvd_best[sym][tf]
                rows.append([
                    sym, tf,
                    p["smooth_length"], p["trend_length"],
                    f"{p['flip_accuracy']:.1%}", f"{p['whipsaw_rate']:.1%}",
                    f"{p['avg_trend_duration']:.0f}",
                    f"{p['composite_score']:.3f}",
                    p["total_flips"], p["bars_tested"],
                ])
        headers = ["Symbol", "TF", "Smooth", "Trend",
                    "FlipAcc", "Whipsaw", "AvgDur", "Score", "Flips", "Bars"]
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print(f"{'Symbol':<8} {'TF':<6} {'Smooth':<7} {'Trend':<6} "
                  f"{'FlipAcc':<8} {'Whipsaw':<8} {'AvgDur':<7} "
                  f"{'Score':<7} {'Flips':<6} {'Bars':<8}")
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<6} {r[2]:<7} {r[3]:<6} "
                      f"{r[4]:<8} {r[5]:<8} {r[6]:<7} "
                      f"{r[7]:<7} {r[8]:<6} {r[9]:<8}")

    if qcvd_best:
        print(f"\n{'='*80}")
        print("PINE SCRIPT — QCVD AUTO-OPTIMIZATION LOOKUP (copy-paste ready)")
        print(f"{'='*80}")
        print(generate_pine_qcvd_lookup(qcvd_best))

    if qsmc_best:
        print(f"\n{'─'*70}")
        print("QSMC — BEST PARAMETERS PER SYMBOL/TIMEFRAME")
        print(f"{'─'*70}")
        rows = []
        for sym in ALL_SYMBOLS:
            if sym not in qsmc_best:
                continue
            for tf in ["1min", "5min", "15min", "1hr"]:
                if tf not in qsmc_best[sym]:
                    continue
                p = qsmc_best[sym][tf]
                rows.append([
                    sym, tf,
                    p["swing_length"], p["ob_strength"],
                    f"{p['flip_accuracy']:.1%}", f"{p['whipsaw_rate']:.1%}",
                    f"{p['avg_trend_duration']:.0f}",
                    f"{p['composite_score']:.3f}",
                    p["total_signals"], p["total_choch"], p["bars_tested"],
                ])
        headers = ["Symbol", "TF", "Swing", "OBStr",
                    "FlipAcc", "Whipsaw", "AvgDur", "Score", "Signals", "CHoCH", "Bars"]
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<6} {r[2]:<6} {r[3]:<6} "
                      f"{r[4]:<8} {r[5]:<8} {r[6]:<7} "
                      f"{r[7]:<7} {r[8]:<8} {r[9]:<6} {r[10]:<8}")

    if qgrid_best:
        print(f"\n{'─'*70}")
        print("QGRID — BEST PARAMETERS PER SYMBOL/TIMEFRAME")
        print(f"{'─'*70}")
        rows = []
        for sym in ALL_SYMBOLS:
            if sym not in qgrid_best:
                continue
            for tf in ["1min", "5min", "15min", "1hr"]:
                if tf not in qgrid_best[sym]:
                    continue
                p = qgrid_best[sym][tf]
                rows.append([
                    sym, tf,
                    p["left_bars"], p["right_bars"],
                    f"{p['bounce_rate']:.1%}", f"{p['interaction_rate']:.1%}",
                    f"{p['density_score']:.2f}",
                    f"{p['composite_score']:.3f}",
                    p["total_levels"], p["total_bounces"], p["bars_tested"],
                ])
        headers = ["Symbol", "TF", "Left", "Right",
                    "Bounce", "IntRate", "Density", "Score", "Levels", "Bounces", "Bars"]
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            for r in rows:
                print(f"{r[0]:<8} {r[1]:<6} {r[2]:<6} {r[3]:<6} "
                      f"{r[4]:<8} {r[5]:<8} {r[6]:<8} "
                      f"{r[7]:<7} {r[8]:<7} {r[9]:<8} {r[10]:<8}")

    if qsmc_best:
        print(f"\n{'='*80}")
        print("PINE SCRIPT — QSMC AUTO-OPTIMIZATION LOOKUP (copy-paste ready)")
        print(f"{'='*80}")
        print(generate_pine_qsmc_lookup(qsmc_best))

    if qgrid_best:
        print(f"\n{'='*80}")
        print("PINE SCRIPT — QGRID AUTO-OPTIMIZATION LOOKUP (copy-paste ready)")
        print(f"{'='*80}")
        print(generate_pine_qgrid_lookup(qgrid_best))

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")

    return {
        "qcloud_best": qcloud_best,
        "qline_best": qline_best,
        "qwave_best": qwave_best,
        "qbands_best": qbands_best,
        "moneyball_best": moneyball_best,
        "qmomentum_best": qmomentum_best,
        "qcvd_best": qcvd_best,
        "qsmc_best": qsmc_best,
        "qgrid_best": qgrid_best,
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YELENA v2 Parameter Optimizer — find optimal indicator settings per symbol/timeframe"
    )
    parser.add_argument("--symbol", type=str, help="Single symbol to optimize (e.g., SPY)")
    parser.add_argument("--timeframe", type=str, help="Single timeframe (1min, 5min, 15min, 1hr)")
    parser.add_argument("--indicator", type=str, help="Single indicator (qcloud, qline, qwave, qbands, moneyball, qmomentum, qcvd, qsmc, qgrid)")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer parameter combos")
    parser.add_argument("--db-pass", type=str, help="Database password (overrides env/SSM)")

    args = parser.parse_args()

    if args.db_pass:
        DB_PASS = args.db_pass

    symbols = [args.symbol] if args.symbol else None
    timeframes = [args.timeframe] if args.timeframe else None
    indicators = [args.indicator] if args.indicator else None

    run_full_optimization(
        symbols=symbols,
        timeframes=timeframes,
        indicators=indicators,
        quick=args.quick,
    )