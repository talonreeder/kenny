"""
YELENA v2 — Training Data Generator (v2)
==========================================
Generates labeled training data for ML models by:
1. Loading OHLCV bars from PostgreSQL
2. Computing ATR for dynamic SL/TP calculation
3. Forward-looking simulation: does price hit TP before SL?
4. Joining with pre-computed v2 features from features table (JSONB)
5. Exporting to CSV → uploading to S3

THIS VERSION pulls all features from the unified Feature Engine v2.
No more inline feature computation — one source of truth.

Labeling Strategy (Hybrid ATR + R:R):
- From each bar, look forward N bars
- Virtual SL = 1.5× ATR against trade direction
- Virtual TP = 2.0× ATR in favorable direction
- CALL_WIN  = price hits TP (upside) before SL (downside)
- CALL_LOSS = price hits SL before TP
- PUT_WIN   = price hits TP (downside) before SL (upside)
- PUT_LOSS  = price hits SL before TP
- NEUTRAL   = neither hit within N bars (excluded from training)

Labels are direction-agnostic: each bar gets BOTH a call_label and put_label.
The model learns: "given these features, is a CALL profitable? Is a PUT profitable?"

Usage:
    cd ~/yelena/backend
    source ../venv/bin/activate
    python -m scripts.generate_training_data                        # All symbols, 5min
    python -m scripts.generate_training_data --symbol SPY           # Single symbol
    python -m scripts.generate_training_data --timeframe 5min       # Specific timeframe
    python -m scripts.generate_training_data --timeframe 1min       # 1-minute bars
    python -m scripts.generate_training_data --upload               # Upload to S3
    python -m scripts.generate_training_data --lookforward 20       # Custom lookforward
    python -m scripts.generate_training_data --sl-mult 1.5 --tp-mult 2.0  # Custom SL/TP

Requirements:
    pip install psycopg2-binary pandas numpy boto3 ta-lib

Changelog:
    v1 (Feb 2026):  Inline feature computation (67 features), 5min only
    v2 (Feb 2026):  Pulls from Feature Engine v2 (163+ features), all timeframes
                     Removed inline compute_features() — single source of truth
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    import talib
except ImportError:
    print("ERROR: TA-Lib not installed. Required for ATR computation.")
    print("Run: pip install ta-lib")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

DB_HOST = os.environ.get("YELENA_DB_HOST", "yelena-db.ci3ykeq8mckh.us-east-1.rds.amazonaws.com")
DB_NAME = os.environ.get("YELENA_DB_NAME", "yelena")
DB_USER = os.environ.get("YELENA_DB_USER", "postgres")
DB_PASS = os.environ.get("YELENA_DB_PASS", "")

if not DB_PASS:
    try:
        import boto3
        ssm = boto3.client("ssm", region_name="us-east-1")
        resp = ssm.get_parameter(Name="/yelena/database-url", WithDecryption=True)
        db_url = resp["Parameter"]["Value"]
        if "@" in db_url and ":" in db_url:
            DB_PASS = db_url.split("://")[1].split(":")[1].split("@")[0]
    except Exception as e:
        print(f"WARNING: Could not read DB password from SSM: {e}")

ALL_SYMBOLS = ["SPY", "QQQ", "TSLA", "NVDA", "META", "NFLX", "AAPL", "GOOGL", "MSFT", "AMZN", "AMD"]

TIMEFRAMES = {
    "1min":  ("bars_1min",  15),   # (table_name, default_lookforward_bars)
    "5min":  ("bars_5min",  15),
    "15min": ("bars_15min", 10),
    "1hr":   ("bars_1hr",   8),
}

S3_BUCKET = "yelena-data-lake"
S3_PREFIX = "training-data"

OUTPUT_DIR = Path.home() / "yelena" / "ml" / "training_data"


# ============================================================================
# DATABASE
# ============================================================================

def get_connection():
    """Create database connection."""
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=5432
    )


def load_bars(conn, table: str, symbol: str) -> pd.DataFrame:
    """Load OHLCV bars for a symbol, filtered to market hours."""
    query = f"""
        SELECT time, open, high, low, close, volume
        FROM {table}
        WHERE symbol = %s
          AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5
          AND EXTRACT(HOUR FROM time AT TIME ZONE 'America/New_York') >= 9
          AND (
              EXTRACT(HOUR FROM time AT TIME ZONE 'America/New_York') < 16
              OR (EXTRACT(HOUR FROM time AT TIME ZONE 'America/New_York') = 9
                  AND EXTRACT(MINUTE FROM time AT TIME ZONE 'America/New_York') >= 30)
          )
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn, params=(symbol,), parse_dates=["time"])
    df = df.set_index("time").sort_index()
    return df


def load_features(conn, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load pre-computed v2 features from JSONB column.

    Returns DataFrame with feature columns, indexed by time.
    Raises ValueError if no features found (v2 backfill required).
    """
    query = """
        SELECT time, features, version
        FROM features
        WHERE symbol = %s AND timeframe = %s
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn, params=(symbol, timeframe), parse_dates=["time"])

    if df.empty:
        raise ValueError(
            f"No features found for {symbol}/{timeframe} in features table. "
            f"Run backfill first: python -m scripts.backfill_features --symbol {symbol} --timeframe {timeframe}"
        )

    # Check version
    avg_version = df["version"].mean()
    if avg_version < 2:
        print(f"  WARNING: Features for {symbol}/{timeframe} are version {avg_version:.1f}. "
              f"Expected v2. Consider re-running backfill.")

    # Unpack JSONB into columns
    features_df = pd.json_normalize(df["features"])
    features_df.index = df["time"]

    # Report stats
    n_features = len(features_df.columns)
    non_null_pct = features_df.notna().mean().mean() * 100
    print(f"  Loaded {len(features_df):,} feature rows ({n_features} features, {non_null_pct:.0f}% non-null)")

    return features_df


# ============================================================================
# LABELING ENGINE
# ============================================================================

def generate_labels(
    df: pd.DataFrame,
    atr: np.ndarray,
    lookforward: int = 15,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Forward-looking SL/TP simulation for labeling.

    For each bar, simulates BOTH call and put directions:
    - CALL: TP = close + tp_mult * ATR, SL = close - sl_mult * ATR
    - PUT:  TP = close - tp_mult * ATR, SL = close + sl_mult * ATR

    Returns DataFrame with columns:
    - call_label:  1 (WIN), 0 (LOSS), -1 (NEUTRAL/timeout)
    - put_label:   1 (WIN), 0 (LOSS), -1 (NEUTRAL/timeout)
    - call_tp:     TP price for call
    - call_sl:     SL price for call
    - put_tp:      TP price for put
    - put_sl:      SL price for put
    - max_favorable_call:  Max price reached within lookforward
    - max_adverse_call:    Min price reached within lookforward
    - bars_to_result_call: Bars until call resolved
    - bars_to_result_put:  Bars until put resolved
    """
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Pre-allocate arrays
    call_label = np.full(n, -1, dtype=np.int8)   # -1 = NEUTRAL
    put_label = np.full(n, -1, dtype=np.int8)
    call_tp = np.full(n, np.nan)
    call_sl = np.full(n, np.nan)
    put_tp = np.full(n, np.nan)
    put_sl = np.full(n, np.nan)
    max_favorable_call = np.full(n, np.nan)
    max_adverse_call = np.full(n, np.nan)
    bars_to_result_call = np.full(n, lookforward, dtype=np.int16)
    bars_to_result_put = np.full(n, lookforward, dtype=np.int16)

    for i in range(n - lookforward):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry = close[i]
        a = atr[i]

        # CALL simulation
        c_tp = entry + tp_mult * a
        c_sl = entry - sl_mult * a
        call_tp[i] = c_tp
        call_sl[i] = c_sl

        # PUT simulation
        p_tp = entry - tp_mult * a
        p_sl = entry + sl_mult * a
        put_tp[i] = p_tp
        put_sl[i] = p_sl

        # Track max favorable / adverse for call
        max_hi = entry
        min_lo = entry

        # Walk forward bar by bar
        for j in range(i + 1, min(i + 1 + lookforward, n)):
            bar_high = high[j]
            bar_low = low[j]
            max_hi = max(max_hi, bar_high)
            min_lo = min(min_lo, bar_low)

            # CALL: check if TP or SL hit
            if call_label[i] == -1:
                if bar_high >= c_tp:
                    call_label[i] = 1   # WIN
                    bars_to_result_call[i] = j - i
                if bar_low <= c_sl:
                    if call_label[i] == 1:
                        # Both hit same bar — check which was hit first
                        # Assume SL hit first if open is closer to SL
                        if abs(df["open"].iloc[j] - c_sl) < abs(df["open"].iloc[j] - c_tp):
                            call_label[i] = 0  # LOSS
                    else:
                        call_label[i] = 0   # LOSS
                        bars_to_result_call[i] = j - i

            # PUT: check if TP or SL hit
            if put_label[i] == -1:
                if bar_low <= p_tp:
                    put_label[i] = 1    # WIN
                    bars_to_result_put[i] = j - i
                if bar_high >= p_sl:
                    if put_label[i] == 1:
                        if abs(df["open"].iloc[j] - p_sl) < abs(df["open"].iloc[j] - p_tp):
                            put_label[i] = 0
                    else:
                        put_label[i] = 0    # LOSS
                        bars_to_result_put[i] = j - i

            # Early exit if both resolved
            if call_label[i] != -1 and put_label[i] != -1:
                break

        max_favorable_call[i] = max_hi
        max_adverse_call[i] = min_lo

    labels = pd.DataFrame({
        "call_label": call_label,
        "put_label": put_label,
        "call_tp": call_tp,
        "call_sl": call_sl,
        "put_tp": put_tp,
        "put_sl": put_sl,
        "max_favorable_call": max_favorable_call,
        "max_adverse_call": max_adverse_call,
        "bars_to_result_call": bars_to_result_call,
        "bars_to_result_put": bars_to_result_put,
    }, index=df.index)

    return labels


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_for_symbol(
    conn,
    symbol: str,
    timeframe: str,
    lookforward: int,
    sl_mult: float,
    tp_mult: float,
) -> Optional[pd.DataFrame]:
    """Generate complete training data for one symbol/timeframe."""

    table_name = TIMEFRAMES[timeframe][0]

    # Load OHLCV
    print(f"  Loading {symbol} {timeframe} bars from {table_name}...")
    df = load_bars(conn, table_name, symbol)
    if df.empty:
        print(f"  WARNING: No bars found for {symbol} in {table_name}")
        return None
    print(f"  Loaded {len(df):,} bars ({df.index.min()} to {df.index.max()})")

    # Compute ATR (needed for labeling — this is the only TA-Lib dependency left)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    atr = talib.ATR(h, l, c, timeperiod=14)

    # Generate labels
    print(f"  Generating labels (lookforward={lookforward}, SL={sl_mult}×ATR, TP={tp_mult}×ATR)...")
    labels = generate_labels(df, atr, lookforward, sl_mult, tp_mult)

    # Load pre-computed v2 features from features table
    print(f"  Loading v2 features from features table...")
    features = load_features(conn, symbol, timeframe)

    # Combine OHLCV + features + labels
    result = pd.DataFrame(index=df.index)
    result["symbol"] = symbol
    result["timeframe"] = timeframe
    result["open"] = df["open"]
    result["high"] = df["high"]
    result["low"] = df["low"]
    result["close"] = df["close"]
    result["volume"] = df["volume"]
    result["atr_14"] = atr

    # Add features (align by index, prefix with f_ for easy identification)
    features_aligned = features.reindex(result.index)
    feature_cols = []
    for col in features_aligned.columns:
        col_name = f"f_{col}"
        result[col_name] = features_aligned[col]
        feature_cols.append(col_name)

    # Add labels
    for col in labels.columns:
        result[col] = labels[col]

    # Filter out rows we can't use
    # Drop: first 200 bars (warmup), last lookforward bars (no labels), NEUTRAL labels
    warmup = 200
    result = result.iloc[warmup:-lookforward] if lookforward > 0 else result.iloc[warmup:]

    # Drop rows where ATR is NaN
    result = result.dropna(subset=["atr_14"])

    # Drop rows with no features (features table didn't cover this time range)
    # Check if at least 50% of feature columns are non-null
    feature_null_pct = result[feature_cols].isnull().mean(axis=1)
    rows_before = len(result)
    result = result[feature_null_pct < 0.5]
    rows_dropped = rows_before - len(result)
    if rows_dropped > 0:
        print(f"  Dropped {rows_dropped:,} rows with >50% null features (no feature coverage)")

    # Stats
    call_wins = (result["call_label"] == 1).sum()
    call_losses = (result["call_label"] == 0).sum()
    call_neutral = (result["call_label"] == -1).sum()
    put_wins = (result["put_label"] == 1).sum()
    put_losses = (result["put_label"] == 0).sum()
    put_neutral = (result["put_label"] == -1).sum()

    total_usable = len(result[result["call_label"] >= 0])
    call_wr = call_wins / max(call_wins + call_losses, 1) * 100
    put_wr = put_wins / max(put_wins + put_losses, 1) * 100

    print(f"  Results for {symbol} {timeframe}:")
    print(f"    Total bars:          {len(result):,}")
    print(f"    Feature columns:     {len(feature_cols)}")
    print(f"    CALL — Win: {call_wins:,}  Loss: {call_losses:,}  Neutral: {call_neutral:,}  WR: {call_wr:.1f}%")
    print(f"    PUT  — Win: {put_wins:,}  Loss: {put_losses:,}  Neutral: {put_neutral:,}  WR: {put_wr:.1f}%")
    print(f"    Usable rows (non-neutral): {total_usable:,}")

    return result


def main():
    parser = argparse.ArgumentParser(description="YELENA v2 — Training Data Generator (v2)")
    parser.add_argument("--symbol", type=str, help="Single symbol (default: all)")
    parser.add_argument("--timeframe", type=str, default="5min", choices=list(TIMEFRAMES.keys()))
    parser.add_argument("--lookforward", type=int, help="Bars to look forward (default: per-timeframe)")
    parser.add_argument("--sl-mult", type=float, default=1.5, help="SL ATR multiplier (default: 1.5)")
    parser.add_argument("--tp-mult", type=float, default=2.0, help="TP ATR multiplier (default: 2.0)")
    parser.add_argument("--upload", action="store_true", help="Upload to S3 after generating")
    parser.add_argument("--output-dir", type=str, help="Custom output directory")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ALL_SYMBOLS
    timeframe = args.timeframe
    lookforward = args.lookforward or TIMEFRAMES[timeframe][1]
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    print("=" * 70)
    print("YELENA v2 — Training Data Generator (v2)")
    print("=" * 70)
    print(f"Symbols:     {', '.join(symbols)}")
    print(f"Timeframe:   {timeframe}")
    print(f"Lookforward: {lookforward} bars")
    print(f"SL:          {args.sl_mult}× ATR")
    print(f"TP:          {args.tp_mult}× ATR")
    print(f"Output:      {output_dir}")
    print(f"Features:    Pre-computed v2 from features table")
    print("=" * 70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect
    print("\nConnecting to database...")
    conn = get_connection()
    print("Connected.\n")

    all_data = []
    start_time = time.time()

    for symbol in symbols:
        print(f"\n{'─' * 50}")
        print(f"Processing {symbol}...")
        print(f"{'─' * 50}")

        try:
            result = generate_for_symbol(
                conn, symbol, timeframe, lookforward,
                args.sl_mult, args.tp_mult,
            )
            if result is not None:
                all_data.append(result)
        except ValueError as e:
            print(f"  SKIPPED: {e}")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    conn.close()

    if not all_data:
        print("\nERROR: No data generated. Ensure features are backfilled first:")
        print(f"  python -m scripts.backfill_features --timeframe {timeframe}")
        sys.exit(1)

    # Combine all symbols
    combined = pd.concat(all_data)
    elapsed = time.time() - start_time

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("COMBINED DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total rows:           {len(combined):,}")
    print(f"Symbols:              {combined['symbol'].nunique()}")
    print(f"Date range:           {combined.index.min()} to {combined.index.max()}")

    feature_cols = [c for c in combined.columns if c.startswith("f_")]
    print(f"Feature columns:      {len(feature_cols)}")

    # Feature completeness check
    feature_null_pct = combined[feature_cols].isnull().mean()
    mostly_null = feature_null_pct[feature_null_pct > 0.5]
    if len(mostly_null) > 0:
        print(f"\nWARNING: {len(mostly_null)} features are >50% null:")
        for col, pct in mostly_null.head(10).items():
            print(f"  {col}: {pct*100:.1f}% null")
        if len(mostly_null) > 10:
            print(f"  ... and {len(mostly_null) - 10} more")

    # Label distribution
    call_usable = combined[combined["call_label"] >= 0]
    put_usable = combined[combined["put_label"] >= 0]
    print(f"\nCALL labels:")
    print(f"  Win:     {(call_usable['call_label'] == 1).sum():,} ({(call_usable['call_label'] == 1).mean()*100:.1f}%)")
    print(f"  Loss:    {(call_usable['call_label'] == 0).sum():,} ({(call_usable['call_label'] == 0).mean()*100:.1f}%)")
    print(f"  Neutral: {(combined['call_label'] == -1).sum():,} (excluded)")
    print(f"\nPUT labels:")
    print(f"  Win:     {(put_usable['put_label'] == 1).sum():,} ({(put_usable['put_label'] == 1).mean()*100:.1f}%)")
    print(f"  Loss:    {(put_usable['put_label'] == 0).sum():,} ({(put_usable['put_label'] == 0).mean()*100:.1f}%)")
    print(f"  Neutral: {(combined['put_label'] == -1).sum():,} (excluded)")

    print(f"\nGeneration time:      {elapsed:.1f}s")

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_data_{timeframe}_{timestamp}.csv"
    filepath = output_dir / filename

    print(f"\nSaving to {filepath}...")
    combined.to_csv(filepath, index=True, index_label="time")
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved: {file_size_mb:.1f} MB")

    # Also save a metadata file
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": 2,
        "feature_engine_version": 2,
        "timeframe": timeframe,
        "lookforward": lookforward,
        "sl_mult": args.sl_mult,
        "tp_mult": args.tp_mult,
        "symbols": symbols,
        "total_rows": len(combined),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "call_win_rate": float((call_usable["call_label"] == 1).mean()) if len(call_usable) > 0 else 0,
        "put_win_rate": float((put_usable["put_label"] == 1).mean()) if len(put_usable) > 0 else 0,
        "date_range_start": str(combined.index.min()),
        "date_range_end": str(combined.index.max()),
    }
    meta_path = output_dir / f"metadata_{timeframe}_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # Upload to S3
    if args.upload:
        print(f"\nUploading to s3://{S3_BUCKET}/{S3_PREFIX}/...")
        try:
            import boto3
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.upload_file(str(filepath), S3_BUCKET, f"{S3_PREFIX}/{filename}")
            s3.upload_file(str(meta_path), S3_BUCKET, f"{S3_PREFIX}/{meta_path.name}")
            print(f"Uploaded to S3 ✓")
        except Exception as e:
            print(f"S3 upload failed: {e}")
            print("Files saved locally — upload manually with:")
            print(f"  aws s3 cp {filepath} s3://{S3_BUCKET}/{S3_PREFIX}/{filename}")

    print(f"\n{'=' * 70}")
    print("TRAINING DATA GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nNext steps:")
    print(f"  1. Upload CSV to Colab or S3: {filepath}")
    print(f"  2. Retrain all 4 models with {len(feature_cols)} v2 features")
    print(f"  3. Target: ≥65% accuracy, ≥1.5 profit factor")
    print(f"  4. Repeat for all timeframes: 1min, 5min, 15min, 1hr")


if __name__ == "__main__":
    main()
