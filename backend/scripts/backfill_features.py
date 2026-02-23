"""
YELENA v2 — Feature Backfill Script (v2)
==========================================
Backfills v2 features (170+) for all symbols across all timeframes.

Key differences from v1 backfill:
- Computes v2 features (SMC, VWAP, Divergence, Derived, MTF)
- Loads higher-timeframe OHLCV data for MTF alignment features
- Processes in batches to manage memory on t3.medium (4GB RAM)
- Upserts with version=2 to track feature version

MTF Timeframe Mapping:
    1min  → uses 5min, 15min, 1hr
    5min  → uses 15min, 1hr
    15min → uses 1hr
    1hr   → no higher TF

Usage:
    cd ~/yelena/backend
    source ../venv/bin/activate

    # Backfill all symbols, all timeframes:
    python -m scripts.backfill_features_v2

    # Single symbol:
    python -m scripts.backfill_features_v2 --symbol SPY

    # Single timeframe:
    python -m scripts.backfill_features_v2 --timeframe 5min

    # Single symbol + timeframe:
    python -m scripts.backfill_features_v2 --symbol SPY --timeframe 1min

    # Dry run (compute but don't write to DB):
    python -m scripts.backfill_features_v2 --dry-run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

# Add parent path so we can import from app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services.feature_engine import FeatureEngine, FEATURE_VERSION


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

# Table names for each timeframe
TIMEFRAME_TABLES = {
    "1min": "bars_1min",
    "5min": "bars_5min",
    "15min": "bars_15min",
    "1hr": "bars_1hr",
}

# Higher timeframe mapping for MTF features
HTF_MAPPING = {
    "1min": ["5min", "15min", "1hr"],
    "5min": ["15min", "1hr"],
    "15min": ["1hr"],
    "1hr": [],
}

# Processing order (highest TF first — no dependency, but good practice)
PROCESSING_ORDER = ["1hr", "15min", "5min", "1min"]

# Batch size for DB inserts (rows per batch)
BATCH_SIZE = 5000


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
    return df


def upsert_features(conn, rows: list, batch_size: int = BATCH_SIZE):
    """
    Upsert feature rows into the features table.

    Each row is a dict with keys: time, symbol, timeframe, features, version
    Uses ON CONFLICT to update existing rows.
    """
    if not rows:
        return 0

    total = len(rows)
    inserted = 0

    cur = conn.cursor()

    for batch_start in range(0, total, batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        values = []
        for row in batch:
            values.append((
                row["time"],
                row["symbol"],
                row["timeframe"],
                json.dumps(row["features"]),
                row["version"],
            ))

        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO features (time, symbol, timeframe, features, version)
            VALUES %s
            ON CONFLICT (symbol, timeframe, time)
            DO UPDATE SET features = EXCLUDED.features, version = EXCLUDED.version
            """,
            values,
            template="(%s, %s, %s, %s::jsonb, %s)",
            page_size=batch_size,
        )
        inserted += len(batch)

        if inserted % 10000 == 0 or inserted == total:
            conn.commit()
            print(f"    Upserted {inserted:,}/{total:,} rows...")

    conn.commit()
    cur.close()
    return inserted


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_features_for_symbol(
    conn,
    engine: FeatureEngine,
    symbol: str,
    timeframe: str,
    dry_run: bool = False,
) -> int:
    """
    Compute v2 features for a single symbol/timeframe combination.

    Returns number of rows processed.
    """
    table = TIMEFRAME_TABLES[timeframe]
    htf_labels = HTF_MAPPING[timeframe]

    # Load base timeframe OHLCV
    print(f"  Loading {symbol} {timeframe} from {table}...")
    df = load_bars(conn, table, symbol)
    if df.empty:
        print(f"  WARNING: No bars found for {symbol} in {table}")
        return 0
    print(f"  Loaded {len(df):,} bars ({df['time'].min()} to {df['time'].max()})")

    # Load higher timeframe data for MTF features
    htf_data = {}
    for htf_label in htf_labels:
        htf_table = TIMEFRAME_TABLES[htf_label]
        print(f"  Loading HTF data: {symbol} {htf_label} from {htf_table}...")
        htf_df = load_bars(conn, htf_table, symbol)
        if not htf_df.empty:
            htf_data[htf_label] = htf_df
            print(f"    {len(htf_df):,} bars loaded")
        else:
            print(f"    WARNING: No HTF data for {htf_label}")

    # Compute features
    print(f"  Computing v2 features ({engine.get_feature_count()['total']} categories)...")
    start = time.time()
    result_df = engine.compute_all(df, htf_data=htf_data if htf_data else None)
    elapsed = time.time() - start
    print(f"  Features computed in {elapsed:.1f}s")

    # Verify feature count on a sample row
    sample_idx = min(250, len(result_df) - 1)  # Past warmup period
    if sample_idx >= 0:
        sample_features = result_df.iloc[sample_idx]["features"]
        non_null = sum(1 for v in sample_features.values() if v is not None)
        total_keys = len(sample_features)
        print(f"  Sample row features: {total_keys} total, {non_null} non-null")

    if dry_run:
        print(f"  DRY RUN — skipping database write")
        return len(result_df)

    # Prepare rows for upsert
    print(f"  Preparing {len(result_df):,} rows for upsert...")
    rows = []
    for _, row in result_df.iterrows():
        rows.append({
            "time": row["time"],
            "symbol": symbol,
            "timeframe": timeframe,
            "features": row["features"],
            "version": FEATURE_VERSION,
        })

    # Upsert to database
    print(f"  Upserting to features table (version={FEATURE_VERSION})...")
    start = time.time()
    inserted = upsert_features(conn, rows)
    elapsed = time.time() - start
    print(f"  Upserted {inserted:,} rows in {elapsed:.1f}s")

    return inserted


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="YELENA v2 — Feature Backfill (v2)")
    parser.add_argument("--symbol", type=str, help="Single symbol (default: all 11)")
    parser.add_argument("--timeframe", type=str, choices=list(TIMEFRAME_TABLES.keys()),
                        help="Single timeframe (default: all 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute features but don't write to DB")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ALL_SYMBOLS
    timeframes = [args.timeframe] if args.timeframe else PROCESSING_ORDER

    print("=" * 70)
    print("YELENA v2 — Feature Backfill (v2)")
    print("=" * 70)
    print(f"Feature Version: {FEATURE_VERSION}")
    print(f"Symbols:         {', '.join(symbols)}")
    print(f"Timeframes:      {', '.join(timeframes)}")
    print(f"Dry Run:         {args.dry_run}")
    print(f"Batch Size:      {BATCH_SIZE:,}")
    print("=" * 70)

    engine = FeatureEngine()
    feature_counts = engine.get_feature_count()
    print(f"\nFeature breakdown:")
    for cat, count in feature_counts.items():
        if cat != "total":
            print(f"  {cat}: {count}")
    print(f"  TOTAL: {feature_counts['total']}")

    # Connect
    print("\nConnecting to database...")
    conn = get_connection()
    print("Connected.\n")

    total_rows = 0
    total_start = time.time()
    results = []

    for tf in timeframes:
        print(f"\n{'═' * 70}")
        print(f"TIMEFRAME: {tf}")
        print(f"  Higher TFs for MTF: {HTF_MAPPING[tf] or 'none (top level)'}")
        print(f"{'═' * 70}")

        for symbol in symbols:
            print(f"\n{'─' * 50}")
            print(f"  {symbol} / {tf}")
            print(f"{'─' * 50}")

            try:
                rows = compute_features_for_symbol(
                    conn, engine, symbol, tf, dry_run=args.dry_run
                )
                total_rows += rows
                results.append({"symbol": symbol, "timeframe": tf, "rows": rows, "status": "OK"})
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append({"symbol": symbol, "timeframe": tf, "rows": 0, "status": f"ERROR: {e}"})
                # Continue with other symbols
                continue

    conn.close()
    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'═' * 70}")
    print("BACKFILL COMPLETE")
    print(f"{'═' * 70}")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Total time:           {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Feature version:      {FEATURE_VERSION}")

    print(f"\nResults by symbol/timeframe:")
    print(f"{'Symbol':<10} {'Timeframe':<10} {'Rows':>10} {'Status'}")
    print(f"{'─'*10} {'─'*10} {'─'*10} {'─'*20}")
    for r in results:
        print(f"{r['symbol']:<10} {r['timeframe']:<10} {r['rows']:>10,} {r['status']}")

    # Verification query suggestion
    print(f"\nVerify with:")
    print(f"  psql -h {DB_HOST} -U {DB_USER} -d {DB_NAME} -c \"SELECT timeframe, COUNT(*) as rows, COUNT(DISTINCT symbol) as symbols, AVG(version) as avg_version FROM features GROUP BY timeframe ORDER BY timeframe;\"")


if __name__ == "__main__":
    main()
