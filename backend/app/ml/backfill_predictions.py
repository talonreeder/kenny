"""
YELENA v2 — Prediction Backfill (v2 — FeatureEngine)
=====================================================
Runs ML models against historical bar data to generate predictions
for backtesting. Replicates the orchestrator's exact pipeline:
  1. Load historical bars from DB
  2. Feed bars into streaming FeatureEngine
  3. compute_features() → feature vector
  4. get_sequence() → sequence for Transformer/CNN
  5. POST features to prediction service → get prediction
  6. Save prediction to DB

Usage:
    python backfill_predictions.py --symbols SPY,QQQ --days 30
    python backfill_predictions.py --symbols SPY --start 2026-02-01 --end 2026-02-20
    python backfill_predictions.py --symbols SPY,QQQ --days 5 --dry-run
"""

import os
import sys
import json
import argparse
import logging
import time as time_module
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import psycopg2
import psycopg2.extras
import numpy as np
import requests

# Add this directory so we can import feature_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engine import FeatureEngine, OHLCVBar

logger = logging.getLogger("yelena.backfill_predictions")

# Prediction service URL
PREDICT_URL = os.getenv("YELENA_PREDICT_URL", "http://localhost:8001")

# FeatureEngine needs 210 bars for warmup (SMA-200 + buffer)
WARMUP_BARS = 210


def get_db_url(args_db_url: Optional[str] = None) -> str:
    """Get database URL from args or SSM."""
    if args_db_url:
        return args_db_url
    try:
        import boto3
        ssm = boto3.client("ssm", region_name="us-east-1")
        resp = ssm.get_parameter(Name="/yelena/database-url", WithDecryption=True)
        return resp["Parameter"]["Value"]
    except Exception as e:
        logger.error(f"Failed to get DB URL: {e}")
        sys.exit(1)


def load_bars(conn, symbol: str, timeframe: str, start: datetime, end: datetime,
              warmup_days: int = 60) -> List[dict]:
    """
    Load historical bars from database.
    Extends start backward by warmup_days to have enough data for feature computation.
    """
    table = f"bars_{timeframe}"
    warmup_start = start - timedelta(days=warmup_days)

    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(f"""
        SELECT time, symbol, open, high, low, close, volume
        FROM {table}
        WHERE symbol = %s AND time >= %s AND time <= %s
        ORDER BY time ASC
    """, (symbol, warmup_start, end))

    bars = []
    for row in cur.fetchall():
        bars.append({
            "time": row["time"],
            "symbol": row["symbol"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    cur.close()
    return bars


def is_market_hours(bar_time) -> bool:
    """Check if bar is during market hours (14:30-21:00 UTC = 9:30-4:00 ET)."""
    if not hasattr(bar_time, "hour"):
        return True
    hour = bar_time.hour
    minute = bar_time.minute
    if hour < 14 or hour >= 21:
        return False
    if hour == 14 and minute < 30:
        return False
    return True


def save_prediction(conn, symbol: str, timeframe: str, bar_time: datetime, prediction: dict):
    """Save a prediction to the database."""
    cur = conn.cursor()

    direction = prediction.get("direction", "HOLD")
    confidence = prediction.get("confidence", 0)

    probabilities = json.dumps({
        "direction": direction,
        "probability": prediction.get("probability", 0),
        "confidence": confidence,
        "grade": prediction.get("grade", ""),
        "models_agreeing": prediction.get("models_agreeing", 0),
        "unanimous": prediction.get("unanimous", False),
        "individual": prediction.get("individual", {}),
        "backfilled": True,
    })

    cur.execute("""
        INSERT INTO predictions (time, symbol, timeframe, model_name, signal, confidence, probabilities, model_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (
        bar_time, symbol, timeframe, "ensemble_v2",
        direction, confidence, probabilities, "2.0.0-backfill",
    ))
    conn.commit()
    cur.close()


def check_prediction_service() -> bool:
    """Check if the prediction service is running."""
    try:
        resp = requests.get(f"{PREDICT_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def backfill_symbol_timeframe(
    conn, engine: FeatureEngine, symbol: str, timeframe: str,
    bars: List[dict], target_start: datetime, dry_run: bool,
    batch_log_interval: int = 100,
) -> dict:
    """
    Backfill predictions for one symbol/timeframe.

    Strategy:
    - Load ALL bars (including warmup period before target_start)
    - Feed bars into FeatureEngine via update_bars_bulk for warmup
    - Then iterate through bars in the target range, updating one at a time
    - At each bar, compute features + sequence → POST to prediction service
    """
    stats = {"predictions": 0, "skipped_hold": 0, "skipped_market": 0, "errors": 0}

    if len(bars) < WARMUP_BARS + 10:
        logger.warning(f"  Not enough bars for {symbol} {timeframe}: {len(bars)} (need {WARMUP_BARS}+)")
        return stats

    # Find the index where target_start begins
    target_start_idx = 0
    for i, bar in enumerate(bars):
        if bar["time"] >= target_start:
            target_start_idx = i
            break

    # Need at least WARMUP_BARS before the target range
    if target_start_idx < WARMUP_BARS:
        logger.warning(
            f"  Only {target_start_idx} warmup bars for {symbol} {timeframe} "
            f"(need {WARMUP_BARS}). Adjusting start index."
        )
        target_start_idx = WARMUP_BARS

    # Bulk load warmup bars into FeatureEngine
    warmup_ohlcv = []
    for bar in bars[:target_start_idx]:
        warmup_ohlcv.append(OHLCVBar(
            timestamp=bar["time"].timestamp(),
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"],
        ))

    engine.update_bars_bulk(symbol, timeframe, warmup_ohlcv)
    bar_count = engine.bar_count(symbol, timeframe)
    logger.info(f"  Warmup: {bar_count} bars loaded (need {WARMUP_BARS})")

    if not engine.has_enough_bars(symbol, timeframe):
        logger.warning(f"  Not enough warmup bars for feature computation")
        return stats

    # Now iterate through target range bars
    total_target = len(bars) - target_start_idx
    start_time = time_module.time()

    for i in range(target_start_idx, len(bars)):
        bar = bars[i]

        # Update the feature engine with this bar
        ohlcv = OHLCVBar(
            timestamp=bar["time"].timestamp(),
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"],
        )
        engine.update_bar(symbol, timeframe, ohlcv)

        # Skip non-market hours
        if not is_market_hours(bar["time"]):
            stats["skipped_market"] += 1
            continue

        # Compute features
        try:
            features, feature_names = engine.compute_features(symbol, timeframe)
            sequence = engine.get_sequence(symbol, timeframe, seq_len=30)
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 3:
                logger.warning(f"  Feature computation error at {bar['time']}: {e}")
            continue

        # Build prediction payload (matches orchestrator's _run_prediction)
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "features": features.tolist(),
            "feature_names": feature_names,
        }
        if sequence is not None:
            payload["sequence"] = sequence.tolist()

        # POST to prediction service
        try:
            resp = requests.post(f"{PREDICT_URL}/predict", json=payload, timeout=10)
            if resp.status_code != 200:
                stats["errors"] += 1
                if stats["errors"] <= 3:
                    logger.warning(f"  Prediction service error at {bar['time']}: {resp.text[:200]}")
                continue
            result = resp.json()
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 3:
                logger.warning(f"  Request error at {bar['time']}: {e}")
            continue

        # Skip HOLD predictions
        direction = result.get("direction", "HOLD")
        if direction == "HOLD":
            stats["skipped_hold"] += 1
            continue

        # Save prediction
        if not dry_run:
            save_prediction(conn, symbol, timeframe, bar["time"], result)
        stats["predictions"] += 1

        # Progress logging
        progress = stats["predictions"] + stats["skipped_hold"] + stats["errors"]
        if progress % batch_log_interval == 0:
            elapsed = time_module.time() - start_time
            bar_idx = i - target_start_idx
            pct = bar_idx / total_target * 100 if total_target > 0 else 0
            rate = progress / elapsed if elapsed > 0 else 0
            logger.info(
                f"  {symbol} {timeframe}: {stats['predictions']} saved, "
                f"{stats['skipped_hold']} HOLD, {stats['errors']} errors "
                f"({pct:.0f}% done, {rate:.1f}/sec)"
            )

    elapsed = time_module.time() - start_time
    logger.info(
        f"  ✅ {symbol} {timeframe}: {stats['predictions']} predictions, "
        f"{stats['skipped_hold']} HOLD, {stats['skipped_market']} off-hours, "
        f"{stats['errors']} errors ({elapsed:.1f}s)"
    )

    return stats


def main():
    parser = argparse.ArgumentParser(description="YELENA v2 Prediction Backfill (v2)")
    parser.add_argument("--symbols", default="SPY,QQQ", help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backfill")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframes", default="1min,5min,15min,1hr", help="Timeframes to predict")
    parser.add_argument("--batch-size", type=int, default=100, help="Bars between progress logs")
    parser.add_argument("--db-url", default=None, help="Database URL (or uses SSM)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just count")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Check prediction service
    if not check_prediction_service():
        logger.error(f"Prediction service not running at {PREDICT_URL}")
        logger.error("Start it with: sudo systemctl start yelena-predict")
        sys.exit(1)
    logger.info(f"✅ Prediction service healthy at {PREDICT_URL}")

    # Database
    db_url = get_db_url(args.db_url)
    conn = psycopg2.connect(db_url)
    logger.info("✅ Connected to database")

    # Parse dates
    if args.start:
        target_start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        target_start = datetime.now(timezone.utc) - timedelta(days=args.days)

    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(
            hour=23, minute=59, tzinfo=timezone.utc
        )
    else:
        end = datetime.now(timezone.utc)

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]

    logger.info("=" * 60)
    logger.info(f"PREDICTION BACKFILL (v2 — FeatureEngine)")
    logger.info(f"  Date range: {target_start.date()} to {end.date()}")
    logger.info(f"  Symbols:    {symbols}")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Dry run:    {args.dry_run}")
    logger.info("=" * 60)

    totals = {"predictions": 0, "skipped_hold": 0, "skipped_market": 0, "errors": 0}

    for symbol in symbols:
        for tf in timeframes:
            logger.info(f"\n{'=' * 40}")
            logger.info(f"Processing {symbol} {tf}...")

            # Create fresh FeatureEngine per symbol/tf
            engine = FeatureEngine(max_bars=500)

            # Load bars with warmup window
            bars = load_bars(conn, symbol, tf, target_start, end, warmup_days=60)
            if not bars:
                logger.warning(f"  No bars found for {symbol} {tf}")
                continue
            logger.info(f"  Loaded {len(bars)} bars (including warmup)")

            stats = backfill_symbol_timeframe(
                conn, engine, symbol, tf, bars, target_start,
                dry_run=args.dry_run, batch_log_interval=args.batch_size,
            )

            for key in totals:
                totals[key] += stats[key]

    conn.close()

    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"  Predictions saved:  {totals['predictions']}")
    logger.info(f"  Skipped (HOLD):     {totals['skipped_hold']}")
    logger.info(f"  Skipped (off-hours):{totals['skipped_market']}")
    logger.info(f"  Errors:             {totals['errors']}")
    logger.info("=" * 60)

    if not args.dry_run and totals["predictions"] > 0:
        logger.info(f"\nNow run the backtest:")
        logger.info(f"  python backtest.py --symbols {','.join(symbols)} --days {args.days}")


if __name__ == "__main__":
    main()
