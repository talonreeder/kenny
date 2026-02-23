"""Quick test of the Feature Engine on real data."""

import asyncio
import logging
import sys
from datetime import datetime, date, timedelta, timezone

import asyncpg
import pandas as pd
from app.config import get_settings
from app.services.feature_engine import FeatureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

settings = get_settings()


async def main():
    conn = await asyncpg.connect(settings.DATABASE_URL)

    # Fetch 500 recent 1-min bars for SPY
    rows = await conn.fetch("""
        SELECT time, symbol, open, high, low, close, volume
        FROM bars_1min
        WHERE symbol = 'SPY'
        ORDER BY time DESC
        LIMIT 500
    """)
    await conn.close()

    if not rows:
        logger.error("No data found!")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    df = df.sort_values("time").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} bars for {df['symbol'].iloc[0]}")
    logger.info(f"Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    engine = FeatureEngine()
    result = engine.compute_all(df)

    # Check the last row's features
    last_features = result["features"].iloc[-1]
    non_null = {k: v for k, v in last_features.items() if v is not None}
    null_count = sum(1 for v in last_features.values() if v is None)

    logger.info(f"Total features computed: {len(last_features)}")
    logger.info(f"Non-null features: {len(non_null)}")
    logger.info(f"Null features: {null_count}")
    logger.info(f"Feature version: {engine.version}")

    # Print a sample of features
    logger.info("--- Sample Features (last bar) ---")
    sample_keys = ["ema_8", "ema_21", "rsi_14", "macd_hist", "atr_14", "bb_percent", "volume_ratio", "obv", "candle_direction", "hour_of_day"]
    for key in sample_keys:
        val = last_features.get(key)
        logger.info(f"  {key}: {val}")


if __name__ == "__main__":
    asyncio.run(main())