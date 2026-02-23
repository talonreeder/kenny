"""
Backfill 5min, 15min, and 1hr bars from Polygon.io.
Uses the generic get_bars() method on PolygonClient.

Usage:
    cd ~/yelena/backend
    python -m scripts.backfill_multiframe --timeframe 5min
    python -m scripts.backfill_multiframe --timeframe 15min
    python -m scripts.backfill_multiframe --timeframe 1hr
    python -m scripts.backfill_multiframe --timeframe all
    python -m scripts.backfill_multiframe --timeframe 5min --symbol SPY
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, date, timedelta, timezone

import asyncpg
from app.config import get_settings
from app.services.polygon_client import PolygonClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

settings = get_settings()

SYMBOLS = [
    "SPY", "QQQ", "TSLA", "NVDA", "META", "NFLX",
    "AAPL", "GOOGL", "MSFT", "AMZN", "AMD",
]

# Timeframe configurations: (multiplier, timespan, table_name, chunk_days)
# chunk_days = how many days per API request to manage memory
TIMEFRAME_CONFIG = {
    "5min": (5, "minute", "bars_5min", 30),
    "15min": (15, "minute", "bars_15min", 60),
    "1hr": (1, "hour", "bars_1hr", 90),
}


async def backfill_timeframe(
    timeframe: str,
    symbols: list[str],
    months: int = 24,
):
    """Backfill bars for a specific timeframe."""
    if timeframe not in TIMEFRAME_CONFIG:
        logger.error(f"Unknown timeframe: {timeframe}. Options: {list(TIMEFRAME_CONFIG.keys())}")
        return

    multiplier, timespan, table_name, chunk_days = TIMEFRAME_CONFIG[timeframe]

    polygon = PolygonClient()
    conn = await asyncpg.connect(settings.DATABASE_URL)

    end_date = date.today()
    start_date = end_date - timedelta(days=months * 30)

    logger.info(f"=== {timeframe.upper()} BACKFILL: {start_date} to {end_date} ({months} months) ===")
    logger.info(f"    Table: {table_name} | Polygon params: {multiplier} {timespan}")
    logger.info(f"    Symbols: {', '.join(symbols)}")

    total_inserted = 0

    for symbol in symbols:
        symbol_inserted = 0
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)

            try:
                bars = await polygon.get_bars(
                    symbol,
                    multiplier,
                    timespan,
                    current_start.strftime("%Y-%m-%d"),
                    current_end.strftime("%Y-%m-%d"),
                )

                if not bars:
                    current_start = current_end + timedelta(days=1)
                    continue

                # Build records for bulk insert
                records = []
                for bar in bars:
                    ts = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc)
                    records.append((
                        ts, symbol,
                        bar["o"], bar["h"], bar["l"], bar["c"],
                        bar["v"],
                        bar.get("vw"),
                        bar.get("n"),
                        "polygon",
                    ))

                await conn.executemany(
                    f"""
                    INSERT INTO {table_name} (time, symbol, open, high, low, close, volume, vwap, trade_count, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (symbol, time) DO NOTHING
                    """,
                    records,
                )

                symbol_inserted += len(records)
                logger.info(f"  {symbol} [{current_start} to {current_end}]: {len(records)} {timeframe} bars inserted")

            except Exception as e:
                logger.error(f"Error fetching {timeframe} bars for {symbol} ({current_start} to {current_end}): {e}")

            current_start = current_end + timedelta(days=1)

        total_inserted += symbol_inserted
        logger.info(f"  {symbol}: {symbol_inserted} total {timeframe} bars inserted")

    await conn.close()
    await polygon.close()
    logger.info(f"=== {timeframe.upper()} BACKFILL COMPLETE: {total_inserted} total bars inserted ===")
    return total_inserted


async def main():
    parser = argparse.ArgumentParser(description="Backfill 5min/15min/1hr bars from Polygon.io")
    parser.add_argument("--timeframe", choices=["5min", "15min", "1hr", "all"], required=True, help="Timeframe to backfill")
    parser.add_argument("--months", type=int, default=24, help="Number of months to backfill (default: 24)")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol (default: all 11)")

    args = parser.parse_args()
    symbols = [args.symbol] if args.symbol else SYMBOLS

    if args.timeframe == "all":
        grand_total = 0
        for tf in ["5min", "15min", "1hr"]:
            count = await backfill_timeframe(tf, symbols, args.months)
            grand_total += (count or 0)
        logger.info(f"=== ALL TIMEFRAMES COMPLETE: {grand_total} total bars across all timeframes ===")
    else:
        await backfill_timeframe(args.timeframe, symbols, args.months)


if __name__ == "__main__":
    asyncio.run(main())
