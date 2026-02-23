# backend/scripts/backfill_historical.py

"""
Historical data backfill script.
Downloads daily and minute bars from Polygon.io and inserts into PostgreSQL.

Usage:
    python -m scripts.backfill_historical --type daily --months 24
    python -m scripts.backfill_historical --type minute --months 6
    python -m scripts.backfill_historical --symbol MSFT --start-date 2025-12-17 --end-date 2026-01-16
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

# All symbols to backfill
SYMBOLS = [
    "SPY", "QQQ", "TSLA", "NVDA", "META", "NFLX",
    "AAPL", "GOOGL", "MSFT", "AMZN", "AMD",
]
# Note: SPX is an index and may not have standard bars on Polygon


async def backfill_daily(months: int = 24):
    """Backfill daily bars for all symbols."""
    polygon = PolygonClient()

    db_url = settings.DATABASE_URL
    conn = await asyncpg.connect(db_url)

    end_date = date.today()
    start_date = end_date - timedelta(days=months * 30)

    logger.info(f"=== DAILY BACKFILL: {start_date} to {end_date} ({months} months) ===")

    total_inserted = 0

    for symbol in SYMBOLS:
        try:
            bars = await polygon.get_daily_bars(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            if not bars:
                logger.warning(f"No daily bars returned for {symbol}")
                continue

            # Insert bars into database
            inserted = 0
            for bar in bars:
                try:
                    ts = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc)
                    await conn.execute(
                        """
                        INSERT INTO bars_daily (time, symbol, open, high, low, close, volume, vwap, trade_count)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (symbol, time) DO NOTHING
                        """,
                        ts, symbol,
                        bar["o"], bar["h"], bar["l"], bar["c"],
                        bar["v"],
                        bar.get("vw"),
                        bar.get("n"),
                    )
                    inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting daily bar for {symbol}: {e}")

            total_inserted += inserted
            logger.info(f"  {symbol}: {inserted}/{len(bars)} daily bars inserted")

        except Exception as e:
            logger.error(f"Error fetching daily bars for {symbol}: {e}")

    await conn.close()
    await polygon.close()
    logger.info(f"=== DAILY BACKFILL COMPLETE: {total_inserted} total bars inserted ===")


async def backfill_minute(months: int = 6):
    """Backfill 1-minute bars for all symbols, one month at a time."""
    polygon = PolygonClient()

    db_url = settings.DATABASE_URL
    conn = await asyncpg.connect(db_url)

    end_date = date.today()
    start_date = end_date - timedelta(days=months * 30)

    logger.info(f"=== MINUTE BACKFILL: {start_date} to {end_date} ({months} months) ===")

    total_inserted = 0

    for symbol in SYMBOLS:
        symbol_inserted = 0

        # Process one month at a time to manage memory
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=30), end_date)

            try:
                bars = await polygon.get_minute_bars(
                    symbol,
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

                # Use executemany for bulk insert with ON CONFLICT
                await conn.executemany(
                    """
                    INSERT INTO bars_1min (time, symbol, open, high, low, close, volume, vwap, trade_count, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (symbol, time) DO NOTHING
                    """,
                    records,
                )

                inserted = len(records)
                symbol_inserted += inserted

                logger.info(f"  {symbol} [{current_start} to {current_end}]: {inserted} minute bars inserted")

            except Exception as e:
                logger.error(f"Error fetching minute bars for {symbol} ({current_start} to {current_end}): {e}")

            current_start = current_end + timedelta(days=1)

        total_inserted += symbol_inserted
        logger.info(f"  {symbol}: {symbol_inserted} total minute bars inserted")

    await conn.close()
    await polygon.close()
    logger.info(f"=== MINUTE BACKFILL COMPLETE: {total_inserted} total bars inserted ===")


async def backfill_minute_range(symbol: str, start: str, end: str):
    """Backfill minute bars for a single symbol over a specific date range."""
    polygon = PolygonClient()
    db_url = settings.DATABASE_URL
    conn = await asyncpg.connect(db_url)

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    logger.info(f"=== TARGETED MINUTE BACKFILL: {symbol} from {start_date} to {end_date} ===")

    total_inserted = 0
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=30), end_date)

        try:
            bars = await polygon.get_minute_bars(
                symbol,
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )

            if not bars:
                current_start = current_end + timedelta(days=1)
                continue

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
                """
                INSERT INTO bars_1min (time, symbol, open, high, low, close, volume, vwap, trade_count, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (symbol, time) DO NOTHING
                """,
                records,
            )

            total_inserted += len(records)
            logger.info(f"  {symbol} [{current_start} to {current_end}]: {len(records)} minute bars inserted")

        except Exception as e:
            logger.error(f"Error fetching minute bars for {symbol} ({current_start} to {current_end}): {e}")

        current_start = current_end + timedelta(days=1)

    await conn.close()
    await polygon.close()
    logger.info(f"=== TARGETED BACKFILL COMPLETE: {total_inserted} bars inserted for {symbol} ===")


async def main():
    parser = argparse.ArgumentParser(description="Backfill historical market data from Polygon.io")
    parser.add_argument("--type", choices=["daily", "minute", "both"], default="both", help="Type of data to backfill")
    parser.add_argument("--months", type=int, default=None, help="Number of months to backfill (default: 24 for daily, 6 for minute)")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to backfill (used with --start-date and --end-date)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date for targeted backfill (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date for targeted backfill (YYYY-MM-DD)")

    args = parser.parse_args()

    # Targeted single-symbol backfill
    if args.symbol and args.start_date and args.end_date:
        await backfill_minute_range(args.symbol, args.start_date, args.end_date)
        return

    if args.type in ("daily", "both"):
        months = args.months or 24
        await backfill_daily(months)

    if args.type in ("minute", "both"):
        months = args.months or 6
        await backfill_minute(months)


if __name__ == "__main__":
    asyncio.run(main())