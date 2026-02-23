"""
KENNY Database — asyncpg connection pool for PostgreSQL.
"""

import logging
from typing import Optional, Any

import asyncpg

from app.config import get_settings

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self._settings = get_settings()

    async def connect(self):
        db_params = self._settings.parse_database_url()
        if not db_params:
            logger.error("No database URL configured. Check SSM parameter /yelena/database-url")
            raise ValueError("Database URL not configured")
        try:
            self.pool = await asyncpg.create_pool(
                min_size=2, max_size=10, command_timeout=30, **db_params
            )
            logger.info(f"Database pool created: {db_params['host']}:{db_params['port']}/{db_params['database']}")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def health_check(self) -> dict:
        if not self.pool:
            return {"status": "disconnected", "error": "No pool"}
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return {
                    "status": "healthy",
                    "pool_size": self.pool.get_size(),
                    "pool_free": self.pool.get_idle_size()
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def fetch(self, query: str, *args) -> list:
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args) -> str:
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def executemany(self, query: str, args: list) -> None:
        async with self.pool.acquire() as conn:
            await conn.executemany(query, args)

    async def get_table_counts(self) -> dict:
        tables = [
            "bars_1min", "bars_5min", "bars_15min", "bars_1hr", "bars_daily",
            "features", "symbols",
            "tv_signals", "verdicts", "predictions", "trade_plans", "trades"
        ]
        counts = {}
        for table in tables:
            try:
                count = await self.fetchval(f"SELECT COUNT(*) FROM {table}")
                counts[table] = count
            except Exception:
                counts[table] = None
        return counts


db = Database()
