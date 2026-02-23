# backend/app/services/polygon_client.py

import httpx
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

BASE_URL = "https://api.polygon.io"

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds: 2, 4, 8


class PolygonClient:
    """Async client for Polygon.io REST API with retry logic."""

    def __init__(self):
        self.api_key = settings.POLYGON_API_KEY
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def _get(self, url: str, params: dict = None) -> dict:
        """Make an authenticated GET request to Polygon.io with retry logic."""
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.get(f"{BASE_URL}{url}", params=params)
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(f"Timeout on {url} (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s...")
                if attempt == MAX_RETRIES:
                    logger.error(f"Max retries reached for {url}")
                    raise
                await asyncio.sleep(wait)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(f"HTTP {e.response.status_code} on {url} (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s...")
                    if attempt == MAX_RETRIES:
                        logger.error(f"Max retries reached for {url}")
                        raise
                    await asyncio.sleep(wait)
                else:
                    raise
            except httpx.RequestError as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(f"Request error on {url}: {e} (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s...")
                if attempt == MAX_RETRIES:
                    logger.error(f"Max retries reached for {url}")
                    raise
                await asyncio.sleep(wait)

    async def _get_url(self, full_url: str) -> dict:
        """Make an authenticated GET to a full URL (for pagination) with retry logic."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.get(full_url, params={"apiKey": self.api_key})
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.RequestError) as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(f"Error on paginated request (attempt {attempt}/{MAX_RETRIES}): {e}, retrying in {wait}s...")
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(wait)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(f"HTTP {e.response.status_code} on paginated request (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s...")
                    if attempt == MAX_RETRIES:
                        raise
                    await asyncio.sleep(wait)
                else:
                    raise

    async def get_daily_bars(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> list[dict]:
        """
        Fetch daily OHLCV bars for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            from_date: Start date 'YYYY-MM-DD'
            to_date: End date 'YYYY-MM-DD'

        Returns:
            List of bar dicts with t, o, h, l, c, v, vw, n fields
        """
        all_results = []
        url = f"/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}

        data = await self._get(url, params)
        results = data.get("results", [])
        all_results.extend(results)

        while data.get("next_url"):
            data = await self._get_url(data["next_url"])
            results = data.get("results", [])
            all_results.extend(results)

        logger.info(f"Fetched {len(all_results)} daily bars for {symbol} ({from_date} to {to_date})")
        return all_results

    async def get_minute_bars(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> list[dict]:
        """
        Fetch 1-minute OHLCV bars for a symbol.
        Polygon returns max 50,000 results per request.
        For large date ranges, we paginate automatically.

        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            from_date: Start date 'YYYY-MM-DD'
            to_date: End date 'YYYY-MM-DD'

        Returns:
            List of bar dicts
        """
        all_results = []
        url = f"/v2/aggs/ticker/{symbol}/range/1/minute/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}

        data = await self._get(url, params)
        results = data.get("results", [])
        all_results.extend(results)

        while data.get("next_url"):
            data = await self._get_url(data["next_url"])
            results = data.get("results", [])
            all_results.extend(results)
            logger.info(f"  ... {symbol} minute bars: {len(all_results)} fetched so far")

        logger.info(f"Fetched {len(all_results)} minute bars for {symbol} ({from_date} to {to_date})")
        return all_results

    async def get_bars(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str,
    ) -> list[dict]:
        """
        Fetch OHLCV bars for any timeframe from Polygon.io.

        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            multiplier: Size of the timespan multiplier (e.g., 5 for 5-minute)
            timespan: Timespan unit: 'minute', 'hour', 'day', 'week', 'month'
            from_date: Start date 'YYYY-MM-DD'
            to_date: End date 'YYYY-MM-DD'

        Returns:
            List of bar dicts with t, o, h, l, c, v, vw, n fields
        """
        all_results = []
        url = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}

        data = await self._get(url, params)
        results = data.get("results", [])
        all_results.extend(results)

        while data.get("next_url"):
            data = await self._get_url(data["next_url"])
            results = data.get("results", [])
            all_results.extend(results)
            logger.info(f"  ... {symbol} {multiplier}{timespan} bars: {len(all_results)} fetched so far")

        logger.info(f"Fetched {len(all_results)} {multiplier}{timespan} bars for {symbol} ({from_date} to {to_date})")
        return all_results

    async def get_options_chain(
        self,
        underlying: str,
        expiry_date: Optional[str] = None,
        option_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch options chain snapshot for an underlying symbol.

        Args:
            underlying: Underlying ticker (e.g., 'SPY')
            expiry_date: Optional filter by expiry 'YYYY-MM-DD'
            option_type: Optional 'call' or 'put'
        """
        url = f"/v3/snapshot/options/{underlying}"
        params = {"limit": 250}
        if expiry_date:
            params["expiration_date"] = expiry_date
        if option_type:
            params["contract_type"] = option_type

        all_results = []
        data = await self._get(url, params)
        results = data.get("results", [])
        all_results.extend(results)

        while data.get("next_url"):
            data = await self._get_url(data["next_url"])
            results = data.get("results", [])
            all_results.extend(results)

        logger.info(f"Fetched {len(all_results)} options contracts for {underlying}")
        return all_results