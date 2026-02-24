"""
YELENA v2 — Real-Time Trading Orchestrator
Connects Polygon.io → Feature Engine → Prediction Service → Verdict Engine.

The orchestrator feeds raw ML predictions into the Verdict Engine,
which consolidates multi-timeframe signals into GO/SKIP trade verdicts.
Only actionable verdicts are pushed to the dashboard.
"""

import os
import sys
import json
import time
import asyncio
import signal
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

import numpy as np
import requests
import websockets

from app.ml.feature_engine import FeatureEngine, OHLCVBar, HTF_DROPS
from app.ml.verdict_engine import VerdictEngine, VerdictConfig, Prediction

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger("yelena.orchestrator")

@dataclass
class OrchestratorConfig:
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    timeframes: List[str] = field(default_factory=lambda: ["1min", "5min", "15min", "1hr"])
    polygon_api_key: str = ""
    predict_url: str = "http://localhost:8001"
    backend_url: str = "http://localhost:8000"
    warmup_bars: int = 300
    log_level: str = "INFO"

    ws_url: str = "wss://delayed.polygon.io/stocks"
    ws_reconnect_delay: float = 5.0
    ws_max_reconnects: int = 10

    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0


TF_TO_POLYGON = {"1min": "A", "5min": None, "15min": None, "1hr": None}
TF_AGGREGATE = {"5min": 5, "15min": 15, "1hr": 60}


# ============================================================================
# BAR AGGREGATOR
# ============================================================================

class BarAggregator:
    def __init__(self):
        self.state: Dict[str, Dict[str, dict]] = {}

    def init_symbol(self, symbol: str):
        self.state[symbol] = {}
        for tf, count in TF_AGGREGATE.items():
            self.state[symbol][tf] = {"count": count, "buffer": [], "current": None}

    def add_1min_bar(self, symbol: str, bar: OHLCVBar) -> Dict[str, Optional[OHLCVBar]]:
        if symbol not in self.state:
            self.init_symbol(symbol)

        completed = {}
        for tf, info in self.state[symbol].items():
            info["buffer"].append(bar)
            if len(info["buffer"]) >= info["count"]:
                bars = info["buffer"]
                agg_bar = OHLCVBar(
                    timestamp=bars[0].timestamp, open=bars[0].open,
                    high=max(b.high for b in bars), low=min(b.low for b in bars),
                    close=bars[-1].close, volume=sum(b.volume for b in bars)
                )
                completed[tf] = agg_bar
                info["buffer"] = []
            else:
                bars = info["buffer"]
                info["current"] = OHLCVBar(
                    timestamp=bars[0].timestamp, open=bars[0].open,
                    high=max(b.high for b in bars), low=min(b.low for b in bars),
                    close=bars[-1].close, volume=sum(b.volume for b in bars)
                )
                completed[tf] = None
        return completed


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.feature_engine = FeatureEngine(max_bars=500)
        self.aggregator = BarAggregator()
        self.verdict_engine = VerdictEngine(VerdictConfig())
        self.verdict_engine.tv_api_url = config.backend_url  # For TV confluence lookups
        self.running = False
        self.ws = None

        self.bars_received = 0
        self.predictions_made = 0
        self.start_time = None
        self.last_prediction: Dict[str, Dict[str, dict]] = {}
        self._shutdown_event = asyncio.Event()

        # Expiration check task
        self._expiry_task = None

    # ----------------------------------------------------------------
    # STARTUP
    # ----------------------------------------------------------------

    def warmup(self):
        logger.info(f"Warming up {len(self.config.symbols)} symbols...")
        for symbol in self.config.symbols:
            self.aggregator.init_symbol(symbol)
            self._fetch_historical(symbol)

        for symbol in self.config.symbols:
            for tf in self.config.timeframes:
                count = self.feature_engine.bar_count(symbol, tf)
                ready = self.feature_engine.has_enough_bars(symbol, tf)
                status = "✅" if ready else "⚠️"
                logger.info(f"  {status} {symbol} {tf}: {count} bars {'(ready)' if ready else '(warming up)'}")

    def _fetch_historical(self, symbol: str):
        api_key = self.config.polygon_api_key
        if not api_key:
            logger.warning(f"No Polygon API key for {symbol}")
            return

        end = datetime.now(timezone.utc)
        days_needed = max(45, self.config.warmup_bars // 300)
        start = end - timedelta(days=days_needed)

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
            f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
        )

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("resultsCount", 0) == 0:
                logger.warning(f"No historical data for {symbol}")
                return

            results = data["results"]
            logger.info(f"  {symbol}: fetched {len(results)} 1min bars")

            bars_1min = []
            for r in results:
                bar = OHLCVBar(
                    timestamp=r["t"] / 1000, open=r["o"], high=r["h"],
                    low=r["l"], close=r["c"], volume=r["v"]
                )
                bars_1min.append(bar)

            self.feature_engine.update_bars_bulk(symbol, "1min", bars_1min)

            for bar in bars_1min:
                completed = self.aggregator.add_1min_bar(symbol, bar)
                for tf, agg_bar in completed.items():
                    if agg_bar is not None:
                        self.feature_engine.update_bar(symbol, tf, agg_bar)

            for tf in ["1hr", "15min", "5min"]:
                if self.feature_engine.has_enough_bars(symbol, tf):
                    self.feature_engine.compute_htf_summary(symbol, tf)

            # Seed verdict engine with recent price data
            for bar in bars_1min[-20:]:
                self.verdict_engine.update_price(symbol, bar.close, bar.high, bar.low)

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")

    # ----------------------------------------------------------------
    # WEBSOCKET
    # ----------------------------------------------------------------

    async def run(self):
        self.running = True
        self.start_time = time.time()
        self.warmup()

        # Start expiration checker
        self._expiry_task = asyncio.create_task(self._check_expirations_loop())

        reconnects = 0
        while self.running and reconnects < self.config.ws_max_reconnects:
            conn_start = time.time()
            try:
                await self._ws_connect()
            except (websockets.exceptions.ConnectionClosed, ConnectionError) as e:
                logger.warning(f"WebSocket disconnected ({reconnects}): {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)

            conn_duration = time.time() - conn_start
            if conn_duration > 60:
                reconnects = 0
                logger.info("Reconnecting after long session...")
            else:
                reconnects += 1
                logger.warning(f"Short connection ({conn_duration:.0f}s), reconnect {reconnects}/{self.config.ws_max_reconnects}")

            if self.running:
                delay = min(self.config.ws_reconnect_delay * reconnects, 30)
                await asyncio.sleep(delay)

        if self._expiry_task:
            self._expiry_task.cancel()
        logger.info("Orchestrator stopped")

    async def _check_expirations_loop(self):
        """Periodically check for expired verdicts."""
        while self.running:
            try:
                closed = self.verdict_engine.check_expirations()
                for v in closed:
                    await self._push_verdict_to_backend(v)
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Expiration check error: {e}")
                await asyncio.sleep(10)

    async def _ws_connect(self):
        logger.info("Connecting to Polygon WebSocket...")

        async with websockets.connect(
            self.config.ws_url, ping_interval=30, ping_timeout=10, close_timeout=5
        ) as ws:
            self.ws = ws

            auth_msg = json.dumps({"action": "auth", "params": self.config.polygon_api_key})
            await ws.send(auth_msg)

            auth_ok = False
            for _ in range(5):
                resp = await asyncio.wait_for(ws.recv(), timeout=10)
                events = json.loads(resp)
                if not isinstance(events, list):
                    events = [events]
                for ev in events:
                    if ev.get("status") == "connected":
                        logger.info("WebSocket connected")
                    elif ev.get("status") == "auth_success":
                        auth_ok = True
                        logger.info("Authenticated with Polygon")
                    elif ev.get("status") == "auth_failed":
                        logger.error(f"Auth failed: {ev.get('message', '')}")
                        return
                if auth_ok:
                    break

            if not auth_ok:
                logger.error("Did not receive auth_success")
                return

            subs = ",".join([f"AM.{s}" for s in self.config.symbols])
            sub_msg = json.dumps({"action": "subscribe", "params": subs})
            await ws.send(sub_msg)
            logger.info(f"Subscribed to: {subs}")

            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=10)
                events = json.loads(resp)
                if isinstance(events, list):
                    for ev in events:
                        if ev.get("status") == "success":
                            logger.info(f"  ✅ {ev.get('message', '')}")
            except asyncio.TimeoutError:
                logger.warning("No subscription confirmation received")

            logger.info("Listening for bars... (delayed data, first bar may take ~60s)")

            bar_count = 0
            async for message in ws:
                if not self.running:
                    break
                try:
                    data = json.loads(message)
                    if isinstance(data, list):
                        for event in data:
                            if event.get("ev") in ("A", "AM"):
                                bar_count += 1
                            await self._handle_event(event)
                    else:
                        if data.get("ev") in ("A", "AM"):
                            bar_count += 1
                        await self._handle_event(data)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

            logger.info(f"WebSocket loop ended after {bar_count} bars")

    async def _handle_event(self, event: dict):
        ev_type = event.get("ev")
        if ev_type in ("A", "AM"):
            await self._handle_minute_bar(event)
        elif ev_type == "status":
            logger.debug(f"Status: {event.get('message', '')}")

    async def _handle_minute_bar(self, event: dict):
        symbol = event.get("sym", "")
        if symbol not in self.config.symbols:
            return

        bar = OHLCVBar(
            timestamp=event["s"] / 1000 if "s" in event else time.time(),
            open=event["o"], high=event["h"], low=event["l"],
            close=event["c"], volume=event["v"]
        )

        self.bars_received += 1
        self.feature_engine.update_bar(symbol, "1min", bar)

        # Feed price data to verdict engine
        self.verdict_engine.update_price(symbol, bar.close, bar.high, bar.low)

        completed = self.aggregator.add_1min_bar(symbol, bar)
        for tf, agg_bar in completed.items():
            if agg_bar is not None:
                self.feature_engine.update_bar(symbol, tf, agg_bar)
                if self.feature_engine.has_enough_bars(symbol, tf):
                    self.feature_engine.compute_htf_summary(symbol, tf)

        tfs_to_predict = ["1min"]
        for tf, agg_bar in completed.items():
            if agg_bar is not None:
                tfs_to_predict.append(tf)

        for tf in tfs_to_predict:
            if self.feature_engine.has_enough_bars(symbol, tf):
                await self._run_prediction(symbol, tf)

    # ----------------------------------------------------------------
    # PREDICTION → VERDICT
    # ----------------------------------------------------------------

    async def _run_prediction(self, symbol: str, timeframe: str):
        try:
            t0 = time.time()
            features, feature_names = self.feature_engine.compute_features(symbol, timeframe)
            sequence = self.feature_engine.get_sequence(symbol, timeframe, seq_len=30)
            feature_ms = (time.time() - t0) * 1000

            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "features": features.tolist(),
                "feature_names": feature_names,
            }
            if sequence is not None:
                payload["sequence"] = sequence.tolist()

            t1 = time.time()
            resp = requests.post(f"{self.config.predict_url}/predict", json=payload, timeout=5)
            predict_ms = (time.time() - t1) * 1000
            total_ms = (time.time() - t0) * 1000

            if resp.status_code != 200:
                logger.error(f"Prediction failed for {symbol} {timeframe}: {resp.text[:200]}")
                return

            result = resp.json()
            self.predictions_made += 1

            if symbol not in self.last_prediction:
                self.last_prediction[symbol] = {}
            self.last_prediction[symbol][timeframe] = result

            direction = result["direction"]
            probability = result["probability"]
            confidence = result["confidence"]
            grade = result["grade"]
            unanimous = result.get("unanimous", False)

            # Log raw prediction
            if direction != "HOLD":
                emoji = "🟢" if direction == "CALL" else "🔴"
                logger.info(
                    f"{emoji} {symbol} {timeframe} {direction} "
                    f"{probability:.1f}% [{grade}] "
                    f"agree={result.get('models_agreeing', 0)}/3 "
                    f"[{total_ms:.0f}ms]"
                )

            # ── Feed to Verdict Engine ──
            pred = Prediction(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                probability=probability,
                confidence=confidence,
                grade=grade,
                models_agreeing=result.get("models_agreeing", 0),
                unanimous=unanimous,
                individual=result.get("individual", {}),
                timestamp=time.time(),
                feature_ms=feature_ms,
                predict_ms=predict_ms,
                total_ms=total_ms,
            )

            verdict = self.verdict_engine.ingest_prediction(pred)
            if verdict:
                await self._push_verdict_to_backend(verdict)

        except Exception as e:
            logger.error(f"Prediction error {symbol} {timeframe}: {e}")

    async def _push_verdict_to_backend(self, verdict):
        """Push verdict to backend API for dashboard delivery."""
        try:
            resp = requests.post(
                f"{self.config.backend_url}/api/verdicts",
                json=verdict.to_dict(),
                timeout=3
            )
            if resp.status_code == 200:
                data = resp.json()
                bc = data.get("broadcast_to", 0)
                if bc > 0:
                    logger.info(f"  → Verdict pushed to {bc} dashboard clients")
            else:
                logger.warning(f"Backend verdict API error: {resp.status_code}")
        except Exception as e:
            logger.debug(f"Backend unavailable for verdict push: {e}")

    # ----------------------------------------------------------------
    # STATUS
    # ----------------------------------------------------------------

    def status(self) -> dict:
        uptime = time.time() - self.start_time if self.start_time else 0
        return {
            "running": self.running,
            "uptime_seconds": round(uptime, 1),
            "bars_received": self.bars_received,
            "predictions_made": self.predictions_made,
            "verdict_stats": self.verdict_engine.get_stats(),
            "active_verdicts": self.verdict_engine.get_active_verdicts(),
            "symbols": self.config.symbols,
        }

    def shutdown(self):
        logger.info("Shutting down orchestrator...")
        self.running = False
        self._shutdown_event.set()
        if self.ws:
            asyncio.ensure_future(self.ws.close())


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="YELENA v2 Real-Time Orchestrator")
    parser.add_argument("--symbols", default=os.getenv("YELENA_SYMBOLS", "SPY,QQQ"),
                        help="Comma-separated symbols")
    parser.add_argument("--polygon-key", default=os.getenv("POLYGON_API_KEY", ""),
                        help="Polygon.io API key")
    parser.add_argument("--predict-url", default=os.getenv("YELENA_PREDICT_URL", "http://localhost:8001"),
                        help="Prediction service URL")
    parser.add_argument("--log-level", default=os.getenv("YELENA_LOG_LEVEL", "INFO"),
                        help="Logging level")
    parser.add_argument("--warmup-only", action="store_true",
                        help="Only fetch historical data, don't connect WebSocket")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    config = OrchestratorConfig(
        symbols=[s.strip().upper() for s in args.symbols.split(",")],
        polygon_api_key=args.polygon_key,
        predict_url=args.predict_url,
        log_level=args.log_level
    )

    logger.info("=" * 60)
    logger.info("YELENA v2 Real-Time Orchestrator + Verdict Engine")
    logger.info(f"  Symbols: {config.symbols}")
    logger.info(f"  Timeframes: {config.timeframes}")
    logger.info(f"  Predict URL: {config.predict_url}")
    logger.info(f"  Polygon key: {'✅ set' if config.polygon_api_key else '❌ missing'}")
    logger.info(f"  Verdict: 15min anchor + 1 confirming TF, min 70% confidence")
    logger.info("=" * 60)

    orchestrator = Orchestrator(config)

    def signal_handler(sig, frame):
        orchestrator.shutdown()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.warmup_only:
        orchestrator.warmup()
        status = orchestrator.status()
        logger.info(f"Warmup complete.")
        return

    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
