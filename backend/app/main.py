"""
KENNY — AI Options Day Trading Platform
FastAPI Application Entry Point

CRITICAL: ALL routers are mounted here from day one. No disconnected modules.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import db
from app.api.signals import router as signals_router
from app.api.tv_alerts import router as tv_alerts_router
from app.api.verdicts import router as verdicts_router

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("KENNY — AI Options Day Trading Platform")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Symbols: {', '.join(settings.symbols)}")
    logger.info(f"Timeframes: {', '.join(settings.timeframes)}")
    logger.info("=" * 60)

    # 1. Database connection
    try:
        await db.connect()
        health = await db.health_check()
        logger.info(f"Database: {health['status']} (pool: {health.get('pool_size', '?')})")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.warning("Continuing without database — some features will be unavailable")

    # 2. ML ModelManager — Phase 3
    logger.info("ML ModelManager: [Phase 3 — not yet initialized]")

    # 3. Wire TV store to Verdict Engine — Phase 4
    # CRITICAL: This was MISSING in YELENA. Wire it here when ready.
    logger.info("TV-ML Synergy: [Phase 4 — not yet wired]")

    logger.info("-" * 60)
    logger.info("KENNY startup complete. All routers mounted.")
    logger.info("-" * 60)

    yield

    logger.info("KENNY shutting down...")
    await db.disconnect()
    logger.info("KENNY shutdown complete.")


app = FastAPI(
    title="KENNY",
    description="AI Options Day Trading Platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.4f}"
    return response


# === MOUNT ALL ROUTERS ===
app.include_router(signals_router, prefix="/api", tags=["signals"])
app.include_router(tv_alerts_router, prefix="/api", tags=["tradingview"])
app.include_router(verdicts_router, prefix="/api", tags=["verdicts"])


# === CORE ENDPOINTS ===

@app.get("/health")
async def health_check():
    settings = get_settings()
    db_health = await db.health_check()
    return {
        "status": "healthy" if db_health["status"] == "healthy" else "degraded",
        "service": "kenny",
        "version": "1.0.0",
        "environment": settings.env,
        "components": {
            "database": db_health,
            "ml_models": {"status": "not_loaded", "note": "Phase 3"},
            "schwab": {"status": "not_connected", "note": "Phase 4"},
            "tv_alerts": {"status": "ready"},
        },
        "symbols": settings.symbols,
        "timeframes": settings.timeframes
    }


@app.get("/api/symbols")
async def get_symbols():
    settings = get_settings()
    try:
        rows = await db.fetch("SELECT * FROM symbols ORDER BY symbol")
        if rows:
            return {"symbols": [dict(row) for row in rows], "count": len(rows)}
    except Exception:
        pass
    return {"symbols": [{"symbol": s} for s in settings.symbols], "count": len(settings.symbols)}


@app.get("/api/status")
async def system_status():
    settings = get_settings()
    try:
        counts = await db.get_table_counts()
    except Exception:
        counts = {}
    return {
        "kenny": {"version": "1.0.0", "env": settings.env},
        "data": {"table_counts": counts, "symbols": len(settings.symbols)},
        "ml": {"models_loaded": 0, "ensemble_threshold": settings.ensemble_threshold},
        "risk": {
            "max_position_pct": settings.max_position_pct,
            "daily_loss_limit": settings.daily_loss_limit,
            "max_concurrent": settings.max_concurrent_positions,
        }
    }
