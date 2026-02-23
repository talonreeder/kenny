"""
YELENA v2 — Main FastAPI Application
Port 8000 — serves dashboard API + WebSocket signals + TradingView webhooks.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.api.signals import router as signals_router
from app.api.tv_alerts import router as tv_alerts_router

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(signals_router)
app.include_router(tv_alerts_router)


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint - verifies API and database connection."""
    try:
        result = await db.execute(text("SELECT COUNT(*) FROM symbols"))
        symbol_count = result.scalar()
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "database": "connected",
            "symbols_tracked": symbol_count,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "database": f"error: {str(e)}",
        }


@app.get("/api/symbols")
async def get_symbols(db: AsyncSession = Depends(get_db)):
    """Get all tracked symbols."""
    result = await db.execute(
        text("SELECT symbol, name, sector, is_active FROM symbols ORDER BY symbol")
    )
    rows = result.fetchall()
    return [
        {"symbol": r[0], "name": r[1], "sector": r[2], "is_active": r[3]}
        for r in rows
    ]
