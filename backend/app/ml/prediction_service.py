"""
YELENA v2 — Prediction Service (FastAPI)
Real-time ML inference endpoint for algorithmic trading signals.

Endpoints:
    POST /predict              — Single timeframe ensemble prediction
    POST /predict/cross-tf     — Cross-timeframe agreement prediction
    GET  /health               — System health check
    GET  /models               — Model inventory and status
    POST /models/reload        — Hot-swap models from disk
    GET  /models/performance   — Cached performance metrics

Architecture: EC2 in-process inference
    - All 44 models loaded at startup (~54MB RAM)
    - Target: <50ms total prediction latency
    - No external API calls (no SageMaker endpoints)
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from model_manager import (
    ModelManager, init_model_manager, get_model_manager,
    EnsemblePrediction, CrossTFPrediction, ModelPrediction
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = os.environ.get("YELENA_MODEL_DIR", "/home/ubuntu/yelena/models/v2")
LOG_LEVEL = os.environ.get("YELENA_LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("yelena.prediction")


# ============================================================================
# FASTAPI LIFESPAN — Load models at startup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup, cleanup on shutdown."""
    logger.info("=" * 60)
    logger.info("YELENA v2 Prediction Service starting...")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info("=" * 60)

    try:
        manager = init_model_manager(MODEL_DIR)
        health = manager.health()
        logger.info(f"Models loaded: {health['total_models']} across {health['loaded_timeframes']}")
        logger.info(f"Load times: {health['load_times_ms']}")
    except Exception as e:
        logger.error(f"FATAL: Failed to load models: {e}")
        raise

    logger.info("Prediction service ready.")
    yield

    logger.info("Prediction service shutting down.")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="YELENA v2 Prediction Service",
    description="Real-time ML ensemble predictions for options trading",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class PredictRequest(BaseModel):
    """Single timeframe prediction request."""
    symbol: str = Field(..., description="Trading symbol (e.g., SPY, QQQ, TSLA)")
    timeframe: str = Field(..., description="Timeframe (1min, 5min, 15min, 1hr)")
    features: List[float] = Field(..., description="156-element feature vector (latest bar)")
    sequence: Optional[List[List[float]]] = Field(
        None,
        description="Sequence array (seq_len × n_features) for Transformer/CNN. "
                    "If omitted, only XGBoost prediction is returned."
    )
    feature_names: Optional[List[str]] = Field(
        None, description="Feature column names for XGBoost DMatrix compatibility"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "SPY",
                "timeframe": "15min",
                "features": [0.1] * 156,
                "sequence": None,
                "feature_names": None
            }
        }


class CrossTFRequest(BaseModel):
    """Cross-timeframe prediction request."""
    symbol: str = Field(..., description="Trading symbol")
    primary_tf: str = Field("15min", description="Primary timeframe for trade decision")
    timeframes: Dict[str, List[float]] = Field(
        ...,
        description="Map of timeframe → 156-element feature vector"
    )
    sequences: Optional[Dict[str, List[List[float]]]] = Field(
        None,
        description="Map of timeframe → sequence array"
    )
    feature_names: Optional[List[str]] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "SPY",
                "primary_tf": "15min",
                "timeframes": {
                    "5min": [0.1] * 156,
                    "15min": [0.1] * 156,
                    "1hr": [0.1] * 156
                }
            }
        }


class ModelPredictionResponse(BaseModel):
    model_name: str
    direction: str
    probability: float
    confidence: float
    signal: str


class PredictResponse(BaseModel):
    """Single timeframe prediction response."""
    symbol: str
    timeframe: str
    direction: str
    probability: float
    confidence: float
    grade: str
    models_agreeing: int
    unanimous: bool
    individual: Dict[str, ModelPredictionResponse]
    latency_ms: float
    timestamp: str


class CrossTFResponse(BaseModel):
    """Cross-timeframe prediction response."""
    symbol: str
    primary_direction: str
    primary_confidence: float
    primary_grade: str
    tf_agreement: Dict[str, str]
    agreement_count: int
    total_tfs: int
    confidence_bonus: float
    adjusted_confidence: float
    timeframe_predictions: Dict[str, PredictResponse]
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    models_loaded: int
    loaded_timeframes: List[str]
    load_times_ms: Dict[str, float]
    version: str


class ReloadRequest(BaseModel):
    timeframe: Optional[str] = Field(None, description="Specific TF to reload, or null for all")


# ============================================================================
# STARTUP TIME TRACKING
# ============================================================================

_startup_time = time.time()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _ensemble_to_response(pred: EnsemblePrediction, latency_ms: float) -> PredictResponse:
    """Convert EnsemblePrediction dataclass to API response."""
    individual = {}
    for key, mp in pred.individual.items():
        individual[key] = ModelPredictionResponse(
            model_name=mp.model_name,
            direction=mp.direction,
            probability=mp.probability,
            confidence=mp.confidence,
            signal=mp.signal
        )

    return PredictResponse(
        symbol=pred.symbol,
        timeframe=pred.timeframe,
        direction=pred.direction,
        probability=pred.probability,
        confidence=pred.confidence,
        grade=pred.grade,
        models_agreeing=pred.models_agreeing,
        unanimous=pred.unanimous,
        individual=individual,
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run ensemble prediction for a single timeframe.

    Returns direction (CALL/PUT/HOLD), probability, confidence grade,
    model agreement status, and individual model predictions.
    """
    start = time.time()
    manager = get_model_manager()

    if request.timeframe not in manager.loaded_timeframes:
        raise HTTPException(
            status_code=400,
            detail=f"Timeframe '{request.timeframe}' not loaded. "
                   f"Available: {manager.loaded_timeframes}"
        )

    # Convert inputs to numpy
    features = np.array(request.features, dtype=np.float32)
    sequence = None
    if request.sequence is not None:
        sequence = np.array(request.sequence, dtype=np.float32)

    # Validate dimensions — ModelManager handles per-TF feature selection
    # but reject obviously wrong inputs (< 100 features means bad data)
    if len(features) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Too few features: {len(features)}. Expected 139-163 depending on timeframe."
        )

    try:
        pred = manager.predict(
            symbol=request.symbol,
            timeframe=request.timeframe,
            features=features,
            sequence=sequence,
            feature_names=request.feature_names
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    latency_ms = (time.time() - start) * 1000
    logger.info(
        f"PREDICT {request.symbol} {request.timeframe}: "
        f"{pred.direction} {pred.confidence}% {pred.grade} "
        f"({pred.models_agreeing} agree, unanimous={pred.unanimous}) "
        f"[{latency_ms:.1f}ms]"
    )

    return _ensemble_to_response(pred, latency_ms)


@app.post("/predict/cross-tf", response_model=CrossTFResponse)
async def predict_cross_tf(request: CrossTFRequest):
    """
    Run cross-timeframe agreement prediction.

    Runs ensemble predictions across multiple timeframes and computes
    agreement bonus. Unanimous agreement across 3+ timeframes adds
    +4-7 percentage points to confidence.
    """
    start = time.time()
    manager = get_model_manager()

    # Convert inputs
    features_by_tf = {
        tf: np.array(feats, dtype=np.float32)
        for tf, feats in request.timeframes.items()
    }
    sequences_by_tf = None
    if request.sequences:
        sequences_by_tf = {
            tf: np.array(seq, dtype=np.float32)
            for tf, seq in request.sequences.items()
        }

    try:
        cross = manager.predict_cross_tf(
            symbol=request.symbol,
            features_by_tf=features_by_tf,
            sequences_by_tf=sequences_by_tf,
            feature_names=request.feature_names,
            primary_tf=request.primary_tf
        )
    except Exception as e:
        logger.error(f"Cross-TF prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cross-TF prediction failed: {str(e)}")

    total_latency = (time.time() - start) * 1000

    # Build per-TF responses
    tf_responses = {}
    for tf, pred in cross.predictions.items():
        tf_responses[tf] = _ensemble_to_response(pred, 0)  # Individual latency not tracked

    logger.info(
        f"CROSS-TF {request.symbol}: {cross.primary_direction} "
        f"{cross.adjusted_confidence}% (base {cross.primary_confidence} + "
        f"bonus {cross.confidence_bonus}) "
        f"Agreement: {cross.agreement_count}/{cross.total_tfs} "
        f"[{total_latency:.1f}ms]"
    )

    return CrossTFResponse(
        symbol=request.symbol,
        primary_direction=cross.primary_direction,
        primary_confidence=cross.primary_confidence,
        primary_grade=cross.primary_grade,
        tf_agreement=cross.tf_agreement,
        agreement_count=cross.agreement_count,
        total_tfs=cross.total_tfs,
        confidence_bonus=cross.confidence_bonus,
        adjusted_confidence=cross.adjusted_confidence,
        timeframe_predictions=tf_responses,
        latency_ms=round(total_latency, 2),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    manager = get_model_manager()
    h = manager.health()

    return HealthResponse(
        status="healthy" if h["is_loaded"] else "degraded",
        uptime_seconds=round(time.time() - _startup_time, 1),
        models_loaded=h["total_models"],
        loaded_timeframes=h["loaded_timeframes"],
        load_times_ms=h["load_times_ms"],
        version="2.0.0"
    )


@app.get("/models")
async def models_info():
    """Detailed model inventory."""
    manager = get_model_manager()
    h = manager.health()

    return {
        "total_models": h["total_models"],
        "model_dir": h["model_dir"],
        "loaded_timeframes": h["loaded_timeframes"],
        "models_per_tf": h["models_per_tf"],
        "tf_metadata": h.get("tf_metadata", {}),
        "load_times_ms": h["load_times_ms"],
        "threshold": manager.THRESHOLD,
        "grade_thresholds": manager.GRADE_THRESHOLDS
    }


@app.post("/models/reload")
async def reload_models(request: ReloadRequest):
    """
    Hot-swap models by reloading from disk.
    Use after deploying updated model files to the model directory.
    """
    manager = get_model_manager()

    try:
        result = manager.reload(timeframe=request.timeframe)
        logger.info(f"Model reload complete: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Model reload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "prediction_service:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False  # No auto-reload in production
    )
