"""
MLOps Pipeline - FastAPI Prediction Service
Serves the trained model with monitoring, auth, and rate limiting.
"""

import os
import time
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR    = Path(os.getenv("MODEL_OUTPUT_DIR", "/app/model/artifacts"))
API_TOKEN    = os.getenv("API_TOKEN", "mlops-secret-token")
MODEL_NAME   = os.getenv("MODEL_NAME", "fraud-detector")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ─── Prometheus Metrics ───────────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "mlops_predictions_total",
    "Total number of predictions",
    ["result", "model_version"]
)
PREDICTION_LATENCY = Histogram(
    "mlops_prediction_latency_seconds",
    "Time spent processing prediction",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
MODEL_ACCURACY_GAUGE = Gauge(
    "mlops_model_accuracy",
    "Current model accuracy score"
)
ACTIVE_REQUESTS = Gauge(
    "mlops_active_requests",
    "Number of active requests being processed"
)

# ─── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MLOps Prediction API",
    description="Production-grade ML model serving with monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Auth ─────────────────────────────────────────────────────────────────────
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials


# ─── Models ───────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector for prediction")
    model_version: Optional[str] = Field("latest", description="Model version to use")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

class BatchPredictionRequest(BaseModel):
    instances: List[List[float]]


# ─── Model Loading ────────────────────────────────────────────────────────────
model  = None
scaler = None
model_version = "unknown"

@app.on_event("startup")
async def load_model():
    global model, scaler, model_version
    try:
        model  = joblib.load(MODEL_DIR / "model.joblib")
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        model_version = os.getenv("MODEL_VERSION", "1.0.0")
        MODEL_ACCURACY_GAUGE.set(0.95)  # Will be updated from metrics.json
        logger.info(f"✅ Model loaded successfully (version: {model_version})")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model_version,
    )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("100/minute")
async def predict(
    request: Request,
    payload: PredictionRequest,
    token: str = Depends(verify_token),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_REQUESTS.inc()
    start = time.time()

    try:
        features = np.array(payload.features).reshape(1, -1)
        scaled   = scaler.transform(features)

        prediction = int(model.predict(scaled)[0])
        probability = float(model.predict_proba(scaled)[0][prediction])

        latency_ms = (time.time() - start) * 1000

        PREDICTION_COUNTER.labels(
            result=str(prediction),
            model_version=model_version
        ).inc()
        PREDICTION_LATENCY.observe(time.time() - start)

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            model_version=model_version,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", tags=["Prediction"])
@limiter.limit("20/minute")
async def predict_batch(
    request: Request,
    payload: BatchPredictionRequest,
    token: str = Depends(verify_token),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array(payload.instances)
    scaled   = scaler.transform(features)
    preds    = model.predict(scaled).tolist()
    probas   = model.predict_proba(scaled).max(axis=1).tolist()

    return {
        "predictions": preds,
        "probabilities": [round(p, 4) for p in probas],
        "count": len(preds),
        "model_version": model_version,
    }


@app.get("/model/info", tags=["Model"])
async def model_info(token: str = Depends(verify_token)):
    """Return model metadata."""
    import json
    metrics_path = MODEL_DIR / "metrics.json"
    params_path  = MODEL_DIR / "params.json"

    info = {"version": model_version, "model_name": MODEL_NAME}
    if metrics_path.exists():
        info["metrics"] = json.loads(metrics_path.read_text())
    if params_path.exists():
        info["hyperparameters"] = json.loads(params_path.read_text())
    return info
