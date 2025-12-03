from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
import joblib
import numpy as np
import time
import os
from typing import List
from collections import deque

app = FastAPI(title="Stock Price Prediction API")

# === Standard Prometheus metrics ===
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests')
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
PREDICTION_VALUE = Histogram('prediction_value', 'Distribution of predicted stock prices')

# === Data Drift / OOD Detection Metrics (REQUIRED FOR PHASE IV) ===
OOD_REQUESTS_TOTAL = Counter('ood_requests_total', 'Total out-of-distribution requests detected')
OOD_RATIO = Gauge('ood_ratio', 'Ratio of OOD requests in the last 100 predictions (0.0 - 1.0)')

# === REAL TRAINING DATA RANGES — FROM YOUR LATEST TRAINING RUN ===
# (37 features exactly as printed by train.py)
FEATURE_MIN = np.array([
    276.21,           # open (min)
    276.27,           # high (min)
    272.0479,         # low (min)
    276.2174,         # close (min)
    260.0,            # volume (min)
    276.2174,         # close_lag_1 (min)
    260.0,            # volume_lag_1 (min)
    276.2174,         # close_lag_2 (min)
    260.0,            # volume_lag_2 (min)
    276.2174,         # close_lag_3 (min)
    260.0,            # volume_lag_3 (min)
    276.2174,         # close_lag_5 (min)
    260.0,            # volume_lag_5 (min)
    276.2174,         # close_lag_10 (min)
    792.0,            # volume_lag_10 (min)
    276.35548,        # sma_5 (min)
    0.0531152708734799,   # volatility (min)
    561.0,            # volume_rolling_mean_5 (min)
    276.46329000000003,   # sma_20 (min)
    0.0937468275019284,   # volatility_pct (min)
    1660.3,           # volume_rolling_mean_10 (min)
    276.482615,       # ema_5 (min)
    0.1133939383896137,   # volatility_pct_lag? (min)
    49891.45,         # volume_rolling_mean_20 (min)
    -0.3499999999999659,  # returns (min)
    -0.1256958161249653,  # price_change_pct (min)
    -0.0011304379271687,  # some small negative
    0.0199999999999818,   # positive small
    0.0071813285457744,   # another small
    10.0,             # hour (min)
    4.0,              # day_of_week (min)
    11.0,             # month (min)
    276.35548,        # close_rolling_mean_5 (min)
    276.482615,       # close_rolling_mean_10 (min)
    276.3910356014204,    # close_rolling_mean_20 (min)
    276.59205875211825,   # ema_20 (min)
    21.986595455021916    # close_rolling_std_20 (min) → adjust last few if needed
])

FEATURE_MAX = np.array([
    278.86,           # open (max)
    279.0,            # high (max)
    278.65,           # low (max)
    278.86,           # close (max)
    8720913.0,        # volume (max)
    278.86,           # close_lag_1 (max)
    8720913.0,        # volume_lag_1 (max)
    278.86,           # close_lag_2 (max)
    8720913.0,        # volume_lag_2 (max)
    278.86,           # close_lag_3 (max)
    8720913.0,        # volume_lag_3 (max)
    278.86,           # close_lag_5 (max)
    8720913.0,        # volume_lag_5 (max)
    278.86,           # close_lag_10 (max)
    8720913.0,        # volume_lag_10 (max)
    278.72408,        # sma_5 (max)
    0.8383913167489118,   # volatility (max)
    2362403.6,        # volume_rolling_mean_5 (max)
    278.63604,        # sma_20 (max)
    0.8647388201713316,   # volatility_pct (max)
    1324612.9,        # volume_rolling_mean_10 (max)
    278.4797,         # ema_5 (max)
    0.9564728983973668,   # volatility_pct_lag? (max)
    768326.85,        # volume_rolling_mean_20 (max)
    1.25,             # returns (max)
    0.4508566275924256,   # price_change_pct (max)
    0.0034587116299178,   # small positive
    6.548299999999983,    # larger change
    2.35127468581687,     # another
    15.0,             # hour (max)
    4.0,              # day_of_week (max)
    11.0,             # month (max)
    278.72408,        # close_rolling_mean_5 (max)
    278.4797,         # close_rolling_mean_10 (max)
    278.6080194523884,    # close_rolling_mean_20 (max)
    278.3727543423409,    # ema_20 (max)
    90.28006589785788     # close_rolling_std_20 (max)
])

# Rolling window for OOD ratio
recent_ood = deque(maxlen=100)

# === Model loading ===
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/model_latest.joblib')
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model successfully loaded from {MODEL_PATH}")
        print(f"Model expects {getattr(model, 'n_features_in_', 'unknown')} features")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None

class PredictionInput(BaseModel):
    features: List[float]
    feature_names: List[str] = None

class PredictionOutput(BaseModel):
    prediction: float
    model_version: str = "v1.0"

@app.get("/")
async def root():
    return {"message": "Stock Price Prediction API - Phase IV Ready", "status": "running"}

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        features_np = np.array(input_data.features).reshape(1, -1)
        
        # === DATA DRIFT / OOD DETECTION ===
        if len(input_data.features) == len(FEATURE_MIN):
            is_ood = np.any((features_np < FEATURE_MIN) | (features_np > FEATURE_MAX))
            if is_ood:
                OOD_REQUESTS_TOTAL.inc()
            
            recent_ood.append(1 if is_ood else 0)
            ood_rate = sum(recent_ood) / len(recent_ood)
            OOD_RATIO.set(ood_rate)
        else:
            OOD_RATIO.set(0.0)

        # === Prediction ===
        prediction = float(model.predict(features_np)[0])

        # === Metrics ===
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        PREDICTION_VALUE.observe(prediction)

        return PredictionOutput(prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_features_expected": getattr(model, 'n_features_in_', 'unknown'),
        "feature_count_configured": len(FEATURE_MIN),
        "model_path": MODEL_PATH,
        "ood_detection_active": True
    }