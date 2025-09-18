# src/pearlmind/deployment/api.py
"""FastAPI service for model serving."""

from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from pearlmind.models import load_model
from pearlmind.evaluation import FairnessAuditor

app = FastAPI(title="PearlMind ML API", version="2.0.0")

# Global model registry
MODELS = {}

class PredictionRequest(BaseModel):
    features: List[List[float]]
    model_name: str = "default"
    include_fairness: bool = False
    sensitive_features: Optional[List[int]] = None

class PredictionResponse(BaseModel):
    predictions: List[float]
    probabilities: Optional[List[List[float]]]
    fairness_metrics: Optional[Dict]
    model_version: str

@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    MODELS["default"] = load_model("models/production/xgboost_v2.pkl")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions with optional fairness audit."""
    
    if request.model_name not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = MODELS[request.model_name]
    X = np.array(request.features)
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X).tolist()
    
    fairness_metrics = None
    if request.include_fairness and request.sensitive_features:
        fairness_metrics = model.audit_fairness(
            X, predictions, request.sensitive_features
        )
    
    return PredictionResponse(
        predictions=predictions.tolist(),
        probabilities=probabilities,
        fairness_metrics=fairness_metrics,
        model_version=model.version
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": list(MODELS.keys())}
