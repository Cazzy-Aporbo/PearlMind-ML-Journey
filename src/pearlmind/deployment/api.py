"""Local teaching API: bounded numeric inputs, real labels required for an audit."""

from contextlib import asynccontextmanager
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from pearlmind.models import load_model


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    features: list[list[float]] = Field(min_length=1, max_length=1000)
    include_fairness: bool = False
    labels: list[int] | None = None
    sensitive_features: list[str] | None = None


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path = model_path or os.environ.get("PEARLMIND_MODEL_PATH")
        app.state.model = load_model(path) if path else None
        yield

    app = FastAPI(title="PearlMind local teaching API", lifespan=lifespan)

    @app.get("/health")
    def health():
        loaded = app.state.model is not None
        if not loaded:
            raise HTTPException(503, "No model configured; train and set PEARLMIND_MODEL_PATH")
        return {"status": "ready", "model_version": app.state.model.version}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        model = app.state.model
        if model is None:
            raise HTTPException(503, "No model loaded")
        try:
            X = np.asarray(request.features, dtype=float)
            if X.ndim != 2 or X.shape[1] != model._model.n_features_in_ or not np.isfinite(X).all():
                raise ValueError(
                    "Feature matrix must be finite and match the trained feature count"
                )
            metrics = None
            if request.include_fairness:
                if request.labels is None or request.sensitive_features is None:
                    raise ValueError(
                        "Audit requires observed labels and group labels; predictions are not ground truth"
                    )
                metrics = model.audit_fairness(X, request.labels, request.sensitive_features)
            return {
                "predictions": model.predict(X).tolist(),
                "probabilities": model.predict_proba(X).tolist(),
                "fairness_metrics": metrics,
                "model_version": model.version,
            }
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
