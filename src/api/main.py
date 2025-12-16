"""FastAPI inference service that loads the best model from the MLflow registry."""

import os
from typing import List, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionRequest, PredictionResponse

MODEL_URI = os.getenv("MODEL_URI", "models:/credit-risk-best-model/Production")


def _load_model():
    """Load the model from MLflow registry using the configured URI."""
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
    except Exception as exc:  # pragma: no cover - defensive startup
        raise RuntimeError(f"Failed to load model from {MODEL_URI}: {exc}") from exc

    feature_names: Optional[List[str]] = None
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    return model, feature_names


app = FastAPI(title="Credit Risk API", version="0.1.0")
model, feature_names = _load_model()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Return risk probabilities for the provided customer feature rows."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame(payload.instances)
    if feature_names:
        missing = sorted(set(feature_names) - set(df.columns))
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        df = df[feature_names]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[:, 1]
    else:
        preds = model.predict(df)
        probs = preds if isinstance(preds, pd.Series) else pd.Series(preds)

    return PredictionResponse(risk_probabilities=list(map(float, probs)))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_uri": MODEL_URI}
