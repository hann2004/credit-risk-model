"""FastAPI inference service that loads the best model from the MLflow registry."""

import os

from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.constants import DEFAULT_MODEL_URI
from src.predict import load_model, predict_instances


MODEL_URI = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)

app = FastAPI(title="Credit Risk API", version="0.1.0")
model, feature_names = load_model(MODEL_URI)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Return risk probabilities for the provided customer feature rows."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        probs = predict_instances(model, payload.instances, feature_names)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(risk_probabilities=probs)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_uri": MODEL_URI}
