from fastapi import FastAPI
import pandas as pd

from .pydantic_models import PredictRequest, PredictResponse
from ..data_processing import engineer_features

app = FastAPI(title="Credit Risk Model API")


# Placeholder: in real use, load a trained model here
model = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    df = engineer_features(df)
    # Placeholder score until a real model is loaded
    score = float(0.5)
    return PredictResponse(score=score)
