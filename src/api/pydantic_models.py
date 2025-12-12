from pydantic import BaseModel


class PredictRequest(BaseModel):
    # Example schema: adjust fields to your dataset
    a: float
    b: float
    c: str


class PredictResponse(BaseModel):
    score: float
