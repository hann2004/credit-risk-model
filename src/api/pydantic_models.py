"""Pydantic schemas for the FastAPI credit risk service."""

from typing import Dict, List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request payload containing one or more feature dictionaries."""

    instances: List[Dict[str, float]] = Field(
        ..., description="List of feature mappings; each entry represents one customer"
    )


class PredictionResponse(BaseModel):
    """Response containing risk probabilities for each provided instance."""

    risk_probabilities: List[float]
