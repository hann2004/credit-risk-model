from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
import os

API_KEYS = os.getenv("API_KEYS", "dev-key-123,prod-key-456").split(",")
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Usage in FastAPI endpoint:
# @app.post("/predict", dependencies=[Depends(verify_api_key)])
# async def predict(request: PredictionRequest):
#     ...
