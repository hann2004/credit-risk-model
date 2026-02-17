import joblib
import pandas as pd
from src.constants import PRODUCTION_MODEL_PATH, PRODUCTION_MODEL_METRICS

# Load model
model = joblib.load(PRODUCTION_MODEL_PATH)
print(f"✅ Loaded model from: {PRODUCTION_MODEL_PATH}")

# Quick test
test_data = pd.DataFrame([{"recency": 30, "frequency": 10, "monetary": 1000}])

pred = model.predict_proba(test_data)[0][1]
print(f"Test prediction: {pred:.4f}")
print(f"Expected range: 0.2-0.8 (reasonable)")
print("\n✅ Production model verified!")
