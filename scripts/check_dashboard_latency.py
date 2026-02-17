import time
import requests
import json

API_URL = "http://localhost:8000/predict"

# Example input (replace with realistic features from your model)
instances = [{"feature_1": 0.0, "feature_2": 0.0}]  # Add all required features

payload = {"instances": instances}

start = time.time()
try:
    response = requests.post(API_URL, json=payload, timeout=5)
    response.raise_for_status()
    elapsed = time.time() - start
    print(f"Prediction returned in {elapsed:.2f} seconds.")
    if elapsed < 3:
        print("Dashboard meets SLA (< 3 seconds).")
    else:
        print("Dashboard does NOT meet SLA.")
except Exception as exc:
    print(f"Error: {exc}")
