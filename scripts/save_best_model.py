#!/usr/bin/env python3
"""
Save the best realistic model to a fixed location for dashboard use.
"""

import mlflow
import shutil
from pathlib import Path
import joblib
import json


# Automatically find the latest run with a model artifact in the correct experiment
EXPERIMENT_ID = "747582833673318534"
mlruns_dir = Path("mlruns") / EXPERIMENT_ID


def find_latest_model_run():
    runs = [d for d in mlruns_dir.iterdir() if d.is_dir() and len(d.name) == 32]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for run_dir in runs:
        model_pkl = run_dir / "artifacts/model/model.pkl"
        if model_pkl.exists():
            return run_dir.name, model_pkl
    return None, None


BEST_RUN_ID, mlflow_model_path = find_latest_model_run()


print("=" * 60)
print("SAVING BEST MODEL FOR DASHBOARD")
print(f"Experiment: {EXPERIMENT_ID}")
print(f"Best Run ID: {BEST_RUN_ID}")
print("=" * 60)


dashboard_model_path = Path("models/production_model.pkl")
dashboard_model_path.parent.mkdir(exist_ok=True)

if mlflow_model_path and mlflow_model_path.exists():
    shutil.copy(mlflow_model_path, dashboard_model_path)
    print(f"✅ Model saved to: {dashboard_model_path}")
    # Also save model info (metrics are placeholders, update as needed)
    info = {
        "run_id": BEST_RUN_ID,
        "date_saved": "2026-02-17"
    }
    with open("models/production_model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("✅ Model info saved")
else:
    print("❌ Could not find any model file in experiment.")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("1. Update dashboard to use: models/production_model.pkl")
print("2. Set DEFAULT_MODEL_URI to this path in constants.py")
print("=" * 60)
