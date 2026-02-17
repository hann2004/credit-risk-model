from pathlib import Path

import mlflow
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Minimal MLflow artifact test
mlruns_dir = Path("mlruns/test_artifact")
mlruns_dir.mkdir(parents=True, exist_ok=True)

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
model = LogisticRegression().fit(X, y)

with mlflow.start_run(run_name="test_model_artifact") as run:
    mlflow.sklearn.log_model(model, "model")
    artifact_uri = mlflow.get_artifact_uri("model")
    print(f"[TEST] Model logged at: {artifact_uri}")
    # Check if model file exists
    model_pkl = Path(artifact_uri.replace("file://", "")) / "model.pkl"
    print(f"[TEST] Model file path: {model_pkl}")
    print(f"[TEST] Model file exists: {model_pkl.exists()}")
