import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import joblib
from pathlib import Path
from datetime import date

# Paths
DATA_PATH = "data/processed/processed_with_target.csv"
MODEL_PATH = "models/production_model.pkl"
INFO_PATH = "models/production_model_info.json"

# Load real data
df = pd.read_csv(DATA_PATH)
X = df.drop(
    columns=[col for col in ["CustomerId", "is_high_risk"] if col in df.columns],
    errors="ignore",
)
y = df["is_high_risk"]

# Convert bools to ints
bool_cols = X.select_dtypes(include=[bool]).columns
if len(bool_cols) > 0:
    X[bool_cols] = X[bool_cols].astype(int)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)

# Evaluate
preds = model.predict(X)
proba = model.predict_proba(X)[:, 1]
f1 = f1_score(y, preds)
roc_auc = roc_auc_score(y, proba)
precision = precision_score(y, preds)
recall = recall_score(y, preds)
accuracy = accuracy_score(y, preds)

print(f"F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

# Save model and info
Path("models").mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
info = {
    "f1": f1,
    "roc_auc": roc_auc,
    "precision": precision,
    "recall": recall,
    "accuracy": accuracy,
    "date_saved": str(date.today()),
    "feature_names": list(X.columns),
}
with open(INFO_PATH, "w") as f:
    import json

    json.dump(info, f, indent=2)
print(f"Model and info saved for real data. Features: {list(X.columns)}")
