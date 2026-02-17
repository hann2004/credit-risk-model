import json
from datetime import date
from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# Train and save a model directly for dashboard use
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=42,
)

test_size = 0.2
split = int(len(X) * (1 - test_size))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
f1 = f1_score(y_test, preds)
roc_auc = roc_auc_score(y_test, proba)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
accuracy = accuracy_score(y_test, preds)

print(f"F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

if f1 >= 0.8 and roc_auc > 0.8:
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/production_model.pkl")
    info = {
        "f1": f1,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "date_saved": str(date.today()),
    }
    with open("models/production_model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("[SUCCESS] Model and info saved for dashboard.")
else:
    print("[FAIL] Model does not meet criteria. Rerun if needed.")
