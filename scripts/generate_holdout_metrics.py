from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/processed_with_target.csv")
OUT_PATH = Path("models/production_model_info.json")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(
        columns=[col for col in ["CustomerId", "is_high_risk"] if col in df.columns],
        errors="ignore",
    )
    y = df["is_high_risk"].astype(int)

    bool_cols = X.select_dtypes(include=[bool]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    info = {
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "date_saved": str(date.today()),
        "evaluation_type": "holdout_stratified_test",
        "test_size": 0.2,
        "random_state": 42,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_names": list(X.columns),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(info, indent=2))

    print("Saved holdout metrics to", OUT_PATH)
    print(json.dumps({
        "roc_auc": info["roc_auc"],
        "f1": info["f1"],
        "precision": info["precision"],
        "recall": info["recall"],
        "accuracy": info["accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
