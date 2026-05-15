"""Model training with MLflow tracking for credit risk proxy target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

if __name__ == "__main__" and __package__ is None:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import TrainingConfig
from src.constants import DEFAULT_EXPERIMENT_NAME, PROCESSED_WITH_TARGET_PATH, RAW_DATA_PATH


def _prepare_data(path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset with target not found at {path}. "
            "Run `python src/data_processing.py --with-target` first."
        )
    labels_df = pd.read_csv(path)
    if "is_high_risk" not in labels_df.columns:
        raise ValueError("Column is_high_risk missing. Recreate processed_with_target dataset.")

    # Build unscaled customer-level features from raw data.
    raw_df = pd.read_csv(RAW_DATA_PATH)
    customer_df = (
        raw_df.groupby("CustomerId")
        .agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            txn_count=("Amount", "count"),
            std_amount=("Amount", "std"),
        )
        .reset_index()
    )
    customer_df["std_amount"] = customer_df["std_amount"].fillna(0.0)

    categorical_cols = [
        c
        for c in [
            "CurrencyCode",
            "CountryCode",
            "ProviderId",
            "ProductId",
            "ProductCategory",
            "ChannelId",
            "PricingStrategy",
        ]
        if c in raw_df.columns
    ]

    if categorical_cols:
        cat_modes = (
            raw_df[["CustomerId"] + categorical_cols]
            .groupby("CustomerId")
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
            .reset_index()
        )
        customer_df = customer_df.merge(cat_modes, on="CustomerId", how="left")

    features_df = pd.get_dummies(customer_df, columns=categorical_cols, dummy_na=True)
    labels = labels_df[["CustomerId", "is_high_risk"]].drop_duplicates("CustomerId")
    df = features_df.merge(labels, on="CustomerId", how="inner")

    y = df["is_high_risk"].astype(int)
    X = df.drop(columns=["is_high_risk", "CustomerId"], errors="ignore")

    # Convert bools to ints to avoid estimator warnings
    bool_cols = X.select_dtypes(include=[bool]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y


def _evaluate(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        proba = model.decision_function(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    if proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, proba)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def _train_log_reg(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"))
    ])
    param_grid = {"model__C": [0.1, 1.0, 10.0]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def _train_random_forest(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])
    param_distributions = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=4,
        cv=3,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_and_log(
    data_path: str | Path = PROCESSED_WITH_TARGET_PATH,
    config: TrainingConfig | None = None,
) -> None:
    if config is None:
        config = TrainingConfig()
    mlflow.set_experiment(config.experiment)
    X, y = _prepare_data(data_path)

    # Ensure both train and test have at least one sample of each class
    # Removed unused import Counter

    max_tries = 10
    for i in range(max_tries):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state + i,
            stratify=y,
        )
        train_classes = set(y_train)
        test_classes = set(y_test)
        if len(train_classes) == 2 and len(test_classes) == 2:
            break
    else:
        raise ValueError(
            "Could not create a split with both classes in train and test after multiple tries."
        )

    # Oversample minority class in training set
    try:
        from imblearn.over_sampling import RandomOverSampler
    except ImportError:
        raise ImportError(
            "imblearn is required for oversampling. Install with 'pip install imbalanced-learn'."
        )
    ros = RandomOverSampler(random_state=config.random_state)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    import os

    candidates = []

    # Logistic Regression
    with mlflow.start_run(run_name="log_reg") as run:
        model, params = _train_log_reg(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        artifact_uri = mlflow.get_artifact_uri("model")
        print(f"[INFO] Logistic Regression model logged at: {artifact_uri}")
        model_pkl_path = os.path.join(mlflow.get_artifact_uri("model"), "model.pkl")
        print(f"[DEBUG] Checking model file: {model_pkl_path}")
        candidates.append(("log_reg", metrics.get("roc_auc", -np.inf), run.info.run_id))

    # Random Forest
    with mlflow.start_run(run_name="random_forest") as run:
        model, params = _train_random_forest(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        artifact_uri = mlflow.get_artifact_uri("model")
        print(f"[INFO] Random Forest model logged at: {artifact_uri}")
        model_pkl_path = os.path.join(mlflow.get_artifact_uri("model"), "model.pkl")
        print(f"[DEBUG] Checking model file: {model_pkl_path}")
        candidates.append(("random_forest", metrics.get("roc_auc", -np.inf), run.info.run_id))

    # Select best by roc_auc (fallback to f1 if nan)
    best_run_id = None
    best_model_name = None
    best_score = -np.inf
    for model_name, score, run_id in candidates:
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_run_id = run_id
    if best_run_id is None and candidates:
        best_model_name = candidates[0][0]
        best_run_id = candidates[0][2]

    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        try:
            mlflow.register_model(model_uri=model_uri, name="credit-risk-best-model")
        except Exception:
            # Registry may not be available; log a message instead
            print("Model Registry not available; model logged but not registered.")
    
    # Retrain best model on full training set and save as production model
    if best_run_id and best_model_name:
        print(f"\n[INFO] Retraining best model on full training set and saving...")

        # Train final production model on full labeled dataset (oversampled)
        X_full, y_full = ros.fit_resample(X, y)
        if best_model_name == "random_forest":
            best_model, _ = _train_random_forest(X_full, y_full)
            best_name = "Random Forest"
        else:
            best_model, _ = _train_log_reg(X_full, y_full)
            best_name = "Logistic Regression"

        # Save to both locations
        dump(best_model, "app/production_model.pkl")
        dump(best_model, "models/production_model.pkl")
        print(f"[INFO] Best model ({best_name}) saved as production_model.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk models with MLflow logging")
    parser.add_argument(
        "--data-path",
        default=str(PROCESSED_WITH_TARGET_PATH),
        help="Path to labeled data containing CustomerId and is_high_risk",
    )
    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT_NAME, help="MLflow experiment name"
    )
    parser.add_argument("--test-size", type=float, default=TrainingConfig().test_size)
    parser.add_argument("--random-state", type=int, default=TrainingConfig().random_state)
    args = parser.parse_args()
    config = TrainingConfig(
        experiment=args.experiment,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_and_log(data_path=args.data_path, config=config)


if __name__ == "__main__":
    main()
