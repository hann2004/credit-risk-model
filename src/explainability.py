"""SHAP explainability utilities for global and local insights."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.constants import DEFAULT_MODEL_URI, PROCESSED_WITH_TARGET_PATH
from src.predict import align_features, load_model


def load_feature_matrix(
    data_path: str | Path = PROCESSED_WITH_TARGET_PATH,
    max_samples: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["is_high_risk", "CustomerId"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
    X = X.astype(float)
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=random_state)
    return X


def _build_explainer(
    model: object,
    background: pd.DataFrame,
) -> shap.Explainer:
    return shap.Explainer(model, background)


def generate_global_shap_artifacts(
    model_uri: str = DEFAULT_MODEL_URI,
    data_path: str | Path = PROCESSED_WITH_TARGET_PATH,
    output_dir: str | Path = "reports/figures",
    max_samples: int = 500,
) -> Dict[str, Path]:
    model, feature_names = load_model(model_uri)
    X = load_feature_matrix(data_path=data_path, max_samples=max_samples)
    X = align_features(X, feature_names)

    explainer = _build_explainer(model, X)
    shap_values = explainer(X.values)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "shap_summary.png"
    bar_path = output_path / "shap_bar.png"

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()

    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    return {"summary": summary_path, "bar": bar_path}


def generate_local_shap_plot(
    instance: Dict[str, float],
    model_uri: str = DEFAULT_MODEL_URI,
    data_path: str | Path = PROCESSED_WITH_TARGET_PATH,
    max_samples: int = 200,
) -> plt.Figure:
    model, feature_names = load_model(model_uri)
    background = load_feature_matrix(data_path=data_path, max_samples=max_samples)
    background = align_features(background, feature_names)
    background = background.astype(float)

    instance_df = pd.DataFrame([instance]).apply(pd.to_numeric, errors="coerce")
    instance_df = align_features(instance_df, feature_names)
    instance_df = instance_df.astype(float)

    explainer = _build_explainer(model, background)
    shap_values = explainer(instance_df.values)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    return fig
