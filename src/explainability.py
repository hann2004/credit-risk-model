"""SHAP explainability utilities for global and local insights."""

from __future__ import annotations
from src.predict import align_features, load_model
from src.constants import DEFAULT_MODEL_URI, PROCESSED_WITH_TARGET_PATH
import shap
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
plt.style.use("dark_background")


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
    # Reduce max_samples for speed
    X = load_feature_matrix(data_path=data_path, max_samples=200)
    missing = set(feature_names or []) - set(X.columns)
    if missing:
        raise ValueError(
            f"Input data is missing required features: {sorted(missing)}.\n\nDownload and use the template_features.csv to ensure all columns are present.")
    X = align_features(X, feature_names)

    explainer = _build_explainer(model, X)
    shap_values = explainer(X)

    # Fix for classifier: use positive class SHAP values
    if hasattr(
            shap_values,
            "values") and shap_values.values.ndim == 3 and shap_values.values.shape[2] == 2:
        shap_values = shap.Explanation(
            shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=X,
            feature_names=list(X.columns)
        )

    import matplotlib.pyplot as plt
    plt.clf()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "shap_summary.png"
    bar_path = output_path / "shap_bar.png"
    pie_path = output_path / "shap_pie.png"

    # SHAP summary plot (dot)
    shap.summary_plot(
        shap_values, X, show=False, plot_size=(8, 6)
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200, facecolor='black')
    plt.close()

    plt.clf()
    # SHAP summary plot (bar)
    shap.summary_plot(
        shap_values, X, plot_type='bar', show=False, plot_size=(8, 6)
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, facecolor='black')
    plt.close()

    values = getattr(shap_values, "values", shap_values)
    mean_abs = np.abs(values).mean(axis=0)
    feature_importance = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    top_n = 6
    top_features = feature_importance.head(top_n)
    other_total = feature_importance.iloc[top_n:].sum()
    if other_total > 0:
        top_features["Other"] = other_total

    plt.clf()
    plt.figure(figsize=(6, 6), facecolor='black')
    wedges, texts, autotexts = plt.pie(
        top_features.values,
        labels=top_features.index,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        textprops={'color': 'white'}
    )
    plt.setp(texts, color='white')
    plt.setp(autotexts, color='white')
    plt.title("Top risk drivers (global)", color='white')
    plt.tight_layout()
    plt.savefig(pie_path, dpi=200, facecolor='black')
    plt.close()

    return {"summary": summary_path, "bar": bar_path, "pie": pie_path}


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
    missing = set(feature_names or []) - set(instance_df.columns)
    if missing:
        raise ValueError(
            f"Input instance is missing required features: {sorted(missing)}.\n\nDownload and use the template_features.csv to ensure all columns are present.")
    instance_df = align_features(instance_df, feature_names)
    instance_df = instance_df.astype(float)

    explainer = _build_explainer(model, background)
    shap_values = explainer(instance_df)

    # Handle classifier SHAP output: extract positive class values
    import shap
    if hasattr(shap_values, "values"):
        values = shap_values.values[0]
        # If values are 2D (n_features, 2), take positive class
        if values.ndim == 2 and values.shape[1] == 2:
            values = values[:, 1]
        # SHAP waterfall expects values and feature names as a SHAP Explanation object or tuple
        # If feature_names argument is not supported, use SHAP's Explanation object
        try:
            shap.plots.waterfall(values, show=False)
        except TypeError:
            # fallback: create SHAP Explanation object
            # Ensure base_values is a scalar
            base_value = shap_values.base_values[0]
            if hasattr(base_value, '__len__') and len(base_value) == 2:
                base_value = base_value[1]
            explanation = shap.Explanation(
                values,
                base_values=base_value,
                data=instance_df.iloc[0],
                feature_names=list(
                    instance_df.columns))
            shap.plots.waterfall(explanation, show=False)
    else:
        shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    fig = plt.gcf()
    return fig
