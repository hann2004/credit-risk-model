"""Streamlit dashboard for credit risk scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.constants import DEFAULT_MODEL_URI, PROCESSED_WITH_TARGET_PATH
from src.explainability import generate_global_shap_artifacts, generate_local_shap_plot
from src.predict import load_model, predict_instances

API_TIMEOUT_SECONDS = 5


@st.cache_resource
def _load_local_model(model_uri: str) -> Tuple[object, Optional[List[str]]]:
    return load_model(model_uri)


@st.cache_data
def _load_reference_scores(model_uri: str, data_path: str) -> List[float]:
    model, feature_names = _load_local_model(model_uri)
    df = pd.read_csv(data_path)
    df = df.drop(columns=["is_high_risk", "CustomerId"], errors="ignore")
    if feature_names:
        missing = sorted(set(feature_names) - set(df.columns))
        if missing:
            raise ValueError(f"Missing features in dataset: {missing}")
        df = df[feature_names]
    if df.empty:
        raise ValueError("Reference dataset has no rows")
    return predict_instances(model, df.to_dict(orient="records"), feature_names)


def _api_health(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/health", timeout=API_TIMEOUT_SECONDS)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _predict_via_api(base_url: str, instances: Sequence[Dict[str, float]]) -> List[float]:
    response = requests.post(
        f"{base_url}/predict",
        json={"instances": instances},
        timeout=API_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise ValueError(response.text)
    payload = response.json()
    return [float(x) for x in payload.get("risk_probabilities", [])]


def _predict_via_local(model_uri: str, instances: Sequence[Dict[str, float]]) -> List[float]:
    model, feature_names = _load_local_model(model_uri)
    return predict_instances(model, instances, feature_names)


def _render_branding() -> None:
    st.set_page_config(
        page_title="Credit Risk Control Room",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
            background: radial-gradient(circle at top left, #f7f2e8 0%, #f1efe6 40%, #ecf1f2 100%);
            color: #1f2a32;
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }
        .hero {
            background: linear-gradient(120deg, #0f3d3e 0%, #1f5c5d 55%, #e0b44c 100%);
            padding: 2rem;
            border-radius: 18px;
            color: #fdfcf8;
            box-shadow: 0 18px 45px rgba(15, 61, 62, 0.25);
        }
        .hero h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .hero p {
            font-size: 1.05rem;
            opacity: 0.9;
        }
        .metric-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.4rem;
            border: 1px solid rgba(31, 42, 50, 0.08);
            box-shadow: 0 8px 20px rgba(20, 35, 45, 0.08);
        }
        .metric-card h3 {
            margin: 0;
            font-size: 1.1rem;
            color: #0f3d3e;
        }
        .metric-card p {
            margin: 0.4rem 0 0;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.2rem;
        }
        .section-title {
            font-size: 1.2rem;
            margin: 1.2rem 0 0.6rem;
            color: #0f3d3e;
        }
        .stButton>button {
            background: #0f3d3e;
            color: #fdfcf8;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            border: none;
            font-weight: 600;
        }
        .stButton>button:hover {
            background: #124b4c;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _default_instance(feature_names: Optional[List[str]]) -> Dict[str, float]:
    if not feature_names:
        return {"feature_1": 0.0, "feature_2": 0.0}
    return {name: 0.0 for name in feature_names}


def _parse_json_instances(raw_text: str) -> List[Dict[str, float]]:
    parsed = json.loads(raw_text)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    raise ValueError("JSON input must be an object or list of objects")


def _score_instances(
    instances: List[Dict[str, float]],
    api_url: str,
    use_api: bool,
    allow_fallback: bool,
    model_uri: str,
) -> Tuple[List[float], str]:
    if use_api and _api_health(api_url):
        return _predict_via_api(api_url, instances), "API"
    if allow_fallback:
        return _predict_via_local(model_uri, instances), "Local"
    raise ValueError("API unavailable and local fallback disabled")


_render_branding()

st.markdown(
    """
    <div class="hero">
        <h1>Credit Risk Control Room</h1>
        <p>Reliable credit risk scoring with traceable decisions and transparent fallbacks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Deployment Controls")
api_url = st.sidebar.text_input("FastAPI base URL", value="http://localhost:8000")
model_uri = st.sidebar.text_input("MLflow model URI", value=DEFAULT_MODEL_URI)
data_path = st.sidebar.text_input(
    "Feature dataset path",
    value=str(PROCESSED_WITH_TARGET_PATH),
)
use_api = st.sidebar.toggle("Use API when available", value=True)
allow_fallback = st.sidebar.toggle("Allow local fallback", value=True)
threshold = st.sidebar.slider("Risk threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
decision_mode = st.sidebar.selectbox(
    "Decision policy",
    options=["Top percentile", "Probability threshold"],
)
percentile = st.sidebar.slider("High-risk percentile", 80, 99, 90, 1)

status = "Online" if _api_health(api_url) else "Offline"
st.sidebar.markdown(f"**API Status:** {status}")

kpi_cols = st.columns(3)
with kpi_cols[0]:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Target ROC-AUC</h3>
            <p>≥ 0.80</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_cols[1]:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Decision SLA</h3>
            <p>< 3 sec</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_cols[2]:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Risk Guardrail</h3>
            <p>Explainable</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-title'>Score Inputs</div>", unsafe_allow_html=True)

try:
    local_model, feature_names = _load_local_model(model_uri)
except Exception:
    feature_names = None

input_tabs = st.tabs(["Single Applicant", "Batch CSV", "JSON"])

with input_tabs[0]:
    if not feature_names:
        st.info("Upload a CSV or provide JSON to score. Feature schema not loaded.")
    else:
        st.caption("Enter realistic values; all-zero inputs usually score very low risk.")
        with st.form("single_prediction"):
            form_cols = st.columns(2)
            instance: Dict[str, float] = {}
            for idx, name in enumerate(feature_names):
                col = form_cols[idx % 2]
                instance[name] = col.number_input(name, value=0.0)
            submitted = st.form_submit_button("Score applicant")
        if submitted:
            probs, channel = _score_instances(
                [instance], api_url, use_api, allow_fallback, model_uri
            )
            risk = probs[0]
            st.session_state["last_instance"] = instance
            if decision_mode == "Top percentile":
                try:
                    ref_scores = _load_reference_scores(model_uri, data_path)
                    threshold = float(np.quantile(ref_scores, percentile / 100))
                except Exception as exc:
                    st.warning(f"Percentile threshold unavailable: {exc}")
            st.metric("Risk probability", f"{risk:.2f}")
            st.write(f"Decision: {'High risk' if risk >= threshold else 'Acceptable'}")
            st.caption(f"Scored via {channel} channel")

with input_tabs[1]:
    csv_file = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        if feature_names and "CustomerId" in df.columns and "CustomerId" not in feature_names:
            df = df.drop(columns=["CustomerId"])
        st.write(df.head())
        if st.button("Score batch"):
            instances = df.to_dict(orient="records")
            probs, channel = _score_instances(
                instances, api_url, use_api, allow_fallback, model_uri
            )
            if decision_mode == "Top percentile":
                threshold = float(np.quantile(probs, percentile / 100))
            df["risk_probability"] = probs
            df["decision"] = df["risk_probability"].apply(
                lambda p: "High risk" if p >= threshold else "Acceptable"
            )
            high_risk_count = int((df["decision"] == "High risk").sum())
            st.caption(f"High-risk flagged: {high_risk_count} of {len(df)}")
            st.write(df)
            st.caption(f"Scored via {channel} channel")

with input_tabs[2]:
    default_json = json.dumps([_default_instance(feature_names)], indent=2)
    raw_json = st.text_area("Paste JSON", value=default_json, height=220)
    if st.button("Score JSON"):
        try:
            instances = _parse_json_instances(raw_json)
            probs, channel = _score_instances(
                instances, api_url, use_api, allow_fallback, model_uri
            )
            st.success("Scoring complete")
            st.json({"risk_probabilities": probs, "channel": channel})
        except Exception as exc:
            st.error(str(exc))

st.markdown("<div class='section-title'>Explainability</div>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])
with col_left:
    st.write("Global explanations show which features drive risk across the portfolio.")
    if st.button("Generate global SHAP plots"):
        try:
            if not Path(data_path).exists():
                raise FileNotFoundError("Feature dataset not found. Run data_processing.py first.")
            outputs = generate_global_shap_artifacts(
                model_uri=model_uri,
                data_path=data_path,
            )
            st.session_state["global_shap"] = outputs
            st.success("Global SHAP plots updated")
        except Exception as exc:
            st.error(str(exc))

with col_right:
    st.write("Local explanations apply to the last scored applicant.")
    if st.button("Generate local SHAP explanation"):
        instance = st.session_state.get("last_instance")
        if not instance:
            st.warning("Score a single applicant first.")
        else:
            try:
                if not Path(data_path).exists():
                    raise FileNotFoundError("Feature dataset not found. Run data_processing.py first.")
                fig = generate_local_shap_plot(
                    instance=instance,
                    model_uri=model_uri,
                    data_path=data_path,
                )
                st.pyplot(fig)
            except Exception as exc:
                st.error(str(exc))

global_outputs = st.session_state.get("global_shap")
if global_outputs:
    summary_path = global_outputs.get("summary")
    bar_path = global_outputs.get("bar")
    pie_path = global_outputs.get("pie")
    if summary_path:
        st.image(str(summary_path), caption="SHAP summary plot")
    if bar_path:
        st.image(str(bar_path), caption="SHAP feature importance")
    if pie_path:
        st.image(str(pie_path), caption="Top risk drivers (simple view)")

st.markdown("<div class='section-title'>Operational Notes</div>", unsafe_allow_html=True)

st.write(
    """
    - Decisions are built for auditability: data, features, and labels are logged in MLflow.
    - Use the temporal cutoff pipeline to prevent target leakage during training.
    - For production, connect the dashboard to a deployed FastAPI endpoint.
    """
)
