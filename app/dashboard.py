"""
EqubScore -- Alternative Credit Scoring for Ethiopia's Equb Groups
Run: streamlit run dashboard_v2.py
Place production_model.pkl in the same folder as this file.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

MODEL_PATH = Path(__file__).parent / "production_model.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).parent.parent / "production_model.pkl"
RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "data.csv"


st.set_page_config(
    page_title="EqubScore",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,300&display=swap');

html, body, .stApp {
    background: #0a0c0f;
    font-family: 'Space Grotesk', sans-serif;
    color: #e2ddd6;
}
[data-testid="stSidebar"] {
    background: #070809;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #9ca3af !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.88em; padding: 6px 10px; }

h1,h2,h3,h4 {
    font-family: 'Fraunces', serif !important;
    color: #f5f0e8 !important;
    letter-spacing: -0.02em;
}

.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 28px 0;
    margin-bottom: 32px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.logo-box {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif;
    font-weight: 700; font-size: 1.2em; color: #fff;
}
.logo-name {
    font-family: 'Fraunces', serif;
    font-weight: 700; font-size: 1.5em;
    color: #f5f0e8; letter-spacing: -0.03em;
}
.logo-tagline {
    font-size: 0.72em; color: #6b7280;
    letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px;
}
.status-pill {
    font-size: 0.72em; font-weight: 600;
    color: #22c55e; background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 100px; padding: 5px 14px; letter-spacing: 0.04em;
}

.sec-label {
    font-size: 0.65em; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #22c55e; margin: 28px 0 14px 0;
}

.card {
    background: #111318;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 22px;
    position: relative; overflow: hidden;
}
.card-accent {
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #22c55e 0%, transparent 70%);
}
.card-label {
    font-size: 0.68em; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #6b7280; margin-bottom: 10px;
}
.card-value {
    font-family: 'Fraunces', serif;
    font-size: 2.2em; font-weight: 600;
    color: #f5f0e8; line-height: 1;
}
.card-sub { font-size: 0.74em; color: #4b5563; margin-top: 6px; }

.signal-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin: 18px 0;
}

.profile-strip {
    display: flex; align-items: center; gap: 16px;
    background: #111318;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 18px 22px; margin-bottom: 22px;
}
.avatar {
    width: 48px; height: 48px; border-radius: 50%;
    background: linear-gradient(135deg, #22c55e, #15803d);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif;
    font-weight: 700; font-size: 1.1em; color: #fff; flex-shrink: 0;
}
.profile-name {
    font-family: 'Fraunces', serif;
    font-size: 1.1em; font-weight: 600; color: #f5f0e8;
}
.profile-meta { font-size: 0.8em; color: #6b7280; margin-top: 3px; }

.pill {
    display: inline-flex; align-items: center; gap: 8px;
    border-radius: 100px; padding: 9px 18px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; font-size: 0.88em; letter-spacing: 0.04em;
}
.dot { width: 7px; height: 7px; border-radius: 50%; }
.pill-low    { background: rgba(34,197,94,0.1);  color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
.pill-medium { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.pill-high   { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

.rec {
    border-radius: 12px; padding: 16px 20px;
    font-size: 0.86em; line-height: 1.7; border-left: 3px solid;
}
.rec-approve { background: rgba(34,197,94,0.05);  border-color: #22c55e; color: #86efac; }
.rec-review  { background: rgba(251,191,36,0.05); border-color: #fbbf24; color: #fde68a; }
.rec-reject  { background: rgba(239,68,68,0.05);  border-color: #ef4444; color: #fca5a5; }

.explain-wrap {
    background: #111318;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 20px;
}
.explain-row { display: flex; gap: 14px; margin-bottom: 16px; align-items: flex-start; }
.explain-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: rgba(34,197,94,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95em; flex-shrink: 0;
}
.explain-title { font-weight: 600; font-size: 0.86em; color: #d1cbc0; margin-bottom: 2px; }
.explain-body { font-size: 0.80em; color: #6b7280; line-height: 1.6; }

.batch-row {
    display: flex; align-items: center; gap: 14px;
    border-radius: 12px; padding: 12px 16px; margin-bottom: 8px; border: 1px solid;
}
.batch-low    { background: rgba(34,197,94,0.04);  border-color: rgba(34,197,94,0.14); }
.batch-medium { background: rgba(251,191,36,0.04); border-color: rgba(251,191,36,0.14); }
.batch-high   { background: rgba(239,68,68,0.04);  border-color: rgba(239,68,68,0.14); }

.info-box {
    background: #111318;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px 18px; margin-bottom: 18px;
    font-size: 0.83em; color: #6b7280; line-height: 1.7;
}

.stSelectbox>div>div,
.stNumberInput input,
.stTextInput input {
    background: #111318 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2ddd6 !important;
    border-radius: 10px !important;
}
.stButton>button {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95em !important;
    padding: 12px 28px !important; width: 100% !important;
    letter-spacing: 0.02em !important; transition: opacity 0.15s !important;
}
.stButton>button:hover { opacity: 0.85 !important; }
.stSelectbox label, .stTextInput label, .stNumberInput label {
    font-size: 0.78em !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    color: #6b7280 !important;
}
hr, .stDivider { border-color: rgba(255,255,255,0.06) !important; }
.stAlert {
    background: #111318 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="metric-container"] {
    background: #111318;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #6b7280 !important; font-size: 0.74em !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f5f0e8 !important;
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


CATEGORY_MAP = {
    "Airtime top-up":     "ProductCategory_airtime",
    "Utility bills":      "ProductCategory_utility_bill",
    "Financial services": "ProductCategory_financial_services",
    "Data bundles":       "ProductCategory_data_bundles",
    "Transport":          "ProductCategory_transport",
    "Other":              "ProductCategory_other",
}
PROVIDER_MAP = {
    "TeleBirr":  "ProviderId_ProviderId_4",
    "CBE Birr":  "ProviderId_ProviderId_1",
    "HelloCash": "ProviderId_ProviderId_2",
    "M-Pesa":    "ProviderId_ProviderId_3",
    "amole":     "ProviderId_ProviderId_5",
    "Other":     "ProviderId_ProviderId_6",
}
CHANNEL_MAP = {
    "Mobile App":    "ChannelId_ChannelId_2",
    "USSD (*127#)":  "ChannelId_ChannelId_3",
    "Agent":         "ChannelId_ChannelId_5",
    "Web browser":   "ChannelId_ChannelId_1",
}
FEATURE_LABELS = {
    "txn_count":                          "Transaction Frequency",
    "total_amount":                       "Total Monthly Volume",
    "avg_amount":                         "Avg Transaction Size",
    "std_amount":                         "Payment Consistency",
    "ProductCategory_utility_bill":       "Pays Utility Bills",
    "ProductCategory_financial_services": "Uses Financial Services",
    "ProductCategory_airtime":            "Airtime Pattern",
    "ProductCategory_data_bundles":       "Data Bundle Use",
    "ChannelId_ChannelId_2":              "Uses Mobile App",
    "ChannelId_ChannelId_3":              "Uses USSD Code",
    "ChannelId_ChannelId_5":              "Agent-Only Transactions",
    "ProviderId_ProviderId_4":            "TeleBirr Account",
    "ProviderId_ProviderId_1":            "CBE Birr Account",
    "ProviderId_ProviderId_2":            "HelloCash Account",
}
REF = {"txn_count": 50.0, "total_amount": 30_000.0, "avg_amount": 800.0, "std_amount": 600.0}
NUMERIC_DIR = {"txn_count": -1, "total_amount": -1, "avg_amount": -1, "std_amount": +1}
CAT_DIR = {
    "ProductCategory_utility_bill": -1, "ProductCategory_financial_services": -1,
    "ProductCategory_data_bundles": -1, "ProductCategory_airtime": 0,
    "ChannelId_ChannelId_2": -1, "ChannelId_ChannelId_3": 0, "ChannelId_ChannelId_5": +1,
}

SAMPLE_MEMBERS = [
    {"name":"Abebe Girma",     "occupation":"Merchant",         "city":"Arba Minch",
     "txn_count":210,"avg_amount":3200, "total_amount":672_000,"std_amount":420,
     "category":"Financial services","provider":"TeleBirr","channel":"Mobile App"},
    {"name":"Tigist Haile",    "occupation":"Teacher",          "city":"Arba Minch",
     "txn_count":38, "avg_amount":1100, "total_amount":41_800, "std_amount":180,
     "category":"Utility bills","provider":"CBE Birr","channel":"Mobile App"},
    {"name":"Dawit Mekonnen",  "occupation":"Daily Laborer",    "city":"Sodo",
     "txn_count":9,  "avg_amount":400,  "total_amount":3_600,  "std_amount":900,
     "category":"Airtime top-up","provider":"TeleBirr","channel":"USSD (*127#)"},
    {"name":"Sara Tadesse",    "occupation":"Nurse",            "city":"Arba Minch",
     "txn_count":55, "avg_amount":1800, "total_amount":99_000, "std_amount":260,
     "category":"Utility bills","provider":"CBE Birr","channel":"Mobile App"},
    {"name":"Yonas Bekele",    "occupation":"Cafe Owner",       "city":"Arba Minch",
     "txn_count":145,"avg_amount":2100, "total_amount":304_500,"std_amount":1100,
     "category":"Financial services","provider":"HelloCash","channel":"Mobile App"},
    {"name":"Hiwot Alemu",     "occupation":"Farmer",           "city":"Chencha",
     "txn_count":6,  "avg_amount":600,  "total_amount":3_600,  "std_amount":1400,
     "category":"Airtime top-up","provider":"Other","channel":"Agent"},
    {"name":"Mulugeta Tesfaye","occupation":"Government Worker","city":"Arba Minch",
     "txn_count":42, "avg_amount":2400, "total_amount":100_800,"std_amount":150,
     "category":"Utility bills","provider":"CBE Birr","channel":"Mobile App"},
    {"name":"Meron Assefa",    "occupation":"Student",          "city":"Arba Minch",
     "txn_count":22, "avg_amount":350,  "total_amount":7_700,  "std_amount":280,
     "category":"Data bundles","provider":"TeleBirr","channel":"USSD (*127#)"},
    {"name":"Biruk Kebede",    "occupation":"Electrician",      "city":"Sawla",
     "txn_count":31, "avg_amount":950,  "total_amount":29_450, "std_amount":720,
     "category":"Airtime top-up","provider":"TeleBirr","channel":"Agent"},
    {"name":"Liya Tsegaye",    "occupation":"Shop Owner",       "city":"Arba Minch",
     "txn_count":88, "avg_amount":1750, "total_amount":154_000,"std_amount":340,
     "category":"Financial services","provider":"TeleBirr","channel":"Mobile App"},
    {"name":"Hanna Wolde",     "occupation":"Market Trader",    "city":"Chencha",
     "txn_count":1,  "avg_amount":150,  "total_amount":150,    "std_amount":5000,
     "category":"Other","provider":"Other","channel":"Agent"},
    {"name":"Tadesse Asefa",   "occupation":"Casual Laborer",   "city":"Sodo",
     "txn_count":2,  "avg_amount":150,  "total_amount":300,    "std_amount":2500,
     "category":"Other","provider":"Other","channel":"Agent"},
]

if "custom_members" not in st.session_state:
    st.session_state.custom_members = []


@st.cache_resource
def load_model():
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def model_feature_count(model) -> int:
    if model is None:
        return 0
    if hasattr(model, "feature_names_in_"):
        return len(model.feature_names_in_)
    try:
        if hasattr(model, "named_steps"):
            final_est = model.named_steps.get("model", list(model.named_steps.values())[-1])
            if hasattr(final_est, "feature_names_in_"):
                return len(final_est.feature_names_in_)
    except Exception:
        pass
    return 0


def initials(name: str) -> str:
    p = name.strip().split()
    return (p[0][0] + p[-1][0]).upper() if len(p) >= 2 else name[:2].upper()


@st.cache_data
def load_numeric_reference_stats() -> dict[str, dict[str, float]]:
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        if not {"CustomerId", "Amount"}.issubset(raw_df.columns):
            raise ValueError("raw data missing required columns")
        customer_df = (
            raw_df.groupby("CustomerId")
            .agg(
                total_amount=("Amount", "sum"),
                avg_amount=("Amount", "mean"),
                txn_count=("Amount", "count"),
                std_amount=("Amount", "std"),
            )
            .reset_index(drop=True)
        )
        customer_df["std_amount"] = customer_df["std_amount"].fillna(0.0)
        numeric = customer_df[["total_amount", "avg_amount", "txn_count", "std_amount"]]
        return {
            "means": numeric.mean().to_dict(),
            "stds": numeric.std(ddof=0).replace(0, 1).to_dict(),
        }
    except Exception:
        return {
            "means": {"total_amount": 0.0, "avg_amount": 0.0, "txn_count": 0.0, "std_amount": 0.0},
            "stds": {"total_amount": 1.0, "avg_amount": 1.0, "txn_count": 1.0, "std_amount": 1.0},
        }

def get_risk(p: float, **kwargs) -> str:
    if p < 0.17:  return "LOW"
    if p < 0.22:  return "MEDIUM"
    return "HIGH"

def risk_color(risk: str) -> str:
    return {"LOW": "#22c55e", "MEDIUM": "#fbbf24", "HIGH": "#ef4444"}[risk]

def recommendation(risk: str) -> str:
    return {
        "LOW":    "Approve for standard Equb participation. Consistent mobile money usage signals reliable financial behavior.",
        "MEDIUM": "Approve at a lower initial contribution tier. Monitor for the first two cycles before raising limits.",
        "HIGH":   "Request a guarantor or a one-cycle probationary period before granting full membership.",
    }[risk]

def build_vector(model, txn_count, avg_amount, total_amount, std_amount,
                 category, provider, channel) -> pd.DataFrame:
    # Resolve feature names from a Pipeline or a bare estimator
    try:
        if hasattr(model, "named_steps"):
            final_est = model.named_steps.get("model", list(model.named_steps.values())[-1])
            feature_names = getattr(final_est, "feature_names_in_", getattr(model, "feature_names_in_", []))
        else:
            feature_names = getattr(model, "feature_names_in_", [])
    except Exception:
        feature_names = getattr(model, "feature_names_in_", [])

    vec = {f: 0.0 for f in feature_names}

    # Do NOT standardize here. The saved pipeline (if present) contains the scaler
    # so we must supply raw numeric values exactly as collected from the user.
    for feat, val in [
        ("txn_count", txn_count),
        ("total_amount", total_amount),
        ("avg_amount", avg_amount),
        ("std_amount", std_amount),
    ]:
        if feat in vec:
            vec[feat] = float(val)
    if "CurrencyCode_UGX" in vec:        vec["CurrencyCode_UGX"] = 1.0
    if "CountryCode_0.0" in vec:         vec["CountryCode_0.0"]  = 1.0
    if "PricingStrategy_-0.2933549531698252" in vec:
        vec["PricingStrategy_-0.2933549531698252"] = 1.0
    if "ProductId_ProductId_6" in vec:   vec["ProductId_ProductId_6"] = 1.0
    for val, mp in [(category, CATEGORY_MAP), (provider, PROVIDER_MAP), (channel, CHANNEL_MAP)]:
        f = mp.get(val)
        if f and f in vec:
            vec[f] = 1.0
    return pd.DataFrame([vec])

def score(model, txn_count, avg_amount, total_amount, std_amount,
          category, provider, channel) -> float:
    if model is None:
        r = min(txn_count / 100.0, 1.0)
        m = min(total_amount / 200_000.0, 1.0)
        s = min(std_amount / 2000.0, 1.0)
        return float(np.clip(0.5 - r * 0.25 - m * 0.15 + s * 0.2, 0.05, 0.95))
    df = build_vector(model, txn_count, avg_amount, total_amount,
                      std_amount, category, provider, channel)
    proba = model.predict_proba(df)[0]
    # Model classes: [0=low_risk, 1=high_risk]
    # Return probability of HIGH RISK (class 1)
    prob_high_risk = proba[1]
    return float(prob_high_risk)

def get_contributions(model, txn_count, avg_amount, total_amount, std_amount,
                      category, provider, channel):
    if model is None:
        return []

    # Resolve final estimator when using a Pipeline
    try:
        if hasattr(model, "named_steps"):
            final_est = model.named_steps.get("model", list(model.named_steps.values())[-1])
        else:
            final_est = model
    except Exception:
        final_est = model

    # Feature names: prefer estimator.feature_names_in_ if available, else pipeline attribute
    feature_names = getattr(final_est, "feature_names_in_", getattr(model, "feature_names_in_", []))

    # Feature importances: some estimators (RandomForest) expose them; fall back to zeros
    importances = getattr(final_est, "feature_importances_", None)
    if importances is None:
        # Not all models have importances (e.g., LogisticRegression); try coefficients
        coef = getattr(final_est, "coef_", None)
        if coef is not None:
            # coef_ may be 2D (n_classes-1, n_features) for multi-class; sum absolute
            try:
                import numpy as _np
                importances = _np.abs(_np.ravel(coef)).astype(float)
            except Exception:
                importances = None
    if importances is None:
        importances = [0.0] * len(feature_names)

    imp = dict(zip(feature_names, importances))

    out = []
    for feat, val in [("txn_count", txn_count), ("total_amount", total_amount),
                      ("avg_amount", avg_amount), ("std_amount", std_amount)]:
        dev = (val - REF.get(feat, 1.0)) / max(REF.get(feat, 1.0), 1.0)
        out.append((FEATURE_LABELS.get(feat, feat),
                    imp.get(feat, 0) * dev * NUMERIC_DIR.get(feat, 0) * -1))

    for feat in [CATEGORY_MAP.get(category), PROVIDER_MAP.get(provider), CHANNEL_MAP.get(channel)]:
        if not feat:
            continue
        d = CAT_DIR.get(feat, 0)
        if d == 0:
            continue
        out.append((FEATURE_LABELS.get(feat, feat), imp.get(feat, 0) * d * -1))

    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out[:8]


def gauge_chart(prob: float, risk: str) -> go.Figure:
    col = risk_color(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "color": "#f5f0e8", "family": "Fraunces"}},
        title={"text": "DEFAULT PROBABILITY",
               "font": {"size": 10, "color": "#6b7280", "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#374151", "size": 10}},
            "bar": {"color": col, "thickness": 0.25},
            "bgcolor": "#111318", "borderwidth": 0,
            "steps": [
                {"range": [0,  25], "color": "rgba(34,197,94,0.07)"},
                {"range": [25, 40], "color": "rgba(251,191,36,0.07)"},
                {"range": [40,100], "color": "rgba(239,68,68,0.07)"},
            ],
            "threshold": {"line": {"color": col, "width": 3}, "value": prob * 100},
        }
    ))
    fig.update_layout(
        height=265, margin=dict(l=20, r=20, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def waterfall_chart(contrib_list) -> go.Figure | None:
    if not contrib_list: return None
    labels = [c[0] for c in contrib_list]
    values = [c[1] for c in contrib_list]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=["#22c55e" if v > 0 else "#ef4444" for v in values],
        text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values],
        textposition="outside",
        textfont={"size": 10, "color": "#9ca3af"},
    ))
    fig.update_layout(
        height=max(260, len(contrib_list) * 40),
        margin=dict(l=10, r=90, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f5f0e8", "family": "Space Grotesk"},
        xaxis=dict(showgrid=True, gridcolor="#1a1e26",
                   zeroline=True, zerolinecolor="#2d3344", zerolinewidth=2,
                   tickfont={"size": 10, "color": "#374151"},
                   title={"text": "Contribution to risk score",
                          "font": {"size": 10, "color": "#6b7280"}}),
        yaxis=dict(tickfont={"size": 11, "color": "#9ca3af"}, autorange="reversed"),
        bargap=0.38,
    )
    return fig

def donut_chart(counts: dict) -> go.Figure:
    labels = list(counts.keys())
    values = list(counts.values())
    clrs   = {"LOW": "#22c55e", "MEDIUM": "#fbbf24", "HIGH": "#ef4444"}
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker_colors=[clrs.get(l, "#6b7280") for l in labels],
        textinfo="label+percent",
        textfont={"size": 12, "color": "#e2ddd6"},
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        height=270, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f5f0e8", "family": "Space Grotesk"},
        showlegend=False,
    )
    return fig


model = load_model()
feature_count = model_feature_count(model)

with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 8px 0;'>
        <div style='font-family:Fraunces,serif;font-size:1.35em;font-weight:700;color:#f5f0e8;'>EqubScore</div>
        <div style='font-size:0.7em;letter-spacing:0.1em;text-transform:uppercase;color:#374151;margin-top:2px;'>Alternative Credit Scoring</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["Assess Member", "Batch Assessment", "About"])
    st.markdown("---")
    loaded = model is not None
    st.markdown(f"""
    <div style='font-size:0.77em;color:#4b5563;line-height:2;'>
        <div style='color:#22c55e;font-weight:600;margin-bottom:4px;'>SYSTEM</div>
        <div style='color:{"#22c55e" if loaded else "#fbbf24"};'>{"Model loaded" if loaded else "Fallback mode"}</div>
        <div>Random Forest, {feature_count} features</div>
        <div>ROC-AUC 0.923</div>
        <br>
        <div style='color:#22c55e;font-weight:600;margin-bottom:4px;'>SESSION</div>
        <div>{len(st.session_state.custom_members)} member(s) assessed</div>
        <div>Shows in Batch tab</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="top-bar">
    <div style='display:flex;align-items:center;gap:12px;'>
        <div class="logo-box">E</div>
        <div>
            <div class="logo-name">EqubScore</div>
            <div class="logo-tagline">Alternative Credit Scoring for Ethiopia</div>
        </div>
    </div>
    <div class="status-pill">Live Scoring</div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# PAGE: ASSESS MEMBER
# ============================================================================

if page == "Assess Member":

    st.markdown('<div class="sec-label">Applicant Profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Full Name", placeholder="e.g. Abebe Girma")
    with c2:
        occupation = st.selectbox("Occupation", [
            "Merchant / Trader", "Government Employee", "Private Sector Worker",
            "Daily Laborer", "Farmer", "Student", "Self-Employed",
            "Healthcare Worker", "Teacher", "Other",
        ])
    with c3:
        city = st.selectbox("City / Town", [
            "Arba Minch", "Addis Ababa", "Hawassa", "Dire Dawa",
            "Bahir Dar", "Sodo", "Chencha", "Sawla", "Other",
        ])

    st.markdown('<div class="sec-label">Mobile Money Behavior</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Answer based on what you know about the applicant. Ask them to show their
        transaction history, or use your best estimate. The system maps these answers
        to behavioral risk signals automatically.
    </div>
    """, unsafe_allow_html=True)

    q1, q2 = st.columns(2)
    with q1:
        freq_label = st.selectbox("How often do they use mobile money?", [
            "Very frequent  (100+ transactions)",
            "Frequent  (40-100 transactions)",
            "Occasional  (10-40 transactions)",
            "Rare  (under 10 transactions)",
        ])
        freq_map = {
            "Very frequent  (100+ transactions)":   200,
            "Frequent  (40-100 transactions)":       80,
            "Occasional  (10-40 transactions)":      18,
            "Rare  (under 10 transactions)":          6,
        }
        txn_count = freq_map[freq_label]

        amt_label = st.selectbox("Typical transaction size?", [
            "Above ETB 5,000",
            "ETB 1,000 to 5,000",
            "ETB 300 to 1,000",
            "Below ETB 300",
        ])
        amt_map = {
            "Above ETB 5,000":     7_500,
            "ETB 1,000 to 5,000": 2_500,
            "ETB 300 to 1,000":     600,
            "Below ETB 300":        150,
        }
        avg_amount = amt_map[amt_label]

        cons_label = st.selectbox("How consistent are their transaction amounts?", [
            "Very consistent -- similar amounts each time",
            "Mostly consistent -- occasional variation",
            "Irregular -- amounts change a lot",
            "Very unpredictable",
        ])
        cons_map = {
            "Very consistent -- similar amounts each time": 150,
            "Mostly consistent -- occasional variation":    450,
            "Irregular -- amounts change a lot":          1_200,
            "Very unpredictable":                         2_500,
        }
        std_amount = cons_map[cons_label]

    with q2:
        category = st.selectbox("What do they mainly spend on?", list(CATEGORY_MAP.keys()))
        provider  = st.selectbox("Which mobile money service?",   list(PROVIDER_MAP.keys()))
        channel   = st.selectbox("How do they usually transact?", list(CHANNEL_MAP.keys()))

    total_amount = txn_count * avg_amount

    st.markdown('<div class="sec-label">Behavioral Signals Preview</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="signal-grid">
        <div class="card">
            <div class="card-accent"></div>
            <div class="card-label">Frequency</div>
            <div class="card-value">~{txn_count}</div>
            <div class="card-sub">transactions / month</div>
        </div>
        <div class="card">
            <div class="card-accent"></div>
            <div class="card-label">Avg Transaction</div>
            <div class="card-value">ETB {avg_amount:,}</div>
            <div class="card-sub">per transaction</div>
        </div>
        <div class="card">
            <div class="card-accent"></div>
            <div class="card-label">Monthly Volume</div>
            <div class="card-value">ETB {total_amount:,}</div>
            <div class="card-sub">estimated total</div>
        </div>
        <div class="card">
            <div class="card-accent"></div>
            <div class="card-label">Consistency</div>
            <div class="card-value">{'High' if std_amount < 400 else 'Medium' if std_amount < 1000 else 'Low'}</div>
            <div class="card-sub">payment regularity</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate Risk Score"):
        prob = score(model, txn_count, avg_amount, total_amount,
                     std_amount, category, provider, channel)
        risk = get_risk(prob)
        rec  = recommendation(risk)

        display_name = name.strip() if name.strip() else "Applicant"
        new_entry = {
            "name": display_name, "occupation": occupation, "city": city,
            "txn_count": txn_count, "avg_amount": avg_amount,
            "total_amount": total_amount, "std_amount": std_amount,
            "category": category, "provider": provider, "channel": channel,
            "prob": prob, "risk": risk,
        }
        existing = [m for m in st.session_state.custom_members
                    if m["name"] != display_name]
        st.session_state.custom_members = existing + [new_entry]

        st.markdown(f"""
        <div class="profile-strip">
            <div class="avatar">{initials(display_name)}</div>
            <div>
                <div class="profile-name">{display_name}</div>
                <div class="profile-meta">{occupation} &nbsp;·&nbsp; {city} &nbsp;·&nbsp; {provider}</div>
            </div>
            <div style='margin-left:auto;font-size:0.78em;color:#4b5563;text-align:right;'>
                Saved to session<br>
                <span style='color:#22c55e;'>Appears in Batch tab</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Risk Assessment</div>', unsafe_allow_html=True)
        g_col, v_col = st.columns([1, 1], gap="large")
        with g_col:
            st.plotly_chart(gauge_chart(prob, risk), width='stretch')
        with v_col:
            st.markdown("<br>", unsafe_allow_html=True)
            col  = risk_color(risk)
            pcls = f"pill-{risk.lower()}"
            rcls = {"LOW": "rec-approve", "MEDIUM": "rec-review", "HIGH": "rec-reject"}[risk]
            lbl  = {"LOW": "Approve", "MEDIUM": "Review", "HIGH": "Escalate"}[risk]
            st.markdown(f"""
            <div style='margin-bottom:18px;'>
                <div style='font-size:0.68em;color:#6b7280;text-transform:uppercase;
                            letter-spacing:0.12em;margin-bottom:10px;'>Risk Level</div>
                <div class="pill {pcls}">
                    <div class="dot" style='background:{col};'></div>
                    {risk} RISK
                </div>
            </div>
            <div>
                <div style='font-size:0.68em;color:#6b7280;text-transform:uppercase;
                            letter-spacing:0.12em;margin-bottom:10px;'>Recommendation</div>
                <div class="rec {rcls}"><strong>{lbl}:</strong> {rec}</div>
            </div>
            <div style='margin-top:16px;font-size:0.76em;color:#374151;line-height:1.8;'>
                Score: {prob*100:.1f}% default probability<br>
                Model: Random Forest · {feature_count} features · ROC-AUC 0.923
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Why This Score?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Green bars reduce the risk score (good signals). Red bars raise it (warning signals).
            Bar length shows how much each factor influenced the prediction.
        </div>
        """, unsafe_allow_html=True)
        ct = get_contributions(model, txn_count, avg_amount, total_amount,
                               std_amount, category, provider, channel)
        wf = waterfall_chart(ct)
        if wf:
            st.plotly_chart(wf, width='stretch')

        st.markdown('<div class="sec-label">Signal Breakdown</div>', unsafe_allow_html=True)
        freq_note = (
            "Very active account -- strong positive signal."
            if txn_count > 80 else
            "Moderate activity -- acceptable for Equb."
            if txn_count > 20 else
            "Low activity -- limited data to assess reliability."
        )
        vol_note = (
            f"ETB {total_amount:,}/month is strong cash flow."
            if total_amount > 80_000 else
            f"ETB {total_amount:,}/month is sufficient for standard Equb."
            if total_amount > 15_000 else
            f"ETB {total_amount:,}/month is limited. Consider a smaller contribution tier."
        )
        con_note = (
            "Very consistent payments -- strongest trust signal."
            if std_amount < 300 else
            "Mostly consistent -- minor variation is acceptable."
            if std_amount < 800 else
            "High variation detected -- suggests irregular income or spending."
        )
        st.markdown(f"""
        <div class="explain-wrap">
            <div class="explain-row">
                <div class="explain-icon">⚡</div>
                <div>
                    <div class="explain-title">Frequency -- ~{txn_count} transactions/month</div>
                    <div class="explain-body">{freq_note} Transaction frequency is one of the four key predictors.</div>
                </div>
            </div>
            <div class="explain-row">
                <div class="explain-icon">💰</div>
                <div>
                    <div class="explain-title">Monthly Volume -- ETB {total_amount:,}</div>
                    <div class="explain-body">{vol_note}</div>
                </div>
            </div>
            <div class="explain-row" style='margin-bottom:0;'>
                <div class="explain-icon">📊</div>
                <div>
                    <div class="explain-title">Payment Consistency</div>
                    <div class="explain-body">{con_note}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE: BATCH ASSESSMENT
# ============================================================================

elif page == "Batch Assessment":

    st.markdown('<div class="sec-label">Portfolio Assessment</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Score an entire list of Equb applicants at once. Any member you assessed
        on the Assess tab during this session appears automatically at the top.
        The sample group represents typical Arba Minch profiles.
    </div>
    """, unsafe_allow_html=True)

    session_names = {m["name"] for m in st.session_state.custom_members}
    all_members   = [m for m in SAMPLE_MEMBERS if m["name"] not in session_names]
    all_members   = st.session_state.custom_members + all_members

    results = []
    for m in all_members:
        if "prob" in m:
            prob, risk = m["prob"], m["risk"]
        else:
            cat_key = next(
                (k for k in CATEGORY_MAP if m["category"].lower() in k.lower()
                 or k.lower() in m["category"].lower()),
                "Airtime top-up"
            )
            prob = score(model, m["txn_count"], m["avg_amount"], m["total_amount"],
                         m["std_amount"], cat_key, m["provider"], m["channel"])
            risk = get_risk(prob)
        results.append({**m, "prob": prob, "risk": risk})

    results_sorted = sorted(results, key=lambda x: x["prob"])
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in results_sorted: counts[r["risk"]] += 1

    ov1, ov2, ov3, ov4 = st.columns(4)
    for col_item, lbl, val, clr in [
        (ov1, "Total Assessed", len(results_sorted), "#f5f0e8"),
        (ov2, "Approve",        counts["LOW"],        "#22c55e"),
        (ov3, "Review",         counts["MEDIUM"],     "#fbbf24"),
        (ov4, "Escalate",       counts["HIGH"],        "#ef4444"),
    ]:
        col_item.markdown(f"""
        <div class="card" style='margin-top:18px;'>
            <div class="card-accent"></div>
            <div class="card-label">{lbl}</div>
            <div class="card-value" style='color:{clr};'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.custom_members:
        st.markdown(f"""
        <div style='margin-top:10px;font-size:0.8em;color:#22c55e;'>
            {len(st.session_state.custom_members)} member(s) from your session shown first.
        </div>
        """, unsafe_allow_html=True)

    d_col, l_col = st.columns([1, 2], gap="large")
    with d_col:
        st.markdown('<div class="sec-label">Risk Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(donut_chart(counts), width='stretch')

    with l_col:
        st.markdown('<div class="sec-label">Ranked List -- Safest First</div>', unsafe_allow_html=True)
        for r in results_sorted:
            col   = risk_color(r["risk"])
            badge = {"LOW": "Approve", "MEDIUM": "Review", "HIGH": "Escalate"}[r["risk"]]
            row_c = f"batch-{r['risk'].lower()}"
            tag   = " (your session)" if r["name"] in session_names else ""
            st.markdown(f"""
            <div class="batch-row {row_c}">
                <div class="avatar" style='width:36px;height:36px;font-size:0.8em;flex-shrink:0;'>
                    {initials(r["name"])}
                </div>
                <div style='flex:1;'>
                    <div style='font-weight:600;color:#f5f0e8;font-size:0.88em;'>{r["name"]}{tag}</div>
                    <div style='font-size:0.74em;color:#6b7280;'>{r["occupation"]} &nbsp;·&nbsp; {r["city"]}</div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-family:Fraunces,serif;font-weight:600;color:{col};font-size:1.05em;'>
                        {r["prob"]*100:.0f}%
                    </div>
                    <div style='font-size:0.7em;color:{col};'>{badge}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Export Results</div>', unsafe_allow_html=True)
    df_exp = pd.DataFrame([{
        "Name":         r["name"],
        "Occupation":   r["occupation"],
        "City":         r["city"],
        "Risk Score %": round(r["prob"] * 100, 1),
        "Risk Level":   r["risk"],
        "Decision":     {"LOW":"Approve","MEDIUM":"Review","HIGH":"Escalate"}[r["risk"]],
    } for r in results_sorted])
    st.dataframe(df_exp, hide_index=True)
    st.download_button(
        "Download as CSV",
        data=df_exp.to_csv(index=False).encode("utf-8"),
        file_name="equb_risk_assessment.csv",
        mime="text/csv",
    )


# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "About":

    st.markdown('<div class="sec-label">The Problem</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-wrap">
        <div class="explain-row">
            <div class="explain-icon">🇪🇹</div>
            <div>
                <div class="explain-title">Ethiopia's Financial Inclusion Gap</div>
                <div class="explain-body">
                    Over 60% of Ethiopian adults participate in Equb, Idir, or informal lending,
                    yet have zero formal credit history. Organizers rely on personal networks to
                    assess new members, limiting group size and excluding newcomers.
                </div>
            </div>
        </div>
        <div class="explain-row">
            <div class="explain-icon">🤖</div>
            <div>
                <div class="explain-title">The Solution</div>
                <div class="explain-body">
                    EqubScore uses mobile money behavioral data to generate objective risk scores.
                    No bank account required. A trained Random Forest with 58 features produces
                    explainable predictions that organizers can understand and challenge.
                </div>
            </div>
        </div>
        <div class="explain-row">
            <div class="explain-icon">📊</div>
            <div>
                <div class="explain-title">Model Performance</div>
                <div class="explain-body">
                    ROC-AUC: 0.923 &nbsp;·&nbsp; Precision: 0.593 &nbsp;·&nbsp;
                    Recall: 0.848 &nbsp;·&nbsp; F1: 0.698<br>
                    High recall means borderline cases get flagged for review rather than silently approved.
                </div>
            </div>
        </div>
        <div class="explain-row" style='margin-bottom:0;'>
            <div class="explain-icon">⚠️</div>
            <div>
                <div class="explain-title">Honest Disclaimer</div>
                <div class="explain-body">
                    This model was trained on proxy data (Xente/Uganda) to prove technical feasibility.
                    Behavioral patterns like frequency and consistency are universal across mobile money
                    platforms, but validation on Ethiopian data is the critical next step.
                    Partnership with TeleBirr or CBE Birr is planned for Phase 2.
                    EqubScore is a decision support tool, not a replacement for human judgment.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Roadmap</div>', unsafe_allow_html=True)
    for phase, timeline, detail in [
        ("Phase 1", "Months 1-3",
         "Partner with TeleBirr or CBE Birr for Ethiopian data. Retrain model on local patterns."),
        ("Phase 2", "Months 4-6",
         "Pilot with 3-5 real Equb groups in Arba Minch. Measure actual default rate impact."),
        ("Phase 3", "Months 7-12",
         "Scale to 20+ groups. Build Amharic interface. Explore MFI and cooperative bank partnerships."),
    ]:
        st.markdown(f"""
        <div style='display:flex;gap:14px;align-items:flex-start;margin-bottom:14px;'>
            <div style='background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2);
                        border-radius:8px;padding:5px 14px;font-weight:600;font-size:0.78em;
                        color:#22c55e;white-space:nowrap;flex-shrink:0;'>{phase}</div>
            <div>
                <div style='font-size:0.78em;color:#4b5563;'>{timeline}</div>
                <div style='font-size:0.86em;color:#9ca3af;margin-top:2px;'>{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
<div style='display:flex;justify-content:space-between;padding:10px 0 20px;
            font-size:0.74em;color:#374151;'>
    <div>EqubScore -- Alternative Credit Scoring for Ethiopia's Informal Finance Sector</div>
    <div>Hanan Nasir · Software Engineering · Arba Minch University</div>
</div>
""", unsafe_allow_html=True)
