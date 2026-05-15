from __future__ import annotations
"""Professional Streamlit dashboard for mobile money credit scoring — modern redesign."""

import sys
from pathlib import Path
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Fingo | Credit Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — Dark luxury with amber accents
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    /* ── Base ──────────────────────────────────── */
    html, body, .stApp {
        background-color: #0d0f14;
        font-family: 'DM Sans', sans-serif;
        color: #e8e4dc;
    }

    /* ── Sidebar ───────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #080a0e;
        border-right: 1px solid #1e2330;
    }
    [data-testid="stSidebar"] * {
        color: #c8c2b8 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9em;
    }

    /* ── Typography ────────────────────────────── */
    h1, h2, h3, h4 {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.03em;
        color: #f0ece3 !important;
    }

    /* ── Wordmark / Masthead ───────────────────── */
    .masthead {
        display: flex;
        align-items: flex-end;
        gap: 14px;
        padding: 0 0 6px 0;
        margin-bottom: 36px;
        border-bottom: 1px solid #1e2330;
    }
    .masthead-logo {
        width: 42px;
        height: 42px;
        background: #e8a020;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.3em;
        color: #0d0f14;
        flex-shrink: 0;
    }
    .masthead-text {
        flex: 1;
    }
    .masthead-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.7em;
        color: #f0ece3;
        line-height: 1;
        letter-spacing: -0.04em;
    }
    .masthead-subtitle {
        font-size: 0.8em;
        color: #6b7280;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 3px;
    }
    .masthead-badge {
        font-size: 0.72em;
        font-weight: 500;
        color: #e8a020;
        background: rgba(232,160,32,0.1);
        border: 1px solid rgba(232,160,32,0.25);
        border-radius: 20px;
        padding: 4px 12px;
        letter-spacing: 0.05em;
    }

    /* ── Section labels ───────────────────────── */
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.68em;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #e8a020;
        margin-bottom: 14px;
        margin-top: 32px;
    }

    /* ── Stat cards ───────────────────────────── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin: 20px 0;
    }
    .stat-card {
        background: #13161e;
        border: 1px solid #1e2330;
        border-radius: 14px;
        padding: 22px 20px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #e8a020, transparent);
    }
    .stat-card:hover {
        border-color: #2e3548;
    }
    .stat-label {
        font-size: 0.72em;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 10px;
    }
    .stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.1em;
        font-weight: 700;
        color: #f0ece3;
        line-height: 1;
    }
    .stat-sub {
        font-size: 0.76em;
        color: #4b5563;
        margin-top: 6px;
    }

    /* ── User profile card ────────────────────── */
    .profile-card {
        background: #13161e;
        border: 1px solid #1e2330;
        border-radius: 14px;
        padding: 22px 24px;
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 24px;
    }
    .profile-avatar {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: linear-gradient(135deg, #e8a020, #c47a10);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.3em;
        color: #0d0f14;
        flex-shrink: 0;
    }
    .profile-name {
        font-family: 'Syne', sans-serif;
        font-size: 1.15em;
        font-weight: 700;
        color: #f0ece3;
    }
    .profile-meta {
        font-size: 0.82em;
        color: #6b7280;
        margin-top: 3px;
    }
    .profile-tag {
        margin-left: auto;
        font-size: 0.72em;
        color: #4ade80;
        background: rgba(74,222,128,0.08);
        border: 1px solid rgba(74,222,128,0.2);
        border-radius: 20px;
        padding: 4px 12px;
        letter-spacing: 0.05em;
        font-weight: 500;
    }

    /* ── Risk badge ───────────────────────────── */
    .risk-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 100px;
        padding: 10px 20px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.9em;
        letter-spacing: 0.05em;
    }
    .risk-pill-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .risk-low { background: rgba(74,222,128,0.1); color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }
    .risk-medium { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
    .risk-high { background: rgba(248,113,113,0.1); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }

    /* ── Recommendation block ─────────────────── */
    .rec-block {
        border-radius: 14px;
        padding: 20px 22px;
        font-size: 0.88em;
        line-height: 1.7;
        font-weight: 400;
        border-left: 3px solid;
    }
    .rec-approve { background: rgba(74,222,128,0.05); border-color: #4ade80; color: #a7f3c0; }
    .rec-review  { background: rgba(251,191,36,0.05); border-color: #fbbf24; color: #fde68a; }
    .rec-reject  { background: rgba(248,113,113,0.05); border-color: #f87171; color: #fecaca; }

    /* ── Divider ──────────────────────────────── */
    hr, .stDivider { border-color: #1e2330 !important; }

    /* ── Comparison metrics ───────────────────── */
    [data-testid="metric-container"] {
        background: #13161e;
        border: 1px solid #1e2330;
        border-radius: 12px;
        padding: 16px !important;
    }
    [data-testid="metric-container"] label {
        color: #6b7280 !important;
        font-size: 0.75em !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f0ece3 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ── Selectbox / inputs ───────────────────── */
    .stSelectbox > div > div {
        background: #13161e !important;
        border: 1px solid #1e2330 !important;
        color: #e8e4dc !important;
        border-radius: 10px !important;
    }
    .stNumberInput input {
        background: #13161e !important;
        border: 1px solid #1e2330 !important;
        color: #e8e4dc !important;
        border-radius: 10px !important;
    }

    /* ── Button ───────────────────────────────── */
    .stButton > button {
        background: #e8a020 !important;
        color: #0d0f14 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95em !important;
        letter-spacing: 0.02em !important;
        padding: 12px 28px !important;
        width: 100% !important;
        transition: opacity 0.15s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* ── Success/Info/Warning ─────────────────── */
    .stAlert {
        background: #13161e !important;
        border: 1px solid #1e2330 !important;
        color: #c8c2b8 !important;
        border-radius: 10px !important;
    }

    /* ── Plotly chart bg ──────────────────────── */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* ── Explanation block ────────────────────── */
    .explain-block {
        background: #13161e;
        border: 1px solid #1e2330;
        border-radius: 14px;
        padding: 22px 24px;
    }
    .explain-row {
        display: flex;
        gap: 14px;
        margin-bottom: 16px;
        align-items: flex-start;
    }
    .explain-icon {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: rgba(232,160,32,0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1em;
        flex-shrink: 0;
    }
    .explain-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.87em;
        font-weight: 700;
        color: #d1cbc0;
        margin-bottom: 3px;
    }
    .explain-body {
        font-size: 0.83em;
        color: #6b7280;
        line-height: 1.6;
    }

    /* ── Input labels ─────────────────────────── */
    .stNumberInput label, .stSelectbox label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8em !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #6b7280 !important;
    }

    /* ── Sidebar radio ────────────────────────── */
    .stRadio > div {
        gap: 4px !important;
    }
    .stRadio label {
        padding: 8px 12px;
        border-radius: 8px;
        transition: background 0.15s;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPERS
# ============================================================================

def get_risk_level(p: float) -> str:
    if p < 0.4:   return "LOW"
    if p < 0.6:   return "MEDIUM"
    return "HIGH"

def get_recommendation(p: float) -> str:
    return {
        "LOW":    "Approve for standard loan. Strong transaction history signals reliable repayment capacity.",
        "MEDIUM": "Approve at reduced amount with quarterly monitoring. Income signals are mixed.",
        "HIGH":   "Refer for manual review. Request collateral or guarantor before disbursement.",
    }[get_risk_level(p)]

def get_risk_color(p: float) -> str:
    if p < 0.4: return "#4ade80"
    if p < 0.6: return "#fbbf24"
    return "#f87171"

def calculate_fallback_score(recency: float, frequency: float, monetary: float) -> float:
    r = min(recency / 30, 1.0)
    f = 1 - min(frequency / 300, 1.0)
    m = 1 - min(monetary / 200_000, 1.0)
    return float(np.clip(r * 0.3 + f * 0.4 + m * 0.3, 0, 1))

def initials(name: str) -> str:
    parts = name.strip().split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else name[:2].upper()

def build_gauge(prob: float) -> go.Figure:
    color = get_risk_color(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": "#f0ece3", "family": "Syne"}},
        title={"text": "DEFAULT PROBABILITY", "font": {"size": 11, "color": "#6b7280", "family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2e3548",
                     "tickfont": {"color": "#4b5563", "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#13161e",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(74,222,128,0.06)"},
                {"range": [40, 60], "color": "rgba(251,191,36,0.06)"},
                {"range": [60,100], "color": "rgba(248,113,113,0.06)"},
            ],
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f0ece3"},
    )
    return fig

# ============================================================================
# TRY to load model (graceful fallback)
# ============================================================================

@st.cache_resource
def _load_model_and_features():
    try:
        from src.predict import load_model, predict_instances
        from src.constants import DEFAULT_MODEL_URI
        model, feature_names = load_model(DEFAULT_MODEL_URI)
        return model, feature_names
    except Exception:
        return None, None

@st.cache_data
def _load_sample_users() -> pd.DataFrame:
    p = Path(__file__).parent.parent / "data" / "sample_mobile_money_users.csv"
    if p.exists():
        return pd.read_csv(p)
    # Synthetic fallback
    np.random.seed(42)
    n = 10
    return pd.DataFrame({
        "user_name":  ["Abebe Girma","Tigist Haile","Dawit Mekonnen","Sara Tadesse",
                       "Yonas Bekele","Hiwot Alemu","Mulugeta Tesfaye","Meron Assefa",
                       "Biruk Kebede","Liya Tsegaye"],
        "occupation": ["Merchant","Teacher","Driver","Farmer","Engineer","Nurse",
                       "Trader","Student","Mechanic","Café Owner"],
        "city":       ["Addis Ababa","Hawassa","Dire Dawa","Bahir Dar","Addis Ababa",
                       "Mekelle","Jimma","Gondar","Adama","Addis Ababa"],
        "recency":    np.random.randint(1, 60, n),
        "frequency":  np.random.randint(20, 350, n),
        "monetary":   np.random.randint(8000, 180000, n),
    })

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 8px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.4em; font-weight: 800;
                    color: #f0ece3; letter-spacing:-0.04em;'>Fingo</div>
        <div style='font-size: 0.72em; letter-spacing: 0.1em; text-transform: uppercase;
                    color: #4b5563; margin-top: 2px;'>Credit Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    demo_mode = st.radio("Assessment mode", ["Sample Users", "Manual Entry"])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78em; color:#4b5563; line-height:1.8;'>
        <div style='color:#e8a020; font-weight:600; margin-bottom:6px; font-family:Syne,sans-serif;'>
            ABOUT FINGO</div>
        Alternative credit scoring for Ethiopia's 40M+ unbanked mobile money users.
        RFM transaction analysis powers real-time risk assessment for microfinance decisions.<br><br>
        <span style='color:#e8a020;'>◈</span> ML-powered • <span style='color:#e8a020;'>◈</span> Ethiopia-focused
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MASTHEAD
# ============================================================================

st.markdown("""
<div class="masthead">
    <div class="masthead-logo">F</div>
    <div class="masthead-text">
        <div class="masthead-title">Fingo Credit Intelligence</div>
        <div class="masthead-subtitle">Mobile Money Risk Assessment · Ethiopia</div>
    </div>
    <div class="masthead-badge">◈ Live Scoring</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# DEMO MODE
# ============================================================================

if demo_mode == "Sample Users":

    st.markdown('<div class="section-label">Select applicant</div>', unsafe_allow_html=True)

    sample_users = _load_sample_users()
    model, feature_names = _load_model_and_features()

    user_options = [
        f"{r['user_name']}  ·  {r['occupation']}, {r['city']}"
        for _, r in sample_users.iterrows()
    ]
    selected_idx = st.selectbox("Select applicant", range(len(user_options)),
                                format_func=lambda x: user_options[x],
                                label_visibility="collapsed")
    u = sample_users.iloc[selected_idx]

    # Profile card
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-avatar">{initials(u['user_name'])}</div>
        <div>
            <div class="profile-name">{u['user_name']}</div>
            <div class="profile-meta">{u['occupation']} · {u['city']}, Ethiopia</div>
        </div>
        <div class="profile-tag">● Active Account</div>
    </div>
    """, unsafe_allow_html=True)

    # RFM metrics
    st.markdown('<div class="section-label">Transaction signals</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Recency</div>
            <div class="stat-value">{int(u['recency'])}</div>
            <div class="stat-sub">days since last txn</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Frequency</div>
            <div class="stat-value">{int(u['frequency'])}</div>
            <div class="stat-sub">monthly transactions</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Monetary</div>
            <div class="stat-value">ETB {int(u['monetary']):,}</div>
            <div class="stat-sub">avg monthly volume</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Score
    if model and feature_names:
        try:
            from src.predict import predict_instances
            inst = pd.DataFrame([{col: u.get(col, 0) for col in feature_names}])
            prob = predict_instances(model, inst, feature_names)[0]
        except Exception:
            prob = calculate_fallback_score(u['recency'], u['frequency'], u['monetary'])
    else:
        prob = calculate_fallback_score(u['recency'], u['frequency'], u['monetary'])

    risk = get_risk_level(prob)
    rec  = get_recommendation(prob)

    st.markdown('<div class="section-label">Risk assessment</div>', unsafe_allow_html=True)

    col_gauge, col_verdict = st.columns([1, 1], gap="large")

    with col_gauge:
        st.plotly_chart(build_gauge(prob), width='stretch')

    with col_verdict:
        st.markdown("<br>", unsafe_allow_html=True)
        dot_color = get_risk_color(prob)
        risk_class = f"risk-{risk.lower()}"
        rec_class  = {"LOW":"rec-approve","MEDIUM":"rec-review","HIGH":"rec-reject"}[risk]
        label      = {"LOW":"Approve","MEDIUM":"Review","HIGH":"Escalate"}[risk]

        st.markdown(f"""
        <div style='margin-bottom:18px;'>
            <div style='font-size:0.72em; color:#6b7280; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:10px;'>Risk Level</div>
            <div class="risk-pill {risk_class}">
                <div class="risk-pill-dot" style='background:{dot_color};'></div>
                {risk} RISK
            </div>
        </div>
        <div>
            <div style='font-size:0.72em; color:#6b7280; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:10px;'>Recommendation</div>
            <div class="rec-block {rec_class}">
                <strong>{label}:</strong> {rec}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Rationale
    st.markdown('<div class="section-label">Assessment rationale</div>', unsafe_allow_html=True)

    icons = {"recency": "🕐", "frequency": "⚡", "monetary": "💰"}

    recency_note = (
        f"Last transaction {int(u['recency'])} days ago — account is actively used."
        if u['recency'] < 14 else
        f"Last transaction {int(u['recency'])} days ago — moderate recency, borderline activity."
        if u['recency'] < 30 else
        f"Last transaction {int(u['recency'])} days ago — elevated dormancy risk."
    )
    freq_note = (
        f"{int(u['frequency'])} monthly transactions signal a high-volume income stream."
        if u['frequency'] > 150 else
        f"{int(u['frequency'])} monthly transactions reflect moderate economic activity."
        if u['frequency'] > 60 else
        f"{int(u['frequency'])} monthly transactions — low engagement pattern detected."
    )
    mon_note = (
        f"ETB {int(u['monetary']):,}/mo indicates strong cash flow for debt servicing."
        if u['monetary'] > 80_000 else
        f"ETB {int(u['monetary']):,}/mo is sufficient for modest loan obligations."
        if u['monetary'] > 30_000 else
        f"ETB {int(u['monetary']):,}/mo — limited cash flow, constrain loan size."
    )

    st.markdown(f"""
    <div class="explain-block">
        <div class="explain-row">
            <div class="explain-icon">🕐</div>
            <div>
                <div class="explain-title">Recency Signal</div>
                <div class="explain-body">{recency_note}</div>
            </div>
        </div>
        <div class="explain-row">
            <div class="explain-icon">⚡</div>
            <div>
                <div class="explain-title">Frequency Signal</div>
                <div class="explain-body">{freq_note}</div>
            </div>
        </div>
        <div class="explain-row" style='margin-bottom:0;'>
            <div class="explain-icon">💰</div>
            <div>
                <div class="explain-title">Monetary Signal</div>
                <div class="explain-body">{mon_note}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Portfolio comparison
    st.markdown('<div class="section-label">Portfolio comparison</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    avg_r = sample_users['recency'].mean()
    avg_f = sample_users['frequency'].mean()
    avg_m = sample_users['monetary'].mean()

    with c1:
        pct = (avg_r - u['recency']) / avg_r * 100
        st.metric("Recency vs Avg", f"{abs(pct):.0f}%",
                  f"{'More recent' if pct > 0 else 'Less recent'}")
    with c2:
        pct = (u['frequency'] - avg_f) / avg_f * 100
        st.metric("Frequency vs Avg", f"{abs(pct):.0f}%",
                  f"{'Above avg' if pct > 0 else 'Below avg'}")
    with c3:
        pct = (u['monetary'] - avg_m) / avg_m * 100
        st.metric("Monetary vs Avg", f"{abs(pct):.0f}%",
                  f"{'Above avg' if pct > 0 else 'Below avg'}")

# ============================================================================
# MANUAL ENTRY MODE
# ============================================================================

else:
    st.markdown('<div class="section-label">New applicant assessment</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#13161e; border:1px solid #1e2330; border-radius:12px;
                padding:16px 20px; margin-bottom:24px; font-size:0.85em; color:#6b7280;'>
        Enter mobile money activity metrics to generate an instant credit risk score
        using the trained RFM model.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("Recency — days since last txn",
                                  min_value=0, max_value=365, value=10)
    with c2:
        frequency = st.number_input("Frequency — monthly transactions",
                                    min_value=0, max_value=500, value=120)
    with c3:
        monetary = st.number_input("Monetary — avg monthly volume (ETB)",
                                   min_value=0, max_value=500_000, value=60_000, step=5_000)

    if st.button("Generate Credit Score ◈"):
        model, feature_names = _load_model_and_features()

        if model and feature_names:
            try:
                from src.predict import predict_instances
                inst = pd.DataFrame([{"recency": recency, "frequency": frequency, "monetary": monetary}])
                prob = predict_instances(model, inst, feature_names)[0]
            except Exception:
                prob = calculate_fallback_score(recency, frequency, monetary)
        else:
            prob = calculate_fallback_score(recency, frequency, monetary)

        risk = get_risk_level(prob)
        rec  = get_recommendation(prob)

        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

        col_g, col_v = st.columns([1, 1], gap="large")
        with col_g:
            st.plotly_chart(build_gauge(prob), width='stretch')

        with col_v:
            st.markdown("<br>", unsafe_allow_html=True)
            dot_color = get_risk_color(prob)
            risk_class = f"risk-{risk.lower()}"
            rec_class  = {"LOW":"rec-approve","MEDIUM":"rec-review","HIGH":"rec-reject"}[risk]
            label      = {"LOW":"Approve","MEDIUM":"Review","HIGH":"Escalate"}[risk]

            st.markdown(f"""
            <div style='margin-bottom:18px;'>
                <div style='font-size:0.72em; color:#6b7280; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:10px;'>Risk Level</div>
                <div class="risk-pill {risk_class}">
                    <div class="risk-pill-dot" style='background:{dot_color};'></div>
                    {risk} RISK
                </div>
            </div>
            <div>
                <div style='font-size:0.72em; color:#6b7280; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:10px;'>Recommendation</div>
                <div class="rec-block {rec_class}">
                    <strong>{label}:</strong> {rec}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            padding: 12px 0 24px; font-size:0.76em; color:#4b5563;">
    <div>Fingo Credit Intelligence · ML-Powered Risk Assessment</div>
    <div>Built for Ethiopia's Financial Inclusion · Hackathon 2025</div>
</div>
""", unsafe_allow_html=True)