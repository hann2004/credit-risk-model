import streamlit as st
import numpy as np
import pandas as pd

def analyze_errors(y_true, y_pred, features):
    """
    Find patterns in model mistakes
    """
    fp_idx = (y_pred == 1) & (y_true == 0)
    fn_idx = (y_pred == 0) & (y_true == 1)
    st.subheader("🔍 Error Patterns")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**False Positives (False Alarms)**")
        if fp_idx.any():
            fp_features = features[fp_idx].mean()
            st.dataframe(fp_features.sort_values(ascending=False).head())
            st.caption("These customers were flagged high-risk but were actually good")
        else:
            st.write("None.")
    with col2:
        st.markdown("**False Negatives (Missed Risks)**")
        if fn_idx.any():
            fn_features = features[fn_idx].mean()
            st.dataframe(fn_features.sort_values(ascending=False).head())
            st.caption("These customers were missed but actually defaulted")
        else:
            st.write("None.")


st.title("Error Analysis Dashboard")
st.write("Upload your predictions and features to analyze model mistakes.")
st.info("""
**Important:** Your CSV must include all model features as columns, matching the template below, plus `y_true` and `y_pred`.
Download and fill in the template to avoid feature mismatch errors.
""")
import os
template_path = os.path.join(os.path.dirname(__file__), "../data/processed/template_features.csv")
if os.path.exists(template_path):
    import streamlit as st
    import io
    with open(template_path, "rb") as f:
        template_bytes = f.read()
    st.download_button(
        label="Download feature template CSV",
        data=template_bytes,
        file_name="template_features.csv",
        mime="text/csv"
    )

uploaded = st.file_uploader("Upload CSV with y_true, y_pred, and features", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    required_cols = set()
    # Try to load feature_names from template
    import csv
    template_path = os.path.join(os.path.dirname(__file__), "../data/processed/template_features.csv")
    if os.path.exists(template_path):
        with open(template_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            required_cols = set(header) - {"CustomerId"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Your CSV is missing required features: {sorted(missing)}. Please use the template.")
    else:
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        features = df.drop(columns=['y_true', 'y_pred'])
        analyze_errors(y_true, y_pred, features)
