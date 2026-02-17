"""Proxy target creation based on RFM clustering."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import RfmClusteringConfig
from src.constants import (PROCESSED_DATA_PATH, PROCESSED_WITH_TARGET_PATH,
                           RAW_DATA_PATH)
from src.data.rfm import compute_rfm


def add_proxy_target(
    raw_csv_path: str | Path = RAW_DATA_PATH,
    processed_csv_path: str | Path = PROCESSED_DATA_PATH,
    output_csv_path: str | Path = PROCESSED_WITH_TARGET_PATH,
    config: RfmClusteringConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = RfmClusteringConfig()

    raw_csv_path = Path(raw_csv_path)
    processed_csv_path = Path(processed_csv_path)
    output_csv_path = Path(output_csv_path)

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_csv_path}")
    if not processed_csv_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_csv_path}")

    raw_df = pd.read_csv(raw_csv_path)
    processed_df = pd.read_csv(processed_csv_path)
    print("[DEBUG] processed_df CustomerIds:", len(processed_df["CustomerId"].unique()))

    # --- NEW LOGIC: match cutoff search script exactly ---
    import re

    m = re.search(r"processed_with_target_(\d{4}-\d{2}-\d{2})_(\d+)d", str(output_csv_path))
    if m:
        cutoff_date = pd.to_datetime(m.group(1))
        outcome_days = int(m.group(2))
        dates = pd.to_datetime(raw_df["TransactionStartTime"])
        outcome_mask = (dates >= cutoff_date) & (
            dates < cutoff_date + pd.Timedelta(days=outcome_days)
        )
        outcome_raw = raw_df[outcome_mask].copy()
        print(
            f"[DEBUG] Outcome window: {cutoff_date.date()} to {(cutoff_date + pd.Timedelta(days=outcome_days)).date()}"
        )
        print(
            "[DEBUG] Unique CustomerIds in outcome window:",
            len(outcome_raw["CustomerId"].unique()),
        )
        # Compute RFM and risk only for outcome window
        rfm = compute_rfm(outcome_raw, snapshot_date=cutoff_date + pd.Timedelta(days=outcome_days))
        print("[DEBUG] Unique CustomerIds after RFM:", len(rfm["CustomerId"].unique()))
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
        rfm["risk_score"] = -rfm_scaled[:, 0] + rfm_scaled[:, 1] - rfm_scaled[:, 2]
        min_high_risk = 2
        quantile = 0.85
        while quantile > 0.0:
            threshold = rfm["risk_score"].quantile(quantile)
            rfm["is_high_risk"] = (rfm["risk_score"] >= threshold).astype(int)
            if rfm["is_high_risk"].sum() >= min_high_risk:
                break
            quantile -= 0.05
        else:
            max_idx = rfm["risk_score"].idxmax()
            rfm["is_high_risk"] = 0
            rfm.loc[max_idx, "is_high_risk"] = 1
        # Only outcome window customers get high risk, all others 0
        proxy = pd.DataFrame({"CustomerId": processed_df["CustomerId"]})
        rfm_map = rfm.set_index("CustomerId")["is_high_risk"].to_dict()
        proxy["is_high_risk"] = proxy["CustomerId"].map(rfm_map).fillna(0).astype(int)
        merged = processed_df.merge(proxy, on="CustomerId", how="left")
        merged["is_high_risk"] = merged["is_high_risk"].fillna(0).astype(int)
    else:
        # fallback: label on all data
        rfm = compute_rfm(raw_df)
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
        rfm["risk_score"] = -rfm_scaled[:, 0] + rfm_scaled[:, 1] - rfm_scaled[:, 2]
        min_high_risk = 2
        quantile = 0.85
        while quantile > 0.0:
            threshold = rfm["risk_score"].quantile(quantile)
            rfm["is_high_risk"] = (rfm["risk_score"] >= threshold).astype(int)
            if rfm["is_high_risk"].sum() >= min_high_risk:
                break
            quantile -= 0.05
        else:
            max_idx = rfm["risk_score"].idxmax()
            rfm["is_high_risk"] = 0
            rfm.loc[max_idx, "is_high_risk"] = 1
        proxy = pd.DataFrame({"CustomerId": processed_df["CustomerId"]})
        rfm_map = rfm.set_index("CustomerId")["is_high_risk"].to_dict()
        proxy["is_high_risk"] = proxy["CustomerId"].map(rfm_map).fillna(0).astype(int)
        merged = processed_df.merge(proxy, on="CustomerId", how="left")
        merged["is_high_risk"] = merged["is_high_risk"].fillna(0).astype(int)

    # Print class counts for verification
    counts = merged["is_high_risk"].value_counts().to_dict()
    print(f"High risk count: {counts.get(1, 0)}, Low risk count: {counts.get(0, 0)}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv_path, index=False)
    return merged
