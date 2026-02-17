"""Temporal train-test dataset construction to reduce leakage."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import RfmClusteringConfig
from src.constants import (PROCESSED_DATA_PATH, PROCESSED_WITH_TARGET_PATH,
                           RAW_DATA_PATH)
from src.data.features import build_feature_dataset
from src.data.rfm import compute_rfm, pick_high_risk_cluster


def build_temporal_dataset(
    raw_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    outcome_days: int = 30,
    config: RfmClusteringConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = RfmClusteringConfig()

    if outcome_days <= 0:
        raise ValueError("outcome_days must be positive")

    if "TransactionStartTime" not in raw_df.columns:
        raise ValueError("TransactionStartTime column is required")

    df = raw_df.copy()
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], errors="coerce", utc=True
    )
    cutoff_date = pd.to_datetime(cutoff_date, utc=True)

    feature_df = df[df["TransactionStartTime"] < cutoff_date]
    if feature_df.empty:
        raise ValueError("No transactions available before cutoff_date")

    outcome_end = cutoff_date + pd.Timedelta(days=outcome_days)
    outcome_df = df[
        (df["TransactionStartTime"] >= cutoff_date) & (df["TransactionStartTime"] < outcome_end)
    ]

    features = build_feature_dataset(feature_df)

    if outcome_df.empty:
        features["is_high_risk"] = 0
        return features

    rfm = compute_rfm(outcome_df, snapshot_date=outcome_end)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    n_clusters = min(config.n_clusters, len(rfm))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.random_state,
        n_init=config.n_init,
    )
    labels = kmeans.fit_predict(scaled)
    rfm["cluster"] = labels
    high_risk_cluster = pick_high_risk_cluster(rfm)
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    proxy = rfm[["CustomerId", "is_high_risk"]]
    merged = features.merge(proxy, on="CustomerId", how="left")
    merged["is_high_risk"] = merged["is_high_risk"].fillna(0).astype(int)
    return merged


def run_temporal_and_save(
    raw_csv_path: str | Path = RAW_DATA_PATH,
    output_csv_path: str | Path = PROCESSED_WITH_TARGET_PATH,
    feature_output_path: str | Path = PROCESSED_DATA_PATH,
    cutoff_date: pd.Timestamp | None = None,
    outcome_days: int = 30,
    config: RfmClusteringConfig | None = None,
) -> pd.DataFrame:
    if cutoff_date is None:
        raise ValueError("cutoff_date is required")

    raw_csv_path = Path(raw_csv_path)
    output_csv_path = Path(output_csv_path)
    feature_output_path = Path(feature_output_path)

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_csv_path}")

    raw_df = pd.read_csv(raw_csv_path)

    labeled = build_temporal_dataset(
        raw_df=raw_df,
        cutoff_date=cutoff_date,
        outcome_days=outcome_days,
        config=config,
    )

    feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    feature_only = labeled.drop(columns=["is_high_risk"], errors="ignore")
    feature_only.to_csv(feature_output_path, index=False)
    labeled.to_csv(output_csv_path, index=False)
    return labeled
