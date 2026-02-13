"""Proxy target creation based on RFM clustering."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import RfmClusteringConfig
from src.constants import PROCESSED_DATA_PATH, PROCESSED_WITH_TARGET_PATH, RAW_DATA_PATH
from src.data.rfm import compute_rfm, pick_high_risk_cluster


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

    rfm = compute_rfm(raw_df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    kmeans = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        n_init=config.n_init,
    )
    labels = kmeans.fit_predict(scaled)
    rfm["cluster"] = labels
    high_risk_cluster = pick_high_risk_cluster(rfm)
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    proxy = rfm[["CustomerId", "is_high_risk"]]
    merged = processed_df.merge(proxy, on="CustomerId", how="left")
    merged["is_high_risk"] = merged["is_high_risk"].fillna(0).astype(int)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv_path, index=False)
    return merged
