"""RFM computation utilities."""

from typing import Optional

import pandas as pd


def compute_rfm(
    raw_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    if "TransactionStartTime" not in raw_df.columns:
        raise ValueError("TransactionStartTime column is required for RFM computation")
    if "CustomerId" not in raw_df.columns:
        raise ValueError("CustomerId column is required for RFM computation")

    df = raw_df.copy()
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], errors="coerce"
    )
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max()
        if pd.isna(snapshot_date):
            raise ValueError(
                "Cannot compute snapshot_date from empty TransactionStartTime"
            )
        snapshot_date = snapshot_date.normalize() + pd.Timedelta(days=1)

    if "Value" in df.columns:
        monetary_source = "Value"
    elif "Amount" in df.columns:
        df["_abs_amount"] = df["Amount"].abs()
        monetary_source = "_abs_amount"
    else:
        raise ValueError("No monetary column (Value or Amount) found for RFM")

    rfm = (
        df.groupby("CustomerId")
        .agg(
            recency=("TransactionStartTime", lambda s: (snapshot_date - s.max()).days),
            frequency=(
                ("TransactionId", "count")
                if "TransactionId" in df.columns
                else ("CustomerId", "size")
            ),
            monetary=(monetary_source, "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = rfm["recency"].fillna(rfm["recency"].max())
    rfm["monetary"] = rfm["monetary"].fillna(0)
    return rfm


def pick_high_risk_cluster(rfm_with_labels: pd.DataFrame) -> int:
    agg = rfm_with_labels.groupby("cluster").agg(
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
    )
    agg["score"] = (
        agg["recency_mean"].rank(ascending=False)
        + agg["frequency_mean"].rank(ascending=True)
        + agg["monetary_mean"].rank(ascending=True)
    )
    return int(agg["score"].idxmax())
