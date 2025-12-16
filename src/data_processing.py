import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _extract_datetime_features(
    df: pd.DataFrame, datetime_col: str = "TransactionStartTime"
) -> pd.DataFrame:
    if datetime_col in df.columns:
        dt = pd.to_datetime(df[datetime_col], errors="coerce")
        df = df.copy()
        df["transaction_hour"] = dt.dt.hour
        df["transaction_day"] = dt.dt.day
        df["transaction_month"] = dt.dt.month
        df["transaction_year"] = dt.dt.year
    return df


def _aggregate_customer(
    df: pd.DataFrame, customer_id_col: str = "CustomerId", amount_col: str = "Amount"
) -> pd.DataFrame:
    if customer_id_col not in df.columns:
        raise ValueError(f"Missing '{customer_id_col}' column for aggregation")
    if amount_col not in df.columns:
        raise ValueError(f"Missing '{amount_col}' column for aggregation")

    grouped = (
        df.groupby(customer_id_col)
        .agg(
            total_amount=(amount_col, "sum"),
            avg_amount=(amount_col, "mean"),
            txn_count=(amount_col, "count"),
            std_amount=(amount_col, "std"),
        )
        .reset_index()
    )
    grouped["std_amount"] = grouped["std_amount"].fillna(0.0)
    return grouped


def build_feature_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = _extract_datetime_features(raw_df)
    cust_df = _aggregate_customer(df)

    # Per-customer categorical modes (optional)
    categorical_cols = [
        c
        for c in [
            "CurrencyCode",
            "CountryCode",
            "ProviderId",
            "ProductId",
            "ProductCategory",
            "ChannelId",
            "PricingStrategy",
        ]
        if c in df.columns
    ]
    if categorical_cols:
        cat_modes = (
            df[["CustomerId"] + categorical_cols]
            .groupby("CustomerId")
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
            .reset_index()
        )
        cust_df = cust_df.merge(cat_modes, on="CustomerId", how="left")

    # Impute simple
    for col in ["total_amount", "avg_amount", "txn_count", "std_amount"]:
        cust_df[col] = cust_df[col].fillna(cust_df[col].median())

    # Standardize numeric
    num_cols = [
        c
        for c in cust_df.columns
        if c != "CustomerId" and pd.api.types.is_numeric_dtype(cust_df[c])
    ]
    if num_cols:
        means = cust_df[num_cols].mean()
        stds = cust_df[num_cols].std(ddof=0).replace(0, 1)
        cust_df[num_cols] = (cust_df[num_cols] - means) / stds

    # One-hot encode categoricals
    if categorical_cols:
        cust_df = pd.get_dummies(cust_df, columns=categorical_cols, dummy_na=True)

    return cust_df


def run_and_save(
    raw_csv_path: str = os.path.join("data", "raw", "data.csv"),
    output_csv_path: str = os.path.join("data", "processed", "processed.csv"),
) -> pd.DataFrame:
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw data not found at {raw_csv_path}")
    raw_df = pd.read_csv(raw_csv_path)
    processed_df = build_feature_dataset(raw_df)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    processed_df.to_csv(output_csv_path, index=False)
    return processed_df


def compute_information_value(
    df: pd.DataFrame, target_col: str, bins: int = 10
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError("Target column not found in DataFrame")
    if set(df[target_col].dropna().unique()) - {0, 1}:
        raise ValueError("Target must be binary {0,1}")
    target = df[target_col]
    features = [
        c
        for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    def _iv_for_feature(x: pd.Series) -> float:
        if x.nunique() <= 1:
            return 0.0
        try:
            qcuts = pd.qcut(x.rank(method="first"), q=bins, duplicates="drop")
        except Exception:
            return 0.0
        tmp = pd.DataFrame({"bin": qcuts, "target": target})
        grouped = tmp.groupby("bin").agg(
            good=("target", lambda s: (s == 0).sum()),
            bad=("target", lambda s: (s == 1).sum()),
        )
        grouped["good_prop"] = grouped["good"] / grouped["good"].sum()
        grouped["bad_prop"] = grouped["bad"] / grouped["bad"].sum()
        eps = 1e-9
        woe = np.log((grouped["good_prop"] + eps) / (grouped["bad_prop"] + eps))
        iv = ((grouped["good_prop"] - grouped["bad_prop"]) * woe).sum()
        return float(iv)

    rows = [{"feature": f, "iv": _iv_for_feature(df[f])} for f in features]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def _compute_rfm(
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


def _pick_high_risk_cluster(rfm_with_labels: pd.DataFrame) -> int:
    agg = rfm_with_labels.groupby("cluster").agg(
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
    )
    # High recency (stale), low frequency, low monetary => higher score
    agg["score"] = (
        agg["recency_mean"].rank(ascending=False)
        + agg["frequency_mean"].rank(ascending=True)
        + agg["monetary_mean"].rank(ascending=True)
    )
    return int(agg["score"].idxmax())


def add_proxy_target(
    raw_csv_path: str = os.path.join("data", "raw", "data.csv"),
    processed_csv_path: str = os.path.join("data", "processed", "processed.csv"),
    output_csv_path: str = os.path.join(
        "data", "processed", "processed_with_target.csv"
    ),
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw data not found at {raw_csv_path}")
    if not os.path.exists(processed_csv_path):
        raise FileNotFoundError(f"Processed data not found at {processed_csv_path}")

    raw_df = pd.read_csv(raw_csv_path)
    processed_df = pd.read_csv(processed_csv_path)

    rfm = _compute_rfm(raw_df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled)
    rfm["cluster"] = labels
    high_risk_cluster = _pick_high_risk_cluster(rfm)
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    proxy = rfm[["CustomerId", "is_high_risk"]]
    merged = processed_df.merge(proxy, on="CustomerId", how="left")
    merged["is_high_risk"] = merged["is_high_risk"].fillna(0).astype(int)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    merged.to_csv(output_csv_path, index=False)
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data processing and proxy target creation"
    )
    parser.add_argument(
        "--with-target",
        action="store_true",
        help="Create processed_with_target.csv including is_high_risk",
    )
    args = parser.parse_args()

    processed = run_and_save()
    if args.with_target:
        add_proxy_target()
    else:
        print("Processed data written to data/processed/processed.csv")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if num_cols:
        means = out[num_cols].mean()
        stds = out[num_cols].std(ddof=0).replace(0, 1)
        out[num_cols] = (out[num_cols] - means) / stds
    return out