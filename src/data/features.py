"""Feature engineering utilities."""

from typing import List

import numpy as np
import pandas as pd


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


def _categorical_modes(
    df: pd.DataFrame, customer_id_col: str, categorical_cols: List[str]
) -> pd.DataFrame:
    cat_modes = (
        df[[customer_id_col] + categorical_cols]
        .groupby(customer_id_col)
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .reset_index()
    )
    return cat_modes


def build_feature_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = _extract_datetime_features(raw_df)
    cust_df = _aggregate_customer(df)

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
        cat_modes = _categorical_modes(df, "CustomerId", categorical_cols)
        cust_df = cust_df.merge(cat_modes, on="CustomerId", how="left")

    for col in ["total_amount", "avg_amount", "txn_count", "std_amount"]:
        cust_df[col] = cust_df[col].fillna(cust_df[col].median())

    num_cols = [
        c
        for c in cust_df.columns
        if c != "CustomerId" and pd.api.types.is_numeric_dtype(cust_df[c])
    ]
    if num_cols:
        means = cust_df[num_cols].mean()
        stds = cust_df[num_cols].std(ddof=0).replace(0, 1)
        cust_df[num_cols] = (cust_df[num_cols] - means) / stds

    if categorical_cols:
        cust_df = pd.get_dummies(cust_df, columns=categorical_cols, dummy_na=True)

    return cust_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if num_cols:
        means = out[num_cols].mean()
        stds = out[num_cols].std(ddof=0).replace(0, 1)
        out[num_cols] = (out[num_cols] - means) / stds
    return out
