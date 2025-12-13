import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numeric columns to zero mean; keep non-numeric columns unchanged.
    Returns a new DataFrame without modifying the input.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        col_mean = out[col].mean()
        out[col] = out[col] - col_mean
    return out

