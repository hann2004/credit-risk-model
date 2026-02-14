"""Information Value computation for numeric features."""

import numpy as np
import pandas as pd


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
