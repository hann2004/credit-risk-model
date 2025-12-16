import pandas as pd
from src.data_processing import engineer_features, compute_information_value, _compute_rfm


def test_engineer_features_scales_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": ["x", "y", "z"]})
    out = engineer_features(df)
    # Numeric columns should be scaled to mean ~0
    assert abs(out["a"].mean()) < 1e-6
    assert abs(out["b"].mean()) < 1e-6
    # Non-numeric column should remain
    assert set(out.columns) == {"a", "b", "c"}


def test_compute_information_value_outputs_sorted_and_columns():
    df = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1, 0, 1],
            "f2": [1, 1, 1, 0, 0, 0],
            "is_high_risk": [0, 1, 0, 1, 0, 1],
        }
    )
    iv_df = compute_information_value(df, target_col="is_high_risk", bins=2)
    assert list(iv_df.columns) == ["feature", "iv"]
    assert iv_df.iloc[0]["iv"] >= iv_df.iloc[-1]["iv"]


def test_compute_rfm_produces_expected_columns():
    raw = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "TransactionId": ["T1", "T2", "T3"],
            "Amount": [100, -50, 200],
            "TransactionStartTime": [
                "2024-01-01T00:00:00Z",
                "2024-01-05T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
        }
    )
    rfm = _compute_rfm(raw)
    assert set(["CustomerId", "recency", "frequency", "monetary"]).issubset(rfm.columns)
    assert len(rfm) == 2
    assert (rfm["recency"] >= 0).all()
