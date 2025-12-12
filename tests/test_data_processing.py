import pandas as pd
from src.data_processing import engineer_features


def test_engineer_features_scales_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": ["x", "y", "z"]})
    out = engineer_features(df)
    # Numeric columns should be scaled to mean ~0
    assert abs(out["a"].mean()) < 1e-6
    assert abs(out["b"].mean()) < 1e-6
    # Non-numeric column should remain
    assert set(out.columns) == {"a", "b", "c"}
