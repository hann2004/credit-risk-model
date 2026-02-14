
import pandas as pd
import pytest

from src.data.temporal import build_temporal_dataset


def test_build_temporal_dataset_creates_target():
    raw = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2", "C2"],
            "Amount": [100, 50, 200, 20],
            "TransactionId": ["T1", "T2", "T3", "T4"],
            "TransactionStartTime": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-04T00:00:00Z",
                "2024-01-06T00:00:00Z",
            ],
        }
    )
    cutoff = pd.to_datetime("2024-01-03")
    dataset = build_temporal_dataset(raw, cutoff_date=cutoff, outcome_days=5)

    assert "is_high_risk" in dataset.columns
    assert dataset["is_high_risk"].isin([0, 1]).all()
    assert set(dataset["CustomerId"]) == {"C1"}


def test_build_temporal_dataset_requires_pre_cutoff_data():
    raw = pd.DataFrame(
        {
            "CustomerId": ["C1"],
            "Amount": [100],
            "TransactionId": ["T1"],
            "TransactionStartTime": ["2024-01-10T00:00:00Z"],
        }
    )
    cutoff = pd.to_datetime("2024-01-01")
    with pytest.raises(ValueError, match="before cutoff_date"):
        build_temporal_dataset(raw, cutoff_date=cutoff, outcome_days=3)
