from pathlib import Path

import pandas as pd

from src.config import RfmClusteringConfig
from src.data.proxy_target import add_proxy_target


def _write_raw_csv(path: Path) -> None:
    raw = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2", "C2"],
            "TransactionId": ["T1", "T2", "T3", "T4"],
            "Amount": [100, -20, 150, 30],
            "TransactionStartTime": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
                "2024-01-04T00:00:00Z",
            ],
        }
    )
    raw.to_csv(path, index=False)


def _write_processed_csv(path: Path) -> None:
    processed = pd.DataFrame(
        {
            "CustomerId": ["C1", "C2"],
            "total_amount": [80, 180],
        }
    )
    processed.to_csv(path, index=False)


def test_add_proxy_target_creates_label(tmp_path: Path):
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    output_path = tmp_path / "processed_with_target.csv"

    _write_raw_csv(raw_path)
    _write_processed_csv(processed_path)

    config = RfmClusteringConfig(n_clusters=2, random_state=1, n_init=10)
    merged = add_proxy_target(
        raw_csv_path=raw_path,
        processed_csv_path=processed_path,
        output_csv_path=output_path,
        config=config,
    )

    assert output_path.exists()
    assert "is_high_risk" in merged.columns
    assert set(merged["is_high_risk"].unique()) <= {0, 1}
    assert len(merged) == 2
