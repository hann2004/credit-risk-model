from pathlib import Path

import pandas as pd

from src.data_processing import run_and_save


def _write_raw_csv(path: Path) -> None:
    raw = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "Amount": [100, 50, 200],
            "TransactionStartTime": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
        }
    )
    raw.to_csv(path, index=False)


def test_run_and_save_writes_processed_csv(tmp_path: Path):
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.csv"
    _write_raw_csv(raw_path)

    processed = run_and_save(raw_csv_path=raw_path, output_csv_path=output_path)

    assert output_path.exists()
    assert "CustomerId" in processed.columns
    assert len(processed) == 2
