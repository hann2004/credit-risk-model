from pathlib import Path

import pandas as pd

from src.constants import PROCESSED_DATA_PATH, RAW_DATA_PATH
from src.data.features import build_feature_dataset, engineer_features
from src.data.iv import compute_information_value
from src.data.proxy_target import add_proxy_target
from src.data.rfm import compute_rfm, pick_high_risk_cluster


def run_and_save(
    raw_csv_path: str | Path = RAW_DATA_PATH,
    output_csv_path: str | Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    raw_csv_path = Path(raw_csv_path)
    output_csv_path = Path(output_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_csv_path}")
    raw_df = pd.read_csv(raw_csv_path)
    processed_df = build_feature_dataset(raw_df)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_csv_path, index=False)
    return processed_df


def _compute_rfm(
    raw_df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    return compute_rfm(raw_df, snapshot_date=snapshot_date)


def _pick_high_risk_cluster(rfm_with_labels: pd.DataFrame) -> int:
    return pick_high_risk_cluster(rfm_with_labels)


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
