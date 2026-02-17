from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __name__ == "__main__" and __package__ is None:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.constants import PROCESSED_DATA_PATH, RAW_DATA_PATH
from src.data.features import build_feature_dataset
from src.data.proxy_target import add_proxy_target
from src.data.rfm import compute_rfm, pick_high_risk_cluster
from src.data.temporal import run_temporal_and_save


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


def _compute_rfm(raw_df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    return compute_rfm(raw_df, snapshot_date=snapshot_date)


def _pick_high_risk_cluster(rfm_with_labels: pd.DataFrame) -> int:
    return pick_high_risk_cluster(rfm_with_labels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data processing and proxy target creation")
    parser.add_argument(
        "--with-target",
        action="store_true",
        help="Create processed_with_target.csv including is_high_risk",
    )
    parser.add_argument(
        "--temporal-cutoff",
        type=str,
        default=None,
        help="Cutoff date for temporal split (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--outcome-days",
        type=int,
        default=30,
        help="Outcome window size in days after cutoff",
    )
    args = parser.parse_args()

    if args.temporal_cutoff:
        cutoff_date = pd.to_datetime(args.temporal_cutoff)
        if pd.isna(cutoff_date):
            raise ValueError("Invalid --temporal-cutoff date")
        run_temporal_and_save(
            cutoff_date=cutoff_date,
            outcome_days=args.outcome_days,
        )
        # Always call add_proxy_target with correct output path
        from src.constants import PROCESSED_WITH_TARGET_PATH

        add_proxy_target(
            raw_csv_path=RAW_DATA_PATH,
            processed_csv_path=PROCESSED_DATA_PATH,
            output_csv_path=PROCESSED_WITH_TARGET_PATH,
        )
        print("Temporal processed data written with target.")
        # Print class counts
        import pandas as pd

        df = pd.read_csv("data/processed/processed_with_target.csv")
        counts = df["is_high_risk"].value_counts().to_dict()
        print(f"High risk count: {counts.get(1, 0)}, Low risk count: {counts.get(0, 0)}")
    else:
        processed = run_and_save()
        if args.with_target:
            add_proxy_target()
            # Print class counts
            import pandas as pd

            df = pd.read_csv("data/processed/processed_with_target.csv")
            counts = df["is_high_risk"].value_counts().to_dict()
            print(f"High risk count: {counts.get(1, 0)}, Low risk count: {counts.get(0, 0)}")
        else:
            print("Processed data written to data/processed/processed.csv")
