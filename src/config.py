"""Dataclass-based configuration objects."""

from dataclasses import dataclass
from pathlib import Path

from src.constants import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_KMEANS_N_INIT,
    DEFAULT_MODEL_URI,
    DEFAULT_N_CLUSTERS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    PROCESSED_DATA_PATH,
    PROCESSED_WITH_TARGET_PATH,
    RAW_DATA_PATH,
)


@dataclass(frozen=True)
class DataPaths:
    raw_csv: Path
    processed_csv: Path
    processed_with_target_csv: Path

    @classmethod
    def default(cls) -> "DataPaths":
        return cls(
            raw_csv=RAW_DATA_PATH,
            processed_csv=PROCESSED_DATA_PATH,
            processed_with_target_csv=PROCESSED_WITH_TARGET_PATH,
        )


@dataclass(frozen=True)
class RfmClusteringConfig:
    n_clusters: int = DEFAULT_N_CLUSTERS
    random_state: int = DEFAULT_RANDOM_STATE
    n_init: int = DEFAULT_KMEANS_N_INIT


@dataclass(frozen=True)
class TrainingConfig:
    experiment: str = DEFAULT_EXPERIMENT_NAME
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class ApiConfig:
    model_uri: str = DEFAULT_MODEL_URI
