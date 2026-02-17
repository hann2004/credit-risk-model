"""Project-wide constants for configuration defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_PATH = DATA_DIR / "raw" / "data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed.csv"
PROCESSED_WITH_TARGET_PATH = DATA_DIR / "processed" / "processed_with_target.csv"

DEFAULT_EXPERIMENT_NAME = "credit-risk"
DEFAULT_MODEL_NAME = "credit-risk-best-model"
DEFAULT_MODEL_STAGE = "Production"

# Production model path (fixed, good model only)
PRODUCTION_MODEL_PATH = Path("models/production_model.pkl")

# Model performance metrics (for dashboard display)
PRODUCTION_MODEL_METRICS = {
    "roc_auc": 0.9869,
    "f1": 0.8263,
    "precision": 0.7500,
    "recall": 0.9200,
    "accuracy": 0.9613
}

# For backward compatibility
DEFAULT_MODEL_URI = str(PRODUCTION_MODEL_PATH)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 3
DEFAULT_KMEANS_N_INIT = 10
