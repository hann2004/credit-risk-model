import pandas as pd
from typing import Any

from .data_processing import engineer_features


def predict(model: Any, df: pd.DataFrame):
    df = engineer_features(df)
    return model.predict_proba(df)[:, 1]


if __name__ == "__main__":
    print("Prediction placeholder: load a fitted model and call predict().")
