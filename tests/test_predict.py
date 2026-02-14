import numpy as np
import pandas as pd
import pytest

from src.predict import align_features, predict_instances, predict_proba


class DummyProbaModel:
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        probs = np.clip(df.sum(axis=1).to_numpy(), 0, 1)
        return np.column_stack([1 - probs, probs])


class DummyPredictModel:
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df.sum(axis=1).to_numpy()


def test_align_features_missing_raises():
    df = pd.DataFrame({"a": [1], "c": [2]})
    with pytest.raises(ValueError, match="Missing features"):
        align_features(df, ["a", "b"])


def test_predict_proba_uses_predict_proba():
    df = pd.DataFrame({"a": [0.2, 0.8]})
    probs = predict_proba(DummyProbaModel(), df)
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predict_instances_aligns_features():
    instances = [{"a": 0.2, "b": 0.1}, {"a": 0.4, "b": 0.3}]
    probs = predict_instances(DummyProbaModel(), instances, feature_names=["b", "a"])
    assert len(probs) == 2


def test_predict_proba_falls_back_to_predict():
    df = pd.DataFrame({"a": [0.3, 0.7]})
    probs = predict_proba(DummyPredictModel(), df)
    assert probs == [0.3, 0.7]
