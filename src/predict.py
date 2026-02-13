"""Reusable inference utilities for batch and API usage."""

from typing import Dict, List, Optional, Sequence, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd

from src.constants import DEFAULT_MODEL_URI


def load_model(model_uri: str = DEFAULT_MODEL_URI) -> Tuple[object, Optional[List[str]]]:
	"""Load a model from MLflow and return model plus feature names if available."""
	model = mlflow.sklearn.load_model(model_uri)
	feature_names: Optional[List[str]] = None
	if hasattr(model, "feature_names_in_"):
		feature_names = list(model.feature_names_in_)
	return model, feature_names


def align_features(
	df: pd.DataFrame, feature_names: Optional[Sequence[str]]
) -> pd.DataFrame:
	if feature_names:
		missing = sorted(set(feature_names) - set(df.columns))
		if missing:
			raise ValueError(f"Missing features: {missing}")
		return df[list(feature_names)]
	return df


def predict_proba(model: object, df: pd.DataFrame) -> List[float]:
	if hasattr(model, "predict_proba"):
		probs = model.predict_proba(df)[:, 1]
	else:
		preds = model.predict(df)
		probs = preds if isinstance(preds, pd.Series) else pd.Series(preds)
	return list(map(float, probs))


def predict_instances(
	model: object,
	instances: Sequence[Dict[str, float]],
	feature_names: Optional[Sequence[str]] = None,
) -> List[float]:
	df = pd.DataFrame(instances)
	df = align_features(df, feature_names)
	return predict_proba(model, df)


def predict_from_uri(
	instances: Sequence[Dict[str, float]],
	model_uri: str = DEFAULT_MODEL_URI,
) -> List[float]:
	model, feature_names = load_model(model_uri)
	return predict_instances(model, instances, feature_names)
