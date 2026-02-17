import pytest

@pytest.mark.skip(reason="MLflow artifact logging requires HTTP backend; skipping in CI.")
def test_mlflow_artifact_logging():
    pass
