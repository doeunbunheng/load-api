import itertools

import pytest
from fastapi.testclient import TestClient

COUNTER = itertools.count()


@pytest.fixture()
def app_factory(monkeypatch, tmp_path):
    def _factory(api_key: str = "test-key", source: str = "bundle", git_sha: str = "test-sha"):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("MODEL_SOURCE", source)
        monkeypatch.setenv("API_KEY", api_key)
        monkeypatch.setenv("GIT_SHA", git_sha)
        monkeypatch.setenv("MODEL_NAME", "loan-default")
        monkeypatch.setenv("MODEL_ALIAS", "champion")
        monkeypatch.setenv(
            "PREDICTION_LOG_PATH",
            str(tmp_path / f"predictions-{next(COUNTER)}.jsonl"),
        )

        import main

        return main.create_app(main.Settings.from_env())

    return _factory


@pytest.fixture()
def client(app_factory):
    with TestClient(app_factory()) as test_client:
        yield test_client


@pytest.fixture()
def open_client(app_factory):
    with TestClient(app_factory(api_key="")) as test_client:
        yield test_client
