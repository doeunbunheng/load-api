"""Serving contract tests for the deployed API."""

from tests.test_contract import SAMPLE_APPLICANT

EXPECTED_PATHS = {
    "/health": {"get"},
    "/predict": {"post"},
}


def test_health_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_source"] == "bundle"
    assert body["git_sha"] == "test-sha"
    assert body["api_key_required"] is True


def test_api_starts_without_mlflow_tracking_uri(client):
    body = client.get("/health").json()
    assert body["model_source"] == "bundle"
    assert body["version"] == "1"
    assert body["run_id"] == "d400758e6cbe4ed2a0b36d4b85fe32b8"


def test_predict_requires_api_key(client):
    response = client.post("/predict", json=SAMPLE_APPLICANT)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "X-API-Key"


def test_predict_rejects_wrong_api_key(client):
    response = client.post(
        "/predict",
        json=SAMPLE_APPLICANT,
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_predict_succeeds_with_api_key(client):
    response = client.post(
        "/predict",
        json=SAMPLE_APPLICANT,
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "loan-default"
    assert body["alias"] == "champion"
    assert body["version"] == "1"
    assert body["run_id"] == "d400758e6cbe4ed2a0b36d4b85fe32b8"
    assert 0.0 <= body["default_probability"] <= 1.0


def test_predict_validation_error(client):
    response = client.post(
        "/predict",
        json={**SAMPLE_APPLICANT, "term_months": 9},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 422


def test_openapi_matches_module2_contract(client):
    document = client.get("/openapi.json").json()
    actual = {
        path: {method.lower() for method in operations}
        for path, operations in document["paths"].items()
    }
    assert actual == EXPECTED_PATHS
    assert "X-API-Key" in document["components"]["securitySchemes"]
    assert document["paths"]["/predict"]["post"]["security"] == [{"X-API-Key": []}]


def test_local_development_can_run_without_api_key(open_client):
    health = open_client.get("/health").json()
    assert health["api_key_required"] is False
    response = open_client.post("/predict", json=SAMPLE_APPLICANT)
    assert response.status_code == 200
