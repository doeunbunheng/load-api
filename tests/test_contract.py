"""Capstone regression tests.

These tests are deliberately strict about the amount limit and the champion
score. A pull request that changes either value fails CI before it can merge.
"""

import json

import pytest

from main import MAX_LOAN_AMOUNT_USD

SAMPLE_APPLICANT = {
    "amount_usd": 1500,
    "term_months": 12,
    "purpose": "Buy a rice thresher",
    "annual_income_usd": 9000.0,
    "credit_score": 640,
    "existing_debt_usd": 2500.0,
    "employment_years": 4.5,
    "num_late_payments_12m": 1,
    "age": 38,
}
EXPECTED_PROBABILITY = 0.2534030951697195


def _predict(client, **overrides):
    payload = {**SAMPLE_APPLICANT, **overrides}
    return client.post(
        "/predict",
        json=payload,
        headers={"X-API-Key": "test-key"},
    )


def test_loan_amount_limit_is_5000():
    assert MAX_LOAN_AMOUNT_USD == 5000


def test_amount_above_limit_is_rejected(client):
    response = _predict(client, amount_usd=5001)
    assert response.status_code == 422


def test_amount_at_limit_is_accepted(client):
    response = _predict(client, amount_usd=5000)
    assert response.status_code == 200


def test_champion_score_regression(client):
    response = _predict(client)
    assert response.status_code == 200
    probability = response.json()["default_probability"]
    assert probability == pytest.approx(EXPECTED_PROBABILITY, abs=1e-6)


def test_worse_credit_score_increases_default_probability(client):
    good = _predict(client, credit_score=780).json()["default_probability"]
    bad = _predict(client, credit_score=540).json()["default_probability"]
    assert bad > good


def test_prediction_log_redacts_api_key(client, tmp_path):
    response = _predict(client)
    assert response.status_code == 200
    logs = list(tmp_path.glob("predictions-*.jsonl"))
    assert logs
    content = "\n".join(path.read_text(encoding="utf-8") for path in logs)
    assert "test-key" not in content
    records = [json.loads(line) for line in content.splitlines() if line]
    assert records[-1]["git_sha"] == "test-sha"
    assert records[-1]["model_source"] == "bundle"
