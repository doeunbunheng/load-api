"""Hands-on 1 integration evidence: MLflow scoring server vs FastAPI."""

import json
from pathlib import Path

import requests

MLFLOW_URL = "http://127.0.0.1:5001"
FASTAPI_URL = "http://127.0.0.1:8000"
OUTPUT = Path(__file__).resolve().parent / "04_integration.json"

APPLICANT = {
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


def mlflow_request(applicant: dict) -> requests.Response:
    columns = list(applicant.keys())
    body = {"dataframe_split": {"columns": columns, "data": [[applicant[c] for c in columns]]}}
    return requests.post(f"{MLFLOW_URL}/invocations", json=body, timeout=60)


def main() -> int:
    evidence = {}

    health_mlflow = requests.get(f"{MLFLOW_URL}/health", timeout=30)
    health_api = requests.get(f"{FASTAPI_URL}/health", timeout=30)
    evidence["health"] = {
        "mlflow_scoring_server": {"status_code": health_mlflow.status_code, "body": health_mlflow.text.strip()},
        "fastapi": {"status_code": health_api.status_code, "body": health_api.json()},
    }

    valid = mlflow_request(APPLICANT)
    valid_body = valid.json() if valid.headers.get("content-type", "").startswith("application/json") else valid.text
    evidence["mlflow_valid_request"] = {
        "request": {"dataframe_split": {"columns": list(APPLICANT.keys()), "data": [list(APPLICANT.values())]}},
        "status_code": valid.status_code,
        "response": valid_body,
    }

    rejected_applicant = dict(APPLICANT)
    del rejected_applicant["credit_score"]
    rejected = mlflow_request(rejected_applicant)
    rejected_body = (
        rejected.json() if rejected.headers.get("content-type", "").startswith("application/json") else rejected.text
    )
    evidence["mlflow_signature_rejection"] = {
        "request": {
            "dataframe_split": {
                "columns": list(rejected_applicant.keys()),
                "data": [list(rejected_applicant.values())],
            }
        },
        "status_code": rejected.status_code,
        "response": rejected_body,
    }

    api_response = requests.post(f"{FASTAPI_URL}/predict", json=APPLICANT, timeout=60)
    api_body = api_response.json()
    evidence["fastapi_prediction"] = {
        "request": APPLICANT,
        "status_code": api_response.status_code,
        "response": api_body,
    }

    mlflow_probability = None
    if isinstance(valid_body, dict):
        predictions = valid_body.get("predictions")
        if isinstance(predictions, list) and predictions:
            mlflow_probability = float(predictions[0]["default_probability"])
    fastapi_probability = float(api_body["default_probability"])
    difference = None if mlflow_probability is None else abs(mlflow_probability - fastapi_probability)
    evidence["parity"] = {
        "mlflow_probability": mlflow_probability,
        "fastapi_probability": fastapi_probability,
        "absolute_difference": difference,
        "same_probability": difference is not None and difference <= 1e-12,
    }

    OUTPUT.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print("Health")
    print(f"  MLflow scoring server: {health_mlflow.status_code} {health_mlflow.text.strip()!r}")
    print(f"  FastAPI             : {health_api.status_code} {health_api.json()}")
    print("\nMLflow valid request")
    print(f"  status={valid.status_code} response={valid_body}")
    print("\nMLflow signature rejection (missing credit_score)")
    print(f"  status={rejected.status_code} response={rejected_body}")
    print("\nFastAPI prediction")
    print(f"  status={api_response.status_code} response={api_body}")
    print("\nParity")
    print(f"  MLflow={mlflow_probability!r} FastAPI={fastapi_probability!r} diff={difference!r}")
    print(f"  same_probability={evidence['parity']['same_probability']}")
    print(f"\nEvidence: {OUTPUT}")

    if valid.status_code != 200:
        print("FAIL: valid MLflow request did not return 200")
        return 1
    if rejected.status_code != 400:
        print("FAIL: signature rejection did not return 400")
        return 1
    if not evidence["parity"]["same_probability"]:
        print("FAIL: probabilities differ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
