"""Verify a deployed Loan API and write evidence.

Checks:
* /health is public and returns 200.
* The first and second health calls are timed so a Render cold start is visible.
* /predict without X-API-Key returns 401 when the service requires a key.
* /predict with the correct key returns the known champion probability.

The API key is read from --api-key or API_KEY and is never printed.
"""

import argparse
import json
import os
import socket
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

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


def _request(method: str, url: str, **kwargs) -> dict:
    started = time.perf_counter()
    try:
        response = requests.request(method, url, timeout=kwargs.pop("timeout", 90), **kwargs)
        elapsed = time.perf_counter() - started
        try:
            body = response.json()
        except ValueError:
            body = response.text[:500]
        return {
            "status": response.status_code,
            "elapsed_seconds": round(elapsed, 3),
            "body": body,
        }
    except requests.RequestException as exc:
        return {
            "status": None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _first_health(base: str, wait_seconds: int) -> tuple[dict, list[dict]]:
    attempts = []
    deadline = time.time() + wait_seconds
    while True:
        result = _request("GET", f"{base}/health", timeout=90)
        attempts.append(result)
        if result["status"] == 200 or time.time() >= deadline:
            return result, attempts
        time.sleep(10)


def _tls_info(base: str) -> dict:
    parsed = urlparse(base)
    if parsed.scheme != "https" or not parsed.hostname:
        return {"https": False}
    try:
        context = ssl.create_default_context()
        with socket.create_connection((parsed.hostname, parsed.port or 443), timeout=10) as raw:
            with context.wrap_socket(raw, server_hostname=parsed.hostname) as secure:
                certificate = secure.getpeercert()
                return {
                    "https": True,
                    "tls_version": secure.version(),
                    "cipher": secure.cipher()[0] if secure.cipher() else None,
                    "subject": certificate.get("subject"),
                    "issuer": certificate.get("issuer"),
                    "not_after": certificate.get("notAfter"),
                }
    except OSError as exc:
        return {"https": True, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("LIVE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--wait", type=int, default=0, help="Seconds to wait for a cold start.")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.url:
        parser.error("--url or LIVE_URL is required")
    base = args.url.rstrip("/")

    started_at = datetime.now(timezone.utc)
    health, health_attempts = _first_health(base, args.wait)
    second_health = _request("GET", f"{base}/health", timeout=30)

    checks: dict[str, dict] = {
        "health_public": {
            "passed": health["status"] == 200,
            "result": health,
        },
        "health_second_call": {
            "result": second_health,
        },
    }
    passed = health["status"] == 200

    health_body = health.get("body") if isinstance(health.get("body"), dict) else {}
    api_key_required = bool(health_body.get("api_key_required"))

    if not args.health_only:
        no_key = _request(
            "POST",
            f"{base}/predict",
            json=SAMPLE_APPLICANT,
            timeout=60,
        )
        checks["predict_without_key"] = {
            "expected_status": 401 if api_key_required else 200,
            "result": no_key,
        }
        if api_key_required:
            passed = passed and no_key["status"] == 401

        if args.api_key:
            with_key = _request(
                "POST",
                f"{base}/predict",
                json=SAMPLE_APPLICANT,
                headers={"X-API-Key": args.api_key},
                timeout=60,
            )
            wrong_key = _request(
                "POST",
                f"{base}/predict",
                json=SAMPLE_APPLICANT,
                headers={"X-API-Key": "deliberately-wrong-key"},
                timeout=60,
            )
            probability = None
            if isinstance(with_key.get("body"), dict):
                probability = with_key["body"].get("default_probability")
            checks["predict_with_key"] = {
                "result": with_key,
                "probability": probability,
            }
            checks["predict_with_wrong_key"] = {"result": wrong_key}
            passed = passed and with_key["status"] == 200
            passed = passed and wrong_key["status"] == 401
            passed = passed and probability is not None
            passed = passed and abs(probability - EXPECTED_PROBABILITY) < 1e-6
        else:
            checks["predict_with_key"] = {"skipped": "API_KEY not provided"}

    evidence = {
        "live_url": base,
        "checked_at": started_at.isoformat(),
        "health": health_body,
        "cold_start": {
            "first_attempt_seconds": health["elapsed_seconds"],
            "second_attempt_seconds": second_health["elapsed_seconds"],
            "attempts": health_attempts,
        },
        "tls": _tls_info(base),
        "checks": checks,
        "passed": passed,
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print(f"Live URL      : {base}")
    print(f"Health        : {health['status']} in {health['elapsed_seconds']}s")
    print(f"Second health : {second_health['status']} in {second_health['elapsed_seconds']}s")
    print(f"Git SHA       : {health_body.get('git_sha')}")
    print(f"Model source  : {health_body.get('model_source')} v{health_body.get('version')}")
    print(f"API key gate  : {'required' if api_key_required else 'not configured'}")
    for name, check in checks.items():
        if "result" in check:
            print(f"{name:<22}: {check['result'].get('status')} in {check['result'].get('elapsed_seconds')}s")
    print(f"Overall       : {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
