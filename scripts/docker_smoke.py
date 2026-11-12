"""Pull/run the production image and verify /health and the API-key gate."""

import argparse
import json
import secrets
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

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


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
        timeout=600,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="ghcr.io/doeunbunheng/load-api:latest")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    api_key = secrets.token_urlsafe(24)
    name = f"loan-api-smoke-{uuid.uuid4().hex[:8]}"
    evidence = {
        "image": args.image,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    passed = False
    logs = ""
    try:
        if not args.skip_pull:
            print(f"Pulling {args.image}")
            _docker("pull", args.image)
        inspect = json.loads(_docker("image", "inspect", args.image).stdout)[0]
        evidence["image_id"] = inspect.get("Id")
        evidence["repo_digests"] = inspect.get("RepoDigests", [])

        _docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{args.port}:8000",
            "-e",
            f"API_KEY={api_key}",
            "-e",
            "MODEL_SOURCE=bundle",
            "-e",
            f"GIT_SHA=smoke-{name}",
            args.image,
        )
        base = f"http://127.0.0.1:{args.port}"
        deadline = time.time() + 90
        health = None
        started = time.perf_counter()
        while time.time() < deadline:
            try:
                response = requests.get(f"{base}/health", timeout=5)
                if response.status_code == 200:
                    health = response.json()
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        startup_seconds = round(time.perf_counter() - started, 3)
        if health is None:
            raise RuntimeError("Container did not become healthy")

        without_key = requests.post(f"{base}/predict", json=SAMPLE_APPLICANT, timeout=30)
        with_key = requests.post(
            f"{base}/predict",
            json=SAMPLE_APPLICANT,
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        evidence.update(
            {
                "health": health,
                "startup_seconds": startup_seconds,
                "predict_without_key_status": without_key.status_code,
                "predict_with_key_status": with_key.status_code,
                "prediction": with_key.json() if with_key.ok else with_key.text,
            }
        )
        passed = (
            health.get("status") == "ok"
            and without_key.status_code == 401
            and with_key.status_code == 200
        )
    except Exception as exc:  # noqa: BLE001 - report Docker failures as evidence
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            logs = _docker("logs", name, check=False).stdout
        except Exception:  # noqa: BLE001
            logs = ""
        _docker("rm", "-f", name, check=False)

    evidence["container_logs"] = "\n".join(logs.splitlines()[-30:])
    evidence["passed"] = passed
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print(json.dumps(evidence, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
