"""Local bad-release and rollback drill.

The script creates two throwaway copies of the serving code:

1. ``amount-limit`` changes ``MAX_LOAN_AMOUNT_USD`` from 5000 to 500.
2. ``scoring`` inverts the champion probability.

It proves that CI fails on both changes, runs the scoring copy as a local
service, records the wrong prediction for one applicant, stops it, starts the
good release again, and measures how long the correct prediction takes to
return.

This is local evidence for the capstone. The course Render service must still
be used for the deployed bad-release and rollback evidence.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence" / "module6"
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
GOOD_PROBABILITY = 0.2534030951697195
BAD_PROBABILITY = 1.0 - GOOD_PROBABILITY
DRILL_API_KEY = "drill-key-not-a-real-secret"
COPY_FILES = [
    "main.py",
    "loan_features.py",
    "loan_model.py",
    "model.joblib",
    "model_metadata.json",
    "pytest.ini",
]
COPY_DIRS = ["tests"]


def _copy_project(destination: Path) -> None:
    for name in COPY_FILES:
        shutil.copy2(ROOT / name, destination / name)
    for name in COPY_DIRS:
        shutil.copytree(ROOT / name, destination / name)


def _patch(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Patch target not found in {path}: {old}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _test_env(root: Path) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_SOURCE": "bundle",
            "API_KEY": DRILL_API_KEY,
            "GIT_SHA": "drill-test",
            "PREDICTION_LOG_PATH": str(root / "logs" / "predictions.jsonl"),
        }
    )
    env.pop("MLFLOW_TRACKING_URI", None)
    return env


def _run_pytest(root: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_contract.py", "-q"],
        cwd=root,
        env=_test_env(root),
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = (result.stdout + result.stderr).strip()
    return {
        "returncode": result.returncode,
        "failed_as_expected": result.returncode != 0,
        "output_tail": "\n".join(output.splitlines()[-12:]),
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(root: Path, git_sha: str, log_path: Path) -> dict:
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = _test_env(root)
    env["GIT_SHA"] = git_sha
    log_handle = log_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(
                f"Server exited with {process.returncode}:\n{log_path.read_text(encoding='utf-8')}"
            )
        try:
            response = requests.get(f"{base}/health", timeout=3)
            if response.status_code == 200:
                startup = time.perf_counter() - started
                return {
                    "process": process,
                    "log_handle": log_handle,
                    "base": base,
                    "startup_seconds": round(startup, 3),
                    "health": response.json(),
                    "log_path": str(log_path),
                }
        except requests.RequestException:
            pass
        time.sleep(0.5)
    process.terminate()
    log_handle.close()
    raise RuntimeError(f"Server did not become healthy:\n{log_path.read_text(encoding='utf-8')}")


def _stop_server(server: dict) -> None:
    process = server["process"]
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)
    server["log_handle"].close()


def _predict(base: str) -> dict:
    started = time.perf_counter()
    response = requests.post(
        f"{base}/predict",
        json=SAMPLE_APPLICANT,
        headers={"X-API-Key": DRILL_API_KEY},
        timeout=30,
    )
    elapsed = time.perf_counter() - started
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    return {
        "status": response.status_code,
        "elapsed_seconds": round(elapsed, 3),
        "default_probability": body.get("default_probability"),
        "body": body,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=EVIDENCE_DIR / "bad_release_drill.json")
    parser.add_argument("--txt", type=Path, default=EVIDENCE_DIR / "bad_release_drill.txt")
    args = parser.parse_args()

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.txt.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    evidence: dict = {
        "generated_at": generated_at,
        "runtime": "local uvicorn processes",
        "sample_applicant": SAMPLE_APPLICANT,
        "expected_probability": GOOD_PROBABILITY,
    }

    with tempfile.TemporaryDirectory(prefix="loan-api-drill-") as temp:
        temp_root = Path(temp)

        good_root = temp_root / "good"
        good_root.mkdir()
        _copy_project(good_root)

        amount_root = temp_root / "amount-limit"
        amount_root.mkdir()
        _copy_project(amount_root)
        _patch(
            amount_root / "main.py",
            "MAX_LOAN_AMOUNT_USD = 5000",
            "MAX_LOAN_AMOUNT_USD = 500",
        )
        evidence["failing_amount_limit_pr"] = _run_pytest(amount_root)

        scoring_root = temp_root / "scoring"
        scoring_root.mkdir()
        _copy_project(scoring_root)
        _patch(
            scoring_root / "main.py",
            'probability = float(output.iloc[0]["default_probability"])',
            'probability = 1.0 - float(output.iloc[0]["default_probability"])',
        )
        evidence["failing_scoring_pr"] = _run_pytest(scoring_root)

        good_server = _start_server(good_root, "good-release", good_root / "good.log")
        try:
            good_prediction = _predict(good_server["base"])
        finally:
            _stop_server(good_server)
        evidence["good_release"] = {
            "git_sha": good_server["health"]["git_sha"],
            "health": good_server["health"],
            "startup_seconds": good_server["startup_seconds"],
            "prediction": good_prediction,
        }

        bad_started = time.perf_counter()
        bad_server = _start_server(scoring_root, "bad-release", scoring_root / "bad.log")
        try:
            bad_prediction = _predict(bad_server["base"])
        finally:
            _stop_server(bad_server)
        bad_live_seconds = time.perf_counter() - bad_started
        evidence["bad_release"] = {
            "git_sha": bad_server["health"]["git_sha"],
            "health": bad_server["health"],
            "startup_seconds": bad_server["startup_seconds"],
            "prediction": bad_prediction,
            "wrong_probability_detected": (
                bad_prediction["default_probability"] is not None
                and abs(bad_prediction["default_probability"] - BAD_PROBABILITY) < 1e-6
            ),
        }

        rollback_root = temp_root / "rollback"
        rollback_root.mkdir()
        _copy_project(rollback_root)
        rollback_started = time.perf_counter()
        rollback_server = _start_server(rollback_root, "good-release", rollback_root / "rollback.log")
        try:
            restored_prediction = _predict(rollback_server["base"])
        finally:
            _stop_server(rollback_server)
        rollback_seconds = time.perf_counter() - rollback_started
        evidence["rollback"] = {
            "git_sha": rollback_server["health"]["git_sha"],
            "health": rollback_server["health"],
            "startup_seconds": rollback_server["startup_seconds"],
            "prediction": restored_prediction,
            "correct_probability_restored": (
                restored_prediction["default_probability"] is not None
                and abs(restored_prediction["default_probability"] - GOOD_PROBABILITY) < 1e-6
            ),
        }
        evidence["timings"] = {
            "bad_release_start_to_wrong_prediction_seconds": round(bad_live_seconds, 3),
            "rollback_start_to_correct_prediction_seconds": round(rollback_seconds, 3),
        }

    passed = (
        evidence["failing_amount_limit_pr"]["failed_as_expected"]
        and evidence["failing_scoring_pr"]["failed_as_expected"]
        and evidence["bad_release"]["wrong_probability_detected"]
        and evidence["rollback"]["correct_probability_restored"]
        and abs(evidence["good_release"]["prediction"]["default_probability"] - GOOD_PROBABILITY) < 1e-6
    )
    evidence["passed"] = passed

    args.json.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    lines = [
        "Module 6 bad-release and rollback drill (local runtime)",
        f"Generated: {generated_at}",
        "",
        f"Good prediction            : {evidence['good_release']['prediction']['default_probability']}",
        f"Bad prediction             : {evidence['bad_release']['prediction']['default_probability']}",
        f"Restored prediction        : {evidence['rollback']['prediction']['default_probability']}",
        f"Amount-limit PR test       : {'FAILED AS EXPECTED' if evidence['failing_amount_limit_pr']['failed_as_expected'] else 'DID NOT FAIL'}",
        f"Scoring PR test            : {'FAILED AS EXPECTED' if evidence['failing_scoring_pr']['failed_as_expected'] else 'DID NOT FAIL'}",
        f"Bad release start -> wrong : {evidence['timings']['bad_release_start_to_wrong_prediction_seconds']}s",
        f"Rollback start -> correct  : {evidence['timings']['rollback_start_to_correct_prediction_seconds']}s",
        "",
        f"Overall                    : {'PASS' if passed else 'FAIL'}",
        "",
        "Amount-limit test output tail:",
        evidence["failing_amount_limit_pr"]["output_tail"],
        "",
        "Scoring test output tail:",
        evidence["failing_scoring_pr"]["output_tail"],
        "",
        "Note: the deployed Render timings must be captured separately.",
    ]
    args.txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
