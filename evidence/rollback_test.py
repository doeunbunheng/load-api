"""Step 6 evidence: move the champion alias, restart FastAPI, verify /health,
then move it back and verify again."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests
from mlflow import MlflowClient

BASE_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = Path(__file__).resolve().parent
SERVER_DIR = EVIDENCE_DIR / "servers"
PID_FILE = SERVER_DIR / "fastapi.pid"
LOG_FILE = SERVER_DIR / "fastapi.rollback.log"
OUTPUT = EVIDENCE_DIR / "06_rollback.json"
TRACKING_URI = "http://127.0.0.1:5000"
API_URL = "http://127.0.0.1:8000"
MODEL_NAME = "loan-default"
PORT = 8000

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


def listening_pids(port: int) -> list[int]:
    pids = []
    for connection in psutil.net_connections(kind="inet"):
        if connection.laddr and connection.laddr.port == port and connection.status == psutil.CONN_LISTEN:
            if connection.pid:
                pids.append(connection.pid)
    return sorted(set(pids))


def stop_api() -> list[int]:
    pids = set(listening_pids(PORT))
    if PID_FILE.exists():
        try:
            pids.add(int(PID_FILE.read_text().strip()))
        except ValueError:
            pass
    stopped = []
    for pid in pids:
        try:
            process = psutil.Process(pid)
            for child in process.children(recursive=True):
                child.terminate()
            process.terminate()
            process.wait(timeout=10)
            stopped.append(pid)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied):
            try:
                process.kill()
                stopped.append(pid)
            except Exception:
                pass
    for _ in range(30):
        if not listening_pids(PORT):
            break
        time.sleep(0.5)
    return stopped


def start_api() -> int:
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = TRACKING_URI
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with LOG_FILE.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(PORT)],
            cwd=BASE_DIR,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=creationflags,
            close_fds=True,
        )
    PID_FILE.write_text(str(process.pid))
    for _ in range(60):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                return process.pid
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("FastAPI did not become healthy after restart")


def health() -> dict:
    return requests.get(f"{API_URL}/health", timeout=10).json()


def predict() -> dict:
    return requests.post(f"{API_URL}/predict", json=APPLICANT, timeout=30).json()


def aliases(client: MlflowClient) -> dict:
    result = {}
    for alias in ("champion", "challenger"):
        version = client.get_model_version_by_alias(MODEL_NAME, alias)
        result[alias] = {"version": str(version.version), "run_id": version.run_id}
    return result


def snapshot(client: MlflowClient, label: str, restart: bool = False) -> dict:
    if restart:
        stopped = stop_api()
        pid = start_api()
    else:
        stopped, pid = [], None
    record = {
        "label": label,
        "aliases": aliases(client),
        "health": health(),
        "prediction": predict(),
        "restarted": restart,
        "stopped_pids": stopped,
        "new_pid": pid,
    }
    print(f"\n[{label}]")
    print(f"  aliases   : {json.dumps(record['aliases'])}")
    print(f"  /health   : {json.dumps(record['health'])}")
    print(f"  /predict  : {json.dumps(record['prediction'])}")
    if restart:
        print(f"  restarted : stopped={stopped} new_pid={pid}")
    return record


def main() -> int:
    SERVER_DIR.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(TRACKING_URI)
    current = aliases(client)
    champion_version = current["champion"]["version"]
    challenger_version = current["challenger"]["version"]
    if champion_version == challenger_version:
        raise RuntimeError("Champion and challenger aliases point to the same version")

    records = [snapshot(client, "before rollback (champion alias)")]

    client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version)
    records.append(snapshot(client, f"after moving champion -> v{challenger_version}", restart=True))

    client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)
    records.append(snapshot(client, f"after moving champion back -> v{champion_version}", restart=True))

    evidence = {
        "model_name": MODEL_NAME,
        "champion_version_before": champion_version,
        "challenger_version_before": challenger_version,
        "steps": records,
        "verified_forward": records[1]["health"]["version"] == challenger_version
        and records[1]["health"]["run_id"] == current["challenger"]["run_id"],
        "verified_back": records[2]["health"]["version"] == champion_version
        and records[2]["health"]["run_id"] == current["champion"]["run_id"],
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"\nForward rollback verified: {evidence['verified_forward']}")
    print(f"Restored champion verified: {evidence['verified_back']}")
    print(f"Evidence: {OUTPUT}")
    return 0 if evidence["verified_forward"] and evidence["verified_back"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
