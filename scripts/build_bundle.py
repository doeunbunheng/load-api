"""Create the deployable model bundle used by the API and the Docker image.

Two sources are supported:

* ``registry`` exports the current ``loan-default@champion`` pipeline from
  MLflow, preserving its version and run ID.
* ``train`` deterministically refits the champion pipeline from
  ``data/loan_default.csv``; this is what CI runs because the runner has no
  MLflow tracking server.

``auto`` tries the registry first and falls back to training.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from loan_features import FEATURE_COLUMNS, TARGET_COLUMN
from train_tracked import (
    ARTIFACTS_DIR,
    BASE_DIR,
    DATA_PATH,
    EXPERIMENT_NAME,
    MODEL_NAME,
    RANDOM_STATE,
    TEST_SIZE,
    _build_pipeline,
    _business_score,
    _metrics,
    _run_specs,
)

CHAMPION_RUN_NAME = "logreg-regularized"
DEFAULT_BUNDLE = BASE_DIR / "model.joblib"
DEFAULT_METADATA = BASE_DIR / "model_metadata.json"
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
EXPECTED_SAMPLE_PROBABILITY = 0.2534030951697195


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _champion_spec() -> dict:
    return next(spec for spec in _run_specs() if spec["run_name"] == CHAMPION_RUN_NAME)


def _train_champion() -> tuple[object, dict, dict]:
    spec = _champion_spec()
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = _build_pipeline(spec)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test.to_numpy(), probabilities)
    metrics["business_score"] = _business_score(metrics)
    info = {
        "source": "retrained",
        "version": "bundled",
        "run_id": "bundled",
        "model_family": spec["family"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    return pipeline, metrics, info


def _registry_champion(tracking_uri: str) -> tuple[object, dict, dict]:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    pyfunc_model = getattr(model, "_model_impl", None)
    pyfunc_model = getattr(pyfunc_model, "python_model", None)
    pipeline = getattr(pyfunc_model, "pipeline", pyfunc_model)
    if not hasattr(pipeline, "predict_proba"):
        raise RuntimeError("Champion artifact does not expose a scikit-learn pipeline")

    run = client.get_run(version.run_id)
    metrics = dict(run.data.metrics)
    metrics.setdefault("business_score", _business_score(metrics))
    info = {
        "source": "mlflow-registry",
        "version": str(version.version),
        "run_id": version.run_id,
        "model_family": run.data.tags.get("model_family", "logistic_regression"),
        "train_rows": int(run.data.params.get("train_rows", 0)),
        "test_rows": int(run.data.params.get("test_rows", 0)),
    }
    return pipeline, metrics, info


def _published_champion() -> dict:
    path = ARTIFACTS_DIR / "selected_versions.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("champion", {})


def build(source: str, tracking_uri: str, output: Path, metadata_path: Path) -> dict:
    published = _published_champion()
    pipeline = None
    metrics = None
    info = None

    if source in {"auto", "registry"}:
        try:
            pipeline, metrics, info = _registry_champion(tracking_uri)
        except Exception as exc:  # noqa: BLE001 - auto mode falls back to training
            if source == "registry":
                raise
            print(f"Registry export unavailable ({exc.__class__.__name__}); training instead")
            pipeline, metrics, info = _train_champion()
    else:
        pipeline, metrics, info = _train_champion()

    if published:
        if info["version"] == "bundled":
            info["version"] = str(published.get("version", "bundled"))
            info["run_id"] = str(published.get("run_id", "bundled"))
        expected = published.get("metrics", {})
        for key in ("recall", "precision", "roc_auc", "pr_auc"):
            if key in expected and abs(expected[key] - metrics[key]) > 1e-9:
                print(
                    f"WARNING: {key} {metrics[key]:.6f} differs from published "
                    f"{expected[key]:.6f}"
                )

    sample_frame = pd.DataFrame([SAMPLE_APPLICANT], columns=FEATURE_COLUMNS)
    sample_probability = float(pipeline.predict_proba(sample_frame)[:, 1][0])
    if abs(sample_probability - EXPECTED_SAMPLE_PROBABILITY) > 1e-6:
        raise RuntimeError(
            "Champion parity check failed: expected "
            f"{EXPECTED_SAMPLE_PROBABILITY}, got {sample_probability}"
        )

    metadata = {
        "model_name": MODEL_NAME,
        "alias": "champion",
        "version": info["version"],
        "run_id": info["run_id"],
        "model_family": info["model_family"],
        "source": info["source"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_NAME,
        "data_path": str(DATA_PATH.relative_to(BASE_DIR)),
        "data_sha256": _sha256(DATA_PATH),
        "train_rows": info["train_rows"],
        "test_rows": info["test_rows"],
        "metrics": {
            key: float(metrics[key])
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "business_score",
            )
            if key in metrics
        },
        "feature_columns": FEATURE_COLUMNS,
        "python_version": platform.python_version(),
        "sklearn_version": __import__("sklearn").__version__,
        "expected_sample": {
            "applicant": SAMPLE_APPLICANT,
            "default_probability": sample_probability,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output, compress=3)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Bundle   : {output} ({output.stat().st_size} bytes)")
    print(f"Metadata : {metadata_path}")
    print(
        f"Source   : {metadata['source']} v{metadata['version']} "
        f"run={metadata['run_id']}"
    )
    print(
        f"Metrics  : recall={metadata['metrics']['recall']:.4f} "
        f"precision={metadata['metrics']['precision']:.4f} "
        f"auc={metadata['metrics']['roc_auc']:.4f}"
    )
    print(f"Sample   : {sample_probability:.16f}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("auto", "registry", "train"),
        default="auto",
        help="Where to obtain the champion pipeline (default: auto).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        help="MLflow tracking URI for --source registry/auto.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    build(args.source, args.tracking_uri, args.output, args.metadata)


if __name__ == "__main__":
    main()
