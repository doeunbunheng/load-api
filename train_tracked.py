"""Train at least six loan-default models, track them in MLflow, and register
the two best candidates under the `loan-default` registry name.

Run against a live tracking server:
    python train_tracked.py --reset-registry

The script logs every run with parameters, metrics, a model signature and an
input example. The top two candidates by the documented business score are
registered, described, and tagged as champion / challenger.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from loan_model import LoanDefaultProbabilityModel

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_default.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_NAME = "loan-default"
EXPERIMENT_NAME = "loan-default-experiments"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
RANDOM_STATE = 42
TEST_SIZE = 0.20
BATCH_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_specs() -> list[dict]:
    return [
        {
            "run_name": "logreg-baseline",
            "family": "logistic_regression",
            "scale": True,
            "estimator": LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE),
            "params": {"C": 1.0, "class_weight": "none", "solver": "lbfgs", "max_iter": 3000},
        },
        {
            "run_name": "logreg-regularized",
            "family": "logistic_regression",
            "scale": True,
            "estimator": LogisticRegression(
                C=0.2, max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE
            ),
            "params": {"C": 0.2, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 3000},
        },
        {
            "run_name": "logreg-balanced",
            "family": "logistic_regression",
            "scale": True,
            "estimator": LogisticRegression(
                C=1.0, max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE
            ),
            "params": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 3000},
        },
        {
            "run_name": "random-forest-shallow",
            "family": "random_forest",
            "scale": False,
            "estimator": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 6,
                "min_samples_leaf": 10,
                "class_weight": "balanced_subsample",
            },
        },
        {
            "run_name": "random-forest-deep",
            "family": "random_forest",
            "scale": False,
            "estimator": RandomForestClassifier(
                n_estimators=500,
                max_depth=14,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": 500,
                "max_depth": 14,
                "min_samples_leaf": 3,
                "class_weight": "balanced_subsample",
            },
        },
        {
            "run_name": "gradient-boosting",
            "family": "gradient_boosting",
            "scale": False,
            "estimator": GradientBoostingClassifier(
                n_estimators=250, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
            ),
            "params": {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 3},
        },
        {
            "run_name": "hist-gradient-boosting",
            "family": "hist_gradient_boosting",
            "scale": False,
            "estimator": HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "params": {
                "max_iter": 300,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
                "class_weight": "balanced",
            },
        },
        {
            "run_name": "extra-trees",
            "family": "extra_trees",
            "scale": False,
            "estimator": ExtraTreesClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=4,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": 400,
                "max_depth": 12,
                "min_samples_leaf": 4,
                "class_weight": "balanced",
            },
        },
    ]


def _build_pipeline(spec: dict) -> Pipeline:
    numeric_transformer = StandardScaler() if spec["scale"] else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", spec["estimator"])])


def _metrics(y_true, probabilities, threshold: float = 0.5) -> dict:
    labels = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def _business_score(metrics: dict) -> float:
    """Recall-heavy score for a default model.

    A missed defaulter (false negative) is assumed to cost about five times a
    declined good applicant (false positive), so recall carries 50% weight,
    precision 30%, and ranking quality (AUC) 20%.
    """
    return 0.5 * metrics["recall"] + 0.3 * metrics["precision"] + 0.2 * metrics["roc_auc"]


def _reset_registry(client: MlflowClient) -> None:
    try:
        client.delete_registered_model(MODEL_NAME)
        print(f"Deleted existing registered model '{MODEL_NAME}'.")
    except Exception as exc:  # noqa: BLE001 - first run has nothing to delete
        print(f"No registered model to delete ({exc.__class__.__name__}).")


def _register_candidate(client: MlflowClient, candidate: dict, alias: str) -> dict:
    registered = mlflow.register_model(candidate["model_uri"], MODEL_NAME)
    version = registered.version
    metrics = candidate["metrics"]
    description = (
        f"{alias.capitalize()}: {candidate['run_name']} ({candidate['family']}). "
        f"Recall={metrics['recall']:.4f}, precision={metrics['precision']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}. "
        f"Trained on {candidate['train_rows']} applications (batch {BATCH_ID}); "
        f"MLflow run {candidate['run_id']}."
    )
    client.update_model_version(MODEL_NAME, version, description=description)
    client.set_model_version_tag(MODEL_NAME, version, "role", alias)
    client.set_model_version_tag(MODEL_NAME, version, "run_id", candidate["run_id"])
    client.set_model_version_tag(MODEL_NAME, version, "model_family", candidate["family"])
    client.set_model_version_tag(MODEL_NAME, version, "business_score", f"{candidate['business_score']:.4f}")
    client.set_model_version_tag(MODEL_NAME, version, "batch_id", BATCH_ID)
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    return {
        "alias": alias,
        "version": str(version),
        "run_id": candidate["run_id"],
        "run_name": candidate["run_name"],
        "family": candidate["family"],
        "metrics": metrics,
        "business_score": candidate["business_score"],
        "description": description,
        "model_uri": f"models:/{MODEL_NAME}@{alias}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-registry",
        action="store_true",
        help="Delete the existing loan-default registered model before registering this batch.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=TRACKING_URI,
        help=f"MLflow tracking URI (default: {TRACKING_URI}).",
    )
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    if args.reset_registry:
        _reset_registry(client)

    data = pd.read_csv(DATA_PATH)
    data_hash = _sha256(DATA_PATH)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Training rows={len(X_train)} test rows={len(X_test)} default rate={y_train.mean():.3f}")
    print(f"Tracking URI={args.tracking_uri} experiment={EXPERIMENT_NAME} batch={BATCH_ID}")

    candidates = []
    for index, spec in enumerate(_run_specs(), start=1):
        pipeline = _build_pipeline(spec)
        with mlflow.start_run(run_name=spec["run_name"]) as run:
            mlflow.set_tags(
                {
                    "model_family": spec["family"],
                    "lab": "week17-mlflow-registry",
                    "batch_id": BATCH_ID,
                    "data_sha256": data_hash,
                }
            )
            mlflow.log_params(
                {
                    **spec["params"],
                    "model_family": spec["family"],
                    "scale_numeric": spec["scale"],
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                }
            )
            mlflow.log_input(
                mlflow.data.from_pandas(data, source=str(DATA_PATH), name="loan-default"),
                context="training",
            )

            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics = _metrics(y_test.to_numpy(), probabilities)
            business_score = _business_score(metrics)
            mlflow.log_metrics({**metrics, "business_score": business_score})

            wrapped = LoanDefaultProbabilityModel(pipeline)
            signature = infer_signature(X_train, wrapped.predict(None, X_train))
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=wrapped,
                code_paths=[str(BASE_DIR / "loan_model.py")],
                signature=signature,
                input_example=X_train.head(3),
                pip_requirements=["mlflow", "scikit-learn", "pandas", "numpy"],
            )
            mlflow.log_dict({"metrics": metrics, "business_score": business_score}, "metrics.json")

            candidate = {
                "run_name": spec["run_name"],
                "family": spec["family"],
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "metrics": metrics,
                "business_score": business_score,
            }
            candidates.append(candidate)
            print(
                f"[{index}/{len(_run_specs())}] {spec['run_name']:<24} "
                f"recall={metrics['recall']:.4f} precision={metrics['precision']:.4f} "
                f"auc={metrics['roc_auc']:.4f} score={business_score:.4f} run={run.info.run_id}"
            )

    leaderboard = pd.DataFrame(
        [
            {
                "run_name": c["run_name"],
                "model_family": c["family"],
                "run_id": c["run_id"],
                "recall": c["metrics"]["recall"],
                "precision": c["metrics"]["precision"],
                "f1": c["metrics"]["f1"],
                "roc_auc": c["metrics"]["roc_auc"],
                "pr_auc": c["metrics"]["pr_auc"],
                "accuracy": c["metrics"]["accuracy"],
                "business_score": c["business_score"],
            }
            for c in candidates
        ]
    ).sort_values("business_score", ascending=False, ignore_index=True)
    leaderboard["rank"] = leaderboard.index + 1
    leaderboard.to_csv(ARTIFACTS_DIR / "leaderboard.csv", index=False)
    (ARTIFACTS_DIR / "leaderboard.json").write_text(leaderboard.to_json(orient="records", indent=2), encoding="utf-8")
    print("\nLeaderboard")
    print(leaderboard.to_string(index=False))

    family_best = leaderboard.drop_duplicates(subset="model_family", keep="first")
    eligible = family_best[(family_best["precision"] >= 0.40) & (family_best["roc_auc"] >= 0.75)]
    if len(eligible) < 2:
        eligible = family_best
    selected_names = eligible.head(2)["run_name"].tolist()
    selected = [next(c for c in candidates if c["run_name"] == name) for name in selected_names]
    champion, challenger = selected[0], selected[1]

    champion_info = _register_candidate(client, champion, "champion")
    challenger_info = _register_candidate(client, challenger, "challenger")

    X_train.to_csv(ARTIFACTS_DIR / "reference_train.csv", index=False)
    run_links = {
        c["run_name"]: f"{args.tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{c['run_id']}"
        for c in candidates
    }
    summary = {
        "batch_id": BATCH_ID,
        "tracking_uri": args.tracking_uri,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": experiment.experiment_id,
        "registered_model": MODEL_NAME,
        "selection_rule": (
            "Best run per model family by business_score = 0.5*recall + 0.3*precision + 0.2*roc_auc; "
            "the top two families become champion and challenger."
        ),
        "champion": champion_info,
        "challenger": challenger_info,
        "run_links": run_links,
    }
    (ARTIFACTS_DIR / "selected_versions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nRegistered candidates")
    print(f"  champion  -> {MODEL_NAME} v{champion_info['version']} ({champion_info['run_name']})")
    print(f"  challenger-> {MODEL_NAME} v{challenger_info['version']} ({challenger_info['run_name']})")
    print(f"  model URI : models:/{MODEL_NAME}@champion")
    print(f"  evidence  : {ARTIFACTS_DIR / 'selected_versions.json'}")


if __name__ == "__main__":
    main()
