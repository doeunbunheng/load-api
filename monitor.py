"""Monitor loan-default input drift against the champion training reference.

The script computes Population Stability Index (PSI) for every model feature
and for the champion's predicted default probability, writes a chart, and
raises an alert when the configured thresholds are crossed.

Thresholds are stored in artifacts/monitor_config.json. The defaults (0.10
warning, 0.25 critical) were calibrated from the chart: the stable features
stay below 0.05 while the deliberately shifted features exceed 0.30.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from loan_features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REFERENCE_PATH = ARTIFACTS_DIR / "reference_train.csv"
PRODUCTION_CSV = BASE_DIR / "data" / "production_window.csv"
PREDICTION_LOG = BASE_DIR / "logs" / "predictions.jsonl"
CONFIG_PATH = ARTIFACTS_DIR / "monitor_config.json"
CHART_PATH = ARTIFACTS_DIR / "drift_report.png"
REPORT_PATH = ARTIFACTS_DIR / "drift_report.json"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "loan-default")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")

DEFAULT_CONFIG = {
    "warning_threshold": 0.10,
    "critical_threshold": 0.25,
    "prediction_warning_threshold": 0.10,
    "prediction_critical_threshold": 0.25,
    "minimum_features_over_warning_for_dataset_alert": 3,
    "rationale": (
        "Calibrated from the drift chart: stable features stayed below PSI 0.05, while the "
        "shifted production features exceeded PSI 0.30. Warning 0.10 sits in the empty gap; "
        "critical 0.25 marks the start of the large-shift cluster and triggers retraining."
    ),
}


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
    CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    return DEFAULT_CONFIG


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    expected_counts = pd.cut(expected, bins=edges, include_lowest=True).value_counts(normalize=True)
    actual_counts = pd.cut(actual, bins=edges, include_lowest=True).value_counts(normalize=True)
    categories = expected_counts.index.union(actual_counts.index)
    expected_pct = expected_counts.reindex(categories, fill_value=0.0)
    actual_pct = actual_counts.reindex(categories, fill_value=0.0)
    epsilon = 1e-6
    expected_pct = expected_pct.clip(lower=epsilon)
    actual_pct = actual_pct.clip(lower=epsilon)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    categories = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    expected_pct = expected.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    actual_pct = actual.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    epsilon = 1e-6
    expected_pct = expected_pct.clip(lower=epsilon)
    actual_pct = actual_pct.clip(lower=epsilon)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _production_frame() -> tuple[pd.DataFrame, str]:
    if PREDICTION_LOG.exists():
        records = [json.loads(line) for line in PREDICTION_LOG.read_text(encoding="utf-8").splitlines() if line]
        if len(records) >= 200:
            return pd.DataFrame(records)[FEATURE_COLUMNS], f"prediction log ({len(records)} rows)"
    return pd.read_csv(PRODUCTION_CSV)[FEATURE_COLUMNS], f"production window CSV ({PRODUCTION_CSV.name})"


def _champion_probabilities(frame: pd.DataFrame) -> np.ndarray:
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    predictions = model.predict(frame[FEATURE_COLUMNS])
    return predictions["default_probability"].to_numpy()


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    warning = float(config["warning_threshold"])
    critical = float(config["critical_threshold"])

    reference = pd.read_csv(REFERENCE_PATH)
    production, production_source = _production_frame()

    results = {}
    for column in NUMERIC_FEATURES:
        results[column] = _psi(reference[column], production[column])
    for column in CATEGORICAL_FEATURES:
        results[column] = _psi_categorical(reference[column], production[column])

    reference_probability = pd.Series(_champion_probabilities(reference))
    production_probability = pd.Series(_champion_probabilities(production))
    prediction_psi = _psi(reference_probability, production_probability)
    results["predicted_default_probability"] = prediction_psi

    def status(value: float) -> str:
        if value >= critical:
            return "CRITICAL"
        if value >= warning:
            return "WARNING"
        return "OK"

    statuses = {name: status(value) for name, value in results.items()}
    feature_alerts = [name for name in FEATURE_COLUMNS if statuses[name] == "CRITICAL"]
    dataset_alert = (
        len([name for name in FEATURE_COLUMNS if results[name] >= warning])
        >= int(config["minimum_features_over_warning_for_dataset_alert"])
    )
    prediction_alert = prediction_psi >= float(config["prediction_critical_threshold"])
    any_alert = bool(feature_alerts) or dataset_alert or prediction_alert

    ordered = sorted(results.items(), key=lambda item: item[1], reverse=True)
    names = [name for name, _ in ordered]
    values = [value for _, value in ordered]
    colors = [
        "#c0392b" if value >= critical else "#f39c12" if value >= warning else "#27ae60"
        for value in values
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(warning, color="#f39c12", linestyle="--", linewidth=1.5, label=f"warning = {warning:.2f}")
    ax.axvline(critical, color="#c0392b", linestyle="--", linewidth=1.5, label=f"critical = {critical:.2f}")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title(
        f"Loan-default drift: {production_source} vs champion training reference\n"
        f"champion = {MODEL_NAME}@{MODEL_ALIAS}"
    )
    ax.legend(loc="lower right")
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(max(values) * 1.18, critical * 1.5))
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    plt.close(fig)

    report = {
        "reference_rows": len(reference),
        "production_rows": len(production),
        "production_source": production_source,
        "champion": f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        "thresholds": {
            "warning": warning,
            "critical": critical,
            "prediction_warning": float(config["prediction_warning_threshold"]),
            "prediction_critical": float(config["prediction_critical_threshold"]),
        },
        "psi": results,
        "status": statuses,
        "feature_alerts": feature_alerts,
        "dataset_alert": dataset_alert,
        "prediction_alert": prediction_alert,
        "any_alert": any_alert,
        "chart": str(CHART_PATH),
        "rationale": config["rationale"],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Reference rows : {len(reference)}")
    print(f"Production rows: {len(production)} ({production_source})")
    print(f"Champion       : models:/{MODEL_NAME}@{MODEL_ALIAS}")
    print(f"Thresholds     : warning PSI >= {warning:.2f}, critical PSI >= {critical:.2f}")
    print("\nPSI by feature")
    for name, value in ordered:
        print(f"  {name:<32} {value:8.4f}  {statuses[name]}")
    print(f"\nChart  : {CHART_PATH}")
    print(f"Report : {REPORT_PATH}")

    if any_alert:
        print("\nALERT: drift thresholds crossed.")
        if feature_alerts:
            print(f"  Critical feature drift: {', '.join(feature_alerts)}")
        if dataset_alert:
            print(f"  Dataset alert: >= {config['minimum_features_over_warning_for_dataset_alert']} features at/above warning.")
        if prediction_alert:
            print(f"  Prediction PSI {prediction_psi:.4f} >= {config['prediction_critical_threshold']:.2f}.")
        print("  Recommended action: investigate the shifted features and retrain a challenger.")
        return 1

    print("\nOK: no drift alert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
