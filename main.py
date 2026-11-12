"""FastAPI loan-default scoring service.

The service resolves the model in one of two ways:

* ``MODEL_SOURCE=mlflow`` loads ``models:/<MODEL_NAME>@<MODEL_ALIAS>`` from the
  configured tracking URI, so moving the registry alias rolls the service
  forward or back on the next restart.
* ``MODEL_SOURCE=bundle`` (or ``auto`` with no tracking URI) loads the bundled
  ``model.joblib`` and ``model_metadata.json`` created by
  ``scripts/build_bundle.py``.

``/health`` is always public. ``/predict`` requires ``X-API-Key`` whenever the
``API_KEY`` environment variable is set. The key is never logged or returned.
"""

import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from loan_features import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent
MAX_LOAN_AMOUNT_USD = 5000
LOGGER = logging.getLogger("loan-api")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    LOGGER.propagate = False
API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key", auto_error=False, scheme_name="X-API-Key"
)


@dataclass(frozen=True)
class Settings:
    model_source: str
    tracking_uri: str
    model_name: str
    model_alias: str
    api_key: str
    git_sha: str
    bundle_path: Path
    metadata_path: Path
    log_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_source=os.environ.get("MODEL_SOURCE", "auto").strip().lower(),
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "").strip(),
            model_name=os.environ.get("MODEL_NAME", "loan-default").strip(),
            model_alias=os.environ.get("MODEL_ALIAS", "champion").strip(),
            api_key=os.environ.get("API_KEY", ""),
            git_sha=(
                os.environ.get("GIT_SHA")
                or os.environ.get("RENDER_GIT_COMMIT")
                or "unknown"
            ).strip(),
            bundle_path=Path(
                os.environ.get("MODEL_BUNDLE_PATH", str(BASE_DIR / "model.joblib"))
            ),
            metadata_path=Path(
                os.environ.get(
                    "MODEL_METADATA_PATH", str(BASE_DIR / "model_metadata.json")
                )
            ),
            log_path=Path(
                os.environ.get(
                    "PREDICTION_LOG_PATH", str(BASE_DIR / "logs" / "predictions.jsonl")
                )
            ),
        )


class Applicant(BaseModel):
    amount_usd: int = Field(ge=100, le=MAX_LOAN_AMOUNT_USD, examples=[1500])
    term_months: Literal[3, 6, 12, 18, 24] = Field(examples=[12])
    purpose: str = Field(min_length=1, max_length=200, examples=["Buy a rice thresher"])
    annual_income_usd: float = Field(gt=0, le=1_000_000, examples=[9000])
    credit_score: int = Field(ge=300, le=850, examples=[640])
    existing_debt_usd: float = Field(ge=0, le=1_000_000, examples=[2500])
    employment_years: float = Field(ge=0, le=60, examples=[4.5])
    num_late_payments_12m: int = Field(ge=0, le=50, examples=[1])
    age: int = Field(ge=18, le=100, examples=[38])


class Health(BaseModel):
    status: str
    git_sha: str
    model_name: str
    alias: str
    version: str
    run_id: str
    model_source: str
    model_loaded: bool
    api_key_required: bool


class Prediction(BaseModel):
    default_probability: float
    model_name: str
    alias: str
    version: str
    run_id: str


class BundleModel:
    """Small adapter so the bundle and the MLflow pyfunc expose one interface."""

    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
        return pd.DataFrame({"default_probability": probabilities})


def _load_bundle(settings: Settings) -> tuple[BundleModel, dict]:
    if not settings.bundle_path.exists():
        raise RuntimeError(
            f"Bundled model missing at {settings.bundle_path}; "
            "run `python scripts/build_bundle.py` or set MODEL_SOURCE=mlflow."
        )
    pipeline = joblib.load(settings.bundle_path)
    metadata = {}
    if settings.metadata_path.exists():
        metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    info = {
        "source": "bundle",
        "model_name": settings.model_name,
        "alias": settings.model_alias,
        "version": str(metadata.get("version", "bundled")),
        "run_id": str(metadata.get("run_id", "bundled")),
        "trained_at": metadata.get("trained_at"),
        "metrics": metadata.get("metrics", {}),
        "path": str(settings.bundle_path),
    }
    return BundleModel(pipeline, metadata), info


def _load_mlflow(settings: Settings) -> tuple[object, dict]:
    if not settings.tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required when MODEL_SOURCE=mlflow")

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.tracking_uri)
    model_uri = f"models:/{settings.model_name}@{settings.model_alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    version = MlflowClient(tracking_uri=settings.tracking_uri).get_model_version_by_alias(
        settings.model_name, settings.model_alias
    )
    info = {
        "source": "mlflow",
        "model_name": settings.model_name,
        "alias": settings.model_alias,
        "version": str(version.version),
        "run_id": version.run_id,
        "trained_at": None,
        "metrics": {},
        "path": model_uri,
    }
    return model, info


def resolve_model(settings: Settings) -> tuple[object, dict]:
    source = settings.model_source
    if source not in {"auto", "mlflow", "bundle"}:
        raise RuntimeError(
            f"MODEL_SOURCE must be auto, mlflow, or bundle (got {source!r})"
        )
    if source == "bundle":
        return _load_bundle(settings)
    if source == "mlflow":
        return _load_mlflow(settings)
    if settings.tracking_uri:
        try:
            return _load_mlflow(settings)
        except Exception as exc:  # noqa: BLE001 - auto mode explicitly falls back
            LOGGER.warning(
                "MLflow load failed (%s); using bundled model.joblib instead",
                exc.__class__.__name__,
            )
    return _load_bundle(settings)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    def require_api_key(provided: str | None = Security(API_KEY_HEADER)) -> None:
        if not settings.api_key:
            return
        if not provided or not secrets.compare_digest(provided, settings.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "X-API-Key"},
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model, info = resolve_model(settings)
        app.state.model = model
        app.state.model_info = info
        if not settings.api_key:
            LOGGER.warning("API_KEY is not set; /predict is not protected")
        LOGGER.info(
            "startup %s",
            json.dumps(
                {
                    "git_sha": settings.git_sha,
                    "model_source": info["source"],
                    "model_version": info["version"],
                    "run_id": info["run_id"],
                    "api_key_required": bool(settings.api_key),
                },
                sort_keys=True,
            ),
        )
        yield

    app = FastAPI(
        title="Loan Default API",
        version="1.0.0",
        description=(
            "Serves the MLflow champion by registry alias when configured, "
            "otherwise the bundled champion model."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        info = request.app.state.model_info
        return {
            "status": "ok",
            "git_sha": settings.git_sha,
            "model_name": info["model_name"],
            "alias": info["alias"],
            "version": info["version"],
            "run_id": info["run_id"],
            "model_source": info["source"],
            "model_loaded": request.app.state.model is not None,
            "api_key_required": bool(settings.api_key),
        }

    @app.post(
        "/predict",
        response_model=Prediction,
        dependencies=[Depends(require_api_key)],
    )
    def predict(applicant: Applicant, request: Request) -> Prediction:
        payload = applicant.model_dump()
        frame = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
        output = request.app.state.model.predict(frame)
        probability = float(output.iloc[0]["default_probability"])
        info = request.app.state.model_info

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
            "default_probability": probability,
            "model_source": info["source"],
            "model_version": info["version"],
            "run_id": info["run_id"],
            "git_sha": settings.git_sha,
        }
        try:
            settings.log_path.parent.mkdir(parents=True, exist_ok=True)
            with settings.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError as exc:
            LOGGER.warning("Prediction log write failed (%s)", exc.__class__.__name__)

        return Prediction(
            default_probability=probability,
            model_name=info["model_name"],
            alias=info["alias"],
            version=info["version"],
            run_id=info["run_id"],
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
