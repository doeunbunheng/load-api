"""FastAPI loan-default scoring service.

The model is always resolved through the MLflow registry alias
(`models:/loan-default@champion`), never through a local file or a hard-coded
version number. Restarting the process after moving the alias is therefore
enough to roll forward or back.
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from loan_features import FEATURE_COLUMNS

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "loan-default")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
LOG_PATH = Path(__file__).resolve().parent / "logs" / "predictions.jsonl"


class Applicant(BaseModel):
    amount_usd: int = Field(ge=100, le=5000, examples=[1500])
    term_months: Literal[3, 6, 12, 18, 24] = Field(examples=[12])
    purpose: str = Field(min_length=1, max_length=200, examples=["Buy a rice thresher"])
    annual_income_usd: float = Field(gt=0, le=1_000_000, examples=[9000])
    credit_score: int = Field(ge=300, le=850, examples=[640])
    existing_debt_usd: float = Field(ge=0, le=1_000_000, examples=[2500])
    employment_years: float = Field(ge=0, le=60, examples=[4.5])
    num_late_payments_12m: int = Field(ge=0, le=50, examples=[1])
    age: int = Field(ge=18, le=100, examples=[38])


class Prediction(BaseModel):
    default_probability: float
    model_name: str
    alias: str
    version: str
    run_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    app.state.model = mlflow.pyfunc.load_model(model_uri)
    app.state.version = MlflowClient(tracking_uri=TRACKING_URI).get_model_version_by_alias(
        MODEL_NAME, MODEL_ALIAS
    )
    print(f"Loaded {model_uri} -> version {app.state.version.version} (run {app.state.version.run_id})")
    yield


app = FastAPI(
    title="Loan Default API",
    version="1.0.0",
    description="Serves the MLflow champion by registry alias.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    version = app.state.version
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(version.version),
        "run_id": version.run_id,
        "model_loaded": app.state.model is not None,
    }


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant) -> Prediction:
    payload = applicant.model_dump()
    frame = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    output = app.state.model.predict(frame)
    probability = float(output.iloc[0]["default_probability"])

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
        "default_probability": probability,
        "model_version": str(app.state.version.version),
        "run_id": app.state.version.run_id,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    return Prediction(
        default_probability=probability,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=str(app.state.version.version),
        run_id=app.state.version.run_id,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
