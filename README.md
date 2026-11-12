# load-api

Loan-default MLflow lifecycle and Module 6 capstone project.

- `capstone-report.md` — Module 6 evidence report (GitHub → GHCR → Render)
- `slides.md` — five-minute capstone presentation
- `report.md` — Module 5 MLflow lifecycle evidence
- `memo.md` — champion promotion memo
- `train_tracked.py` — eight tracked runs, model selection, registry aliases
- `main.py` — FastAPI service; MLflow alias or bundled `model.joblib`
- `monitor.py` — PSI drift monitoring and alerting
- `scripts/build_bundle.py` — export/retrain the champion bundle
- `scripts/verify_live.py` — health, API-key, prediction, cold-start evidence
- `scripts/check_openapi.py` — Module 2 contract comparison
- `scripts/drill_bad_release.py` — local bad-release and rollback drill
- `.github/workflows/ci.yml` — test → build-and-push → deploy → verify
- `Dockerfile`, `render.yaml` — production image and Render service

## Local quick start

```powershell
python -m pytest
# with the Module 5 MLflow server on http://127.0.0.1:5000:
python scripts/build_bundle.py --source registry
# offline / CI equivalent:
# python scripts/build_bundle.py --source train

$env:MODEL_SOURCE="bundle"
$env:API_KEY="local-test-key"
$env:GIT_SHA="local"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -H "X-API-Key: local-test-key" `
  -d '{"amount_usd":1500,"term_months":12,"purpose":"Buy a rice thresher","annual_income_usd":9000,"credit_score":640,"existing_debt_usd":2500,"employment_years":4.5,"num_late_payments_12m":1,"age":38}'
```

With `MLFLOW_TRACKING_URI` unset or `MODEL_SOURCE=bundle`, the API loads the
bundled champion. Set `MODEL_SOURCE=mlflow` and `MLFLOW_TRACKING_URI` to serve
`models:/loan-default@champion`.

## Docker

```powershell
$sha = git rev-parse HEAD
docker build --build-arg GIT_SHA=$sha -t loan-api:$sha .
docker run --rm -p 8000:8000 -e API_KEY="local-test-key" -e GIT_SHA=$sha loan-api:$sha
```

Or verify a pulled production image:

```powershell
python scripts/docker_smoke.py `
  --image ghcr.io/doeunbunheng/load-api:$sha `
  --json evidence/module6/docker_smoke.json
```

## Recreate the Module 5 registry

```powershell
python data/make_dataset.py
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
python train_tracked.py --reset-registry
```

See `report.md` for the full serving, monitoring and rollback commands.
See `docs/render-deployment.md` for GHCR, Render, secrets, DNS, and HTTPS.
See `docs/capstone-runbook.md` for the evidence-collection sequence.

## Deployment path

Every pull request runs `test`. A green push to `main` runs
`test → build-and-push → deploy → verify`:

- `test` trains `model.joblib` and runs the pytest suite with no MLflow server.
- `build-and-push` publishes `ghcr.io/doeunbunheng/load-api:<git-sha>` and `:latest`.
- `deploy` calls the Render deploy hook stored in GitHub secrets.
- `verify` polls the live `/health` until `git_sha` matches the commit.

Required GitHub configuration:

- secret `RENDER_DEPLOY_HOOK_URL`
- variable `RENDER_SERVICE_URL`
- optional secret `API_KEY` for the protected-route verification
- branch protection requiring the `test` check
