# load-api

Week 17 MLflow lifecycle project for the loan-default classifier.

- `report.md` — full evidence report covering all six deliverables
- `memo.md` — champion promotion memo
- `train_tracked.py` — eight tracked runs, model selection, registry aliases
- `main.py` — FastAPI service that loads `models:/loan-default@champion`
- `monitor.py` — PSI drift monitoring and alerting
- `artifacts/` — leaderboard, selected versions, drift chart and monitor config
- `evidence/` — integration, registry, monitoring and rollback evidence

The model artifacts and local MLflow database are excluded from Git because they are large. Recreate them with:

```powershell
python data/make_dataset.py
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
python train_tracked.py --reset-registry
```

See `report.md` for the full serving, monitoring and rollback commands.
