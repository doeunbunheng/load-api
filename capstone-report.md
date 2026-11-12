# Module 6 Capstone — Loan API from pull request to verified release

**Repository:** <https://github.com/doeunbunheng/load-api>
**Live app:** `[paste the Render URL after deployment]`
**Model:** `loan-default@champion`, registry version 1, run `d400758e6cbe4ed2a0b36d4b85fe32b8`
**Status:** serving code, model bundle, tests, container, CI/CD, and local drills are complete. Remote Render/GHCR/SSL Labs fields are marked `[fill]` and must be captured on the live service.

## Architecture and release path

```text
Pull request
    |
    v
GitHub Actions "test"  -- trains model.joblib, runs 14 tests
    |
    v  (merge to main)
"build-and-push"       -- GHCR :<git-sha> and :latest, SHA baked into image
    |
    v
"deploy"               -- Render deploy hook (secret)
    |
    v
"verify"               -- polls /health until git_sha == commit, then checks
                          health, 401 without key, prediction with key
```

The service has two model sources:

- `MODEL_SOURCE=mlflow` loads `models:/loan-default@champion`, so moving the
  alias is the rollback switch.
- `MODEL_SOURCE=bundle` (the Render setting) loads `model.joblib` created from
  the same champion pipeline. This removes the tracking server from the request
  path and from the Render deployment.

`/health` is public. `/predict` requires `X-API-Key` whenever `API_KEY` is set.
The key and the Render deploy hook are supplied only through environment
variables/secrets and never appear in logs.

## 1. Foundations and the Module 2 contract

Local test run with `MLFLOW_TRACKING_URI` unset:

```text
14 passed
```

Evidence: `evidence/module6/pytest_local.txt`.

The test suite covers:

- application startup with no MLflow tracking URI and the bundled model;
- `/health` without a key;
- `/predict` 401 without a key, 401 with a wrong key, 200 with the key;
- the exact champion probability for the known applicant;
- the 5000 amount limit and the 5001 rejection;
- the OpenAPI method/path contract;
- API-key redaction in the prediction log.

The live `/openapi.json` was compared with `spec/module2_contract.json` using
`scripts/check_openapi.py`:

| Check | Result |
|---|---|
| Paths/methods | PASS — only `GET /health` and `POST /predict` |
| Request fields | PASS — all nine Module 2 applicant fields |
| `/predict` response fields | PASS — probability, model, alias, version, run ID |
| Intentional additions | `X-API-Key` security scheme; `git_sha`, `model_source`, `api_key_required` in `/health` |

Local evidence: `evidence/module6/openapi_diff_local.json`.

Remote evidence to attach:

- `[fill]` green `main` CI run URL.
- `[fill]` branch protection screenshot showing required check `test`.
- `[fill]` failing amount-limit PR URL and screenshot of the red `test` job.

## 2. Model lifecycle

The Module 5 evidence is still the model-lifecycle record:

- Eight MLflow runs across five model families: `evidence/01_training.txt`, `evidence/01_leaderboard.txt`.
- Registered champion v1 and challenger v2: `evidence/02_registry.txt`, `evidence/02_registry.json`.
- Promotion memo: `memo.md`.
- Drift chart and PSI report: `artifacts/drift_report.png`, `artifacts/drift_report.json`.

Champion metrics on the 1,600-row stratified holdout:

| Metric | Champion (`logreg-regularized`) | Challenger (`random-forest-shallow`) |
|---|---:|---:|
| Recall | 0.7546 | 0.7232 |
| Precision | 0.4573 | 0.4556 |
| ROC-AUC | 0.8188 | 0.8079 |
| PR-AUC | 0.6113 | 0.5862 |

The bundled model is created from the champion pipeline and preserves its
identity and metrics:

```json
{
  "version": "1",
  "run_id": "d400758e6cbe4ed2a0b36d4b85fe32b8",
  "metrics": {
    "recall": 0.7545691906005222,
    "precision": 0.4572784810126582,
    "roc_auc": 0.818817835236671
  },
  "expected_sample": {
    "default_probability": 0.2534030951697195
  }
}
```

Full metadata: `model_metadata.json`.

Drift policy from `artifacts/monitor_config.json`: warning at PSI ≥ 0.10,
critical at PSI ≥ 0.25, dataset alert at ≥ 3 features over warning. Planned
response: investigate shifted features, retrain a challenger, compare recall,
precision, and AUC, then move the `champion` alias only if the challenger wins.

## 3. Ship and verify a real improvement

The improvement shipped by this module is the regression suite plus the
amount-limit guard and the deployment verifier. The pipeline jobs are:

| Job | What it proves |
|---|---|
| `test` | Clean runner trains `model.joblib`; 14 tests pass with no MLflow server; PRs cannot merge on a red check |
| `build-and-push` | Image is pushed as `ghcr.io/doeunbunheng/load-api:<git-sha>` and `:latest`; the SHA is baked into `GIT_SHA` |
| `deploy` | Render deploy hook is called from a GitHub secret |
| `verify` | Polls `/health` until the live `git_sha` equals the merged commit; runs the key/prediction check |

Local container verification: the Dockerfile built successfully (`loan-api:local-final`, image ID `sha256:c6b0f94919949afa2673731220cc47e3e1e8b23b7d4efac2963af3962f0e1e09`), the container became healthy in 3.141 s, `/predict` returned 401 without a key and 200 with a key, and the probability was `0.2534030951697195`. Evidence: `evidence/module6/docker_smoke_local.json` and `docker_build_local.txt`.

Remote evidence to attach:

- `[fill]` PR URL and passing `test` check.
- `[fill]` pipeline graph screenshot.
- `[fill]` GHCR package page showing the SHA tag and `latest`.
- `[fill]` `verify.json` artifact with `seconds_commit_to_live`.
- `[fill]` live `/health` showing the same `git_sha`.

| Release measurement | Value |
|---|---:|
| Merge → live SHA verified | `[fill]` |
| Workflow start → live SHA verified | `[fill]` |
| GHCR image tag | `[fill]` |

## 4. API-key protection and HTTPS

Local serving check (same code path as the container):

| Request | Status | Time |
|---|---:|---:|
| `GET /health` without key | 200 | 0.025 s |
| `GET /health` second call | 200 | 0.013 s |
| `POST /predict` without key | 401 | 0.034 s |
| `POST /predict` with correct key | 200 | 0.147 s |
| `POST /predict` with wrong key | 401 | 0.005 s |

The correct-key response returned `default_probability = 0.2534030951697195`.
Evidence: `evidence/module6/live_verification_local.json`.

Remote evidence to attach:

- `[fill]` first request after Render idle and the immediate second request.
- `[fill]` `/health` 200 without a key.
- `[fill]` `/predict` 401 without a key.
- `[fill]` `/predict` 200 with the key.
- `[fill]` SSL Labs grade for the live hostname (target A/A+).

Do not screenshot the key, the deploy hook, or request headers that contain the
key.

## 5. Break and recover

### Local drill (completed)

`python scripts/drill_bad_release.py` produced
`evidence/module6/bad_release_drill.json` and `.txt`.

| Stage | Git SHA label | Probability |
|---|---|---:|
| Good release | `good-release` | 0.2534030951697195 |
| Bad scoring release | `bad-release` | 0.7465969048302805 |
| Rollback | `good-release` | 0.2534030951697195 |

The drill also proved both guardrails:

- Changing `MAX_LOAN_AMOUNT_USD` from 5000 to 500 fails
  `test_loan_amount_limit_is_5000` and `test_amount_at_limit_is_accepted`.
- Inverting the probability fails `test_champion_score_regression` and the
  credit-score ordering test.

Local timings:

| Measurement | Value |
|---|---:|
| Bad release start → wrong prediction | 6.250 s |
| Rollback start → correct prediction | 5.884 s |

### Deployed drill (to complete on Render)

Because branch protection blocks a knowingly broken merge, use a separate
`drill-bad` image tag or a separate free Render drill service, then use Render
**Rollback**:

| Measurement | Value |
|---|---:|
| Bad image deploy start → wrong prediction | `[fill]` |
| Rollback start → correct prediction | `[fill]` |
| Same applicant before/after | 0.2534030951697195 / `[fill]` |

The final PR that fixes the formula and keeps the regression test is:
`[fill PR URL]`.

## 6. Five-minute presentation

1. **Architecture (45 s):** FastAPI service, champion bundle, GHCR image,
   Render, the four-job pipeline.
2. **Guardrails (45 s):** CI test on a clean runner, branch protection,
   failing amount-limit PR.
3. **Model lifecycle (45 s):** eight tracked runs, champion/challenger, memo
   metrics, drift thresholds.
4. **Release identity (45 s):** immutable `<git-sha>` image tag, `latest` as a
   pointer, live `/health` SHA match, merge-to-live timing.
5. **Security (45 s):** `/health` public, `/predict` 401/200/401, HTTPS, SSL
   Labs grade, secrets kept out of logs.
6. **Bad release (45 s):** good 0.2534 → bad 0.7466 → rollback 0.2534, plus
   the failing PR and the restoration timing.
7. **Decision, trade-off, next step (30 s):**
   - Decision: deploy by git SHA, not by a floating tag, so the live `git_sha`
     is an auditable release identity.
   - Trade-off: bundling the champion in the image makes Render independent of
     MLflow but means a model promotion requires a new image and pipeline run.
   - Next step: add PostgreSQL persistence for prediction logs and drift
     monitoring, replacing the ephemeral JSONL file.

## Evidence index

| File | Contents |
|---|---|
| `evidence/module6/pytest_local.txt` | 14 passing tests |
| `evidence/module6/openapi_diff_local.json` | Module 2 contract comparison |
| `evidence/module6/live_verification_local.json` | health, 401, 200, timing, TLS |
| `evidence/module6/bad_release_drill.json` | good/bad/rollback predictions and tests |
| `evidence/module6/bad_release_drill.txt` | human-readable drill summary |
| `evidence/module6/docker_smoke_local.json` | built image health, 401/200, prediction |
| `evidence/module6/docker_build_local.txt` | local image ID and size |
| `model_metadata.json` | bundled champion identity and metrics |
| `spec/module2_contract.json` | Module 2 path/method/field contract |
| `.github/workflows/ci.yml` | test → build-and-push → deploy → verify |
| `Dockerfile` | production image with baked `GIT_SHA` |
| `render.yaml` | Render image service, `/health`, API key secret |
| `docs/branch-protection.md` | branch protection setup |
| `docs/render-deployment.md` | GHCR, Render, deploy hook, HTTPS, rollback |
| `docs/capstone-runbook.md` | evidence collection commands |
| `slides.md` | five-minute presentation outline |
