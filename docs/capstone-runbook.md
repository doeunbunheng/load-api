# Capstone evidence runbook

Use this sequence to collect the six capstone evidence sets. Commands assume
PowerShell from the repository root. Keep secrets in environment variables and
out of screenshots.

## 0. Baseline

```powershell
python -m pytest
python scripts/build_bundle.py --source registry
git status
git log --oneline -5
```

Save the pytest output as `evidence/module6/pytest_local.txt`.

## 1. Foundations and API contract

```powershell
$env:LIVE_URL="https://<service>.onrender.com"
python scripts/check_openapi.py --url $env:LIVE_URL `
  --json evidence/module6/openapi_diff.json
```

Expected: only `GET /health` and `POST /predict`; all nine request fields
present. Intentional Module 6 differences are the `X-API-Key` security scheme
and the extra health fields (`git_sha`, `model_source`, `api_key_required`).

Capture:

- CI run on `main` (green).
- Branch protection rule with required check `test`.
- Failing amount-limit PR (see `docs/branch-protection.md`).
- `openapi_diff.json`.

## 2. Model lifecycle

Reuse the Module 5 evidence:

- `evidence/01_training.txt`, `evidence/01_leaderboard.txt`
- `evidence/02_registry.txt`, `evidence/02_registry.json`
- `memo.md`
- `artifacts/drift_report.png`, `artifacts/drift_report.json`
- `model_metadata.json` for the exact champion bundled into the image.

Record the champion recall, precision, AUC, and the drift thresholds from
`artifacts/monitor_config.json`.

## 3. Ship a real improvement

The final regression test and the amount-limit guard are the improvement.
Open a PR from a short-lived branch:

```powershell
git switch -c feat/amount-limit-regression
# edit tests/test_contract.py, main.py, or scripts as needed
git add .
git commit -m "Add amount-limit and champion-score regression tests"
git push -u origin feat/amount-limit-regression
```

Capture:

- PR URL and passing `test` check.
- Pipeline graph `test -> build-and-push -> deploy -> verify`.
- GHCR package page with `<git-sha>` and `latest` tags.
- `verify.json` artifact with `seconds_commit_to_live`.
- Live `/health` showing the same `git_sha`.

## 4. API key and HTTPS

```powershell
$env:API_KEY="<live key>"
python scripts/verify_live.py --url $env:LIVE_URL --wait 600 `
  --json evidence/module6/live_verification.json
```

The evidence contains:

- `/health` 200 with no key.
- `/predict` 401 with no key.
- `/predict` 200 with the key.
- `/predict` 401 with a wrong key.
- First-vs-second health timings.
- TLS version, issuer, and certificate expiry.

Run an SSL Labs test and save the grade screenshot. Do not screenshot the key
or the deploy hook.

## 5. Break and recover

### 5a. Prove the guardrail locally

```powershell
python scripts/drill_bad_release.py
```

This writes `evidence/module6/bad_release_drill.json` and `.txt`. It proves:

- A PR that changes the amount limit to 500 fails the test job.
- A PR that inverts the champion formula fails the score regression.
- The good prediction is `0.2534`, the bad prediction is `0.7466`, and the
  rollback restores `0.2534`.

### 5b. Perform the deployed drill on Render

Because branch protection blocks a knowingly broken merge, perform the Render
drill with a separate drill image/tag and a Render rollback:

1. Create a bad image from a throwaway copy (never commit it):
   - change `MAX_LOAN_AMOUNT_USD` to `500`, and
   - invert `default_probability` in `main.py`.
2. Push it as `ghcr.io/doeunbunheng/load-api:drill-bad` (or use the local
   Dockerfile on a separate free Render service).
3. Point the drill service at `drill-bad`, wait for Live, and send the sample
   applicant. Record the wrong prediction and the time from deploy start.
4. Use Render **Rollback** to the previous good deploy. Poll `/health` and
   resend the same applicant. Record the time from rollback start to the
   correct prediction.
5. Delete the drill service/tag afterward.

Record in the report:

| Measurement | Value |
|---|---:|
| Bad image deploy start → wrong prediction | `[fill]` |
| Rollback start → correct prediction | `[fill]` |
| Good probability | 0.2534030951697195 |
| Bad probability | 0.7465969048302805 |
| Restored probability | 0.2534030951697195 |

## 6. Five-minute presentation

Use `slides.md`:

1. Architecture and release path.
2. CI guardrails and branch protection.
3. GHCR immutable SHA + `latest` pointer.
4. Render hosting and cold start.
5. Secret/HTTPS verification.
6. Bad release and rollback evidence.
7. Decision, trade-off, next step.

Keep the live service up until assessment is complete.
