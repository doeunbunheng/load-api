# Controlled bad-release patches

These patches are for evidence only. Never commit them to `main`.

## Failing amount-limit PR

```powershell
git switch -c drill/amount-limit
git apply drills/amount_limit.patch
git add main.py
git commit -m "drill: lower loan amount limit"
git push -u origin drill/amount-limit
```

Open a PR. The `test` check fails on `test_loan_amount_limit_is_5000` and
`test_amount_at_limit_is_accepted`. Close the PR without merging.

## Deployed scoring drill

```powershell
git switch -c drill/bad-scoring
git apply drills/bad_scoring.patch
docker build --build-arg GIT_SHA=drill-bad -t ghcr.io/doeunbunheng/load-api:drill-bad .
docker push ghcr.io/doeunbunheng/load-api:drill-bad
```

Point the Render drill service at the `drill-bad` tag (or deploy it to a
separate free image service), record the wrong prediction for the sample
applicant, then use Render **Rollback** to the previous good deploy and record
the restored prediction.

Restore the working tree without merging:

```powershell
git restore main.py
```
