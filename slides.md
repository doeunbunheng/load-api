# Module 6 Capstone — Loan API

Repository: https://github.com/doeunbunheng/load-api
Live: https://[fill].onrender.com

---

## 1. Architecture

```text
PR -> CI test -> GHCR image :<sha> -> Render -> /health verify
                         \
                          :latest pointer
```

- FastAPI service, `/health` public, `/predict` key-protected.
- Champion model bundled as `model.joblib`; optional MLflow alias load.
- Render free image service; deploy hook in GitHub secrets.

---

## 2. Guardrails

- CI trains the bundle on a clean runner, then runs 14 tests with no MLflow.
- `main` requires the `test` check.
- Amount-limit PR fails: 5000 → 500.
- Score inversion PR fails: regression test expects `0.2534030951697195`.

Evidence: `evidence/module6/bad_release_drill.txt`

---

## 3. Model lifecycle

- 8 runs, 5 families; champion `logreg-regularized` v1.
- Recall 0.7546, precision 0.4573, ROC-AUC 0.8188.
- Challenger `random-forest-shallow` v2 retained for shadow/rollback.
- Drift: warning PSI 0.10, critical PSI 0.25; retrain challenger on alert.

---

## 4. Release identity

- Image tags: `ghcr.io/doeunbunheng/load-api:<sha>` and `:latest`.
- SHA baked at build; `/health` returns it.
- Pipeline: test → build-and-push → deploy → verify.
- Merge → live: `[fill]` seconds.

---

## 5. Security

| Request | Result |
|---|---|
| `GET /health` no key | 200 |
| `POST /predict` no key | 401 |
| `POST /predict` correct key | 200 |
| SSL Labs | `[fill grade]` |

- Key and deploy hook only in Render/GitHub secrets.
- HTTPS terminates at Render; custom domain optional.

---

## 6. Bad release and rollback

| Stage | Probability |
|---|---:|
| Good | 0.2534030951697195 |
| Bad formula | 0.7465969048302805 |
| Rollback | 0.2534030951697195 |

- Local drill: bad start → wrong 6.250 s; rollback → correct 5.884 s.
- Render drill: `[fill deploy and rollback timings]`.

---

## 7. Decision, trade-off, next step

**Decision:** deploy by immutable git SHA so the live `git_sha` proves the
release.

**Trade-off:** bundling the model removes MLflow from the Render request path,
but a model promotion needs a new image and pipeline run.

**Next step:** PostgreSQL persistence for prediction logs and drift monitoring,
replacing the ephemeral JSONL file.
