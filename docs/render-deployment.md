# Render, GHCR, secrets, and HTTPS

## 1. Publish the image

The `CI` workflow pushes two tags for every green push to `main`:

- `ghcr.io/doeunbunheng/load-api:<full-git-sha>`
- `ghcr.io/doeunbunheng/load-api:latest`

The commit SHA is the immutable release identity. `latest` is only the moving
pointer used by the Render image service.

Make the GHCR package public for Render's free image service:

1. GitHub → your profile → **Packages** → `load-api`.
2. **Package settings → Danger Zone → Change visibility → Public**.

Local container check:

```powershell
$sha = git rev-parse HEAD
docker pull ghcr.io/doeunbunheng/load-api:$sha
docker run --rm -p 8000:8000 `
  -e API_KEY="local-smoke-key" `
  -e GIT_SHA="$sha" `
  ghcr.io/doeunbunheng/load-api:$sha
```

Then open `http://127.0.0.1:8000/health`. Or run:

```powershell
python scripts/docker_smoke.py `
  --image ghcr.io/doeunbunheng/load-api:$sha `
  --json evidence/module6/docker_smoke.json
```

## 2. Create the Render service

Use the public prebuilt image, not Render's Docker build, so the deployed
artifact is exactly the one CI verified.

1. Render Dashboard → **New → Web Service → Deploy an existing image**.
2. Image URL: `ghcr.io/doeunbunheng/load-api:latest`.
3. Name: `load-api`; region: Oregon; plan: Free.
4. Health check path: `/health`.
5. Environment variables:
   - `MODEL_SOURCE=bundle`
   - `API_KEY=<long random value>` (use Render's generate button)
6. Create the service and wait for **Live**.

`render.yaml` in this repository describes the same service for Blueprint
deploys: public GHCR image, free plan, `/health`, and a prompted `API_KEY`.

## 3. Connect continuous deployment

1. Render service → **Settings → Deploy Hook** → copy the hook URL.
2. GitHub repository → **Settings → Secrets and variables → Actions → Secrets**:
   add `RENDER_DEPLOY_HOOK_URL`.
3. Add the live URL as a repository **variable**:
   `RENDER_SERVICE_URL=https://<service>.onrender.com`.
4. Optionally add the live API key as the GitHub Actions secret `API_KEY` so
   the verify job can test the protected route and prediction. If it is absent,
   the verify job still checks `/health` and the matching SHA.

After a merge to `main`, the pipeline is:

```text
test -> build-and-push -> deploy -> verify
```

`deploy` calls the Render deploy hook. `verify` polls `/health` until
`git_sha` equals the pushed commit, then writes `verify.json` and
`verify-smoke.json` as artifacts. The report's merge-to-live number is
`seconds_commit_to_live` from `verify.json`.

## 4. Cold start

A free Render service sleeps after inactivity. For the cold-start evidence,
wait until the service is asleep, then run:

```powershell
$env:LIVE_URL="https://<service>.onrender.com"
$env:API_KEY="<live key>"
python scripts/verify_live.py --url $env:LIVE_URL --wait 600 `
  --json evidence/module6/live_after_idle.json
```

The script records the first successful health call and the immediate second
call so the difference is visible.

## 5. HTTPS and DNS

The `onrender.com` hostname is HTTPS by default. For a custom hostname:

1. Render → service → **Settings → Custom Domains → Add Custom Domain**.
2. Create the CNAME/A records Render shows at your DNS provider.
3. Wait for the certificate status to become **Issued**.
4. Run an SSL Labs test on the hostname and save the grade (aim for A or A+).

Record the TLS evidence from `scripts/verify_live.py` (`tls_version`, issuer,
expiry) plus the SSL Labs screenshot.

## 6. Rollback

Render keeps previous deploys. For a bad release:

1. Render → service → **Deploys** → open the last known-good deploy.
2. Choose **Rollback** and wait for **Live**.
3. Poll `/health` until it reports the good `git_sha`.
4. Re-run the same prediction request and confirm the original probability.

If you deploy by image tag instead, point the service back to the previous
immutable `<git-sha>` tag. Never use `latest` as the rollback identity.

## 7. Secret hygiene

- `API_KEY` and the deploy hook live only in Render and GitHub secrets.
- Never commit `.env`, paste keys into logs, or screenshot environment values.
- `scripts/verify_live.py` prints `api_key_required` but never the key.
