# Production Deployment

## Dashboard frontend

`frontend/` — React 19 + TypeScript + Vite. Real components (`src/components/`) calling
the real backend through a typed client (`src/api.ts`): prediction card (direction,
confidence, regime, modality-contribution bars, top Integrated Gradients attributions,
counterfactual, explanation text), prediction history table, data-source health panel.
No mocked/placeholder data anywhere in the app code — every value rendered comes from a
real API response.

**Verified**: `npm run build` (TypeScript compiles, Vite bundles cleanly) and `npm test`
(16 Vitest/React Testing Library tests, including `PredictionCard.test.tsx`'s fixture,
which is not invented but a literal copy of a real response captured from
`POST /api/v1/predictions/MSFT/refresh` during manual verification against the live
backend). The dev server was started (`npm run dev`) alongside the FastAPI backend
(`uvicorn emafrs.api.app:app`) and every module (`App.tsx` and all components) was
confirmed to transform/serve without error, and the exact HTTP calls the dashboard makes
(`GET /health`, `GET /api/v1/data-sources/health`, `POST .../refresh`,
`GET .../latest`, `GET .../history`) were exercised directly via curl with the frontend's
`Origin` header, confirming CORS is correctly configured and the full pipeline produces
real persisted predictions reachable from the browser's fetch calls.

**Not verified**: an actual rendered-pixels visual check in a real browser window — the
Chrome DevTools browser-automation tool was unavailable in this environment (extension not
connected). The functional verification above (build succeeds, module graph loads without
error, live API calls succeed with correct CORS/data shape, component tests pass against a
real captured API response) is a strong but not identical substitute for an actual
click-through visual check, and this gap is stated explicitly rather than silently assumed
equivalent.

## What exists today

- **FastAPI backend** (`src/emafrs/api/app.py`): `/health`,
  `/api/v1/data-sources/health`, `/api/v1/predictions/{ticker}/latest`,
  `/api/v1/predictions/{ticker}/refresh` (POST, triggers real training), 
  `/api/v1/predictions/{ticker}/history`, `/api/v1/model-versions/{ticker}`.
- **Persistent DB**: SQLite by default (`data/emafrs.db`, gitignored), portable to
  Postgres via `DATABASE_URL` with no ORM changes (`src/emafrs/db/`).
- **Model serving**: checkpoints persisted to `models/checkpoints/` (gitignored,
  regenerable) with DB-indexed versioning (`src/emafrs/mlops/model_registry.py`).
- **Security middleware**: CORS (configurable via `CORS_ORIGINS` env var), security
  headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy), in-memory rate
  limiting, request logging that never logs bodies/headers/secrets.

Run locally:

```bash
uvicorn emafrs.api.app:app --reload
```

## Known limitations of the current implementation (disclosed, not hidden)

1. **The refresh endpoint trains synchronously.** `POST /api/v1/predictions/{ticker}/refresh`
   runs the full training pipeline (~10-30 seconds on CPU) in the request/response cycle.
   This is a demo-scale pattern. **Before any real-traffic deployment**, this must move to
   a background job queue (e.g. `apscheduler` for simple periodic refresh, or Celery/RQ +
   Redis for on-demand jobs with proper task tracking) so a slow training run cannot block
   or time out an HTTP request. `apscheduler` is already a project dependency
   (`requirements.txt`) but is not yet wired into the API for this purpose.
2. **Rate limiting is in-memory, per-process.** Correct for a single uvicorn worker;
   incorrect the moment the app runs with multiple workers or multiple replicas (each
   would have its own independent counter). A production deployment needs a shared store
   (Redis) for the rate limiter.
3. **No authentication/authorization layer yet.** All endpoints are currently open. Build
   brief Section 31 calls for authentication "where appropriate" — for a public read-only
   dashboard API this may be acceptable, but the `/refresh` endpoint (which triggers real
   compute) should be protected before any public deployment (API key header, at minimum).
4. **SQLite is not safe for concurrent multi-process writes.** Fine for a single-process
   demo; a real deployment should switch `DATABASE_URL` to Postgres (no code change
   required — see `db/session.py`) before scaling beyond one process.
5. **No HTTPS/TLS termination configured** — expected to be handled by a reverse proxy
   (nginx, a cloud load balancer) in front of uvicorn, not by the application itself.

## Docker

- `Dockerfile` (repo root): backend image, `python:3.12-slim`, CPU-only torch install,
  editable install of `emafrs`, non-root user, healthcheck against `/health`.
- `frontend/Dockerfile`: multi-stage — `node:22-slim` build, served by `nginx:1.27-alpine`
  with `frontend/nginx.conf` (SPA fallback routing, cache headers, the same security
  headers the API sets).
- `docker-compose.yml`: wires both together, backend on `:8000`, frontend on `:8080`,
  named volumes for `data/` (SQLite + cache) and `models/checkpoints/`, API keys passed
  through from the host environment (never baked into the image).

**Disclosed limitation**: the Docker Desktop daemon was not running in this development
environment (`docker ps` failed to connect), so these Dockerfiles/compose file have **not
been verified with an actual `docker build`/`docker compose up` run**. They were written
carefully and reviewed against the same dependency list and startup command already
verified to work directly (`uvicorn emafrs.api.app:app`, `npm run build`), but "written
carefully" is not the same claim as "verified" — per this project's no-fabrication rule,
this gap is stated explicitly rather than silently assumed to work. Verifying this
(`docker compose up --build` and confirming both healthchecks pass) is the natural
next step once Docker Desktop is available.

## Recommended hardening checklist before any real deployment

- [ ] Move `/refresh` to a background job queue with a job-status endpoint
- [ ] Add API-key or OAuth2 authentication to write/compute endpoints
- [ ] Move rate limiting to a shared store (Redis) if running >1 worker/replica
- [ ] Switch `DATABASE_URL` to Postgres; run Alembic migrations (dependency already
      present, migration scripts not yet written)
- [ ] Pin exact dependency versions (`requirements.txt` currently uses `>=`, appropriate
      for a research reconstruction, not for a release artifact — see
      `docs/REPRODUCIBILITY.md`)
- [ ] Pin Hugging Face model revisions (`revision=<commit_sha>`) instead of tracking the
      latest checkpoint under a model ID
- [ ] Add structured JSON logging + a log aggregator if deployed beyond a single host
- [ ] TLS termination via a reverse proxy
