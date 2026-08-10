# Production Deployment

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

**Not yet built as of this document.** Tracked under task "Dashboard frontend + Docker +
tests." When it lands, this section will describe the actual `Dockerfile`/
`docker-compose.yml`, not a speculative one written in advance.

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
