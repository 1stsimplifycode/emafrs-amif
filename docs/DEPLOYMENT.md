# Public deployment (Render backend + Vercel frontend)

This is the "live public URL" deployment path, separate from `docker compose up`
(local) and `run.bat` (Windows one-click local demo). It splits across two free hosts
because the backend is a real Python/PyTorch service (10-30s synchronous model training
per prediction, needs a persistent process) that does not fit Vercel's serverless
function model (short execution limits, ephemeral filesystem, ~250MB deployment size
cap on the Hobby tier).

## Backend: Render (`render.yaml`)

`render.yaml` at the repo root is a Render Blueprint — it describes the web service
(Docker build from the existing `Dockerfile`, port 8000, health check at `/health`) so
deployment is "connect the repo, click deploy" rather than manual dashboard
configuration.

**Manual steps (require your own Render account — cannot be done on your behalf):**
1. Sign up / log in at https://render.com (GitHub OAuth is the fastest path, and this
   repo is already public at the pushed GitHub URL).
2. New → Blueprint → select the `emafrs-amif` repo → Render detects `render.yaml`.
3. Fill in the `sync: false` API keys (`YOUTUBE_API_KEY`, `NEWS_API_KEY`,
   `ALPHAVANTAGE_API_KEY`, `FINNHUB_API_KEY`, `FRED_API_KEY`) if you want live data
   sources reachable in the deployed demo — every source is optional; missing ones are
   reported as "not configured" by the health panel, never faked.
4. Deploy. First build installs the full CPU dependency set (torch, transformers,
   xgboost, etc.) and will take several minutes.
5. Once live, note the assigned URL (`https://emafrs-backend.onrender.com` unless that
   name was taken, in which case Render appends a suffix — check the dashboard).

**Known, disclosed limitations of this specific free-tier setup:**
- **SQLite is NOT persistent across restarts/redeploys on Render's free tier** (no
  persistent disk without a paid plan) — prediction history resets whenever the service
  restarts. Fine for a demo; not a production data-durability guarantee. A paid Render
  disk, or migrating `DATABASE_URL` to a managed Postgres instance, would fix this.
- **Free-tier services spin down after 15 minutes of inactivity** and take ~30-60s to
  wake on the next request — the first request after idle time will be slow, not broken.
- Update `CORS_ORIGINS` in `render.yaml` (or the Render dashboard) to the real Vercel
  URL once the frontend is deployed (see below), then redeploy — otherwise the browser
  will block the frontend's requests to this backend.

## Frontend: Vercel

Deployed from the `frontend/` subdirectory (Vercel auto-detects the Vite/React
framework preset). Requires `VITE_API_BASE_URL` set to the Render backend's URL from
above — this is baked in at build time (Vite convention), so it must be set as a Vercel
project environment variable before building, not just in `frontend/.env` (which is
gitignored and never deployed).

```bash
cd frontend
npx vercel login        # opens a browser device-code flow, like the GitHub CLI login used for this repo
npx vercel link         # links this directory to a new/existing Vercel project
npx vercel env add VITE_API_BASE_URL production   # paste the Render URL when prompted
npx vercel --prod
```

## What "explainable" means in this deployed dashboard

Every prediction card includes a "How much should I trust this?" panel
(`frontend/src/components/TrustworthinessPanel.tsx`) surfacing this project's own real,
measured findings — explanation faithfulness, calibration status, walk-forward accuracy,
and RAMA's actual (small) regime-adaptivity effect size — next to the numbers they
qualify, sourced from `docs/TRUSTWORTHINESS_AUDIT.md`. This is deliberate: showing
attributions and confidence scores without that context would not be genuinely
explainable, just more numbers.
