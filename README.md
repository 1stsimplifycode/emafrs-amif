# EMAFRS / AMIF — Research Reconstruction

**Explainable Multimodal Affective Financial Reasoning System** (EMAFRS), also referred to in the
source specification as the **AI Market Integrity Framework** (AMIF).

## What this is

This repository is a from-scratch reconstruction of a system previously presented in
[`PROJECT DOCUMENT.pdf`](./PROJECT%20DOCUMENT.pdf) (an Indian patent Complete Specification). The
original implementation was lost; nothing here is copied from a prior codebase. See
[`docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md`](docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md) for the
line-by-line mapping from the specification to this implementation, including the handful of
places where the spec's formulas needed a disclosed correction to be implementable, and
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for what genuinely cannot be built with free-tier data
(true tick-level order book data, live audio/video modeling at pipeline scale).

## Research question

> When machine-learning models are accurate but opaque, can multimodal evidence and
> regime-adaptive fusion, combined with quantitative explanation evaluation, provide richer, more
> faithful, and more auditable decision evidence — without sacrificing predictive performance?

This is treated as an open empirical question, not a foregone conclusion. See
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) once populated.

## Architecture

```
Data Acquisition Layer (DAL)
        -> Preprocessing / Feature Engineering
        -> Affective Signal Extraction Module (ASEM)
        -> Market Regime Detection Engine (MRDE, HMM)
        -> Market Microstructure Intelligence Engine (MMIE, proxy signals)
        -> Regime-Adaptive Multimodal Attention (RAMA) / Volatility-Aware Fusion Engine (VAFE)
        -> Manipulation Risk Model
        -> Explainability (Integrated Gradients, attention, counterfactual)
        -> Decision Intelligence Output / Citizen Alerts
        -> Adaptive Integrity Controller
```

Full detail: [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md).

## Status

The research core is complete and validated against real data: Data Acquisition Layer, feature
engineering + leakage audit, regime detection, affective signal extraction + NLP benchmark,
microstructure proxies, RAMA/VAFE fusion model + baselines A-D, manipulation risk model, XAI layer,
and a full walk-forward/ablation/calibration experiment protocol. Production deployment (FastAPI
backend, dashboard, Docker) is the remaining phase — see `docs/PRODUCTION_DEPLOYMENT.md` once it
lands. Nothing in this repo should be read as a finished result beyond what the corresponding doc
under `docs/` states — every doc says "Not yet evaluated" for anything not yet run, per the
no-fabrication rule this project is built under, and several real *negative* findings are reported
exactly as measured (see `docs/EXPERIMENT_PROTOCOL.md`, `docs/XAI_EVALUATION.md`).

## Documentation index

| Doc | Covers |
|---|---|
| [`PDF_TO_IMPLEMENTATION_TRACEABILITY.md`](docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md) | Every PDF concept mapped to its implementation, dataset, and disclosed corrections |
| [`RECOVERY_AND_RECONSTRUCTION.md`](docs/RECOVERY_AND_RECONSTRUCTION.md) | What was lost, what this reconstruction is/isn't, method |
| [`SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md) | Module map, pipeline/RAMA diagrams |
| [`DATA_SOURCES_AND_APIS.md`](docs/DATA_SOURCES_AND_APIS.md) | Every data source: endpoints, limits, licensing, resilience behavior |
| [`API_CONFIGURATION.md`](docs/API_CONFIGURATION.md) | How to set up `.env`, what's actually configured here |
| [`DATASET.md`](docs/DATASET.md) | What data trains/evaluates each model |
| [`DATA_LEAKAGE_AUDIT.md`](docs/DATA_LEAKAGE_AUDIT.md) | Automated no-lookahead checks, proven to have teeth |
| [`EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) | Walk-forward validation, regime-wise/calibration results, significance tests |
| [`ABLATION_STUDY.md`](docs/ABLATION_STUDY.md) | All 11 build-brief ablation rows, honestly scored evaluated/not-yet-evaluated |
| [`XAI_EVALUATION.md`](docs/XAI_EVALUATION.md) | Faithfulness, stability, sparsity, counterfactual validity — quantified |
| [`NLP_MODEL_BENCHMARK.md`](docs/NLP_MODEL_BENCHMARK.md) | Generic vs. domain-adapted transformer sentiment comparison |
| [`MODEL_CARD.md`](docs/MODEL_CARD.md) | One card per model: architecture, training, real metrics, limitations |
| [`LIMITATIONS.md`](docs/LIMITATIONS.md) | Everything that could not be honestly built, and why |
| [`REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Environment, seeds, exact commands to reproduce every result |

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env       # fill in whichever API keys you have
```

Compute target for this reconstruction is **CPU-only** — model choices, fine-tuning schedules, and
benchmarking are all sized accordingly (see `docs/MODEL_CARD.md` and `configs/nlp_models.yaml`
once populated).

### Running the full stack

```bash
# Backend
pip install -e .
uvicorn emafrs.api.app:app --reload   # http://localhost:8000

# Frontend (separate terminal)
cd frontend
npm install
cp .env.example .env                  # points at the backend above by default
npm run dev                            # http://localhost:5173
```

Or via Docker Compose (backend on :8000, dashboard on :8080) — see
[`docs/PRODUCTION_DEPLOYMENT.md`](docs/PRODUCTION_DEPLOYMENT.md) for the disclosed caveat
that the Docker build itself has not been verified in this development environment:

```bash
docker compose up --build
```

### Helper MCP server (dev aid, not part of the research system)

```bash
python -m emafrs.mcp_server
```

Exposes the DAL (yfinance, Alpha Vantage, Finnhub, NewsAPI, YouTube, FRED, SEC EDGAR) plus
project-inspection tools (cache stats, docs listing, git status, test runner) as MCP tools
over stdio, for use from any MCP client while building/debugging this repo. Point your MCP
client config at `python -m emafrs.mcp_server` run from the repo root with `.venv` active.

## Repository layout

```
src/emafrs/
  data/            data acquisition clients (yfinance, Alpha Vantage, Finnhub, NewsAPI, YouTube, FRED, SEC EDGAR)
  features/        technical indicators, preprocessing, leakage-safe transforms
  regime/          Market Regime Detection Engine (HMM)
  nlp/             Affective Signal Extraction Module, transformer benchmark
  microstructure/  Market Microstructure Intelligence Engine (documented proxies)
  models/          RAMA/VAFE fusion network, baseline models A-E, calibration
  xai/             Integrated Gradients, attention explanation, counterfactuals, NLG
  risk/            manipulation risk scoring
  alerts/          citizen alert generation and delivery
  api/             FastAPI backend
  db/              persistence layer
  mlops/           training/eval pipelines, drift monitoring
  mcp_server/      helper MCP server (data access + project inspection tools; a build aid, not a research claim)
configs/           model registries, hyperparameters
docs/              all required project documentation
tests/             unit / integration / e2e
frontend/          React + TypeScript + Vite dashboard (real components, no mocked data — see docs/PRODUCTION_DEPLOYMENT.md)
scripts/           one-off / CLI utilities
```
