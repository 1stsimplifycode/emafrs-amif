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

Early scaffolding. See open tasks / phased roadmap in project tracking. Nothing in this repo
should be read as a finished result until the corresponding doc under `docs/` says so —
`docs/*.md` state "Not yet evaluated" for anything not yet run, per the no-fabrication rule this
project is built under.

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
frontend/          dashboard
scripts/           one-off / CLI utilities
```
