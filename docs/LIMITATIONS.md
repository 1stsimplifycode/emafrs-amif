# Limitations

This document is updated as each phase lands, per the project's no-fabrication rule: a
capability that cannot be honestly implemented is documented here rather than faked.

## Market microstructure (MMIE)

No free/research-tier data source available to this project provides true Level-2/Level-3
order book depth, bid/ask quotes, or trade-by-trade tick data (see
`docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` row #4 and `docs/DATA_SOURCES_AND_APIS.md`).
Consequently:

- **OBI (Order Book Imbalance), OFI (Order Flow Imbalance), LSI (Liquidity Shock), VPIN**
  are all implemented in `src/emafrs/microstructure/proxies.py` as documented PROXIES
  derived from OHLCV bars (1-minute where available via yfinance/Alpha Vantage), not real
  order-book measurements. Every function's docstring states the specific approximation.
  Every output is named with a `_proxy` suffix so it cannot be silently mistaken for the
  real signal downstream (dashboard, XAI, model cards).
- **Depth Depletion Rate** (PDF Eq. 7) has **no implementation at all** — there is no
  defensible way to approximate per-price-level quoted depth from OHLCV bars, so
  `depth_depletion_rate()` raises `NotImplementedError` rather than returning a fabricated
  number. This is a genuine capability gap, not an oversight.
- Real 1-minute bar history from yfinance is capped at a trailing 7 days by the provider
  regardless of account tier — the MMIE's intraday proxies can only be computed over that
  recent window, not historically over the multi-year windows used for regime detection.

**Implication for research claims**: any manipulation-risk or microstructure finding in this
project must be described as based on OHLCV-bar proxies, never presented as validated
against real order-book dynamics, until/unless a Level-2 data source is added.

## Affective Signal Extraction (ASEM)

- **Emotion detection is zero-shot, not fine-tuned.** No labeled dataset exists for the
  PDF's 7-category financial emotion taxonomy (confidence, uncertainty, optimism, fear,
  caution, assertiveness, evasiveness). `src/emafrs/nlp/emotion.py` uses NLI-based
  zero-shot classification instead, validated only by directional sanity checks (a
  fear-coded sentence scores higher on "fear" than a confidence-coded one), not by accuracy
  against ground truth. See `docs/NLP_MODEL_BENCHMARK.md`.
- **Influencer role stratification** is populated only where structural metadata actually
  supports it (SEC EDGAR filings, a manually-verified YouTube channel allowlist that is
  empty by default). Everything else — including all current NewsAPI/Finnhub article
  sources — is honestly labeled `unclassified` rather than guessed from a byline or
  publication name. Central-bank-official and government-policymaker communications are
  therefore essentially unreachable via the current DAL sources until a verified channel
  allowlist is populated or a dedicated official-communications source is added.
- **Credibility scoring's historical-impact component is unset** (`historical_component=None`
  in `CredibilityScore`) until a real backtest correlating a source's past communications
  with subsequent market moves exists. Currently credibility is authority-tier-only.
- **Audio/video modalities are not modeled in the live pipeline.** The YouTube client only
  extracts transcript text (tagged `modality="text_derived_from_av_source"`); true
  audio/video affect features (pitch, prosody, facial expression) would require downloading
  and processing actual media files, which is out of scope for the live/production pipeline
  and, if attempted at all, would only cover a small hand-picked case-study set.

## Data sources

- Only 3 sources are wired in beyond the core financial/news set (FRED, SEC EDGAR) plus
  YouTube — World Bank, IMF, GDELT, and Stooq are not integrated. Per build brief Section
  12 ("data quality > API quantity"), breadth was intentionally not prioritized over
  provenance-tracked reliability of the sources that are integrated.
- SEC EDGAR enforces strict per-IP rate limiting / bot-defense on this environment; a live
  403 was observed during development (see `docs/DATA_SOURCES_AND_APIS.md`). The DAL
  surfaces this as `ProviderUnavailableError` rather than masking it, but it means SEC EDGAR
  availability in this environment cannot be assumed reliable.

## Environment

- The `datasets` (Hugging Face) Python package is **unusable on this development machine**:
  it unconditionally imports `pyarrow.dataset` at module load time, and that submodule is
  blocked by Windows Application Control policy here (verified: `pyarrow.parquet` imports
  fine, `pyarrow.dataset` does not). Any future use of an HF `datasets`-hosted resource must
  go through `huggingface_hub.hf_hub_download` and manual parsing instead, as
  `emafrs/nlp/sentiment_benchmark.py` already does for Financial PhraseBank.
- This reconstruction targets **CPU-only** compute (no GPU available). Model choices
  throughout (FinBERT-scale, DeBERTa-xsmall, MiniLM) are selected for CPU tractability;
  larger models from the build brief's suggested sweep (full DeBERTa-v3-large, bart-large-mnli)
  are not used. See `docs/NLP_MODEL_BENCHMARK.md`.

## Not yet evaluated

Items that will be added here as later phases land: project-specific fine-tuning of any
NLP model, RAMA vs. static-fusion ablation results, manipulation-risk model validation
against real historical pump-and-dump cases, walk-forward experiment results, and
production deployment hardening.
