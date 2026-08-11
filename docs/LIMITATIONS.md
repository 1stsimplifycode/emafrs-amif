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
- **YouTube transcript fetching is IP-blocked by an undocumented anti-bot heuristic** in
  `youtube-transcript-api`'s underlying (unofficial) endpoint. A real pilot run
  (`docs/TEXT_MODALITY_PILOT.md`) saw 23 of 35 transcript requests blocked with "YouTube
  is blocking requests from your IP," and a much smaller follow-up (a single video)
  triggered the same block shortly after -- this appears to be a sustained IP-level
  restriction on this development environment's network, not simply a per-burst rate
  limit. A courtesy delay between requests (`transcript_request_delay_seconds`) was added
  but does not eliminate it. Practical consequence: real text-modality datasets built from
  this environment will have a much lower video yield than videos found, and this must be
  accounted for (via the `text_available` mask, already handled) rather than assumed away.
- **Text modality is technically wired end-to-end (YouTube + NewsAPI, unified schema) but
  every real run to date has been data-starved.** The YouTube-only pilot got 2 of ~160
  test-set rows with real text signal; a combined YouTube+NewsAPI pilot got **0 of any**
  split row (see `docs/TEXT_MODALITY_PILOT.md` for both breakdowns and a third,
  remediation-pass follow-up run). The zero-availability result had two compounding,
  precisely diagnosed causes: (1) NewsAPI articles published on non-trading days
  (weekends) never aligned to any row in the OHLCV trading calendar and were silently
  dropped by the join, and (2) the leakage-safe rolling z-score warm-up
  (`min_periods=window`) requires more prior text-bearing days than a short pilot
  accumulates. **Cause (1) has since been fixed and verified**: articles now map
  forward-only (never backward — strictly causal) to the next real trading day
  (`aggregate_daily_text_features(..., trading_days=...)`), confirmed on a real re-run to
  correctly merge weekend articles into the following Monday. **The 0-usable-rows outcome
  is unchanged**, because cause (2) alone is now sufficient to block it: NewsAPI's ~29-day
  lookback cap limits a single collection run to only 1-2 unique trading days, and the
  z-score warm-up (`min_periods=3`) mathematically cannot produce a non-NaN value from
  fewer than 3 observations, regardless of how correctly those observations were
  computed. This is now precisely characterized as a genuine additional-data-required
  limitation (needs real elapsed time across multiple collection runs to accumulate 3+
  distinct trading days, or resolving YouTube's IP-blocking to add a second source), not
  a remaining implementation bug.
  **Update: text modality became usable for the first time in a real run, for
  company-specific tickers.** Switching from broad macro queries to publisher-targeted
  (CNBC/Bloomberg via real `domains`/channel-ID filtering) queries **for the specific
  ticker being predicted** gave MSFT enough real, distinct daily NewsAPI coverage to
  clear the z-score warm-up — `modalities=['financial', 'microstructure', 'text']` in a
  real, reproduced run (opt-in via `include_text=True`, off by default; see
  `docs/TEXT_MODALITY_PILOT.md` "Follow-up 3" for the full account, including the honest
  caveats: `signal_consistency` was ambiguous, RAMA's fusion weights stayed near-equal
  across modalities, and this has not yet been checked across many tickers/days to know
  how often it succeeds vs. still lands at zero). All EARLIER 3-modality training results
  in this document remain reported as uninformative; this new result is real, positive,
  and separately caveated, not blended into the earlier null findings.

## Data sources

- Only 3 sources are wired in beyond the core financial/news set (FRED, SEC EDGAR) plus
  YouTube — World Bank, IMF, GDELT, and Stooq are not integrated. Per build brief Section
  12 ("data quality > API quantity"), breadth was intentionally not prioritized over
  provenance-tracked reliability of the sources that are integrated.
- **NewsAPI key resolved.** The first three keys provided were rejected with `401
  apiKeyInvalid` (verified via raw HTTP request bypassing all project code); root cause
  turned out to be key FORMAT, not the account -- those three were UUID-shaped
  (8-4-4-4-12 with dashes), while a real NewsAPI key is a 32-character hex string with no
  dashes. A correctly-formatted key was supplied and verified working end-to-end through
  `emafrs/data/newsapi_client.py` against live articles. News-derived text is now wired
  into `emafrs/nlp/text_dataset.py` alongside YouTube under one unified schema
  (`fetch_news_affective_records`, source-agnostic aggregation proven by
  `test_aggregation_is_source_agnostic_by_design`); see `docs/TEXT_MODALITY_PILOT.md` for
  the real combined-pilot run and why it was still uninformative about text modality
  value.
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
