# Dataset

## Market data (financial modality)

- **Source**: yfinance (`emafrs.data.yfinance_client`), no API key required.
- **Instruments used in experiments so far**: SPY (primary), AAPL, MSFT (smoke tests).
- **Resolution**: daily OHLCV for the main dataset (walk-forward experiments used ~10
  years, 2016-08-10 to 2026-08-10, 2,513 trading days for SPY); 1-minute intraday for the
  microstructure proxies (capped at the trailing 7 days by the provider, ~2,500 bars).
- **Features derived**: returns, realized volatility, max drawdown, SMA10/50, EMA12/26,
  RSI14, MACD (+signal +histogram), Bollinger Bands (mid/upper/lower/%B), rolling
  historical VaR/CVaR (95%), volume z-score — see `emafrs.features.technical` and
  `emafrs.features.risk_metrics`. All rolling/causal by construction; verified via
  `emafrs.features.leakage_audit.assert_no_lookahead` in `tests/unit/test_technical_features.py`
  and `tests/unit/test_risk_metrics.py`.

## Labels

- **Target**: ternary market direction `{-1 (bearish), 0 (neutral), +1 (bullish)}`, from
  the next-day (`horizon=1`) forward return with a disclosed fixed deadzone of 0.25%
  (`emafrs.models.labels.build_ternary_labels`) — not empirically tuned, stated as a
  design choice.
- **Class balance** (SPY, 5y window used in the single-split experiments): roughly
  balanced across the three classes (e.g. train split: 298 bullish / 256 bearish / 192
  neutral out of 746 rows) — not a degenerate majority-class problem.

## Sentiment benchmark dataset

- **Financial PhraseBank** (Malo et al. 2014, JASIST), `sentences_66agree` config: ~66%
  of the full dataset where at least 66% of annotators agreed on the sentiment label.
  Real, human-annotated financial news sentences. Loaded via
  `huggingface_hub.hf_hub_download` (not the `datasets` library — see
  `docs/NLP_MODEL_BENCHMARK.md` and `docs/LIMITATIONS.md` for why) and parsed directly
  from the original release's zip/text format.
- **Sample used**: a seeded random sample of 240 sentences (not a file-order prefix — see
  `docs/NLP_MODEL_BENCHMARK.md`'s "methodological issue found and fixed" section for why
  that distinction mattered in practice).

## Regime detection inputs

- Derived entirely from the market data above: 21-day realized volatility, 21-day Shannon
  entropy of returns, 126-day Hurst exponent (R/S analysis with a DFA cross-check),
  21-day volume z-score. See `emafrs.regime.hmm_regime.build_regime_features`.

## Microstructure (MMIE) inputs

- 1-minute OHLCV bars (yfinance intraday, 7-day trailing cap) — see
  `docs/LIMITATIONS.md` for why this severely limits how this modality can currently be
  evaluated (no walk-forward-able history).

## Text / affective data

- **Not yet populated in any trained dataset.** `emafrs.nlp.asem` is fully implemented
  and validated in isolation against real text (see `tests/integration/test_asem_pipeline.py`
  and `docs/NLP_MODEL_BENCHMARK.md`), but no NewsAPI/YouTube API key was configured during
  this reconstruction, so no real news/communication text has been pulled into a training
  dataset. See `docs/LIMITATIONS.md` and `docs/ABLATION_STUDY.md`.

## No synthetic/fabricated data in any trained model or reported metric

Every dataset that feeds a reported number in this project's docs is real: live yfinance
data, the real Financial PhraseBank release, or real transformer model outputs on real
input text. Synthetic data appears ONLY in unit tests that are explicitly testing a
mechanism's mathematical behavior (e.g. `_synthetic_ohlcv()` in
`tests/unit/test_microstructure_proxies.py`, used to verify a formula's correctness
against known ground truth) — never as a substitute for real data in an experiment result.

## Splits

- **Single train/val/test split** (70/15/15, chronological): used for the primary
  multimodal (financial+microstructure) RAMA-vs-static comparison and all XAI/calibration
  evaluation. See `emafrs.models.dataset.build_daily_dataset`.
- **Walk-forward** (5 expanding-window folds, chronological, non-overlapping test
  windows): used for the financial-only Model A/B/NN comparison. See
  `emafrs.experiments.walk_forward` and `docs/EXPERIMENT_PROTOCOL.md`.
- Every split is verified via `emafrs.features.leakage_audit.check_chronological_split` in
  the test suite, not merely assumed correct because the code "looks" chronological.
