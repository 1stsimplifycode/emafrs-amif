# Data Leakage Audit

Implementation: `src/emafrs/features/leakage_audit.py`. This is not a prose checklist —
every claim below is backed by an automated test in `tests/unit/test_leakage_audit.py` that
is run as part of the normal test suite (`pytest`).

## Method: no-lookahead via truncation

`assert_no_lookahead(feature_fn, df)` recomputes a feature on a truncated history
`df.iloc[:t+1]` for several sampled cutoffs `t` and compares the value at `t` to the value
computed on the full series. If a feature secretly depends on rows after `t` (a centered
window, a negative `.shift()`, a full-sample normalization), the truncated and full
computations diverge and the check fails.

**This check has been proven to have teeth, not just to pass trivially:**
`test_centered_rolling_mean_fails_no_lookahead_check` and
`test_shift_negative_fails_no_lookahead_check` construct deliberately leaky feature
functions and assert the audit catches them. A leakage check that only ever reports "pass"
is worthless; these two tests are the evidence it isn't.

## What is audited

| Risk (build brief Section 25) | Mechanism | Where checked |
|---|---|---|
| Future feature values / lookahead in rolling stats | `assert_no_lookahead` truncation test | Every function in `features/technical.py` (returns, SMA, EMA, RSI, MACD, Bollinger %B, realized volatility, max drawdown) — `test_real_technical_features_pass_no_lookahead_check`, parametrized over all 8 |
| Future normalization (full-sample z-score leaking future distribution) | Same truncation test applied to `preprocessing.rolling_zscore` | `test_rolling_zscore_uses_only_past` |
| Future news / communication leakage | `check_news_before_prediction_time` — every record's `publishedAt` must be `<= prediction_time`; unparseable timestamps fail closed | `test_news_before_prediction_time_*` |
| Random/shuffled train-test splitting | `check_chronological_split` — every test timestamp must be strictly later than every train timestamp, with zero date overlap | `test_chronological_split_detects_overlap_and_order_violation` |
| Timestamp misalignment (ffill introducing future values) | `preprocessing.align_to_business_calendar` uses `.ffill()` only (never `.bfill()`); tested directly | `test_align_to_business_calendar_forward_fills_gaps_only_from_past` |
| Target leakage (label computed from data also used as a feature at the same timestamp) | Not yet applicable — no label construction exists yet (Phase: RAMA/baselines). Will be audited via the same truncation method once `src/emafrs/models/` defines `y_t`. | **Not yet evaluated** |
| Survivorship bias | Not yet applicable — no fixed universe/backtest exists yet. Will be addressed by pulling tickers from the point-in-time index membership where feasible, or explicitly documenting the limitation, when the experiment protocol (`docs/EXPERIMENT_PROTOCOL.md`) is written. | **Not yet evaluated** |

## Design decisions that prevent leakage by construction

- **Rolling, not full-sample, z-scoring.** `rolling_zscore(series, window)` computes mean/std
  over `[t-window+1, t]` only. An earlier design that standardized against the full series'
  mean/std would leak the entire sample's future distribution into every early observation;
  this was avoided from the start rather than fixed after the fact.
- **`min_periods=window`, not `min_periods=1`**, for most indicators — a partially-filled
  window is left as `NaN` rather than silently computed over fewer points, which would
  otherwise make early-period feature scale inconsistent with the rest of the series (a
  subtler distributional-leakage-adjacent bug, not classic look-ahead, but still a validity
  risk for model training).
- **News timestamp is provider-reported `publishedAt`, never ingestion time.** The DAL
  (`docs/DATA_SOURCES_AND_APIS.md`) preserves this verbatim so the leakage check has a
  timestamp to audit against; NewsAPI's free tier itself introduces an ingestion delay,
  which if anything makes the system *more* conservative (data becomes available to us
  later than `publishedAt`, never earlier).

## Not yet evaluated

Per the project's no-fabrication rule: target leakage and survivorship bias checks are
listed above as **not yet evaluated** because the artifacts they'd check (label
construction, a fixed backtest universe) don't exist yet. This section will be updated,
not silently left stale, once `src/emafrs/models/` and `docs/EXPERIMENT_PROTOCOL.md` land.
