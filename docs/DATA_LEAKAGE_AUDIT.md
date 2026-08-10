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
| Target leakage (label computed from data also used as a feature at the same timestamp) | `emafrs.models.labels.build_ternary_labels` uses `close.shift(-horizon)` (strictly future) to build `y_t`; features are built only from data at or before `t`. Verified directly via the same truncation methodology: `build_daily_dataset_from_ohlcv(df.iloc[:cutoff], ...)` vs. the full series produces byte-identical labels AND byte-identical financial/technical features on rows safely before the cutoff (max abs diff 0.0 for both, checked against a synthetic 500-row series). **Evaluated: no target leakage found in the label/feature construction itself.** | **Evaluated** |
| Regime-probability leakage (a leakage pathway this project did not originally anticipate, found during the trustworthiness audit) | The SAME truncation check applied to the full assembled feature set revealed the regime-probability columns (`regime_p1..p4`) were NOT truncation-invariant, unlike every other feature. Root cause: `RegimeModel.predict_proba` used hmmlearn's SMOOTHED (forward-backward) posterior over the entire train+val+test sequence in one call, so a date's regime probability could in principle depend on later dates, even though the HMM's fitted parameters were correctly train-only. Measured to be numerically negligible on real SPY data (<1e-5 beyond 1-2 rows from a sequence boundary, because this fitted HMM produces near-deterministic regime assignments) once the confound of "different train/test split changing the fitted model" was controlled for. **Corrected regardless**: `RegimeModel.predict_proba_causal` (forward-filtering only, no backward pass) is now used by `dataset.py`, proven exactly truncation-invariant in `tests/unit/test_hmm_regime.py::test_predict_proba_causal_is_truncation_invariant`. Full writeup: `docs/EXPERIMENT_PROTOCOL.md`'s "Regime-probability leakage" section. | **Evaluated and corrected** |
| Survivorship bias | Not applicable in this project's current single-fixed-ticker configuration (SPY only, chosen ahead of time and never dropped for underperforming) — survivorship bias in the classic sense requires a multi-ticker universe backtest that excludes delisted/failed constituents, which does not exist here. Would need to be revisited if a multi-ticker universe backtest is ever added. | **N/A for current scope, not a forever-pending item** |

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

- A dedicated regression test proving the FULL assembled feature pipeline (not just
  individual technical-indicator functions) is truncation-invariant for every feature
  group at once, in one place — the target-leakage and regime-probability checks above
  were done as ad-hoc diagnostics during this audit, not yet promoted into a single
  reusable `assert_no_lookahead`-style test covering `build_daily_dataset_from_ohlcv`'s
  full output. The individual pieces (technical features, regime probabilities via
  `predict_proba_causal`, labels) are each proven safe on their own; a combined
  end-to-end regression test would catch future regressions in how they're joined.

## Revision note

This document previously listed target leakage and survivorship bias as "Not yet
evaluated," blocked on artifacts (label construction, a fixed backtest universe) that had
in fact already landed in `src/emafrs/models/` and `docs/EXPERIMENT_PROTOCOL.md` well
before this note was written — the promised follow-up never happened until this audit
revisited it. Both are now resolved above (target leakage: evaluated, none found;
survivorship bias: correctly scoped as N/A for the current single-ticker configuration).
A previously-unknown THIRD leakage risk (regime-probability smoothing) was found in the
process of actually running the check this document had promised, rather than assuming
the original design intent (`fit_regime_hmm`'s docstring already correctly described
train-only fitting) was sufficient on its own — it was necessary but not sufficient, since
the leakage lived in the APPLICATION of the fitted model, not its fitting.
