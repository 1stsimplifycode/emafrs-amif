# Experiment Protocol

Every number in this document comes from an actual run of the scripts named in each
section (`scripts/run_walk_forward_experiment.py`,
`scripts/run_regime_calibration_xai_eval.py`, `scripts/run_baseline_comparison.py`). Where
a claim in the build brief has not been run, this document says "Not yet evaluated"
explicitly rather than omitting it.

## Method

- **Chronological, never shuffled, splits.** Every split in this project (single train/
  val/test and walk-forward folds) is verified via
  `emafrs.features.leakage_audit.check_chronological_split` — see
  `tests/unit/test_walk_forward.py` and `tests/integration/test_dataset_and_models.py`.
- **Walk-forward validation**: `emafrs.experiments.walk_forward.generate_walk_forward_folds`
  produces expanding-window folds (training data grows with each fold; test windows are
  non-overlapping and strictly increasing in time).
- **Regime model fit only on each fold's training window**, never on validation/test dates
  — see `dataset.py`'s `explicit_split` parameter and the leakage-boundary regression test.
- **Multiple seeds** for every neural (RAMA/static fusion) result: 3 seeds per fold/config.
- **Statistical significance**: paired Wilcoxon signed-rank test and paired t-test across
  folds (`emafrs.experiments.significance.compare_paired_results`), 95% confidence
  intervals via the t-distribution (appropriate for n=5 folds, not assumed-normal z-CIs).

## Scope limitation, disclosed up front

A genuine multi-fold walk-forward study needs every modality under comparison to be
available across ALL folds. In this reconstruction:
- **Financial** features: available for the full multi-year history — walk-forward-able.
- **Microstructure** (MMIE) proxies: yfinance only exposes 1-minute bars for the trailing
  7 days regardless of account tier (`docs/LIMITATIONS.md`) — genuinely cannot be walked
  forward across historical folds without fabricating "as of" availability for dates that
  never had real intraday data.
- **Text** (ASEM): not populated at all (no NewsAPI/YouTube key configured as of this
  experiment run).

Consequently: **walk-forward validation (below) uses financial-only features** across
Model A, B, and a single-modality neural baseline. **The multimodal RAMA-vs-static
comparison remains the single train/val/test split** from task #8/#9 (financial +
microstructure, ~5-7 real trading days of microstructure coverage). This is disclosed
here, not glossed over — a full multi-fold multimodal study is listed under "Not yet
evaluated" below, pending either a real intraday data source with deeper history or enough
accumulated live-pipeline history.

## Walk-forward results (real run, SPY, 2016-08-10 to 2026-08-10, 5 expanding-window folds)

Reproduce: `python scripts/run_walk_forward_experiment.py`

| Model | n_folds | Mean Accuracy | 95% CI | Per-fold accuracy |
|---|---:|---:|---|---|
| A: opaque XGBoost | 5 | 0.384 | [0.349, 0.419] | [0.363, 0.355, 0.382, 0.426, 0.393] |
| B: explainable logistic regression | 5 | 0.390 | [0.374, 0.406] | [0.371, 0.402, 0.386, 0.390, 0.401] |
| NN (single-modality fusion, 3 seeds/fold) | 5 | 0.402 | [0.379, 0.426] | [0.402, 0.400, 0.373, 0.424, 0.413] |

**Paired significance** (Wilcoxon signed-rank / paired t-test across the 5 folds):

| Comparison | Mean diff | Wilcoxon p | Paired-t p | Significant (α=0.05) |
|---|---:|---:|---:|---|
| A vs B | −0.006 | n/a (< 6 samples) | 0.656 | No |
| A vs NN | −0.019 | n/a (< 6 samples) | 0.164 | No |
| B vs NN | −0.012 | n/a (< 6 samples) | 0.256 | No |

**Finding, stated plainly**: none of the three models statistically significantly
outperforms the others at 1-day-ahead SPY direction prediction from financial/technical
features alone (all overlapping 95% CIs, all p > 0.16). Accuracy hovers around 38-40%
across a 3-class (bearish/neutral/bullish) problem where a class-frequency baseline would
land somewhere near 33-38% depending on the fold's label balance — i.e., this task is hard
and none of these models meaningfully solve it from financial features alone at a 1-day
horizon. This is consistent with market-efficiency expectations and is reported as a
genuine negative/null result, not adjusted or explained away. Wilcoxon could not be
computed (needs ≥6 paired samples; only 5 folds were run) — the paired t-test is the only
significance test available at this sample size, itself a limitation of only running 5 folds.

## Regime-wise performance, calibration, and XAI evaluation (real run, SPY, single split)

Reproduce: `python scripts/run_regime_calibration_xai_eval.py`. Model: RAMA
(financial+microstructure), single split. **Reproducibility caveat**: this script builds
its dataset from a "trailing 5 years from today" window, so re-running it on a later date
shifts the train/val/test boundary dates slightly and changes exact numbers run-to-run
even with zero code changes — expect small (not zero) drift when reproducing; the model
architecture, training procedure, and qualitative findings below have been confirmed
stable across two real runs a day apart (test accuracy 0.425 → 0.406, same qualitative
pattern in every metric below). This non-reproducibility-by-construction of "single
trailing-window split" evaluations is itself tracked as a limitation — see
`docs/LIMITATIONS.md`.

### Regime-wise performance (most recent real run, using leakage-corrected causal regime probabilities -- see below)

| Regime | N | Accuracy | Macro F1 | Mean fusion weight (financial) |
|---|---:|---:|---:|---:|
| 1 (Low) | 27 | 0.407 | 0.193 | 0.502 |
| 2 (Medium) | 71 | 0.408 | 0.233 | 0.512 |
| 3 (High) | 45 | 0.422 | 0.273 | 0.509 |
| 4 (Extreme) | 17 | 0.471 | 0.306 | 0.518 |

(An earlier run had Regime 4 excluded for n<5; later runs' shifted windows have included
enough Regime-4 test rows to report it.) Accuracy rises mildly
with regime severity but F1 stays low throughout, and — consistent with the finding
already logged in `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` row #19, and now confirmed
across two independent real runs on different date windows — the mean financial-modality
fusion weight stays within a narrow 0.502-0.518 band across ALL FOUR observed regimes
(including the previously-excluded Extreme regime): RAMA's gate shows only a SMALL
absolute range of regime-dependent adjustment in this trained model, even though the
underlying mechanism is proven capable of much larger differentiation in isolation
(`tests/unit/test_fusion_models.py::test_rama_weights_differ_across_regimes`). Seeing the
same near-flat 0.50-0.52 band on a second, independently-trained run (different data
window, same seed=42) upgrades this from "one run's fluke" to a reproducible property of
how this model trains on this data — see the root-cause investigation below. **Important
precision, added during remediation**: "small absolute range" is not the same claim as
"no real effect" — a statistical significance test (below, after the root-cause
investigation) found this small range IS a real, non-random signal, not noise. Both facts
are true simultaneously and are reported together, not conflated.

**FAILURE → ROOT CAUSE → CORRECTION → RE-EVALUATE loop for the flat gate (real, not
hypothetical)**

*Hypothesis 1 — weight decay suppresses the gate.* `train.py`'s optimizer applied
`weight_decay=1e-4` to every parameter via vanilla Adam, including `RAMAFusionModel.gate`
— a small softmax-logit parameter initialized at zero. L2 decay continuously pulls such a
parameter back toward zero (== uniform weighting) every step, which could plausibly
suppress differentiation given the weak gradient signal from a noisy ~40%-accuracy task.

*Correction implemented*: `src/emafrs/models/train.py`'s optimizer now uses
per-parameter-group weight decay, excluding `gate`/`static_logits` from L2
regularization entirely (standard practice for gate/bias-like parameters).

*Re-evaluation*: re-ran the full pipeline with the fix. Result: **no measurable change** —
mean financial-modality weight per regime was 0.502/0.511/0.506/0.518, identical to 3
decimal places to the un-fixed version. **Hypothesis 1 is rejected.**

*Hypothesis 2 — early stopping cuts training short before the gate can differentiate.*
Direct inspection (bypassing the training function to log intermediate state) showed
training stops via early-stopping at epoch ~52 of a 150-epoch budget, with the gate having
moved only slightly off zero by then (regime-1 financial-modality weight ≈0.50 at that
point) — consistent with training simply not running long enough.

*Correction tested (diagnostic only, not adopted)*: forced 1000 epochs of unregularized
training on the training set alone (no early stopping, no validation-based model
selection) to see whether the gate would differentiate given unlimited optimization
pressure.

*Re-evaluation*: the gate DID move substantially off uniform (financial-modality weight
rose to a 0.85-0.90 range from a ≈0.50 start) — but **moved almost identically across all
4 regimes** (0.849, 0.903, 0.870, 0.850 — a ≤5-point band), while train loss collapsed to
0.096, indicating severe overfitting (this run has no early stopping and no
regularization signal left to prevent it). **Hypothesis 2 is also rejected as the primary
explanation**: more training does not produce regime-*specific* differentiation, only a
regime-*invariant* global shift in modality preference, achieved only by badly overfitting.

**Conclusion, stated plainly, not manufactured into a positive result**: neither
optimizer regularization nor training duration explains the flat gate. The best-supported
remaining explanation is that this real dataset's HMM-detected volatility regimes do not
carry strong differential information about which modality (financial vs. microstructure)
is more reliable in which regime — so a model trained with principled, leakage-safe early
stopping correctly declines to invent regime-specific modality preferences the data does
not genuinely support. RAMA behaving like static fusion on this specific
dataset/regime-detection combination is reported as a property of the data, not a bug in
the mechanism (which IS proven capable of differentiation given a synthetic dataset with a
genuine regime-modality relationship, per the passing unit test) and not fixable by the
two most obvious training-side interventions, both of which were actually tried. The
weight-decay exclusion is kept in `train.py` regardless (it is a defensible, harmless
practice on its own merits) but is explicitly not claimed to have fixed anything here.

**Statistical significance test (added during audit remediation) — an important
correction to the framing above.** "The absolute range is small" was previously reported
in a way that could be read as "there is no real effect." That is a distinct claim,
directly testable, and was tested: `scripts/run_rama_gate_significance_test.py` runs a
one-way ANOVA (financial-modality weight, grouped by regime) on the real test set, with
significance assessed via a 5000-shuffle permutation test (not a normality assumption,
given small/unequal per-regime sample sizes) rather than just eyeballing the range.

| | |
|---|---|
| Per-regime mean weight | Regime 1: 0.5019 (n=27), Regime 2: 0.5117 (n=71), Regime 3: 0.5086 (n=45), Regime 4: 0.5179 (n=17) |
| Between-regime variance (of the 4 means) | 0.0000335 |
| Mean within-regime variance (of per-row weight) | 0.00000098 |
| Between/within ratio | 34.0 |
| Observed F-statistic | 1644.6 |
| Permutation p-value (5000 shuffles) | **<0.0002** (0 of 5000 shuffles matched or exceeded the observed F) |

**Verdict: the regime-to-regime differences ARE statistically significant** — far more
extreme than random regime-label shuffling ever produced. This is possible despite a tiny
absolute weight range (0.502-0.518) because the within-regime variance is even more
extreme: the fusion weight is very nearly CONSTANT within a given regime (std as low as
0.0001), a direct consequence of RAMA's gate being a deterministic function of the regime
posterior alone (`regime_probs @ self.gate`) — so even a small between-regime mean
difference stands out sharply against that near-zero noise floor. **Correct, precise
conclusion, superseding the earlier "not meaningfully regime-adaptive" framing**: RAMA HAS
learned a real, reproducible (non-chance), regime-dependent adjustment to its fusion
weights on this data — but the SUBSTANTIVE size of that adjustment (a ~1.6 percentage
point range) is small relative to what the PDF's original design intent (or the
synthetic unit test demonstrating the mechanism's capability) would suggest. Statistical
significance and practical/substantive significance are different questions with
different answers here, and both are reported rather than collapsing them into one
verdict. This does not overturn the root-cause conclusion above (this dataset's regimes
do not carry LARGE differential modality-reliability information) — it refines it: they
carry a real but small amount.
practice on its own merits) but is explicitly not claimed to have fixed anything here.

### Calibration

**FAILURE → ROOT CAUSE → CORRECTION → RE-EVALUATE loop (real, not hypothetical)**

*Original finding (first run)*: regime-dependent temperature scaling made calibration
slightly WORSE on the test split (Brier 0.660→0.673, ECE 0.019→0.049). Hypothesized cause
at the time: a small per-regime validation sample (~160 validation rows split across 4
regimes, roughly 40 rows/regime) used to fit 4 independent temperatures via LBFGS, prone
to overfitting the validation split's own noise.

*Correction implemented*: `src/emafrs/models/calibration.py` gained two additional,
real, testable variants — `fit_global_temperature` (a single temperature shared across
all regimes, i.e. zero per-regime freedom) and `fit_regime_temperature_shrinkage`
(per-regime temperatures regularized via an L2 penalty toward a jointly-fit global
temperature, `shrinkage_weight=2.0`, a disclosed fixed hyperparameter not tuned against
test). If the overfitting hypothesis were correct, at least one of these two
lower-variance corrections should recover some of the lost calibration quality on test.

*Re-evaluation (real re-run, `scripts/run_regime_calibration_xai_eval.py`,
`docs/_regime_calibration_xai_results.md`)* — note this re-run used a data window one day
further forward than the original run (the "trailing 5 years from today" window shifts as
real time passes; see the reproducibility caveat below), so the baseline numbers differ
slightly from the original run (test accuracy 0.406 vs. 0.425) even though no model code
changed:

| Variant | Brier | ECE | Fitted temperature(s) |
|---|---:|---:|---|
| Uncalibrated | 0.660 | 0.031 | n/a |
| Naive per-regime (original) | 0.665 | 0.038 | [1.931, 0.866, 0.394, 0.465] |
| Global-only (1 free parameter) | 0.666 | 0.040 | [0.694 ×4] |
| Shrinkage per-regime (weight=2.0) | 0.667 | 0.042 | [0.675, 0.686, 0.668, 0.666] |

**Finding, stated plainly, correction did NOT work**: neither the shrinkage variant nor
even a single global temperature (the simplest possible calibrator, with no per-regime
freedom to overfit at all) recovered the uncalibrated ECE — all three variants make
calibration worse than doing nothing, and the ranking is uncalibrated > naive-per-regime >
global-only > shrinkage. This **rules out** the original "too many free parameters
overfitting a small per-regime sample" hypothesis: if that were the whole story, the
1-parameter global fit should have clearly outperformed the 4-parameter naive fit, and it
did not (0.040 vs. 0.038 ECE — global was marginally worse, not better).

**Updated root-cause hypothesis at the time**: the validation and test
windows are chronologically disjoint (val precedes test, per this project's leakage-safe
splitting) and potentially non-stationary in volatility/regime composition — post-hoc
temperature scaling assumes the confidence-accuracy relationship learned on validation
transfers to test, which chronological non-stationarity in financial time series can
violate regardless of how many temperature parameters are used.

**Second re-evaluation, after an independently-discovered regime-probability leakage fix
(see "Regime-probability leakage" below)**: the analysis above was run on regime
probabilities later found to carry a real (if numerically small on this dataset) leakage
pathway. After fixing that (switching to `RegimeModel.predict_proba_causal`, a forward-only
filter), the calibration comparison was re-run on the corrected regime assignments — and
the picture changed meaningfully:

| Variant | Brier | ECE | NLL | Fitted temperature(s) |
|---|---:|---:|---:|---|
| Uncalibrated | 0.660 | 0.031 | **1.091** | n/a |
| Naive per-regime | 0.754 | **0.123** | 2.384 | [**0.018**, 1.024, 0.506, 0.361] |
| Global-only (1 free parameter) | 0.666 | **0.027** | 1.106 | [0.694 ×4] |
| Shrinkage per-regime (weight=2.0) | 0.667 | 0.032 | 1.108 | [0.667, 0.685, 0.667, 0.661] |

**Multi-metric picture, reported in full rather than cherry-picking the metric that looks
best (added when NLL was included alongside Brier/ECE during remediation)**: global-only
improves ECE (0.027 < 0.031) but is actually slightly WORSE than doing nothing on BOTH
Brier (0.666 vs. 0.660) and NLL (1.106 vs. 1.091). Naive per-regime became dramatically
worse on every metric (ECE 0.123, NLL 2.384, driven by a degenerate near-zero temperature
of 0.018 fit for the smallest regime group, n=27) — consistent with the ORIGINAL
small-sample overfitting hypothesis, not the non-stationarity one. **Honest caveat on
attribution**: this comparison also used a data window one day later than the previous
run (regime group sizes shifted: 27/71/45/17 vs. the earlier 32/70/46/12), so the change
from the first calibration analysis cannot be attributed to the regime-probability fix
alone with full confidence. What IS a clean, controlled finding (not confounded by date
drift) is documented under "Regime-probability leakage" below.

**Current best-supported conclusion, precise rather than a single verdict**: no
calibration variant tested is unambiguously better than doing nothing — uncalibrated wins
on Brier and NLL, global-only wins on ECE, and naive/shrinkage per-regime lose on every
metric. Different calibration metrics can legitimately disagree (ECE only looks at binned
top-class confidence vs. accuracy; Brier and NLL score the full predicted distribution),
and reporting only ECE would have overstated the case for global-only calibration. Per-regime
temperature freedom is harmful at this sample size (~15-70 rows per regime) on every
metric, regardless of exact regime assignment quality — that part of the conclusion is
robust. Whether ANY post-hoc temperature scaling is worth adopting for this model is,
honestly, not yet resolved by the evidence gathered; it is not adopted as a default.

### Regime-probability leakage (found during this audit, root-caused, fixed, re-verified)

**Finding**: `RegimeModel.predict_proba` (`src/emafrs/regime/hmm_regime.py`) used
hmmlearn's `GaussianHMM.predict_proba`, which computes SMOOTHED posteriors via the
forward-backward algorithm over the entire sequence passed to it in one call. Since
`dataset.py` called this once over the full train+val+test date range, every date's
regime probability — a real model INPUT FEATURE for RAMA, and the basis of the
regime-wise performance table above — was, in principle, influenced by observations
*after* that date, all the way to the end of the multi-year series, despite the HMM's
fitted parameters being correctly train-only. This is a real leakage pathway that none of
the existing `assert_no_lookahead` checks covered, because those only ever audited
individual technical/microstructure feature functions, never this smoothing step.

**Magnitude, measured directly, not assumed**: holding the fitted HMM fixed and varying
only how much future data was included in a `predict_proba` call (isolating the smoothing
effect from the unrelated, confounding effect of a different train/test split changing
which dates the model's *parameters* were fit on), the actual numerical difference on
real SPY data was **negligible beyond 1-2 rows from the sequence boundary** (<1e-5),
because this fitted HMM produces near-deterministic (confidently ~0/1) regime assignments
— a single day's own volatility/entropy/Hurst signature is usually enough to pin down its
regime with near-certainty, so conditioning on a little extra future context via smoothing
barely moves anything. An initial, cruder measurement (varying total series length without
controlling for this confound) showed apparent differences up to 1.0 — that number was
**not real smoothing leakage**; it was almost entirely the separate, expected effect of a
proportional train/val/test split producing a different absolute boundary date (and
therefore a genuinely different fitted HMM) when the total series length changes. This
correction to my own initial (overstated) measurement is disclosed here rather than left
uncorrected.

**Correction implemented regardless**: `RegimeModel.predict_proba_causal`
(`src/emafrs/regime/hmm_regime.py`) implements the forward algorithm directly from the
fitted HMM's parameters (no backward pass), giving a provably exact (not
negligibly-approximate) leakage-safe posterior. Verified via
`tests/unit/test_hmm_regime.py::test_predict_proba_causal_matches_smoothed_at_final_row`
(filtering and smoothing must agree exactly at the last row of a sequence, where there is
no future left to differ on) and
`test_predict_proba_causal_is_truncation_invariant` (the actual leakage-safety property,
proven directly via the same truncation methodology `assert_no_lookahead` uses elsewhere).
`dataset.py` was switched to use this method as the default for regime probabilities used
as a model input feature. Even though the practical impact on this particular dataset is
small, there is no guarantee that holds for every ticker/regime configuration, so the
exact version is used regardless of the measured-negligible impact here.

### XAI evaluation (30-sample subset of the test set; full method in `docs/XAI_EVALUATION.md`)

| Metric | First run | Re-run (independent data window, same code) |
|---|---:|---:|
| Faithfulness (top-5 IG-attributed feature removal vs. random-5 removal) | −0.001 | −0.004 |
| Stability (attribution consistency under small input noise) | 0.946 | 0.958 |
| Sparsity (median # features for 80% attribution coverage, of 26 total) | 8.0 | 8.0 |
| Counterfactual validity rate | 0.80 (12/15) | 0.73 (11/15) |

**Finding, stated plainly, now confirmed across two independent runs**: faithfulness is
statistically indistinguishable from zero in both runs — removing the top-5
Integrated-Gradients-attributed features degrades this model's prediction about the same
amount as removing 5 random features. This does not mean IG is "broken" (its completeness
axiom holds exactly, per `tests/unit/test_integrated_gradients.py`) but it does mean, on
this model architecture/training regime, IG's top-ranked features are not more
functionally load-bearing than average — reproduced, not a one-off artifact of a single
training run. See `docs/XAI_EVALUATION.md`'s diagnostic follow-up for a sample-size
robustness check of this finding. Stability and counterfactual validity are both
reasonably strong in both runs.

## Not yet evaluated

- Multi-fold walk-forward for the multimodal RAMA-vs-static comparison (blocked on
  microstructure/text data depth — see Scope limitation above).
- Multiple tickers (only SPY has been run end-to-end through walk-forward and regime/
  calibration/XAI evaluation; single-split baseline comparison has also been run on SPY
  only via `scripts/run_baseline_comparison.py`).
- Full ablation grid A-K from the build brief (see `docs/ABLATION_STUDY.md` for exactly
  which rows are covered vs. not).
- Cross-dataset / cross-market external validation.
- Adopting a heavier shrinkage weight (or a fully global temperature) as the DEFAULT
  calibrator, now that global-only has outperformed naive per-regime and uncalibrated in
  the corrected-regime re-run above -- not yet wired into any production/serving path,
  only demonstrated in this evaluation script.
- Re-running the calibration comparison on a THIRD occasion with the date window held
  perfectly fixed (e.g. a frozen historical OHLCV snapshot) to cleanly separate the
  regime-probability-leakage fix's effect from ordinary date-window drift, since the two
  were confounded in the before/after comparison above.
