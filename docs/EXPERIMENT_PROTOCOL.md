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

### Regime-wise performance (most recent real run)

| Regime | N | Accuracy | Macro F1 | Mean fusion weight (financial) |
|---|---:|---:|---:|---:|
| 1 (Low) | 32 | 0.344 | 0.175 | 0.502 |
| 2 (Medium) | 70 | 0.386 | 0.206 | 0.511 |
| 3 (High) | 46 | 0.457 | 0.285 | 0.506 |
| 4 (Extreme) | 12 | 0.500 | 0.235 | 0.518 |

(An earlier run had Regime 4 excluded for n<5; this run's slightly shifted window
happened to include 12 Regime-4 test rows, so it is reported here.) Accuracy rises mildly
with regime severity but F1 stays low throughout, and — consistent with the finding
already logged in `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` row #19, and now confirmed
across two independent real runs on different date windows — the mean financial-modality
fusion weight stays within a narrow 0.502-0.518 band across ALL FOUR observed regimes
(including the previously-excluded Extreme regime): RAMA's gate is not meaningfully
regime-adaptive in this trained model, even though the underlying mechanism is proven
capable of it in isolation
(`tests/unit/test_fusion_models.py::test_rama_weights_differ_across_regimes`). Seeing the
same near-flat 0.50-0.52 band on a second, independently-trained run (different data
window, same seed=42) upgrades this from "one run's fluke" to a reproducible property of
how this model trains on this data — see the root-cause investigation below.

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

**Updated root-cause hypothesis (not yet further tested)**: the validation and test
windows are chronologically disjoint (val precedes test, per this project's leakage-safe
splitting) and potentially non-stationary in volatility/regime composition — post-hoc
temperature scaling assumes the confidence-accuracy relationship learned on validation
transfers to test, which chronological non-stationarity in financial time series can
violate regardless of how many temperature parameters are used. This reframes the failure
from "a fixable overfitting bug" to "a harder distribution-shift problem inherent to
calibrating on one chronological period and applying to a later one" — reported honestly
as unresolved, not forced into a false "corrected" state. A rolling/walk-forward
calibration fit (re-fitting temperature per fold rather than once) is the natural next
step to actually test the non-stationarity hypothesis, not yet done (see "Not yet
evaluated").

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
- Rolling/walk-forward temperature calibration (re-fit per fold) to test the
  val/test-non-stationarity hypothesis raised above as the calibration-worsening
  finding's updated root cause, now that per-regime-parameter-count overfitting has been
  ruled out.
