# Model Cards

One card per major model in the pipeline, per build brief Section 18A.18. Metrics shown
are from actual runs recorded in `docs/NLP_MODEL_BENCHMARK.md` and
`docs/EXPERIMENT_PROTOCOL.md` — not re-stated speculatively here.

---

## Financial Sentiment Classifier (production SPC)

- **Architecture / source**: `ProsusAI/finbert` (BERT-base, financial-domain pretrained
  and fine-tuned for sentiment).
- **Fine-tuned by this project**: No — used as pretrained. Chosen over the originally
  planned `yiyanghkust/finbert-tone` after that checkpoint failed to load
  (`config.json` missing `model_type` — a real compatibility issue found during
  implementation, see `docs/NLP_MODEL_BENCHMARK.md`).
- **Task**: 3-class sentiment (negative/neutral/positive) on financial text.
- **Training data (of the pretrained model)**: Financial PhraseBank and related financial
  corpora (per the model's own card on Hugging Face) — meaning its evaluation on
  Financial PhraseBank in this project's benchmark is likely partially in-sample; see the
  explicit caveat in `docs/NLP_MODEL_BENCHMARK.md`.
- **Evaluation (this project, real run)**: accuracy 0.892, macro F1 0.882, n=240
  (seeded random sample), CPU latency ~59ms/example.
- **Limitations**: in-sample bias noted above; not evaluated on out-of-domain financial
  text (e.g. social media) in this project.
- **Intended use**: sentiment polarity component of the ASEM affective feature vector.
- **Out-of-scope use**: standalone trading signal; general-purpose (non-financial) text.
- **Compute**: CPU inference only, ~110M parameters.

## Emotion Detector (zero-shot)

- **Architecture / source**: `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`
  (DeBERTa-v3-xsmall, ~70M params), used via the HF zero-shot-classification pipeline
  (NLI entailment scoring against the 7 PDF emotion category labels).
- **Fine-tuned by this project**: No, and cannot be — no labeled dataset exists for the
  PDF's 7-category financial emotion taxonomy. This is a deliberate, disclosed choice
  (zero-shot rather than fabricating labels), not a shortcut.
- **Task**: multi-label classification over {confidence, uncertainty, optimism, fear,
  caution, assertiveness, evasiveness}.
- **Evaluation**: no accuracy metric exists (no ground truth). Validated only by
  directional sanity checks (`tests/integration/test_asem_pipeline.py`): a fear-coded
  sentence scores measurably higher on "fear" than a confidence-coded sentence, and vice
  versa. This is NOT a substitute for accuracy and must never be cited as one.
- **Limitations**: zero-shot NLI-based classifiers are known to be less accurate than
  fine-tuned classifiers on domain-specific label sets; no quantitative accuracy bound is
  available here.
- **Intended use**: exploratory affective signal within the ASEM pipeline.
- **Out-of-scope use**: any claim of validated emotion-detection accuracy.

## Contextual Semantic Encoder (CSE)

- **Architecture / source**: `sentence-transformers/all-MiniLM-L6-v2` (~22M params,
  384-dim output), pretrained, not fine-tuned by this project.
- **Task**: general-purpose sentence embedding for topical/contextual similarity.
- **Evaluation**: qualitative only (`tests/integration/test_asem_pipeline.py`: two
  paraphrased central-bank-policy sentences embed closer together than either does to an
  unrelated sentence about cooking).
- **Intended use**: the `c_t` component of the ASEM affective feature vector (PDF Eq. 1).

## Market Regime Detection Engine (HMM)

- **Architecture**: 4-state Gaussian HMM (`hmmlearn`, diagonal covariance), Baum-Welch
  fit, over realized volatility / Shannon entropy / Hurst exponent / volume z-score.
- **Fit by this project**: Yes, trained on each experiment's specific training split
  (never on validation/test dates — see the leakage-boundary regression test in
  `tests/integration/test_dataset_and_models.py`).
- **Evaluation (real run, 5y SPY)**: 4 regimes recovered with correct monotonic
  volatility ordering (11.2% -> 26.6% annualized), mean run length ~45 days (not
  noise-chasing), but LOW silhouette separation (-0.004) — reported honestly in
  `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` row #7, not hidden.
- **Limitations**: regime semantics can drift if applied far out-of-sample from the
  training window; quality (silhouette) is regime/feature-set dependent and was found to
  be weak in the one run measured.
- **Intended use**: regime conditioning input to RAMA and the manipulation risk model.

## Baseline Model A (opaque, XGBoost)

- **Architecture**: gradient-boosted trees (`xgboost`), 150-200 estimators, depth 4,
  multi:softprob objective.
- **Trained by this project**: Yes, per-fold in walk-forward validation and on the
  single-split dataset.
- **Evaluation**: walk-forward (SPY, 5 folds) mean accuracy 0.384, 95% CI [0.349, 0.419].
- **Limitations**: minimal interpretability by design (this is its role in the
  accuracy/explainability comparison, per build brief Section 4).

## Baseline Model B (explainable, logistic regression)

- **Architecture**: multinomial logistic regression (`sklearn.linear_model.LogisticRegression`).
- **Trained by this project**: Yes.
- **Evaluation**: walk-forward mean accuracy 0.390, 95% CI [0.374, 0.406] — not
  significantly different from Model A (paired t-test p=0.656).
- **Interpretability**: linear coefficients are the explanation, exposed directly via
  `ModelBExplainableLogReg.coefficients()`.

## Model C (Static Multimodal Fusion)

- **Architecture**: per-modality linear+ReLU+LayerNorm encoders, combined with a single
  learned-but-NOT-regime-conditioned softmax weight vector (`StaticFusionModel`).
- **Trained by this project**: Yes.
- **Evaluation**: single-split (financial+microstructure, SPY) test accuracy 0.412.

## Model D (RAMA -- Regime-Adaptive Multimodal Attention)

- **Architecture**: same per-modality encoders as Model C, but fusion weights are a
  softmax over a learned `(n_regimes x n_modalities)` gate matrix applied to the regime
  posterior (`RAMAFusionModel`) -- the disclosed correction to the PDF's literal (inert)
  formula, see `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` row #19.
- **Trained by this project**: Yes.
- **Evaluation**: single-split test accuracy 0.425 (vs. Model C's 0.412 on the same
  split — not tested for significance, single run only). Walk-forward (financial-only,
  where RAMA degenerates to a single-modality network): mean accuracy 0.402, not
  significantly different from Models A/B.
- **Mechanism verified independent of training outcome**: `test_rama_weights_differ_across_regimes`
  proves the gate CAN produce different fusion weights per regime; the trained run's gate
  happened to stay close to uniform (see finding in `docs/EXPERIMENT_PROTOCOL.md`) — the
  mechanism working and the trained instance using it strongly are reported as separate facts.

## Model E (RAMA + XAI)

- **Architecture**: identical trained weights to Model D; XAI (Integrated Gradients,
  counterfactual search, NLG) is a read-only analysis layer over the frozen model, so
  there is no distinct "Model E" set of weights or accuracy — see `docs/ABLATION_STUDY.md`
  row K.
- **Evaluation**: XAI quality metrics (not predictive accuracy) — see `docs/XAI_EVALUATION.md`.

## Manipulation Risk Model

- **Architecture**: disclosed weighted linear combination of 5 signals (hype score, OBI
  proxy, volume anomaly, OFI proxy, liquidity shock proxy), regime-modulated.
- **Trained by this project**: No — default weights are an equal-weight (0.2 each) prior,
  NOT fit from labeled manipulation-event data, because no such labeled dataset currently
  exists. `fit_weights_from_labels()` exists for when one does but is not called anywhere
  in this project's default pipeline.
- **Intended use**: relative risk scoring / alert triggering, not a validated fraud-
  detection classifier.
- **Out-of-scope use**: any claim that a flagged event constitutes proven manipulation --
  the module's own docstrings and `docs/LIMITATIONS.md` are explicit that this is a risk
  score, not a legal or evidentiary determination.
