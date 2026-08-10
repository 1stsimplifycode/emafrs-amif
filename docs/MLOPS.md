# MLOps

## Pipeline (build brief Section 35)

```
Data ingestion (emafrs.data)
      |
Data validation (emafrs.data.health, leakage_audit)
      |
Feature generation (emafrs.features, emafrs.regime, emafrs.microstructure)
      |
Training (emafrs.models.train)
      |
Evaluation (emafrs.models.baselines.evaluate_predictions, emafrs.models.calibration)
      |
Model versioning (emafrs.mlops.model_registry)
      |
Inference (emafrs.mlops.training_pipeline.run_training_and_prediction_pipeline)
      |
Monitoring (emafrs.data.health for data freshness; emafrs.mlops.drift for model input drift)
      |
Retraining recommendation (emafrs.mlops.drift.DriftReport.retrain_recommended)
```

This is a real, callable pipeline — every stage above corresponds to actual code in the
repository, not a planned-but-unimplemented diagram.

## Model versioning

`emafrs.mlops.model_registry.save_model()` persists a trained model's `state_dict` plus
architecture metadata (modality dimensions, hidden size, class count) and records a
`ModelVersion` row. Saving a new version for the same `(model_type, ticker)` pair
deactivates (not deletes) the prior active version — build brief Section 35's explicit
instruction: **do not automatically replace production models without safeguards.**
Verified in `tests/integration/test_mlops_pipeline.py::test_model_registry_deactivates_old_versions`.

`load_latest_model()` reads back the currently-active version and reconstructs the exact
architecture before loading weights — verified to produce bit-identical inference output
to the in-memory model it was saved from
(`test_model_registry_roundtrip`).

## Drift detection

`emafrs.mlops.drift.detect_feature_drift()` runs a two-sample Kolmogorov-Smirnov test per
feature, comparing a reference (training-window) distribution against a recent window.
Retraining is **recommended** (a `bool` field on the returned report), never triggered
automatically — the actual retraining decision remains a human/operational one. Not yet
wired into a scheduled job; see "Not yet built" below.

## Retraining

There is currently no automatic retraining trigger. The recommended pattern (build brief
Section 35: "do not automatically replace production models without safeguards"):

1. A scheduled job periodically calls `detect_feature_drift()` comparing each active
   model's original training distribution against a recent live window.
2. If `retrain_recommended` is `True`, this is surfaced (e.g. as a flag in
   `/api/v1/data-sources/health`-style endpoint, or a log alert) for a human to review.
3. A human-triggered retrain calls `run_training_and_prediction_pipeline()` again, which
   saves a new model version and deactivates the old one — but the old checkpoint remains
   on disk and in the DB for rollback/comparison.

## Not yet built

- **Scheduled background jobs.** `apscheduler` is a listed dependency but not yet wired
  into the API process for periodic retraining/drift-checking/data-freshness refresh —
  see `docs/PRODUCTION_DEPLOYMENT.md`'s hardening checklist.
- **Automated data validation gate** before training (e.g. reject a training run if the
  fetched OHLCV has suspicious gaps/outliers). Currently only the leakage audit
  (`emafrs.features.leakage_audit`) runs as an automated check, and it runs in the test
  suite, not as a pre-training gate in the live pipeline.
- **A/B or shadow evaluation** of a new model version against the currently active one
  before promoting it — currently a new version becomes active immediately upon saving.
- **Alert delivery.** `emafrs.alerts.citizen_alerts.maybe_generate_alert()` generates the
  alert PAYLOAD (headline, risk score, explanation) when a risk score crosses a threshold
  -- real logic, tested -- but no delivery channel is wired up (no push notification,
  email, or webhook provider is configured). It is also not yet connected to drift/
  data-source-outage monitoring events, only to manipulation-risk scores.
