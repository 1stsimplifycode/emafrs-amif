# Reproducibility

## Environment

- **Python**: 3.12.10 (this repo's `.venv` was built with Python 3.12, not the system's
  default 3.14 — several ML dependencies (hmmlearn, torch, xgboost) needed the more
  broadly-supported 3.12 for reliable wheel availability at the time of writing).
- **Compute**: CPU-only throughout. No GPU was available or used. `pip install torch
  --index-url https://download.pytorch.org/whl/cpu` was used specifically to avoid the
  much larger default CUDA wheel.
- **OS**: Windows 11, tested via Git Bash. One environment-specific constraint is
  recorded and worked around: the `datasets` (Hugging Face) library is unusable on this
  machine because it unconditionally imports `pyarrow.dataset`, which Windows Application
  Control policy blocks here — see `docs/NLP_MODEL_BENCHMARK.md` and
  `docs/LIMITATIONS.md`. `huggingface_hub.hf_hub_download` is used directly instead.

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu  # if not already present
cp .env.example .env  # fill in whichever API keys you have
```

## Reproducing each result

| Result | Command | Notes |
|---|---|---|
| Full test suite | `pytest` | 150 tests as of the experiment-protocol phase; `pytest -m "not integration"` skips network/model-download tests for a fast local loop |
| Sentiment transformer benchmark | `python scripts/run_sentiment_benchmark.py` | Downloads ~3 models (~1.2GB total) on first run; writes `docs/_sentiment_benchmark_table.md` |
| Single-split baseline comparison (Models A-D) | `python scripts/run_baseline_comparison.py [TICKER]` | Defaults to SPY |
| Walk-forward validation (Models A/B/NN) | `python scripts/run_walk_forward_experiment.py` | ~5-10 minutes on CPU: 5 folds x (1 XGBoost + 1 LogReg + 3-seed NN) |
| Regime-wise / calibration / XAI evaluation | `python scripts/run_regime_calibration_xai_eval.py` | Single-split RAMA model, SPY |

## Seeds

Every stochastic component that supports a seed is explicitly seeded and the seed is
recorded in the corresponding result:
- `numpy` / `torch` seeds: 42 (single-split experiments), {0, 1, 2} (walk-forward NN,
  3 seeds/fold).
- HMM fitting: `random_state=42` (`fit_regime_hmm`).
- XGBoost / LogisticRegression: `random_state=42`.
- Financial PhraseBank sampling: `seed=0` (`load_financial_phrasebank`).

Re-running any script above with the same seeds against the same live data source should
reproduce the same numbers up to floating-point nondeterminism in BLAS/torch CPU kernels
(typically negligible at the precision reported) — **it will NOT reproduce identical
numbers if re-run on a different date**, since yfinance/NewsAPI/etc. return live,
continuously-updated data, not a frozen snapshot. Every reported result states the ticker
and date range it was computed over for exactly this reason.

## What is NOT pinned

`requirements.txt` uses `>=` version bounds, not exact pins — this is a research
reconstruction, not a production release artifact (see `docs/PRODUCTION_DEPLOYMENT.md`
once that phase lands for release-pinning practices). If a future dependency upgrade
changes behavior (as already happened once during this project: `sentence-transformers`
renamed `get_sentence_embedding_dimension` to `get_embedding_dimension`, handled with a
runtime fallback in `emafrs/nlp/semantic_encoder.py`), that is expected and should be
handled the same way — a disclosed compatibility shim, documented at the point of change,
not a silent pin-and-forget.

## Known sources of non-reproducibility (disclosed, not hidden)

- Live data changes daily (see above).
- `hmmlearn`'s EM fitting can converge to different local optima across library versions
  even with a fixed `random_state`, if the underlying convergence criteria change between
  releases.
- Model downloads from the Hugging Face Hub could serve an updated checkpoint under the
  same model ID if the upstream maintainer pushes a new version — this project does not
  pin to a specific model revision/commit hash. A stricter reproducibility posture (pin
  `revision=<commit_sha>` in every `from_pretrained` call) is a reasonable hardening step,
  tracked as a follow-up rather than done retroactively here.
