# NLP Transformer Benchmark

Answers the research question posed in build brief Section 21: *does moving from a generic
language model to a financial-domain pretrained one improve sentiment quality on financial
text, before any project-specific fine-tuning happens at all?*

Reproduce with: `python scripts/run_sentiment_benchmark.py` (writes
`docs/_sentiment_benchmark_table.md`, merged into this file below). Model registry:
`configs/nlp_models.yaml`.

## Method

- **Dataset**: [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank)
  (Malo et al. 2014, JASIST), `sentences_66agree` config — real, human-annotated financial
  news sentences, restricted to the subset where ≥66% of annotators agreed on the label.
  Loaded via a **seeded random sample of 240 sentences** (not a file-order prefix — see
  "Methodological issue found and fixed" below).
- **Evaluation**: inference-only (zero fine-tuning) for all three models, per build brief
  Section 19's allowance when fine-tuning every candidate model is CPU-infeasible.
- **Models compared** (full rationale in `configs/nlp_models.yaml`):
  - `generic_baseline` — DistilBERT fine-tuned on SST-2 movie reviews. No domain adaptation
    at all. Binary output only (no neutral class), so evaluated on the non-neutral subset
    (n=100) — a different, smaller sample than the two 3-class models, reported honestly via
    the `Scheme`/`N` columns rather than forced into an apples-to-apples number it can't produce.
  - `other_domain` — Twitter-tuned RoBERTa. Domain-adapted, but to a *different*, non-financial
    domain — tests whether domain adaptation *in general* helps, or whether it specifically
    needs to be financial.
  - `financial_domain` — FinBERT (`ProsusAI/finbert`), pretrained and fine-tuned on financial
    text including Financial PhraseBank itself.

## Results (n as shown; seed=0; run on this machine, CPU-only)

| Model | Domain | Scheme | N | Accuracy | Macro F1 | Precision | Recall | Latency (ms/ex) | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| generic_baseline | generic (movie reviews, SST-2) | binary | 100 | 0.770 | 0.767 | 0.795 | 0.828 | 36.8 | 66M |
| other_domain | social media (Twitter) | 3class | 240 | 0.729 | 0.670 | 0.769 | 0.633 | 69.3 | 125M |
| financial_domain | financial (Financial PhraseBank-trained) | 3class | 240 | 0.892 | 0.882 | 0.856 | 0.923 | 59.1 | 110M |

**Finding**: the financial-domain model (FinBERT) clearly outperforms both the generic and
the other-domain (Twitter) models on financial sentiment, and other-domain adaptation
(Twitter) is *not* uniformly better than no adaptation at all — its macro F1 (0.670) is
below the generic binary baseline's (0.767, though on an easier binary/non-neutral subset).
This supports the hypothesis that domain adaptation specifically to the financial domain
matters, not just "any" domain adaptation.

## Important caveat: this is not a strictly held-out evaluation

`ProsusAI/finbert` was fine-tuned using Financial PhraseBank data. Evaluating it *on*
Financial PhraseBank is therefore likely partially in-sample (the model may have seen these
exact sentences, or close paraphrases, during its own training) — its 0.892 accuracy is an
**optimistic upper bound** on true out-of-domain financial sentiment performance, not a
clean generalization estimate. This is disclosed here rather than presented as a clean
result. A more rigorous follow-up (tracked, not yet done) would evaluate all three models
on a financial sentiment set that is verifiably disjoint from FinBERT's training data (e.g.
a held-out slice of NewsAPI-sourced headlines once real API keys are configured).

## Methodological issue found and fixed during implementation

The first benchmark run took the first 240 lines of the raw Financial PhraseBank file as
the sample. The file is **not randomly ordered** (grouped by source document), and that
prefix happened to contain **zero "negative"-labeled examples** (219 positive, 21 neutral
out of 240) — silently skewing accuracy/F1 upward and producing an internally inconsistent
result (96% accuracy but only 0.58 macro F1, because the "negative" class had zero true
support and the "neutral" class was thin). This was caught by manually checking the
predicted vs. true label distribution rather than accepting the first metric printed, and
fixed by switching `load_financial_phrasebank()` to a seeded random sample
(`emafrs/nlp/sentiment_benchmark.py`). The results above are from the corrected version.

## Environment constraint: `datasets` library unusable on this machine

The Hugging Face `datasets` package unconditionally imports `pyarrow.dataset` at module
load time. On this machine, that submodule is blocked by a Windows Application Control
policy (`ImportError: ... An Application Control policy has blocked this file`), verified
directly: `import pyarrow.parquet` succeeds, `import pyarrow.dataset` fails. `datasets` is
therefore unusable here regardless of which dataset is requested. Financial PhraseBank is
instead downloaded via `huggingface_hub.hf_hub_download` (plain HTTP + local cache, no
pyarrow.dataset dependency) and parsed directly from its original zip/text format — the
underlying data is identical to what `datasets.load_dataset(...)` would return.

## Model compatibility issue found and fixed

`yiyanghkust/finbert-tone` (the originally planned FinBERT checkpoint) fails to load via
`transformers.pipeline`: `ValueError: Unrecognized model in yiyanghkust/finbert-tone.
Should have a 'model_type' key in its config.json.` Verified directly, not assumed.
Switched to `ProsusAI/finbert`, which loads and predicts correctly and is documented in
`configs/nlp_models.yaml`.

## Emotion detection (zero-shot, not benchmarked against ground truth)

No labeled dataset exists for the PDF's 7-category financial emotion taxonomy (confidence,
uncertainty, optimism, fear, caution, assertiveness, evasiveness), so `emafrs/nlp/emotion.py`
implements zero-shot NLI classification rather than a fine-tuned classifier — see that
module's docstring. It is validated by directional sanity checks only (a fear-coded sentence
scores higher on "fear" than a confidence-coded one, and vice versa for "confidence") in
`tests/integration/test_asem_pipeline.py`, not by accuracy against a labeled test set,
because no such set exists. This is a genuine limitation, tracked in `docs/LIMITATIONS.md`.

## Not yet evaluated

- Project-specific fine-tuning of any sentiment/emotion model (build brief Section 4 calls
  for at least one project-trained model). Not done yet — would need a labeled financial
  emotion corpus that doesn't currently exist, or a fine-tune of the sentiment head on a
  held-out split of Financial PhraseBank with FinBERT excluded from comparison (since it
  already trained on it). Tracked as follow-up work.
- DeBERTa and ALBERT/ELECTRA variants in the generic BERT-family sweep (build brief Section
  18A.6's suggested table lists BERT/RoBERTa/DeBERTa/DistilBERT/FinBERT). Three
  representative points (generic/other-domain/financial-domain) were prioritized over an
  exhaustive architecture sweep per Section 18A.8 ("prioritize models that provide a
  scientifically meaningful comparison" over downloading many models). Can be extended by
  adding entries to `configs/nlp_models.yaml`'s `sentiment_benchmark` section.
- Hyperparameter optimization (N/A — no fine-tuning was performed in this phase).
