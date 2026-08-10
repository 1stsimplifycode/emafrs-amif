"""Sentiment transformer benchmark: generic vs other-domain vs financial-domain pretrained
models, evaluated INFERENCE-ONLY (no fine-tuning -- infeasible to fine-tune every model on
CPU per build brief Section 19) on the real, human-annotated Financial PhraseBank dataset.

Research question this answers (build brief Section 21): does moving from a generic
language model to a financial-domain pretrained one improve sentiment quality on financial
text, before any project-specific fine-tuning happens at all?
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Provider-specific output label strings, normalized to {"negative","neutral","positive"}.
_LABEL_NORMALIZERS = {
    "distilbert-base-uncased-finetuned-sst-2-english": {"POSITIVE": "positive", "NEGATIVE": "negative"},
    "cardiffnlp/twitter-roberta-base-sentiment-latest": {
        "negative": "negative", "neutral": "neutral", "positive": "positive",
    },
    "ProsusAI/finbert": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
}

_TRUE_LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass
class BenchmarkResult:
    model_key: str
    hf_id: str
    domain: str
    label_scheme: str  # "binary" (no neutral class) or "3class"
    n_examples: int
    accuracy: float
    macro_f1: float
    precision_macro: float
    recall_macro: float
    mean_latency_ms: float
    params_approx: str


_AGREEMENT_FILES = {
    "sentences_50agree": "Sentences_50Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}
_LABEL_TO_INT = {"negative": 0, "neutral": 1, "positive": 2}


def load_financial_phrasebank(
    config: str = "sentences_66agree", max_examples: int | None = None, seed: int = 0
) -> dict:
    """Real, human-annotated financial sentiment dataset (Malo et al. 2014).

    NOTE: loaded via `huggingface_hub.hf_hub_download` + manual zip/text parsing rather
    than the `datasets` library, because `datasets` unconditionally imports
    `pyarrow.dataset` at module load time, and that submodule is blocked by this machine's
    Windows Application Control policy (verified directly: `import pyarrow.parquet` works,
    `import pyarrow.dataset` is blocked by DLL policy). This is an environment constraint,
    not a code choice -- the underlying data is identical to what `load_dataset(...)` would
    return. Raises on failure (network/repo-gated) rather than substituting synthetic
    sentences -- callers must surface "benchmark not run," never fabricate results.

    Returns {"sentence": [...], "label": [...]} with label as int (0=negative, 1=neutral,
    2=positive), matching the HF dataset's own label encoding.

    IMPORTANT: the raw file is NOT randomly ordered -- it is grouped by source document, and
    an early real run of this benchmark discovered the first 240 lines contain zero
    "negative" examples (219 positive, 21 neutral), which silently skewed accuracy/F1
    metrics. `max_examples` therefore selects a seeded random sample, not a file-order
    prefix, so the subsample's label distribution is representative of the full file.
    """
    import random
    import zipfile

    from huggingface_hub import hf_hub_download

    if config not in _AGREEMENT_FILES:
        raise ValueError(f"config must be one of {list(_AGREEMENT_FILES)}, got {config!r}")

    zip_path = hf_hub_download("financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
    inner_name = f"FinancialPhraseBank-v1.0/{_AGREEMENT_FILES[config]}"

    with zipfile.ZipFile(zip_path) as z:
        raw_bytes = z.read(inner_name)
    text = raw_bytes.decode("latin-1")

    sentences, labels = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        sentence, _, label_str = line.rpartition("@")
        label_str = label_str.strip().lower()
        if label_str not in _LABEL_TO_INT:
            continue
        sentences.append(sentence.strip())
        labels.append(_LABEL_TO_INT[label_str])

    if max_examples and max_examples < len(sentences):
        rng = random.Random(seed)
        idx = list(range(len(sentences)))
        rng.shuffle(idx)
        idx = idx[:max_examples]
        sentences = [sentences[i] for i in idx]
        labels = [labels[i] for i in idx]

    return {"sentence": sentences, "label": labels}


class HFSentimentModel:
    """Wraps a HF text-classification pipeline and normalizes its label vocabulary."""

    def __init__(self, hf_id: str):
        from transformers import pipeline

        self.hf_id = hf_id
        self._pipe = pipeline("text-classification", model=hf_id, top_k=None)
        self._normalizer = _LABEL_NORMALIZERS.get(hf_id)
        if self._normalizer is None:
            raise ValueError(f"No label normalizer registered for {hf_id!r} -- add one to _LABEL_NORMALIZERS")
        self.supports_neutral = "neutral" in self._normalizer.values()

    def predict(self, texts: list[str]) -> tuple[list[str], float]:
        """Returns (normalized_labels, mean_latency_ms_per_example)."""
        start = time.perf_counter()
        raw = self._pipe(texts, truncation=True)
        elapsed_ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)

        labels = []
        for scores in raw:
            # top_k=None returns a list of {label, score} per class; take the argmax
            best = max(scores, key=lambda d: d["score"])
            labels.append(self._normalizer.get(best["label"], best["label"].lower()))
        return labels, elapsed_ms


def _evaluate(true_labels: list[str], pred_labels: list[str], label_set: list[str]) -> dict:
    y_true = [label_set.index(l) for l in true_labels]
    y_pred = [label_set.index(l) if l in label_set else -1 for l in pred_labels]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(range(len(label_set))), average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, labels=list(range(len(label_set))), average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=list(range(len(label_set))), average="macro", zero_division=0),
    }


def run_sentiment_benchmark(registry: dict, max_examples: int = 240) -> list[BenchmarkResult]:
    """registry: the `sentiment_benchmark` section of configs/nlp_models.yaml.
    Runs each configured model against Financial PhraseBank and returns comparable metrics.
    The generic (binary) model is evaluated only on the non-neutral subset, since it
    structurally cannot predict a neutral class -- this is reported via label_scheme, not
    silently forced into an unfair 3-class comparison."""
    dataset = load_financial_phrasebank(max_examples=max_examples)
    texts = list(dataset["sentence"])
    true_label_names = [_TRUE_LABEL_NAMES[int(l)] for l in dataset["label"]]

    results = []
    for model_key, spec in registry.items():
        model = HFSentimentModel(spec["hf_id"])

        if model.supports_neutral:
            eval_texts, eval_true = texts, true_label_names
            label_scheme = "3class"
            label_set = ["negative", "neutral", "positive"]
        else:
            eval_texts, eval_true = zip(
                *[(t, l) for t, l in zip(texts, true_label_names) if l != "neutral"]
            )
            eval_texts, eval_true = list(eval_texts), list(eval_true)
            label_scheme = "binary"
            label_set = ["negative", "positive"]

        pred_labels, latency_ms = model.predict(eval_texts)
        metrics = _evaluate(eval_true, pred_labels, label_set)

        results.append(
            BenchmarkResult(
                model_key=model_key,
                hf_id=spec["hf_id"],
                domain=spec["domain"],
                label_scheme=label_scheme,
                n_examples=len(eval_texts),
                accuracy=metrics["accuracy"],
                macro_f1=metrics["macro_f1"],
                precision_macro=metrics["precision_macro"],
                recall_macro=metrics["recall_macro"],
                mean_latency_ms=latency_ms,
                params_approx=spec.get("params_approx", "unknown"),
            )
        )
    return results


def results_to_markdown_table(results: list[BenchmarkResult]) -> str:
    header = "| Model | Domain | Scheme | N | Accuracy | Macro F1 | Precision | Recall | Latency (ms/ex) | Params |\n"
    header += "|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    rows = [
        f"| {r.model_key} | {r.domain} | {r.label_scheme} | {r.n_examples} | {r.accuracy:.3f} | "
        f"{r.macro_f1:.3f} | {r.precision_macro:.3f} | {r.recall_macro:.3f} | {r.mean_latency_ms:.1f} | {r.params_approx} |"
        for r in results
    ]
    return header + "\n".join(rows) + "\n"
