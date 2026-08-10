"""Integration test for the Financial PhraseBank loader (real network download, no model
inference -- the full 3-model benchmark is run via scripts/run_sentiment_benchmark.py and
its results are checked into docs/NLP_MODEL_BENCHMARK.md rather than re-run on every test
invocation, since it downloads ~1GB of models)."""
import pytest

from emafrs.nlp.sentiment_benchmark import _LABEL_TO_INT, load_financial_phrasebank

pytestmark = pytest.mark.integration


def test_load_financial_phrasebank_returns_real_labeled_sentences():
    ds = load_financial_phrasebank(max_examples=20)
    assert len(ds["sentence"]) == 20
    assert len(ds["label"]) == 20
    assert all(isinstance(s, str) and len(s) > 0 for s in ds["sentence"])
    assert all(label in _LABEL_TO_INT.values() for label in ds["label"])


def test_load_financial_phrasebank_rejects_unknown_config():
    with pytest.raises(ValueError):
        load_financial_phrasebank(config="not_a_real_config")
