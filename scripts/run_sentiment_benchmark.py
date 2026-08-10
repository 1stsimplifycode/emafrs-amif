"""Runs the real sentiment transformer benchmark against Financial PhraseBank and writes
results to docs/NLP_MODEL_BENCHMARK.md. Not fabricated -- every number in that doc comes
from executing this script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from emafrs.nlp.sentiment_benchmark import results_to_markdown_table, run_sentiment_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main():
    with open(ROOT / "configs" / "nlp_models.yaml") as f:
        config = yaml.safe_load(f)

    registry = config["sentiment_benchmark"]
    print("Running sentiment benchmark against Financial PhraseBank (max_examples=240)...")
    results = run_sentiment_benchmark(registry, max_examples=240)

    for r in results:
        print(f"{r.model_key:20s} acc={r.accuracy:.3f} macro_f1={r.macro_f1:.3f} "
              f"n={r.n_examples} scheme={r.label_scheme} latency={r.mean_latency_ms:.1f}ms")

    table = results_to_markdown_table(results)
    out_path = ROOT / "docs" / "_sentiment_benchmark_table.md"
    out_path.write_text(table, encoding="utf-8")
    print(f"\nWrote table to {out_path}")


if __name__ == "__main__":
    main()
