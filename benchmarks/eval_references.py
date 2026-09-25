"""Generate independent metric fixtures; never imports Kayak's implementation.

Run in a separate environment, with no model or dataset downloads:
    uv run --no-project --with scikit-learn==1.9.1 --with scipy==1.18.0 --with numpy==2.5.3 \
        python benchmarks/eval_references.py --output /tmp/eval-reference.json
    cmp /tmp/eval-reference.json tests/fixtures/eval-reference.json

References use explicitly matched conventions. Missing predictions are a separate
category for MCC and incorrect for accuracy/F1. Macro F1 uses every candidate;
balanced accuracy uses supported gold classes. Brier is unhalved. Log-loss
fixtures use interior probabilities because sklearn clips at dtype epsilon,
whereas Kayak documents a caller-selected floor. nDCG has linear gain and strict
ranks, avoiding score-tie conventions. Unknown judgments are tested separately.
"""

import argparse
import importlib
import importlib.metadata
import json
import random
import warnings
from pathlib import Path


def references() -> dict[str, object]:
    metrics = importlib.import_module("sklearn.metrics")
    stats = importlib.import_module("scipy.stats")
    rng = random.Random(20260925)
    classifications = []
    probabilities = []
    rankings = []
    for index in range(30):
        labels = [f"class_{i}" for i in range(2 + index % 6)]
        count = 1 + index * 3
        gold = [rng.choice(labels[: 1 + index % len(labels)]) for _ in range(count)]
        choices = [rng.choice([*labels, None]) for _ in gold]
        if index % 6 == 0:
            choices = list(gold)
        elif index % 6 == 1:
            choices = [None for _ in gold]
        predicted = [choice if choice is not None else "__failed__" for choice in choices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected = {
                "accuracy": float(metrics.accuracy_score(gold, predicted)),
                "macro_f1": float(
                    metrics.f1_score(
                        gold, predicted, labels=labels, average="macro", zero_division=0
                    )
                ),
                "weighted_f1": float(
                    metrics.f1_score(
                        gold, predicted, labels=labels, average="weighted", zero_division=0
                    )
                ),
                "balanced_accuracy": float(metrics.balanced_accuracy_score(gold, predicted)),
                "matthews_correlation": float(metrics.matthews_corrcoef(gold, predicted)),
            }
        classifications.append(
            {
                "labels": labels,
                "gold": gold,
                "choices": choices,
                "expected": expected,
            }
        )

    for index in range(12):
        labels = [f"class_{i}" for i in range(2 + index % 6)]
        gold = [rng.choice(labels) for _ in range(5 + index)]
        vectors = []
        for _ in gold:
            weights = [rng.randint(1, 100) for _ in labels]
            vectors.append([weight / sum(weights) for weight in weights])
        probabilities.append(
            {
                "labels": labels,
                "gold": gold,
                "probabilities": vectors,
                "expected": {
                    "log_loss": float(metrics.log_loss(gold, vectors, labels=labels)),
                    "brier_score": float(
                        metrics.brier_score_loss(gold, vectors, labels=labels, scale_by_half=False)
                    ),
                },
            }
        )

    for index in range(24):
        labels = [f"doc_{i}" for i in range(2 + index % 7)]
        grades = [rng.randint(0, 4) for _ in labels]
        grades[0] = 1  # A nonzero ideal; zero-positive judgments remain undefined in Kayak.
        order = rng.sample(labels, len(labels))
        k = 1 + index % (len(labels) + 2)
        scores = [float(len(labels) - order.index(label)) for label in labels]
        rankings.append(
            {
                "order": order,
                "relevance": dict(zip(labels, grades, strict=True)),
                "k": k,
                "ndcg_at_k": float(metrics.ndcg_score([grades], [scores], k=k, ignore_ties=True)),
            }
        )

    paired = []
    for fixed, regressed in ((0, 1), (1, 1), (0, 10), (20, 9), (113, 102), (187, 33), (3000, 2999)):
        paired.append(
            {
                "fixed": fixed,
                "regressed": regressed,
                "p": float(stats.binomtest(fixed, fixed + regressed, p=0.5).pvalue),
            }
        )
    intervals = []
    for correct, total in ((0, 1), (1, 1), (0, 770), (770, 770), (276, 770), (430, 770), (50, 100)):
        interval = stats.binomtest(correct, total).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        intervals.append(
            {
                "correct": correct,
                "total": total,
                "lower": float(interval.low),
                "upper": float(interval.high),
            }
        )
    return {
        "seed": 20260925,
        "versions": {
            name: importlib.metadata.version(name) for name in ("scikit-learn", "scipy", "numpy")
        },
        "classification": classifications,
        "probabilities": probabilities,
        "ranking": rankings,
        "mcnemar": paired,
        "wilson": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoded = json.dumps(references(), indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


if __name__ == "__main__":
    main()
