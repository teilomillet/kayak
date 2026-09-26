"""Generate OOS references without importing Kayak (optional development packages).

    uv run --no-project --with scikit-learn==1.9.1 --with numpy==2.5.3 \
        python benchmarks/routing_references.py --output /tmp/routing-reference.json
    cmp /tmp/routing-reference.json tests/fixtures/routing-reference.json

OOS is the positive class. AP is non-interpolated; FPR95 uses the first attainable
threshold, retaining every tie group. Undefined populations are tested separately.
"""

import argparse
import importlib
import importlib.metadata
import json
import random
from pathlib import Path


def references() -> dict[str, object]:
    metrics = importlib.import_module("sklearn.metrics")
    rng = random.Random(20260926)
    cases = []
    fixed = [([0, 1], [0.0, 1.0]), ([0, 1], [1.0, 0.0]), ([0, 1], [0.5, 0.5])]
    for index in range(40):
        gold = [0, 1] + [rng.randrange(2) for _ in range(index * 3)]
        scores = [rng.randint(-4, 4) / 4 for _ in gold]
        fixed.append((gold, scores))
    for gold, scores in fixed:
        choices = [int(score >= 0.5) for score in scores]
        fpr, tpr, _ = metrics.roc_curve(gold, scores, drop_intermediate=False)
        cases.append(
            {
                "gold": gold,
                "scores": scores,
                "choices": choices,
                "auroc": float(metrics.roc_auc_score(gold, scores)),
                "average_precision": float(metrics.average_precision_score(gold, scores)),
                "fpr_at_95_recall": next(
                    float(f) for f, t in zip(fpr, tpr, strict=True) if t >= 0.95
                ),
                "fpr": list(map(float, fpr)),
                "recall": list(map(float, tpr)),
                "out_of_scope_f1": float(metrics.f1_score(gold, choices, zero_division=0)),
            }
        )
    return {
        "seed": 20260926,
        "versions": {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy")},
        "cases": cases,
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
