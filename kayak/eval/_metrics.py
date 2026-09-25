"""Quality uses distinct examples; timing uses individual successful calls."""

import math
import statistics

from ._schema import Report


def distribution(seconds: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(seconds)
    p95 = None
    if len(ordered) >= 20:
        position = (len(ordered) - 1) * 0.95
        lower = int(position)
        p95 = ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (
            position - lower
        )
    return {
        "calls": len(ordered),
        "mean_seconds": statistics.mean(ordered) if ordered else None,
        "median_seconds": statistics.median(ordered) if ordered else None,
        "p95_seconds": p95,
        "min_seconds": min(ordered) if ordered else None,
        "max_seconds": max(ordered) if ordered else None,
    }


def summarize(report: Report, *, legacy: bool = False) -> dict[str, object]:
    """Derive metrics from retained attempts, without inference or artifact writes.

    Use ``load_report`` first to verify saved evidence. ``legacy`` reproduces the
    schema 1/2 summary exactly for the artifact verifier.
    """
    labels = list(report.suite.question.criteria)
    examples = {example.id: example for example in report.suite.examples}
    rows = {row.id: row for row in report.observations}
    choices_by_example: list[str | None] = []
    top5 = unstable = call_errors = 0
    successful_seconds: list[float] = []
    attempt_seconds: list[float] = []
    max_score_drift = 0.0
    for identifier, example in examples.items():
        row = rows.get(identifier)
        result = row.attempts[0].result if row else None
        choices_by_example.append(result.answers["intent"].choice if result else None)
        if result is not None:
            answer = result.answers["intent"]
            ranked = sorted(answer.scores, key=answer.scores.__getitem__, reverse=True)
            top5 += example.label in ranked[:5]
        if row:
            choices: set[str] = set()
            reference_scores: dict[str, float] | None = None
            for attempt in row.attempts:
                attempt_seconds.append(attempt.seconds)
                if attempt.result is None:
                    call_errors += 1
                    continue
                successful_seconds.append(attempt.seconds)
                answer = attempt.result.answers["intent"]
                choices.add(answer.choice)
                if reference_scores is None:
                    reference_scores = answer.scores
                else:
                    max_score_drift = max(
                        max_score_drift,
                        max(
                            abs(score - reference_scores[key])
                            for key, score in answer.scores.items()
                        ),
                    )
            unstable += len(choices) > 1
    summary = classification_summary(
        tuple(labels),
        tuple(example.label for example in examples.values()),
        tuple(choices_by_example),
        legacy=legacy,
    )
    summary.update(
        {
            "attempted_examples": len(rows),
            "top5_accuracy": top5 / len(examples),
            "failed_calls": call_errors,
            "failed_warmups": sum(attempt.error is not None for attempt in report.warmups),
            "unstable_examples": unstable,
            "max_repeated_score_drift": max_score_drift,
            "latency": distribution(successful_seconds),
            "attempt_latency": distribution(attempt_seconds),
            "warmup_latency": distribution([attempt.seconds for attempt in report.warmups]),
            "successful_calls_per_inference_second": (
                len(successful_seconds) / sum(attempt_seconds) if sum(attempt_seconds) else None
            ),
        }
    )
    return summary


def classification_summary(
    labels: tuple[str, ...],
    references: tuple[str, ...],
    choices: tuple[str | None, ...],
    *,
    legacy: bool = False,
) -> dict[str, object]:
    """Shared arithmetic for native and external classification predictions."""
    matrix = {label: dict.fromkeys([*labels, "__failed__"], 0) for label in labels}
    for gold, choice in zip(references, choices, strict=True):
        matrix[gold][choice if choice is not None else "__failed__"] += 1
    total = len(references)
    correct = sum(matrix[label][label] for label in labels)
    failures = sum(matrix[label]["__failed__"] for label in labels)
    supports = {label: sum(matrix[label].values()) for label in labels}
    predicted_counts = {label: sum(matrix[gold][label] for gold in labels) for label in labels}
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        support = supports[label]
        predicted_count = predicted_counts[label]
        tp = matrix[label][label]
        per_class[label] = {
            "support": support,
            "precision": tp / predicted_count if predicted_count else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / (support + predicted_count) if support + predicted_count else 0.0,
        }
    summary: dict[str, object] = {
        "examples": total,
        "correct": correct,
        "failed_or_missing_examples": failures,
        "accuracy": correct / total,
        "macro_f1": statistics.mean(float(row["f1"]) for row in per_class.values()),
        "macro_f1_labels": "all candidates, including zero-support labels",
        "labels_with_examples": sum(row["support"] > 0 for row in per_class.values()),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
    if legacy:
        return summary

    # MCC uses the union of gold and predicted categories. Failures form an
    # additional predicted category with zero gold support, never dropped rows.
    covariance = correct * total - sum(
        supports[label] * predicted_counts[label] for label in labels
    )
    gold_variance = total**2 - sum(count**2 for count in supports.values())
    predicted_variance = (
        total**2 - sum(count**2 for count in predicted_counts.values()) - failures**2
    )
    denominator = math.sqrt(gold_variance * predicted_variance)
    summary.update(
        {
            "balanced_accuracy": statistics.mean(
                float(per_class[label]["recall"]) for label in labels if supports[label]
            ),
            "weighted_f1": sum(supports[label] * float(per_class[label]["f1"]) for label in labels)
            / total,
            "matthews_correlation": covariance / denominator if denominator else 0.0,
            "zero_recall_labels": [
                label for label in labels if supports[label] and matrix[label][label] == 0
            ],
            "predicted_label_count": sum(count > 0 for count in predicted_counts.values()),
            "largest_prediction_share": max(predicted_counts.values()) / total,
            "uniform_chance_accuracy": 1.0 / len(labels),
            "majority_class_accuracy": max(supports.values()) / total,
        }
    )
    return summary
