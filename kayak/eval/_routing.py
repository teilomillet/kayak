"""Failure-aware intent routing and optional out-of-scope score diagnostics."""

import math
from collections.abc import Mapping
from itertools import groupby

from ._benchmark import accuracy_interval
from ._predictions import PredictionSet
from ._scoring import MetricInput, classification, metric_input


def _rate(numerator: int, denominator: int, reason: str | None = None) -> dict[str, object]:
    if denominator == 0:
        reason = "denominator is zero"
    return {
        "value": numerator / denominator if reason is None else None,
        "numerator": numerator,
        "denominator": denominator,
        "interval_95": accuracy_interval(numerator, denominator) if reason is None else None,
        "unavailable_reason": reason,
    }


def _detection(
    data: MetricInput, reject_label: str, scores: Mapping[str, float] | None, complete: bool
) -> dict[str, object]:
    result: dict[str, object] = {
        "auroc": None,
        "average_precision": None,
        "fpr_at_95_recall": None,
        "curve": [],
        "score_direction": "higher means out of scope; scores need not be calibrated",
        "unavailable_reason": None,
    }
    if scores is None:
        result["unavailable_reason"] = "no rejection scores supplied"
        return result
    scores = dict(scores)
    if set(scores) != {row.id for row in data.samples}:
        raise ValueError("rejection scores must cover exactly every selected example ID")
    try:
        invalid = any(
            type(value) not in (int, float) or not math.isfinite(value) for value in scores.values()
        )
    except OverflowError:
        invalid = True
    if invalid:
        raise ValueError("rejection scores must be finite machine numbers, not booleans")
    positives = sum(row.label == reject_label for row in data.samples)
    negatives = len(data.samples) - positives
    if not complete:
        result["unavailable_reason"] = "requires complete predictions; failures are never dropped"
        return result
    if positives == 0 or negatives == 0:
        result["unavailable_reason"] = "requires both in-scope and out-of-scope gold examples"
        return result
    ordered = sorted(
        ((scores[row.id], row.label == reject_label) for row in data.samples), reverse=True
    )
    true_positives = false_positives = 0
    previous_recall = previous_fpr = area = average_precision = 0.0
    fpr95 = None
    curve: list[dict[str, float | None]] = [
        {"threshold": None, "recall": 0.0, "false_positive_rate": 0.0, "precision": None}
    ]
    for threshold, tied in groupby(ordered, key=lambda pair: pair[0]):
        # A threshold accepts the entire tie group: no label-dependent tie breaking.
        for _, positive in tied:
            true_positives += positive
            false_positives += not positive
        recall = true_positives / positives
        fpr = false_positives / negatives
        precision = true_positives / (true_positives + false_positives)
        area += (fpr - previous_fpr) * (recall + previous_recall) / 2
        average_precision += (recall - previous_recall) * precision
        if fpr95 is None and recall >= 0.95:
            fpr95 = fpr
        curve.append(
            {
                "threshold": threshold,
                "recall": recall,
                "false_positive_rate": fpr,
                "precision": precision,
            }
        )
        previous_recall, previous_fpr = recall, fpr
    result.update(
        auroc=area, average_precision=average_precision, fpr_at_95_recall=fpr95, curve=curve
    )
    return result


def routing_summary(
    predictions: PredictionSet,
    *,
    reject_label: str,
    rejection_scores: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Report routing decisions without treating missing answers as rejections.

    Scores are optional, ID-keyed scalars with higher meaning out of scope.
    Detection curves group ties and are test diagnostics, not threshold tuning.
    Missing answers count as unsuccessful in accuracy/recall/F1. Harm rates are
    unavailable when missing answers could conceal harm in that population.
    """
    predictions = PredictionSet.model_validate(predictions.model_dump())
    data = metric_input(predictions)
    if reject_label not in data.labels:
        raise ValueError("reject_label must belong to the suite's candidate labels")
    counts = dict.fromkeys(
        [
            "in_scope",
            "out_of_scope",
            "correct_routes",
            "wrong_routes",
            "false_rejections",
            "correct_rejections",
            "false_accepts",
            "in_scope_failures",
            "out_of_scope_failures",
        ],
        0,
    )
    for row in data.samples:
        if row.label == reject_label:
            counts["out_of_scope"] += 1
            key = (
                "out_of_scope_failures"
                if row.choice is None
                else ("correct_rejections" if row.choice == reject_label else "false_accepts")
            )
        else:
            counts["in_scope"] += 1
            key = (
                "in_scope_failures"
                if row.choice is None
                else (
                    "false_rejections"
                    if row.choice == reject_label
                    else ("correct_routes" if row.choice == row.label else "wrong_routes")
                )
            )
        counts[key] += 1
    ins, oos = counts["in_scope"], counts["out_of_scope"]
    tp, fp = counts["correct_rejections"], counts["false_rejections"]
    failures = counts["in_scope_failures"] + counts["out_of_scope_failures"]
    routed = counts["correct_routes"] + counts["wrong_routes"] + counts["false_accepts"]
    complete = (
        failures == 0 and predictions.metadata.get("original_status", "complete") == "complete"
    )
    fn = oos - tp  # Includes missing OOS answers: an execution failure is not successful detection.
    f1_denominator = 2 * tp + fp + fn
    rates = {
        "in_scope_accuracy": _rate(counts["correct_routes"], ins),
        "out_of_scope_recall": _rate(tp, oos),
        "out_of_scope_precision": _rate(tp, tp + fp),
        "out_of_scope_false_accept_rate": _rate(
            counts["false_accepts"],
            oos,
            "missing out-of-scope decisions" if counts["out_of_scope_failures"] else None,
        ),
        "in_scope_false_reject_rate": _rate(
            fp,
            ins,
            "missing in-scope decisions" if counts["in_scope_failures"] else None,
        ),
        "routing_precision": _rate(counts["correct_routes"], routed),
        "routing_coverage": _rate(routed, len(data.samples)),
        "failure_rate": _rate(failures, len(data.samples)),
    }
    return {
        "schema_version": 1,
        "suite_sha256": predictions.suite.sha256,
        "system": predictions.system,
        "method": predictions.method,
        "reject_label": reject_label,
        "complete": complete,
        "counts": counts,
        "rates": rates,
        "out_of_scope_f1": 2 * tp / f1_denominator if f1_denominator else None,
        "classification": classification(data),
        "unobserved_gold_labels": sorted(set(data.labels) - {row.label for row in data.samples}),
        "detection": _detection(data, reject_label, rejection_scores, complete),
        "notes": [
            "Missing predictions remain in the selected denominator; they are not rejections.",
            "Macro F1 includes every candidate, including unobserved gold labels.",
            "Wilson intervals assume independent representative examples; "
            "duplicates weaken this assumption.",
            "AP is recall-increment weighted precision, not trapezoidal PR area.",
            "FPR at 95% OOS recall uses the first attainable tied-score threshold, "
            "without interpolation.",
            "Test curves diagnose scores; select deployment thresholds "
            "on separate labeled development data.",
        ],
    }
