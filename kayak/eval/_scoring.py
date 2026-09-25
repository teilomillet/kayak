"""Pure, extensible metrics over immutable labeled predictions."""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from ._metrics import classification_summary
from ._predictions import PredictionSet


class MetricResult(TypedDict):
    value: float | None
    direction: Literal["higher", "lower"]
    version: str
    description: str
    parameters: dict[str, str | int | float]
    requires: Literal["labels", "probabilities", "ranking"]
    available_examples: int
    total_examples: int
    unavailable_reason: str | None


@dataclass(frozen=True)
class MetricSample:
    id: str
    label: str
    choice: str | None
    probabilities: tuple[float, ...] | None
    ranking: tuple[str, ...] | None


@dataclass(frozen=True)
class MetricInput:
    """All selected examples, including missing predictions; vectors follow labels."""

    labels: tuple[str, ...]
    samples: tuple[MetricSample, ...]


@dataclass(frozen=True)
class Metric:
    """A named, versioned pure function. No registry or dynamic code loading."""

    name: str
    compute: Callable[[MetricInput], float | None]
    description: str
    direction: Literal["higher", "lower"] = "higher"
    version: str = "1"
    requires: Literal["labels", "probabilities", "ranking"] = "labels"
    parameters: tuple[tuple[str, str | int | float], ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("metric names must use lowercase letters, digits, and underscores")
        if not self.version.strip() or not self.description.strip():
            raise ValueError("metrics require a version and description")
        if self.direction not in {"higher", "lower"}:
            raise ValueError("metric direction must be higher or lower")
        if self.requires not in {"labels", "probabilities", "ranking"}:
            raise ValueError("unknown metric requirement")
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("metric parameter names must be unique")
        if any(
            isinstance(value, float) and not math.isfinite(value) for _, value in self.parameters
        ):
            raise ValueError("metric parameters must be finite")


def metric_input(predictions: PredictionSet) -> MetricInput:
    """Snapshot validated external/native data into immutable extension inputs."""
    predictions = PredictionSet.model_validate(predictions.model_dump())
    labels = tuple(predictions.suite.question.criteria)
    rows = {row.id: row for row in predictions.predictions}
    samples = []
    for example in predictions.suite.examples:
        row = rows.get(example.id)
        probabilities = None
        ranking = None
        if row is not None:
            if row.probabilities is not None:
                probabilities = tuple(row.probabilities[label] for label in labels)
                ranking = tuple(sorted(labels, key=row.probabilities.__getitem__, reverse=True))
            if row.ranking is not None:
                ranking = tuple(row.ranking)
        samples.append(
            MetricSample(
                example.id, example.label, row.choice if row else None, probabilities, ranking
            )
        )
    return MetricInput(labels, tuple(samples))


def classification(data: MetricInput) -> dict[str, object]:
    return classification_summary(
        data.labels,
        tuple(row.label for row in data.samples),
        tuple(row.choice for row in data.samples),
    )


def score_metrics(data: MetricInput, metrics: Sequence[Metric]) -> dict[str, MetricResult]:
    """Never drop missing rows or substitute an invented probability distribution."""
    if not metrics or len({metric.name for metric in metrics}) != len(metrics):
        raise ValueError("select at least one metric, with unique names")
    results: dict[str, MetricResult] = {}
    for metric in metrics:
        available = len(data.samples)
        if metric.requires == "probabilities":
            available = sum(row.probabilities is not None for row in data.samples)
        elif metric.requires == "ranking":
            available = sum(row.ranking is not None for row in data.samples)
        value = None
        reason = None
        if available != len(data.samples):
            reason = f"requires {metric.requires} for every selected example"
        else:
            value = metric.compute(data)
            if value is None:
                reason = "undefined for this input (for example, no accepted predictions)"
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"metric {metric.name} must return a finite number or None")
            else:
                value = float(value)
        results[metric.name] = {
            "value": value,
            "direction": metric.direction,
            "version": metric.version,
            "description": metric.description,
            "parameters": dict(metric.parameters),
            "requires": metric.requires,
            "available_examples": available,
            "total_examples": len(data.samples),
            "unavailable_reason": reason,
        }
    return results


def top_k_accuracy(k: int) -> Metric:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    def compute(data: MetricInput) -> float:
        return sum(row.label in (row.ranking or ())[:k] for row in data.samples) / len(data.samples)

    return Metric(
        f"top{k}_accuracy",
        compute,
        "Gold label in the first min(k, labels) candidates.",
        requires="ranking",
        parameters=(("k", k),),
    )


def log_loss(*, epsilon: float = 1e-15) -> Metric:
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be finite and strictly between zero and one")

    def compute(data: MetricInput) -> float:
        losses = []
        for row in data.samples:
            assert row.probabilities is not None
            probability = row.probabilities[data.labels.index(row.label)]
            losses.append(-math.log(max(epsilon, probability)))
        return math.fsum(losses) / len(losses)

    return Metric(
        "log_loss",
        compute,
        "Mean negative natural log of the gold-label probability.",
        direction="lower",
        requires="probabilities",
        parameters=(("epsilon", epsilon),),
    )


def _brier(data: MetricInput) -> float:
    losses = []
    for row in data.samples:
        assert row.probabilities is not None
        losses.append(
            math.fsum(
                (probability - float(label == row.label)) ** 2
                for label, probability in zip(data.labels, row.probabilities, strict=True)
            )
        )
    return math.fsum(losses) / len(losses)


def reliability_bins(data: MetricInput, *, bins: int = 10) -> list[dict[str, float | int | None]]:
    """Equal-width [lower, upper) bins, final bin includes 1; confidence is chosen-label p."""
    if type(bins) is not int or not 1 <= bins <= 1000:
        raise ValueError("bins must be an integer between 1 and 1000")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for row in data.samples:
        if row.probabilities is None or row.choice is None:
            raise ValueError("reliability bins require probabilities for every selected example")
        confidence = row.probabilities[data.labels.index(row.choice)]
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, row.choice == row.label))
    return [
        {
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "examples": len(bucket),
            "mean_confidence": math.fsum(p for p, _ in bucket) / len(bucket) if bucket else None,
            "accuracy": sum(correct for _, correct in bucket) / len(bucket) if bucket else None,
        }
        for index, bucket in enumerate(buckets)
    ]


def expected_calibration_error(*, bins: int = 10) -> Metric:
    if type(bins) is not int or not 1 <= bins <= 1000:
        raise ValueError("bins must be an integer between 1 and 1000")

    def compute(data: MetricInput) -> float:
        total = 0.0
        for bucket in reliability_bins(data, bins=bins):
            accuracy, confidence, count = (
                bucket["accuracy"],
                bucket["mean_confidence"],
                bucket["examples"],
            )
            if accuracy is not None and confidence is not None and count is not None:
                total += count * abs(accuracy - confidence)
        return total / len(data.samples)

    return Metric(
        "ece",
        compute,
        "Chosen-label confidence ECE; bin-dependent, not a calibration proof.",
        direction="lower",
        requires="probabilities",
        parameters=(("bins", bins),),
    )


def confidence_metrics(threshold: float) -> tuple[Metric, Metric]:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be finite and within [0, 1]")

    def accepted(data: MetricInput) -> list[MetricSample]:
        return [
            row
            for row in data.samples
            if row.choice is not None
            and row.probabilities is not None
            and row.probabilities[data.labels.index(row.choice)] >= threshold
        ]

    def accuracy(data: MetricInput) -> float | None:
        selected = accepted(data)
        return (
            sum(row.choice == row.label for row in selected) / len(selected) if selected else None
        )

    def coverage(data: MetricInput) -> float:
        return len(accepted(data)) / len(data.samples)

    parameters = (("threshold", threshold),)
    return (
        Metric(
            "selective_accuracy",
            accuracy,
            "Accuracy among predictions meeting the threshold.",
            requires="probabilities",
            parameters=parameters,
        ),
        Metric(
            "coverage",
            coverage,
            "Accepted predictions divided by all selected examples.",
            requires="probabilities",
            parameters=parameters,
        ),
    )


def default_metrics() -> tuple[Metric, ...]:
    metrics = []
    for name, description in (
        ("accuracy", "Correct predictions divided by all selected examples."),
        ("macro_f1", "Mean F1 over all candidate labels, including zero-support labels."),
        ("balanced_accuracy", "Mean recall over supported labels."),
        ("weighted_f1", "Mean per-label F1 weighted by gold support."),
        ("matthews_correlation", "Multiclass MCC, retaining unanswered examples."),
    ):

        def compute(data: MetricInput, key: str = name) -> float:
            value = classification(data)[key]
            assert isinstance(value, float)
            return value

        metrics.append(Metric(name, compute, description))
    return (
        *metrics,
        top_k_accuracy(3),
        top_k_accuracy(5),
        log_loss(),
        Metric(
            "brier_score",
            _brier,
            "Mean sum of squared class-probability errors, range [0, 2].",
            direction="lower",
            requires="probabilities",
            parameters=(("scale_by_half", "false"),),
        ),
        expected_calibration_error(),
    )
