"""Model-neutral prediction exchange; importing predictions never claims execution."""

import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from ._runner import load_report
from ._schema import Record, Report, Suite


class Prediction(Record):
    """One first-attempt prediction. None means an unanswered/failed example."""

    id: str = Field(min_length=1)
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    ranking: list[str] | None = None


class PredictionSet(Record):
    """Explicit task, system/method identity, and optional probability/ranking evidence."""

    schema_version: Literal[1] = 1
    system: str = Field(min_length=1)
    method: str = Field(min_length=1)
    suite: Suite
    predictions: list[Prediction]
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_predictions(self) -> Self:
        if not self.system.strip() or not self.method.strip():
            raise ValueError("system and method must be nonblank")
        labels = set(self.suite.question.criteria)
        examples = {example.id for example in self.suite.examples}
        seen: set[str] = set()
        for row in self.predictions:
            if row.id not in examples or row.id in seen:
                raise ValueError("prediction IDs must be unique and belong to the suite")
            seen.add(row.id)
            if row.choice is None:
                if row.probabilities is not None or row.ranking is not None:
                    raise ValueError("unanswered examples cannot supply probabilities or rankings")
                continue
            if row.choice not in labels:
                raise ValueError("predicted choices must belong to the candidate labels")
            if row.probabilities is not None:
                values = row.probabilities.values()
                if set(row.probabilities) != labels:
                    raise ValueError("probabilities must cover exactly the candidate labels")
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                    raise ValueError("probabilities must be finite and within [0, 1]")
                if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError("probabilities must sum to one")
            if row.ranking is not None and (
                len(row.ranking) != len(labels)
                or set(row.ranking) != labels
                or row.ranking[0] != row.choice
            ):
                raise ValueError("ranking must list every candidate once, with the choice first")
        return self


def predictions_from_report(report: Report, *, system: str, method: str) -> PredictionSet:
    """Convert first attempts; callers should verify saved reports with load_report first."""
    predictions = []
    for row in report.observations:
        result = row.attempts[0].result
        if result is None:
            predictions.append(Prediction(id=row.id))
            continue
        answer = result.answers["intent"]
        predictions.append(
            Prediction(
                id=row.id,
                choice=answer.choice,
                probabilities=dict(answer.probabilities),
                ranking=sorted(answer.scores, key=answer.scores.__getitem__, reverse=True),
            )
        )
    return PredictionSet(
        system=system,
        method=method,
        suite=report.suite.model_copy(deep=True),
        predictions=predictions,
        metadata={
            "origin": "Kayak report first measured attempts",
            "original_status": report.status,
            "original_schema_version": str(report.schema_version),
            "evidence_kind": str(report.config.get("evidence_kind", "unspecified")),
        },
    )


def export_predictions(
    run: str | Path, output: str | Path, *, system: str, method: str
) -> PredictionSet:
    """Verify a Kayak run and write portable predictions to a new JSON file."""
    predictions = predictions_from_report(load_report(run), system=system, method=method)
    encoded = predictions.model_dump_json(indent=2) + "\n"
    with Path(output).open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return predictions
