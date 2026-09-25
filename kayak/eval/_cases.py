"""Evaluate complete Choice and ranking requests against supplied case judgments.

The caller owns execution and backend lifetime. This module snapshots inputs,
isolates callback failures, and scores retained outputs without sending labels
to a backend. The fixed-classification Suite/Report artifact format is separate.
"""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .. import decisions
from ..decisions import DecisionRequest, DecisionResult, ModelInfo
from ..errors import InferenceError, KayakError, RemoteError
from ..ranking import RankingRequest, RankingResult
from ._ranking import RankedOutput, ranking_metrics
from ._schema import digest


class Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)


def check_labels(labels: list[str], candidates: dict[str, str]) -> None:
    if not labels or len(set(labels)) != len(labels) or not set(labels) <= candidates.keys():
        raise ValueError("labels must be nonempty, unique, and present in the request")


class ChoiceCase(Record):
    id: str
    request: DecisionRequest
    expected: dict[str, list[str]]

    @model_validator(mode="after")
    def valid_labels(self) -> Self:
        if self.expected.keys() != self.request.questions.keys():
            raise ValueError("expected must label every question, with no extra questions")
        for question_id, labels in self.expected.items():
            check_labels(labels, self.request.questions[question_id].criteria)
        return self


class RankingCase(Record):
    id: str
    request: RankingRequest
    relevant: list[str]

    @model_validator(mode="after")
    def valid_labels(self) -> Self:
        check_labels(self.relevant, self.request.candidates)
        return self


def _check_identity(name: str, identifiers: list[str]) -> None:
    if not name.strip() or any(not identifier.strip() for identifier in identifiers):
        raise ValueError("dataset name and case IDs must not be blank")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case IDs must be unique within a dataset")


class ChoiceDataset(Record):
    name: str
    kind: Literal["choice"]
    examples: list[str] = Field(default_factory=list)
    cases: list[ChoiceCase] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_identity(self) -> Self:
        _check_identity(self.name, [case.id for case in self.cases])
        return self


class RankingDataset(Record):
    name: str
    kind: Literal["ranking"]
    examples: list[str] = Field(default_factory=list)
    k: int = Field(ge=1)
    cases: list[RankingCase] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_identity(self) -> Self:
        _check_identity(self.name, [case.id for case in self.cases])
        return self


Dataset = ChoiceDataset | RankingDataset
DATASET: TypeAdapter[Dataset] = TypeAdapter(Annotated[Dataset, Field(discriminator="kind")])
CaseResult = DecisionResult | RankingResult | RankedOutput


def parse_cases(content: str | bytes) -> Dataset:
    """Validate a complete JSON case dataset without filesystem or backend effects."""
    return DATASET.validate_json(content)


def choice_metrics(
    expected: dict[str, list[str]], result: DecisionResult | None
) -> dict[str, float]:
    matches = [
        result is not None and result.answers[question_id].choice in labels
        for question_id, labels in expected.items()
    ]
    return {"exact_match": float(all(matches)), "question_accuracy": sum(matches) / len(matches)}


class CaseReport(Record):
    """One observation bound to its request, judgments, and metric cutoff.

    ``case_sha256`` is an additive consistency check, not authentication of the
    observation or its execution. Retain the original dataset with the report.
    """

    id: str
    case_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected: dict[str, list[str]] | list[str]
    result: CaseResult | None
    error: dict[str, str | int | None] | None
    duration_seconds: float = Field(ge=0)
    metrics: dict[str, float]
    input_order_metrics: dict[str, float] | None


def _case_digest(case: ChoiceCase | RankingCase, k: int) -> str:
    return digest({"case": case.model_dump(mode="json"), "k": k})


@dataclass
class _ResultIdentity:
    """Retain the first observed representation and declared execution identity."""

    kind: type[CaseResult] | None = None
    model: ModelInfo | None = None
    provenance: dict[str, str] | None = None

    def check(self, result: CaseResult) -> str | None:
        if self.kind is None:
            self.kind = type(result)
        elif type(result) is not self.kind:
            return "ResultKindChanged"
        if isinstance(result, (DecisionResult, RankingResult)):
            if self.model is None:
                self.model = result.model
            elif result.model != self.model:
                return "ModelChanged"
        elif result.provenance:
            if self.provenance is None:
                self.provenance = dict(result.provenance)
            elif result.provenance != self.provenance:
                return "ProvenanceChanged"
        return None


def _case_metrics(
    case: ChoiceCase | RankingCase, result: CaseResult | None, failed: bool, k: int
) -> tuple[dict[str, float], dict[str, float] | None]:
    if isinstance(case, ChoiceCase):
        decision = result if isinstance(result, DecisionResult) and not failed else None
        return choice_metrics(case.expected, decision), None
    ordered = []
    if not failed:
        if isinstance(result, RankingResult):
            ordered = [candidate.id for candidate in result.ranked]
        elif isinstance(result, RankedOutput):
            ordered = result.ids
    return (
        ranking_metrics(case.relevant, ordered, k),
        ranking_metrics(case.relevant, list(case.request.candidates), k),
    )


def _checked_result(request: DecisionRequest | RankingRequest, result: CaseResult) -> CaseResult:
    """Snapshot and validate callback outputs against the unmodified request."""
    if isinstance(request, DecisionRequest):
        if not isinstance(result, DecisionResult):
            raise TypeError("decide must return a DecisionResult")
        decision = DecisionResult.model_validate(result.model_dump())
        decisions.check_result(request, decision)
        return decision

    if isinstance(result, RankingResult):
        ranked = RankingResult.model_validate(result.model_dump())
        identifiers = [candidate.id for candidate in ranked.ranked]
        scores = {candidate.id: candidate.score for candidate in ranked.ranked}
        if set(identifiers) != request.candidates.keys():
            raise InferenceError("ranking must contain every supplied candidate exactly once")
        if identifiers != sorted(request.candidates, key=scores.__getitem__, reverse=True):
            raise InferenceError("native ranking ties must preserve candidate order")
        return ranked
    if isinstance(result, RankedOutput):
        output = RankedOutput.model_validate(result.model_dump())
        if set(output.ids) != request.candidates.keys():
            raise InferenceError("ranking must contain every supplied candidate exactly once")
        return output
    raise TypeError("rank must return a RankingResult or RankedOutput")


def _call(
    request: DecisionRequest | RankingRequest,
    decide: Callable[[DecisionRequest], DecisionResult] | None,
    rank: Callable[[RankingRequest], RankingResult | RankedOutput] | None,
    sync: Callable[[], None] | None,
) -> tuple[CaseResult | None, dict[str, str | int | None] | None, float]:
    """Isolate one callback and its completion hook, retaining only bounded errors."""
    callback_request = request.model_copy(deep=True)
    seconds = 0.0
    try:
        if sync is not None:
            sync()
        started = perf_counter()
        call_failed = False
        try:
            try:
                if isinstance(callback_request, DecisionRequest):
                    if decide is None:
                        raise ValueError("Choice datasets require a decide callback")
                    result: CaseResult = decide(callback_request)
                else:
                    if rank is None:
                        raise ValueError("ranking datasets require a rank callback")
                    result = rank(callback_request)
            except BaseException:
                call_failed = True
                raise
            finally:
                if sync is not None:
                    try:
                        sync()
                    except Exception:
                        if not call_failed:
                            raise
        finally:
            seconds = perf_counter() - started
        return _checked_result(request, result), None, seconds
    except Exception as exc:
        request_id = exc.request_id if isinstance(exc, KayakError) else None
        error: dict[str, str | int | None] = {
            "type": type(exc).__name__[:128],
            "request_id": request_id[:128] if request_id is not None else None,
        }
        if isinstance(exc, RemoteError):
            error.update(status_code=exc.status_code, code=exc.code[:128])
        return None, error, seconds


def evaluate_cases(
    dataset: Dataset,
    *,
    decide: Callable[[DecisionRequest], DecisionResult] | None = None,
    rank: Callable[[RankingRequest], RankingResult | RankedOutput] | None = None,
    sync: Callable[[], None] | None = None,
) -> list[CaseReport]:
    """Call once per case; failed outputs remain in the quality denominator.

    Only the callback for the dataset kind is required. It receives a fresh
    request, never case IDs or expected labels. No retries or backend cleanup
    occur here. Optional synchronization runs before and after each callback;
    duration covers the callback and completion hook, excluding preparation,
    the initial synchronization, and output validation. Interruptions propagate.

    Native model identity and nonempty adapter provenance are compared with
    their first observed values; changing output representation is a failure.
    Missing provenance remains unknown. Recorded provenance is the caller's
    declaration, not verified execution identity.
    """
    dataset = DATASET.validate_python(dataset.model_dump())
    if isinstance(dataset, ChoiceDataset) and decide is None:
        raise ValueError("Choice datasets require a decide callback")
    if isinstance(dataset, RankingDataset) and rank is None:
        raise ValueError("ranking datasets require a rank callback")
    reports: list[CaseReport] = []
    identity = _ResultIdentity()
    k = dataset.k if isinstance(dataset, RankingDataset) else 1
    for case in dataset.cases:
        result, error, duration = _call(case.request, decide, rank, sync)
        if result is not None:
            changed = identity.check(result)
            if changed is not None:
                error = {"type": changed}
        metrics, input_order_metrics = _case_metrics(case, result, error is not None, k)
        expected = case.expected if isinstance(case, ChoiceCase) else case.relevant
        reports.append(
            CaseReport(
                id=case.id,
                case_sha256=_case_digest(case, k),
                expected=expected,
                result=result,
                error=error,
                duration_seconds=duration,
                metrics=metrics,
                input_order_metrics=input_order_metrics,
            )
        )
    return reports


def summarize(dataset: Dataset, reports: list[CaseReport]) -> dict[str, float | int]:
    """Verify case bindings and recomputed metrics before summarizing a full run.

    Changed inputs, judgments, cutoffs, and inconsistent metrics are rejected.
    This detects inconsistent records; it cannot authenticate supplied outputs.
    """
    dataset = DATASET.validate_python(dataset.model_dump())
    if [report.id for report in reports] != [case.id for case in dataset.cases]:
        raise ValueError("summary requires one report per dataset case, in dataset order")
    reports = [CaseReport.model_validate(report.model_dump()) for report in reports]
    cases: list[ChoiceCase | RankingCase] = list(dataset.cases)
    k = dataset.k if isinstance(dataset, RankingDataset) else 1
    identity = _ResultIdentity()
    for case, report in zip(cases, reports, strict=True):
        expected = case.expected if isinstance(case, ChoiceCase) else case.relevant
        if report.expected != expected:
            raise ValueError("report judgments differ from the dataset")
        if report.case_sha256 != _case_digest(case, k):
            raise ValueError("report case hash differs from the request, judgments, or cutoff")
        if report.result is None and report.error is None:
            raise ValueError("case report has neither a result nor an error")
        if report.result is not None:
            result = _checked_result(case.request, report.result)
            changed = identity.check(result)
            recorded = report.error.get("type") if report.error is not None else None
            if changed is not None and recorded != changed:
                raise ValueError("report does not retain the observed identity change")
            if changed is None and recorded in {
                "ModelChanged",
                "ResultKindChanged",
                "ProvenanceChanged",
            }:
                raise ValueError("report identity error contradicts the retained outputs")
        measured, baseline = _case_metrics(case, report.result, report.error is not None, k)
        if report.metrics != measured or report.input_order_metrics != baseline:
            raise ValueError("report metrics differ from the retained outputs and judgments")
    metrics = {
        name: sum(report.metrics[name] for report in reports) / len(reports)
        for name in reports[0].metrics
    }
    if isinstance(dataset, ChoiceDataset):
        # Cases can contain different numbers of questions; count each question once.
        questions = sum(len(case.expected) for case in dataset.cases)
        metrics["question_accuracy"] = (
            sum(
                report.metrics["question_accuracy"] * len(case.expected)
                for case, report in zip(dataset.cases, reports, strict=True)
            )
            / questions
        )
    else:
        baselines = [report.input_order_metrics for report in reports]
        metrics.update(
            {
                f"input_order_{name}": sum(
                    baseline[name] for baseline in baselines if baseline is not None
                )
                / len(reports)
                for name in reports[0].metrics
            }
        )
    return {
        "cases": len(reports),
        "failed": sum(report.error is not None for report in reports),
        **metrics,
    }
