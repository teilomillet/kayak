"""Portable RAG experiment records; configuration and observations have explicit owners."""

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from ._rag import RAGAssessment, RAGJudgments, RAGTrace
from ._schema import digest


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)


class _VersionedRecord(_Record):
    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def supported_version(cls, value: object) -> int:
        # Literal equality alone also accepts True and 1.0 in Python.
        if type(value) is not int or value != 1:
            raise ValueError("schema_version must be the integer 1")
        return value


class RAGInput(_Record):
    """Pipeline-visible input. Parameters are application data, never evaluation gold."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input ID and query must not be blank")
        return value


class RAGCase(_Record):
    """Inputs and independently supplied reference material for one task."""

    input: RAGInput
    judgments: RAGJudgments = Field(default_factory=RAGJudgments)
    reference_answer: str | None = None


class RAGDataset(_VersionedRecord):
    name: str = Field(min_length=1)
    cases: list[RAGCase] = Field(min_length=1)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_identity(self) -> Self:
        if not self.name.strip():
            raise ValueError("dataset name must not be blank")
        identifiers = [case.input.id for case in self.cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("case IDs must be unique")
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class RAGGate(_Record):
    """An explicit threshold on a known-value mean, with required observation coverage."""

    metric: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
    minimum: float | None = None
    maximum: float | None = None
    min_coverage: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def valid_bounds(self) -> Self:
        if self.minimum is None and self.maximum is None:
            raise ValueError("a gate needs a minimum or maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("gate minimum must not exceed maximum")
        return self


class RAGEvalConfig(_VersionedRecord):
    """Evaluator-owned knobs and a snapshot of caller-owned system settings.

    `system` is retained for reproduction; it does not configure a backend.
    Application code constructs its pipeline from those settings explicitly.
    Async concurrency is bounded; synchronous execution requires a value of one.
    """

    k: int = Field(default=5, ge=1)
    repeats: int = Field(default=1, ge=1, le=100)
    max_concurrency: int = Field(default=1, ge=1, le=100)
    gates: list[RAGGate] = Field(default_factory=list)
    system: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_gates(self) -> Self:
        names = [gate.metric for gate in self.gates]
        if len(names) != len(set(names)):
            raise ValueError("configure each metric gate once")
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class RAGOutput(_VersionedRecord):
    """One final observation and optional earlier steps, in recorded order.

    Steps retain rewritten queries and their own sources without inventing
    causal edges. Final-task judgments are not applied to intermediate queries.
    The wrapper leaves existing RAGTrace hashes unchanged.
    """

    final: RAGTrace
    steps: list[RAGTrace] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_steps(self) -> Self:
        identifiers = [step.id for step in self.steps] + [self.final.id]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("intermediate and final trace IDs must be distinct")
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class RAGScore(_Record):
    """A judge's native finite score; unavailable values need an explicit reason."""

    value: float | None
    reason: str | None = None

    @model_validator(mode="after")
    def explain_unknown(self) -> Self:
        if self.value is None and (self.reason is None or not self.reason.strip()):
            raise ValueError("an unavailable score needs a reason")
        return self


class RAGReview(_Record):
    """Caller judgments bound to the task, references, and all observed outputs.

    Correctness and grounding are optional independent Boolean judgments.
    Named scores retain their native scale; thresholds are explicit config.
    Provenance records the reviewer, rubric, model, or other declared identity.
    """

    review_input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    correct: bool | None = None
    grounded: bool | None = None
    scores: dict[str, RAGScore] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_scores(self) -> Self:
        if self.correct is None and self.grounded is None and not self.scores:
            raise ValueError("a review needs a Boolean judgment or named score")
        if any(re.fullmatch(r"[a-z][a-z0-9_]*", name) is None for name in self.scores):
            raise ValueError("score names must use lowercase letters, digits, and underscores")
        return self


class RAGReviewInput(_Record):
    """The reviewer can inspect references; the pipeline never receives this record."""

    case: RAGCase
    output: RAGOutput

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class RAGReviewRecord(_Record):
    """One external review addressed to its case and repetition."""

    case_id: str = Field(min_length=1)
    repeat: int = Field(default=0, ge=0)
    review: RAGReview


class RAGFailure(_Record):
    phase: Literal["pipeline", "output", "review"]
    type: str = Field(min_length=1, max_length=128)


class RAGAttempt(_Record):
    """One completed attempt; absent records mean execution was not observed.

    Pipeline and review errors are separate. Raw exception messages are not
    retained. Durations are caller-observed wall time, not verified device time.
    """

    input: RAGInput
    repeat: int = Field(default=0, ge=0)
    system: dict[str, JsonValue] = Field(default_factory=dict)
    output: RAGOutput | None = None
    error: RAGFailure | None = None
    review: RAGReview | None = None
    review_error: RAGFailure | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    review_duration_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def coherent_outcomes(self) -> Self:
        if (self.output is None) == (self.error is None):
            raise ValueError("an attempt needs either a pipeline output or an error")
        if self.error is not None and self.error.phase == "review":
            raise ValueError("review failures belong in review_error")
        if self.review_error is not None and self.review_error.phase != "review":
            raise ValueError("review_error must describe the review phase")
        if self.review is not None and self.review_error is not None:
            raise ValueError("a review cannot have both an output and an error")
        if self.output is None and (self.review is not None or self.review_error is not None):
            raise ValueError("review outcomes require a pipeline output")
        if self.output is None and self.review_duration_seconds is not None:
            raise ValueError("review timing requires a pipeline output")
        return self

    @property
    def case_id(self) -> str:
        return self.input.id


class RAGCaseResult(_Record):
    attempt: RAGAttempt
    assessment: RAGAssessment | None


class RAGMetricSummary(_Record):
    """Descriptive statistics across eligible attempts, not independent task samples."""

    mean: float | None
    observed: int = Field(ge=0)
    unknown: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    minimum: float | None
    maximum: float | None
    stddev: float | None


class RAGGateResult(_Record):
    metric: str
    status: Literal["passed", "failed", "unknown"]
    reason: str


class RAGSummary(_Record):
    planned: int = Field(ge=0)
    recorded: int = Field(ge=0)
    missing: int = Field(ge=0)
    pipeline_failures: int = Field(ge=0)
    review_failures: int = Field(ge=0)
    stage_errors: dict[str, int]
    diagnostic_excluded: int = Field(ge=0)
    metrics: dict[str, RAGMetricSummary]
    unstable_cases: list[str]


class RAGReport(_VersionedRecord):
    """Versioned experiment evidence, separate from the classification Report format."""

    dataset: RAGDataset
    config: RAGEvalConfig
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    attempts_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: Literal["complete", "failed", "incomplete"]
    passed: bool | None
    results: list[RAGCaseResult]
    summary: RAGSummary
    gates: list[RAGGateResult]
