"""Versioned inputs and raw observations for dataset evaluations."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..decisions import Choice, DecisionRequest, DecisionResult, ModelInfo


def digest(value: object) -> str:
    # Mapping order matters: candidate order defines the tie-breaking contract.
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Example(Record):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)


class Suite(Record):
    """One fixed classification task; every example receives the same candidates."""

    name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    question: Choice
    examples: list[Example] = Field(min_length=1)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_examples(self) -> Self:
        if "__failed__" in self.question.criteria:
            raise ValueError("__failed__ is reserved for missing predictions")
        if len({example.id for example in self.examples}) != len(self.examples):
            raise ValueError("example IDs must be unique")
        for example in self.examples:
            if not example.id.strip() or example.label not in self.question.criteria:
                raise ValueError("examples need nonblank IDs and labels from the candidate set")
            DecisionRequest(state=example.text, questions={"intent": self.question})
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class Protocol(Record):
    warmups: int = Field(default=1, ge=0, le=100)
    repeats: int = Field(default=1, ge=1, le=100)
    seed: int = 42


class Failure(Record):
    type: str
    request_id: str | None = None


class Attempt(Record):
    seconds: float = Field(ge=0)
    result: DecisionResult | None = None
    error: Failure | None = None

    @model_validator(mode="after")
    def one_outcome(self) -> Self:
        if (self.result is None) == (self.error is None):
            raise ValueError("an attempt must have exactly one result or error")
        return self


class Observation(Record):
    id: str
    attempts: list[Attempt] = Field(min_length=1)


class PredictionArtifact(Record):
    """Exact prediction bytes committed by the latest metadata checkpoint."""

    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(ge=0)


class Report(Record):
    schema_version: Literal[1, 2, 3] = 1
    created_at: str
    status: Literal["running", "complete", "failed", "interrupted"] = "running"
    suite: Suite
    suite_sha256: str
    protocol: Protocol
    transport: Literal["local", "http", "custom"]
    model: ModelInfo | None = None
    environment: dict[str, object]
    config: dict[str, object] = Field(default_factory=dict)
    warmups: list[Attempt] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    prediction_artifact: PredictionArtifact | None = None
    memory: dict[str, Annotated[int, Field(ge=0)] | None] = Field(default_factory=dict)
    summary: dict[str, object] = Field(default_factory=dict)
    error: str | None = None

    @field_validator("environment", "config", "summary", mode="before")
    @classmethod
    def json_metadata(cls, value: object) -> object:
        # Validate before Pydantic's JSON serializer can turn NaN into null.
        # The round trip also detaches nested containers owned by the caller.
        try:
            encoded = json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain finite JSON values") from exc
        snapshot: object = json.loads(encoded)
        return snapshot

    @model_validator(mode="after")
    def suite_identity(self) -> Self:
        if self.suite.sha256 != self.suite_sha256:
            raise ValueError("suite identity does not match its recorded hash")
        if self.schema_version >= 2 and self.prediction_artifact is None:
            raise ValueError(
                "version 2/3 reports require a prediction artifact hash and byte count"
            )
        if self.schema_version == 1 and self.prediction_artifact is not None:
            raise ValueError("legacy reports do not declare prediction artifact integrity")
        return self
