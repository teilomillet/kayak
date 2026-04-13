"""Explicit candidate-window artifact materialization for Python stage results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageArtifactMaterialization:
    family: str
    document_count: int
    document_vector_count: int = 0
    document_text_count: int = 0

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("materialization family must be non-empty")
        if self.document_count < 0:
            raise ValueError("materialization document_count must be non-negative")
        if self.document_vector_count < 0:
            raise ValueError(
                "materialization document_vector_count must be non-negative"
            )
        if self.document_text_count < 0:
            raise ValueError(
                "materialization document_text_count must be non-negative"
            )
