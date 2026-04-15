"""Counts that make Python search-stage behavior inspectable."""

from __future__ import annotations

from dataclasses import dataclass

from .stage_artifact_materialization import StageArtifactMaterialization


@dataclass(frozen=True, slots=True)
class SearchStageProfile:
    """Measured counts for one explicit search stage in the Python pipeline."""

    stage_name: str
    input_hit_count: int
    output_hit_count: int
    query_vector_count: int
    document_count: int
    document_vector_count: int
    document_text_count: int = 0
    materialized_artifacts: tuple[StageArtifactMaterialization, ...] = ()

    def __post_init__(self) -> None:
        if self.input_hit_count < 0:
            raise ValueError("stage input_hit_count must be non-negative")
        if self.output_hit_count < 0:
            raise ValueError("stage output_hit_count must be non-negative")
        if self.query_vector_count < 0:
            raise ValueError("stage query_vector_count must be non-negative")
        if self.document_count < 0:
            raise ValueError("stage document_count must be non-negative")
        if self.document_vector_count < 0:
            raise ValueError("stage document_vector_count must be non-negative")
        if self.document_text_count < 0:
            raise ValueError("stage document_text_count must be non-negative")
