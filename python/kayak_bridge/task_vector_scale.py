"""Owns deterministic document-vector inflation helpers for benchmark sweeps.

These helpers keep document ids, texts, and judged labels fixed while
increasing only the number of document vectors. They are useful when the
question is "does the engine stay faster as token density grows?" rather than
"does it stay faster as document count grows?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class VectorScaledTaskBuildResult:
    task: dict[str, Any]
    document_vector_multiplier: int
    inflation_policy: str


def build_vector_scaled_task(
    task: Mapping[str, Any],
    *,
    document_vector_multiplier: int,
    inflation_policy: str = "repeat_document_vectors_in_place",
) -> VectorScaledTaskBuildResult:
    """Scale only document vectors by repeating each document's vectors.

    Reason:
    - MaxSim over a multiset of identical document vectors is unchanged
    - this isolates vector-count growth from document-count growth
    - query vectors stay fixed, so the judged task semantics remain stable
    """

    if document_vector_multiplier <= 0:
        raise ValueError("document_vector_multiplier must be positive")

    if document_vector_multiplier == 1:
        return VectorScaledTaskBuildResult(
            task=dict(task),
            document_vector_multiplier=document_vector_multiplier,
            inflation_policy=inflation_policy,
        )

    scaled_documents: list[dict[str, Any]] = []
    total_document_vector_count = 0
    for document in task["documents"]:
        base_vectors = list(document["vectors"])
        scaled_vectors = base_vectors * document_vector_multiplier
        total_document_vector_count += len(scaled_vectors)
        scaled_documents.append(
            {
                **document,
                "vectors": scaled_vectors,
                "vector_count": len(scaled_vectors),
            }
        )

    scaled_task = dict(task)
    scaled_task["documents"] = scaled_documents
    scaled_task["nominal_document_vector_count"] = round(
        total_document_vector_count / float(len(scaled_documents))
    )
    return VectorScaledTaskBuildResult(
        task=scaled_task,
        document_vector_multiplier=document_vector_multiplier,
        inflation_policy=inflation_policy,
    )
