"""Owns deterministic task inflation helpers for scale sweeps.

These helpers grow document count without changing the judged queries. The
policy is intentionally explicit because scale claims are sensitive to how the
extra documents are chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def protected_doc_ids_for_scale_sweep(
    task: Mapping[str, Any],
    *,
    ranked_doc_ids_by_query: Sequence[Sequence[str]],
) -> frozenset[str]:
    protected: set[str] = set()

    for query in task["queries"]:
        protected.update(str(doc_id) for doc_id in query["relevant_doc_ids"])

    for ranked_doc_ids in ranked_doc_ids_by_query:
        protected.update(str(doc_id) for doc_id in ranked_doc_ids)

    return frozenset(protected)


def choose_repeatable_distractor_doc_ids(
    task: Mapping[str, Any],
    *,
    protected_doc_ids: frozenset[str],
) -> tuple[str, ...]:
    distractor_doc_ids = tuple(
        str(document["doc_id"])
        for document in task["documents"]
        if str(document["doc_id"]) not in protected_doc_ids
    )
    if not distractor_doc_ids:
        raise ValueError(
            "scale sweep needs at least one non-protected distractor document"
        )
    return distractor_doc_ids


def _mean_document_vector_count(documents: Sequence[Mapping[str, Any]]) -> int:
    total = sum(int(document["vector_count"]) for document in documents)
    return round(total / float(len(documents)))


@dataclass(frozen=True, slots=True)
class ScaledTaskBuildResult:
    task: dict[str, Any]
    base_document_count: int
    duplicated_document_count: int
    distractor_source_count: int
    inflation_policy: str


def build_scaled_task_with_document_copies(
    task: Mapping[str, Any],
    *,
    target_document_count: int,
    repeatable_doc_ids: Sequence[str],
    inflation_policy: str = "repeat_nonprotected_documents_with_unique_doc_ids",
) -> ScaledTaskBuildResult:
    base_documents = list(task["documents"])
    base_document_count = len(base_documents)
    if target_document_count < base_document_count:
        raise ValueError("target_document_count must be at least the base size")

    source_by_doc_id = {
        str(document["doc_id"]): document for document in base_documents
    }
    source_documents = [
        source_by_doc_id[str(doc_id)] for doc_id in repeatable_doc_ids
    ]
    if not source_documents:
        raise ValueError("repeatable_doc_ids must not be empty")

    scaled_documents = [dict(document) for document in base_documents]
    duplicated_document_count = 0
    while len(scaled_documents) < target_document_count:
        source = source_documents[duplicated_document_count % len(source_documents)]
        copy_index = duplicated_document_count + 1
        scaled_documents.append(
            {
                **source,
                "doc_id": f"{source['doc_id']}::scale_copy::{copy_index:06d}",
            }
        )
        duplicated_document_count += 1

    scaled_task = dict(task)
    scaled_task["documents"] = scaled_documents
    scaled_task["nominal_document_vector_count"] = _mean_document_vector_count(
        scaled_documents
    )

    return ScaledTaskBuildResult(
        task=scaled_task,
        base_document_count=base_document_count,
        duplicated_document_count=duplicated_document_count,
        distractor_source_count=len(source_documents),
        inflation_policy=inflation_policy,
    )
