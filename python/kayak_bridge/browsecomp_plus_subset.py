from __future__ import annotations

from .browsecomp_plus_decrypt import (
    DEFAULT_DATASET_ID,
    iter_decrypted_browsecomp_plus_rows,
)
from .colbert_encoder import DEFAULT_MODEL_NAME
from .retrieval_task_builder import build_retrieval_subset_task


def _document_record(document: dict[str, str]) -> dict[str, str]:
    return {
        "doc_id": str(document["docid"]),
        "text": str(document["text"]),
    }


def build_browsecomp_plus_colbert_subset(
    query_limit: int = 4,
    negative_doc_limit: int = 16,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict:
    selected_queries = []
    documents_by_id: dict[str, dict[str, str]] = {}

    for row in iter_decrypted_browsecomp_plus_rows(query_limit, dataset_id):
        evidence_doc_ids: list[str] = []

        for document in row["evidence_docs"]:
            doc_id = str(document["docid"])
            if doc_id in evidence_doc_ids:
                continue

            evidence_doc_ids.append(doc_id)
            documents_by_id.setdefault(doc_id, _document_record(document))

        negative_count = 0
        for document in row["negative_docs"]:
            if negative_count == negative_doc_limit:
                break

            doc_id = str(document["docid"])
            if doc_id in evidence_doc_ids:
                continue

            documents_by_id.setdefault(doc_id, _document_record(document))
            negative_count += 1

        selected_queries.append(
            {
                "query_id": str(row["query_id"]),
                "text": str(row["query"]),
                "relevant_doc_ids": evidence_doc_ids,
            }
        )

    if not selected_queries:
        raise ValueError("BrowseComp-Plus selection produced zero queries")

    return build_retrieval_subset_task(
        family="browsecomp_plus",
        slice_name="browsecomp_plus_evidence_slice",
        why=(
            "Official BrowseComp-Plus retrieval slice using decrypted benchmark "
            "queries, human-verified evidence documents, and the benchmark's "
            "curated hard negatives. This is a light slice, not the full 100k-doc "
            "agent benchmark."
        ),
        primary_metric="ndcg",
        k=10,
        dataset_id=dataset_id + "/evidence-slice",
        model_name=model_name,
        documents=list(documents_by_id.values()),
        queries=selected_queries,
    )
