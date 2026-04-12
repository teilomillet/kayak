from __future__ import annotations

from .colbert_encoder import DEFAULT_MODEL_NAME, encode_query_text
from .browsecomp_plus_decrypt import (
    DEFAULT_DATASET_ID,
    iter_decrypted_browsecomp_plus_rows,
)
from .retrieval_task_builder import encode_documents


def _document_record(document: dict[str, str]) -> dict[str, str]:
    return {
        "doc_id": str(document["docid"]),
        "text": str(document["text"]),
    }


def _mean_vector_count(items: list[dict]) -> int:
    if not items:
        return 0

    total = 0
    for item in items:
        total += int(item["vector_count"])

    return round(total / len(items))


def _unique_doc_ids(documents: list[dict[str, str]]) -> list[str]:
    doc_ids: list[str] = []

    for document in documents:
        doc_id = str(document["docid"])
        if doc_id in doc_ids:
            continue

        doc_ids.append(doc_id)

    return doc_ids


def build_browsecomp_plus_encoded_slice(
    query_limit: int = 4,
    negative_doc_limit: int = 16,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict:
    raw_queries = []
    documents_by_id: dict[str, dict[str, str]] = {}

    for row in iter_decrypted_browsecomp_plus_rows(query_limit, dataset_id):
        for document in row["evidence_docs"]:
            documents_by_id.setdefault(
                str(document["docid"]), _document_record(document)
            )

        for document in row["gold_docs"]:
            documents_by_id.setdefault(
                str(document["docid"]), _document_record(document)
            )

        negative_count = 0
        for document in row["negative_docs"]:
            if negative_count == negative_doc_limit:
                break

            doc_id = str(document["docid"])
            if doc_id in documents_by_id:
                continue

            documents_by_id.setdefault(doc_id, _document_record(document))
            negative_count += 1

        raw_queries.append(
            {
                "query_id": str(row["query_id"]),
                "text": str(row["query"]),
                "evidence_doc_ids": _unique_doc_ids(row["evidence_docs"]),
                "gold_doc_ids": _unique_doc_ids(row["gold_docs"]),
            }
        )

    if not raw_queries:
        raise ValueError("BrowseComp-Plus selection produced zero queries")

    encoded_documents = encode_documents(list(documents_by_id.values()), model_name)
    encoded_queries = []

    for query in raw_queries:
        vectors = encode_query_text(str(query["text"]), model_name)
        encoded_queries.append(
            {
                "query_id": str(query["query_id"]),
                "text": str(query["text"]),
                "vector_count": len(vectors),
                "vectors": vectors,
                "evidence_doc_ids": list(query["evidence_doc_ids"]),
                "gold_doc_ids": list(query["gold_doc_ids"]),
            }
        )

    vector_dim = 0
    if encoded_documents and encoded_documents[0]["vectors"]:
        vector_dim = len(encoded_documents[0]["vectors"][0])

    return {
        "family": "browsecomp_plus",
        "dataset_id": dataset_id,
        "model_name": model_name,
        "documents": encoded_documents,
        "queries": encoded_queries,
        "nominal_query_vector_count": _mean_vector_count(encoded_queries),
        "nominal_document_vector_count": _mean_vector_count(encoded_documents),
        "vector_dim": vector_dim,
    }


def build_browsecomp_plus_task_from_encoded_slice(
    encoded_slice: dict,
    *,
    relevance_kind: str,
) -> dict:
    if relevance_kind not in {"evidence", "gold"}:
        raise ValueError(f"unknown BrowseComp-Plus relevance kind: {relevance_kind}")

    slice_name = f"browsecomp_plus_{relevance_kind}_slice"
    why = (
        "Official BrowseComp-Plus retrieval slice using decrypted benchmark "
        "queries, human-verified evidence documents, gold documents, and the "
        "benchmark's curated hard negatives. This is a light slice, not the "
        "full 100k-doc agent benchmark."
    )

    if relevance_kind == "gold":
        why = (
            "Official BrowseComp-Plus gold retrieval slice using decrypted "
            "benchmark queries, answer-containing gold documents, supporting "
            "evidence documents as distractors, and the benchmark's curated "
            "hard negatives. This is a light slice, not the full 100k-doc "
            "agent benchmark."
        )

    queries = []
    for query in encoded_slice["queries"]:
        queries.append(
            {
                "query_id": str(query["query_id"]),
                "text": str(query["text"]),
                "relevant_doc_ids": list(query[f"{relevance_kind}_doc_ids"]),
                "vector_count": int(query["vector_count"]),
                "vectors": query["vectors"],
            }
        )

    return {
        "family": "browsecomp_plus",
        "slice_name": slice_name,
        "why": why,
        "primary_metric": "ndcg",
        "k": 10,
        "nominal_query_vector_count": int(
            encoded_slice["nominal_query_vector_count"]
        ),
        "nominal_document_vector_count": int(
            encoded_slice["nominal_document_vector_count"]
        ),
        "vector_dim": int(encoded_slice["vector_dim"]),
        "dataset_id": str(encoded_slice["dataset_id"]) + f"/{relevance_kind}-slice",
        "model_name": str(encoded_slice["model_name"]),
        "documents": encoded_slice["documents"],
        "queries": queries,
    }


def build_browsecomp_plus_colbert_subset(
    query_limit: int = 4,
    negative_doc_limit: int = 16,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict:
    encoded_slice = build_browsecomp_plus_encoded_slice(
        query_limit, negative_doc_limit, model_name, dataset_id
    )
    return build_browsecomp_plus_task_from_encoded_slice(
        encoded_slice, relevance_kind="evidence"
    )


def build_browsecomp_plus_gold_colbert_subset(
    query_limit: int = 4,
    negative_doc_limit: int = 16,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict:
    encoded_slice = build_browsecomp_plus_encoded_slice(
        query_limit, negative_doc_limit, model_name, dataset_id
    )
    return build_browsecomp_plus_task_from_encoded_slice(
        encoded_slice, relevance_kind="gold"
    )
