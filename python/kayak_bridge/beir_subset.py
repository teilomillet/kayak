from __future__ import annotations

from collections import defaultdict

from .cache_paths import configure_local_caches

configure_local_caches()

import ir_datasets

from .colbert_encoder import DEFAULT_MODEL_NAME, encode_document_text, encode_query_text


def _format_document_text(document) -> str:
    parts: list[str] = []

    title = getattr(document, "title", None)
    text = getattr(document, "text", None)

    if title:
        parts.append(title.strip())
    if text:
        parts.append(text.strip())

    return "\n".join(parts)


def _positive_qrels_by_query(dataset) -> dict[str, list[str]]:
    qrels_by_query: dict[str, list[str]] = defaultdict(list)

    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels_by_query[str(qrel.query_id)].append(str(qrel.doc_id))

    return dict(qrels_by_query)


def _select_queries(dataset, qrels_by_query: dict[str, list[str]], query_limit: int):
    selected_queries = []

    for query in dataset.queries_iter():
        query_id = str(query.query_id)
        if query_id not in qrels_by_query:
            continue

        selected_queries.append(query)
        if len(selected_queries) == query_limit:
            break

    return selected_queries


def _collect_positive_doc_ids(
    selected_queries, qrels_by_query: dict[str, list[str]]
) -> list[str]:
    seen_doc_ids: set[str] = set()
    doc_ids: list[str] = []

    for query in selected_queries:
        for doc_id in qrels_by_query[str(query.query_id)]:
            if doc_id in seen_doc_ids:
                continue

            seen_doc_ids.add(doc_id)
            doc_ids.append(doc_id)

    return doc_ids


def _collect_negative_doc_ids(
    dataset, excluded_doc_ids: set[str], negative_doc_limit: int
) -> list[str]:
    negative_doc_ids: list[str] = []

    for document in dataset.docs_iter():
        doc_id = str(document.doc_id)
        if doc_id in excluded_doc_ids:
            continue

        negative_doc_ids.append(doc_id)
        if len(negative_doc_ids) == negative_doc_limit:
            break

    return negative_doc_ids


def _encode_documents(dataset, doc_ids: list[str], model_name: str):
    store = dataset.docs_store()
    encoded_documents = []

    for doc_id in doc_ids:
        document = store.get(doc_id)
        text = _format_document_text(document)
        vectors = encode_document_text(text, model_name)
        encoded_documents.append(
            {
                "doc_id": doc_id,
                "text": text,
                "vector_count": len(vectors),
                "vectors": vectors,
            }
        )

    return encoded_documents


def _encode_queries(selected_queries, qrels_by_query: dict[str, list[str]], model_name: str):
    encoded_queries = []

    for query in selected_queries:
        query_id = str(query.query_id)
        vectors = encode_query_text(query.text, model_name)
        encoded_queries.append(
            {
                "query_id": query_id,
                "text": query.text,
                "relevant_doc_ids": list(qrels_by_query[query_id]),
                "vector_count": len(vectors),
                "vectors": vectors,
            }
        )

    return encoded_queries


def _mean_vector_count(items: list[dict]) -> int:
    if not items:
        return 0

    total = 0
    for item in items:
        total += int(item["vector_count"])

    return round(total / len(items))


def build_beir_colbert_subset(
    *,
    dataset_id: str,
    slice_name: str,
    why: str,
    query_limit: int,
    negative_doc_limit: int,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    dataset = ir_datasets.load(dataset_id)
    qrels_by_query = _positive_qrels_by_query(dataset)
    selected_queries = _select_queries(dataset, qrels_by_query, query_limit)

    positive_doc_ids = _collect_positive_doc_ids(selected_queries, qrels_by_query)
    negative_doc_ids = _collect_negative_doc_ids(
        dataset, set(positive_doc_ids), negative_doc_limit
    )

    documents = _encode_documents(dataset, positive_doc_ids + negative_doc_ids, model_name)
    queries = _encode_queries(selected_queries, qrels_by_query, model_name)

    vector_dim = 0
    if documents and documents[0]["vectors"]:
        vector_dim = len(documents[0]["vectors"][0])

    return {
        "family": "beir",
        "slice_name": slice_name,
        "why": why,
        "primary_metric": "mrr",
        "k": 10,
        "nominal_query_vector_count": _mean_vector_count(queries),
        "nominal_document_vector_count": _mean_vector_count(documents),
        "vector_dim": vector_dim,
        "dataset_id": dataset_id,
        "model_name": model_name,
        "documents": documents,
        "queries": queries,
    }
