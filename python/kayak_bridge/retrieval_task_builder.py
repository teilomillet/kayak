from __future__ import annotations

from .colbert_encoder import DEFAULT_MODEL_NAME, encode_document_text, encode_query_text


def _mean_vector_count(items: list[dict]) -> int:
    if not items:
        return 0

    total = 0
    for item in items:
        total += int(item["vector_count"])

    return round(total / len(items))


def encode_documents(
    documents: list[dict[str, str]], model_name: str = DEFAULT_MODEL_NAME
) -> list[dict]:
    encoded_documents = []

    for document in documents:
        text = str(document["text"])
        vectors = encode_document_text(text, model_name)
        encoded_documents.append(
            {
                "doc_id": str(document["doc_id"]),
                "text": text,
                "vector_count": len(vectors),
                "vectors": vectors,
            }
        )

    return encoded_documents


def encode_queries(
    queries: list[dict[str, object]], model_name: str = DEFAULT_MODEL_NAME
) -> list[dict]:
    encoded_queries = []

    for query in queries:
        text = str(query["text"])
        vectors = encode_query_text(text, model_name)
        encoded_queries.append(
            {
                "query_id": str(query["query_id"]),
                "text": text,
                "relevant_doc_ids": list(query["relevant_doc_ids"]),
                "vector_count": len(vectors),
                "vectors": vectors,
            }
        )

    return encoded_queries


def build_retrieval_subset_task(
    *,
    family: str,
    slice_name: str,
    why: str,
    primary_metric: str,
    k: int,
    documents: list[dict[str, str]],
    queries: list[dict[str, object]],
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str,
) -> dict:
    encoded_documents = encode_documents(documents, model_name)
    encoded_queries = encode_queries(queries, model_name)

    vector_dim = 0
    if encoded_documents and encoded_documents[0]["vectors"]:
        vector_dim = len(encoded_documents[0]["vectors"][0])

    return {
        "family": family,
        "slice_name": slice_name,
        "why": why,
        "primary_metric": primary_metric,
        "k": k,
        "nominal_query_vector_count": _mean_vector_count(encoded_queries),
        "nominal_document_vector_count": _mean_vector_count(encoded_documents),
        "vector_dim": vector_dim,
        "dataset_id": dataset_id,
        "model_name": model_name,
        "documents": encoded_documents,
        "queries": encoded_queries,
    }
