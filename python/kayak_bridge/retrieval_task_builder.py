from __future__ import annotations

from .colbert_encoder import (
    DEFAULT_MODEL_NAME,
    encode_document_texts,
    encode_document_texts_with_token_ids,
    encode_query_text,
)


def _mean_vector_count(items: list[dict]) -> int:
    if not items:
        return 0

    total = 0
    for item in items:
        total += int(item["vector_count"])

    return round(total / len(items))


def encode_documents(
    documents: list[dict[str, str]],
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    include_token_ids: bool = False,
    batch_size: int = 8,
) -> list[dict]:
    texts = [str(document["text"]) for document in documents]
    if include_token_ids:
        encoded_rows = encode_document_texts_with_token_ids(
            texts,
            model_name,
            batch_size=batch_size,
        )
        return [
            _encoded_document_row(
                document=document,
                text=text,
                vectors=vectors,
                token_ids=token_ids,
            )
            for document, text, (vectors, token_ids) in zip(
                documents,
                texts,
                encoded_rows,
                strict=True,
            )
        ]

    encoded_rows_without_ids = encode_document_texts(
        texts,
        model_name,
        batch_size=batch_size,
    )
    return [
        _encoded_document_row(
            document=document,
            text=text,
            vectors=vectors,
            token_ids=None,
        )
        for document, text, vectors in zip(
            documents,
            texts,
            encoded_rows_without_ids,
            strict=True,
        )
    ]


def _encoded_document_row(
    *,
    document: dict[str, str],
    text: str,
    vectors: list[list[float]],
    token_ids: list[int] | None,
) -> dict:
    row = {
        "doc_id": str(document["doc_id"]),
        "text": text,
        "vector_count": len(vectors),
        "vectors": vectors,
    }
    if token_ids is not None:
        row["token_ids"] = token_ids
    return row


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
    include_document_token_ids: bool = False,
    document_batch_size: int = 8,
) -> dict:
    encoded_documents = encode_documents(
        documents,
        model_name,
        include_token_ids=include_document_token_ids,
        batch_size=document_batch_size,
    )
    encoded_queries = encode_queries(queries, model_name)

    vector_dim = 0
    if encoded_documents and encoded_documents[0]["vectors"]:
        vector_dim = len(encoded_documents[0]["vectors"][0])

    task = {
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
    if include_document_token_ids:
        task["document_token_ids"] = "colbert_doc_tokenizer_input_ids"
    return task
