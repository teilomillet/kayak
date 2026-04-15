from __future__ import annotations

from .cache_paths import configure_local_caches

configure_local_caches()

from datasets import load_dataset

from .colbert_encoder import DEFAULT_MODEL_NAME
from .retrieval_task_builder import build_retrieval_subset_task


DEFAULT_DATASET_ID = "isaacus/legal-rag-bench"


def _render_document_text(title: object, text: object) -> str:
    rendered_title = str(title).strip()
    rendered_text = str(text)
    if not rendered_title:
        return rendered_text
    return rendered_title + "\n" + rendered_text


def build_legal_rag_bench_colbert_subset(
    query_limit: int = 8,
    negative_doc_limit: int = 128,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict:
    queries_dataset = load_dataset(dataset_id, "qa", split="test")
    documents_dataset = load_dataset(dataset_id, "corpus", split="test")

    documents_by_id = {
        str(row["id"]): _render_document_text(row.get("title", ""), row["text"])
        for row in documents_dataset
    }

    selected_queries: list[dict[str, object]] = []
    positive_doc_ids: list[str] = []
    seen_positive_doc_ids: set[str] = set()

    for row in queries_dataset:
        doc_id = str(row["relevant_passage_id"])
        if doc_id not in documents_by_id:
            continue

        selected_queries.append(
            {
                "query_id": str(row["id"]),
                "text": str(row["question"]),
                "relevant_doc_ids": [doc_id],
            }
        )

        if doc_id not in seen_positive_doc_ids:
            seen_positive_doc_ids.add(doc_id)
            positive_doc_ids.append(doc_id)

        if len(selected_queries) == query_limit:
            break

    documents = [
        {"doc_id": doc_id, "text": documents_by_id[doc_id]}
        for doc_id in positive_doc_ids
    ]

    excluded_doc_ids = set(positive_doc_ids)
    negative_doc_count = 0
    for row in documents_dataset:
        doc_id = str(row["id"])
        if doc_id in excluded_doc_ids:
            continue

        documents.append(
            {
                "doc_id": doc_id,
                "text": documents_by_id[doc_id],
            }
        )
        negative_doc_count += 1
        if negative_doc_count == negative_doc_limit:
            break

    return build_retrieval_subset_task(
        family="legal_rag_bench",
        slice_name="legal_rag_bench_real_subset",
        why=(
            "Real Legal RAG Bench subset encoded with ColBERTv2 on CPU. "
            "This adds a legal reasoning retrieval slice while keeping the "
            "encoded task small enough for repeated local benchmarking."
        ),
        primary_metric="mrr",
        k=10,
        dataset_id=dataset_id + "/qa",
        model_name=model_name,
        documents=documents,
        queries=selected_queries,
    )
