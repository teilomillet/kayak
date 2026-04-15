from __future__ import annotations

from .cache_paths import configure_local_caches

configure_local_caches()

from datasets import load_dataset

from .colbert_encoder import DEFAULT_MODEL_NAME
from .retrieval_task_builder import build_retrieval_subset_task


DEFAULT_DATASET_ID = "R2MED/Biology"


def build_r2med_biology_colbert_subset(
    query_limit: int = 8,
    negative_doc_limit: int = 128,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict:
    queries_dataset = load_dataset(dataset_id, "query", split="query")
    documents_dataset = load_dataset(dataset_id, "corpus", split="corpus")
    qrels_dataset = load_dataset(dataset_id, "qrels", split="qrels")

    documents_by_id = {str(row["id"]): str(row["text"]) for row in documents_dataset}

    relevant_doc_ids_by_query: dict[str, list[str]] = {}
    for row in qrels_dataset:
        doc_id = str(row["p_id"])
        if int(row["score"]) <= 0 or doc_id not in documents_by_id:
            continue

        query_id = str(row["q_id"])
        relevant_doc_ids = relevant_doc_ids_by_query.setdefault(query_id, [])
        if doc_id not in relevant_doc_ids:
            relevant_doc_ids.append(doc_id)

    selected_queries: list[dict[str, object]] = []
    positive_doc_ids: list[str] = []
    seen_positive_doc_ids: set[str] = set()

    for row in queries_dataset:
        query_id = str(row["id"])
        relevant_doc_ids = relevant_doc_ids_by_query.get(query_id, [])
        if not relevant_doc_ids:
            continue

        selected_queries.append(
            {
                "query_id": query_id,
                "text": str(row["text"]),
                "relevant_doc_ids": relevant_doc_ids,
            }
        )

        for doc_id in relevant_doc_ids:
            if doc_id in seen_positive_doc_ids:
                continue
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
                "text": str(row["text"]),
            }
        )
        negative_doc_count += 1
        if negative_doc_count == negative_doc_limit:
            break

    return build_retrieval_subset_task(
        family="r2med",
        slice_name="r2med_biology_real_subset",
        why=(
            "Real R2MED Biology subset encoded with ColBERTv2 on CPU. "
            "This adds a biomedical reasoning retrieval slice while keeping "
            "the encoded task compact enough for repeated local benchmarking."
        ),
        primary_metric="ndcg",
        k=10,
        dataset_id=dataset_id,
        model_name=model_name,
        documents=documents,
        queries=selected_queries,
    )
