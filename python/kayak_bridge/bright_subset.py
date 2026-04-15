from __future__ import annotations

from .cache_paths import configure_local_caches

configure_local_caches()

from datasets import load_dataset

from .colbert_encoder import DEFAULT_MODEL_NAME
from .retrieval_task_builder import build_retrieval_subset_task


DEFAULT_DATASET_ID = "xlangai/BRIGHT"
DEFAULT_DOMAIN = "stackoverflow"


def build_bright_colbert_subset(
    query_limit: int = 8,
    negative_doc_limit: int = 128,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
    domain: str = DEFAULT_DOMAIN,
) -> dict:
    queries_dataset = load_dataset(dataset_id, "examples", split=domain)
    documents_dataset = load_dataset(dataset_id, "documents", split=domain)

    documents_by_id = {
        str(row["id"]): str(row["content"]) for row in documents_dataset
    }

    selected_queries: list[dict[str, object]] = []
    positive_doc_ids: list[str] = []
    seen_positive_doc_ids: set[str] = set()

    for row in queries_dataset:
        relevant_doc_ids = [
            str(doc_id)
            for doc_id in row["gold_ids"]
            if str(doc_id) in documents_by_id
        ]
        if not relevant_doc_ids:
            continue

        selected_queries.append(
            {
                "query_id": str(row["id"]),
                "text": str(row["query"]),
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
                "text": str(row["content"]),
            }
        )
        negative_doc_count += 1
        if negative_doc_count == negative_doc_limit:
            break

    return build_retrieval_subset_task(
        family="bright",
        slice_name="bright_stackoverflow_real_subset",
        why=(
            "Real BRIGHT StackOverflow subset encoded with ColBERTv2 on CPU. "
            "This adds a reasoning-oriented coding retrieval slice while keeping "
            "the public benchmark loop small enough for repeated runs."
        ),
        primary_metric="ndcg",
        k=10,
        dataset_id=dataset_id + "/" + domain,
        model_name=model_name,
        documents=documents,
        queries=selected_queries,
    )
