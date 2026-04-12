from __future__ import annotations

from collections import defaultdict

from .cache_paths import configure_local_caches

configure_local_caches()

from datasets import load_dataset

from .colbert_encoder import DEFAULT_MODEL_NAME
from .retrieval_task_builder import build_retrieval_subset_task


DEFAULT_DATASET_ID = "orionweller/LIMIT-small"


def _positive_qrels_by_query(qrels_dataset) -> dict[str, list[str]]:
    qrels_by_query: dict[str, list[str]] = defaultdict(list)

    for row in qrels_dataset:
        if int(row["score"]) <= 0:
            continue

        qrels_by_query[str(row["query-id"])].append(str(row["corpus-id"]))

    return dict(qrels_by_query)


def build_limit_small_colbert_subset(
    query_limit: int = 32,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict:
    corpus_dataset = load_dataset(dataset_id, "corpus", split="corpus")
    queries_dataset = load_dataset(dataset_id, "queries", split="queries")
    qrels_dataset = load_dataset(dataset_id, split="test")

    qrels_by_query = _positive_qrels_by_query(qrels_dataset)
    selected_queries = []

    for row in queries_dataset:
        query_id = str(row["_id"])
        if query_id not in qrels_by_query:
            continue

        selected_queries.append(
            {
                "query_id": query_id,
                "text": str(row["text"]),
                "relevant_doc_ids": list(qrels_by_query[query_id]),
            }
        )
        if len(selected_queries) == query_limit:
            break

    documents = [
        {"doc_id": str(row["_id"]), "text": str(row["text"])} for row in corpus_dataset
    ]

    return build_retrieval_subset_task(
        family="limit",
        slice_name="limit_small_real_subset",
        why=(
            "Official LIMIT-small public slice encoded with ColBERTv2 on CPU. "
            "This keeps the MTEB-style retrieval setting while staying light enough "
            "for repeated smoke runs."
        ),
        primary_metric="ndcg",
        k=10,
        dataset_id=dataset_id,
        model_name=model_name,
        documents=documents,
        queries=selected_queries,
    )
