from __future__ import annotations

from .cache_paths import configure_local_caches

configure_local_caches()

from datasets import load_dataset

from .colbert_encoder import DEFAULT_MODEL_NAME
from .retrieval_task_builder import build_retrieval_subset_task


DEFAULT_DATASET_ID = "mteb/LEMBNarrativeQARetrieval"


def _render_document_text(title: object, text: object) -> str:
    rendered_title = str(title).strip()
    rendered_text = str(text)
    if not rendered_title:
        return rendered_text
    return rendered_title + "\n" + rendered_text


def build_lemb_narrativeqa_colbert_subset(
    query_limit: int = 8,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict:
    queries_dataset = load_dataset(dataset_id, "queries", split="test")
    documents_dataset = load_dataset(dataset_id, "corpus", split="test")
    qrels_dataset = load_dataset(dataset_id, "qrels", split="test")

    documents = []
    document_ids: set[str] = set()
    for row in documents_dataset:
        doc_id = str(row["_id"])
        document_ids.add(doc_id)
        documents.append(
            {
                "doc_id": doc_id,
                "text": _render_document_text(row.get("title", ""), row["text"]),
            }
        )

    relevant_doc_ids_by_query: dict[str, list[str]] = {}
    for row in qrels_dataset:
        doc_id = str(row["corpus-id"])
        if int(row["score"]) <= 0 or doc_id not in document_ids:
            continue

        query_id = str(row["query-id"])
        relevant_doc_ids = relevant_doc_ids_by_query.setdefault(query_id, [])
        if doc_id not in relevant_doc_ids:
            relevant_doc_ids.append(doc_id)

    selected_queries: list[dict[str, object]] = []
    for row in queries_dataset:
        query_id = str(row["_id"])
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
        if len(selected_queries) == query_limit:
            break

    return build_retrieval_subset_task(
        family="lemb",
        slice_name="lemb_narrativeqa_real_subset",
        why=(
            "Real LEMB NarrativeQA subset encoded with ColBERTv2 on CPU. "
            "This adds a long-document retrieval slice from the LongEmbed "
            "benchmark family while keeping the public benchmark loop small "
            "enough for repeated runs."
        ),
        primary_metric="ndcg",
        k=10,
        dataset_id=dataset_id + "/test",
        model_name=model_name,
        documents=documents,
        queries=selected_queries,
    )
