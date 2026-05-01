"""Builds encoded Kayak tasks from local MS MARCO passage files.

This module owns local-file parsing for the official MS MARCO passage ranking
TSV/qrels files. It does not download the corpus or choose benchmark budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .colbert_encoder import DEFAULT_MODEL_NAME
from .retrieval_task_builder import build_retrieval_subset_task


DEFAULT_DATASET_ID = "msmarco-passage/local"


@dataclass(frozen=True, slots=True)
class MsmarcoPassagePaths:
    collection: Path
    queries: Path
    qrels: Path


@dataclass(frozen=True, slots=True)
class MsmarcoPassageSelection:
    documents: list[dict[str, str]]
    queries: list[dict[str, object]]
    source_document_count: int
    selected_document_count: int
    selected_query_count: int
    required_relevant_doc_count: int
    missing_required_relevant_doc_count: int

    def to_json_ready(self) -> dict[str, object]:
        return {
            "source_document_count": self.source_document_count,
            "selected_document_count": self.selected_document_count,
            "selected_query_count": self.selected_query_count,
            "required_relevant_doc_count": self.required_relevant_doc_count,
            "missing_required_relevant_doc_count": (
                self.missing_required_relevant_doc_count
            ),
        }


def parse_qrels_line(line: str) -> tuple[str, str, int]:
    parts = line.strip().split()
    if len(parts) == 4:
        query_id, _, doc_id, score = parts
    elif len(parts) == 3:
        query_id, doc_id, score = parts
    else:
        raise ValueError("MS MARCO qrels rows must have 3 or 4 columns")
    return query_id, doc_id, int(float(score))


def parse_tsv_id_text_line(line: str, *, row_name: str) -> tuple[str, str]:
    parts = line.rstrip("\n").split("\t", 1)
    if len(parts) != 2:
        raise ValueError(f"{row_name} rows must have id and text columns")
    return parts[0], parts[1]


def load_positive_doc_ids_by_query(qrels_path: Path) -> dict[str, list[str]]:
    positives_by_query: dict[str, list[str]] = {}
    with qrels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query_id, doc_id, score = parse_qrels_line(line)
            if score <= 0:
                continue
            positives = positives_by_query.setdefault(query_id, [])
            if doc_id not in positives:
                positives.append(doc_id)
    return positives_by_query


def load_queries_by_id(queries_path: Path) -> dict[str, str]:
    queries_by_id: dict[str, str] = {}
    with queries_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query_id, text = parse_tsv_id_text_line(line, row_name="query")
            queries_by_id[query_id] = text
    return queries_by_id


def choose_query_ids(
    positives_by_query: dict[str, list[str]],
    queries_by_id: dict[str, str],
    *,
    query_limit: int | None,
) -> list[str]:
    query_ids: list[str] = []
    for query_id in positives_by_query:
        if query_id not in queries_by_id:
            continue
        query_ids.append(query_id)
        if query_limit is not None and len(query_ids) == query_limit:
            break
    return query_ids


def select_msmarco_passage_documents_and_queries(
    paths: MsmarcoPassagePaths,
    *,
    document_limit: int | None = None,
    query_limit: int | None = None,
    include_relevant_documents: bool = True,
) -> MsmarcoPassageSelection:
    if document_limit is not None and document_limit <= 0:
        raise ValueError("document_limit must be positive when provided")
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive when provided")

    positives_by_query = load_positive_doc_ids_by_query(paths.qrels)
    queries_by_id = load_queries_by_id(paths.queries)
    selected_query_ids = choose_query_ids(
        positives_by_query,
        queries_by_id,
        query_limit=query_limit,
    )
    required_doc_ids = _required_doc_ids(
        positives_by_query,
        selected_query_ids,
        include_relevant_documents=include_relevant_documents,
    )
    documents, source_document_count = _select_documents(
        paths.collection,
        document_limit=document_limit,
        required_doc_ids=required_doc_ids,
    )
    selected_doc_ids = {document["doc_id"] for document in documents}
    queries = _selected_queries(
        selected_query_ids,
        positives_by_query,
        queries_by_id,
        selected_doc_ids,
    )
    if not queries:
        raise ValueError(
            "MS MARCO selection produced no judged queries; increase the "
            "document limit or enable include_relevant_documents"
        )

    missing_required = len(required_doc_ids - selected_doc_ids)
    return MsmarcoPassageSelection(
        documents=documents,
        queries=queries,
        source_document_count=source_document_count,
        selected_document_count=len(documents),
        selected_query_count=len(queries),
        required_relevant_doc_count=len(required_doc_ids),
        missing_required_relevant_doc_count=missing_required,
    )


def build_msmarco_passage_task(
    paths: MsmarcoPassagePaths,
    *,
    document_limit: int | None = None,
    query_limit: int | None = None,
    include_relevant_documents: bool = True,
    include_document_token_ids: bool = False,
    document_batch_size: int = 8,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> tuple[dict, MsmarcoPassageSelection]:
    selection = select_msmarco_passage_documents_and_queries(
        paths,
        document_limit=document_limit,
        query_limit=query_limit,
        include_relevant_documents=include_relevant_documents,
    )
    task = build_retrieval_subset_task(
        family="msmarco",
        slice_name=_slice_name(document_limit=document_limit, query_limit=query_limit),
        why=(
            "Local MS MARCO passage ranking subset encoded for Kayak PLAID "
            "candidate-generation and serving-scale profiling."
        ),
        primary_metric="mrr",
        k=10,
        dataset_id=dataset_id,
        model_name=model_name,
        documents=selection.documents,
        queries=selection.queries,
        include_document_token_ids=include_document_token_ids,
        document_batch_size=document_batch_size,
    )
    return task, selection


def _required_doc_ids(
    positives_by_query: dict[str, list[str]],
    selected_query_ids: Iterable[str],
    *,
    include_relevant_documents: bool,
) -> set[str]:
    if not include_relevant_documents:
        return set()
    required: set[str] = set()
    for query_id in selected_query_ids:
        required.update(positives_by_query[query_id])
    return required


def _select_documents(
    collection_path: Path,
    *,
    document_limit: int | None,
    required_doc_ids: set[str],
) -> tuple[list[dict[str, str]], int]:
    documents: list[dict[str, str]] = []
    selected_doc_ids: set[str] = set()
    source_document_count = 0
    with collection_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_document_count += 1
            doc_id, text = parse_tsv_id_text_line(line, row_name="collection")
            under_limit = (
                document_limit is None or len(documents) < document_limit
            )
            required = doc_id in required_doc_ids
            if (not under_limit and not required) or doc_id in selected_doc_ids:
                continue
            documents.append({"doc_id": doc_id, "text": text})
            selected_doc_ids.add(doc_id)
    return documents, source_document_count


def _selected_queries(
    selected_query_ids: Iterable[str],
    positives_by_query: dict[str, list[str]],
    queries_by_id: dict[str, str],
    selected_doc_ids: set[str],
) -> list[dict[str, object]]:
    queries: list[dict[str, object]] = []
    for query_id in selected_query_ids:
        relevant_doc_ids = [
            doc_id
            for doc_id in positives_by_query[query_id]
            if doc_id in selected_doc_ids
        ]
        if not relevant_doc_ids:
            continue
        queries.append(
            {
                "query_id": query_id,
                "text": queries_by_id[query_id],
                "relevant_doc_ids": relevant_doc_ids,
            }
        )
    return queries


def _slice_name(*, document_limit: int | None, query_limit: int | None) -> str:
    document_scope = "docs_full" if document_limit is None else f"docs_{document_limit}"
    query_scope = "queries_full" if query_limit is None else f"queries_{query_limit}"
    return f"msmarco_passage_{document_scope}_{query_scope}"
