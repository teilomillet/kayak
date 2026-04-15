"""Owns the full-corpus R2MED/Biology exact benchmark path.

This module owns:
- loading the public R2MED/Biology corpus and qrels
- building one reusable packed snapshot for exact Kayak search
- benchmarking full-corpus exact search and emitting ranked hits

This module does not own:
- generic public benchmark registration
- LanceDB comparisons
- external leaderboard scraping

Assumptions:
- the benchmark uses the public `R2MED/Biology` Hugging Face dataset
- document and query vector budgets come from the active ColBERT checkpoint
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import kayak

from kayak.stores import DirectoryLateStore

from .cache_paths import REPO_ROOT, configure_local_caches
from .colbert_encoder import (
    DEFAULT_MODEL_NAME,
    encode_document_text,
    encode_document_texts,
    encode_query_text,
    get_checkpoint,
)
from .directory_snapshot_builder import (
    StreamedPackedSnapshotDocument,
    StreamedPackedSnapshotSummary,
    write_streamed_packed_directory_snapshot,
)
from .judged_metrics import summarize_ranked_task
from .prepared_index_cache import clear_prepared_packed_index_cache

configure_local_caches()

from datasets import load_dataset


DEFAULT_DATASET_ID = "R2MED/Biology"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / ".cache" / "kayak" / "r2med_biology_full"
DEFAULT_SNAPSHOT_ROOT = DEFAULT_ARTIFACT_ROOT / "store"
DEFAULT_QUERY_CACHE_PATH = DEFAULT_ARTIFACT_ROOT / "queries.json"
DEFAULT_K = 10
DEFAULT_DOCUMENT_BATCH_SIZE = 8


@dataclass(frozen=True, slots=True)
class EncodedJudgedQuery:
    query_id: str
    text: str
    relevant_doc_ids: tuple[str, ...]
    vector_count: int
    vectors: list[list[float]]

    def to_json_ready(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "relevant_doc_ids": list(self.relevant_doc_ids),
            "vector_count": self.vector_count,
            "vectors": self.vectors,
        }


@dataclass(frozen=True, slots=True)
class EncodedJudgedQueryCache:
    dataset_id: str
    model_name: str
    vector_dim: int
    query_count: int
    nominal_query_vector_count: int
    queries: tuple[EncodedJudgedQuery, ...]

    def to_json_ready(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "vector_dim": self.vector_dim,
            "query_count": self.query_count,
            "nominal_query_vector_count": self.nominal_query_vector_count,
            "queries": [query.to_json_ready() for query in self.queries],
        }


@dataclass(frozen=True, slots=True)
class R2MEDFullExactBenchmarkSummary:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    primary_metric: str
    primary_value: float
    mean_ndcg_at_k: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    success_rate_at_k: float
    cold_first_search_seconds: float
    cold_first_search_seconds_per_query: float
    mean_search_seconds: float
    query_encode_seconds: float
    snapshot_build_seconds: float
    snapshot_reused: bool
    query_cache_reused: bool
    k: int
    query_count: int
    document_count: int
    nominal_query_vector_count: int
    nominal_document_vector_count: int
    document_vector_count_total: int
    reserved_vector_capacity: int
    vector_dim: int
    backend: str
    query_warmup_iterations: int
    query_measurement_iterations: int
    snapshot_storage_byte_size: int

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _split_name(base_split: str, limit: int | None) -> str:
    if limit is None:
        return base_split
    return f"{base_split}[:{limit}]"


def _load_positive_doc_ids_by_query(
    *,
    dataset_id: str,
) -> dict[str, tuple[str, ...]]:
    qrels_dataset = load_dataset(dataset_id, "qrels", split="qrels")
    positives_by_query: dict[str, list[str]] = {}

    for row in qrels_dataset:
        doc_id = str(row["p_id"])
        if int(row["score"]) <= 0:
            continue

        query_id = str(row["q_id"])
        positives = positives_by_query.setdefault(query_id, [])
        if doc_id not in positives:
            positives.append(doc_id)

    return {
        query_id: tuple(doc_ids)
        for query_id, doc_ids in positives_by_query.items()
    }


def _expected_query_count(
    *,
    dataset_id: str,
    query_limit: int | None,
    available_doc_ids: set[str] | None = None,
) -> int:
    positives_by_query = _load_positive_doc_ids_by_query(dataset_id=dataset_id)
    queries_dataset = load_dataset(
        dataset_id,
        "query",
        split=_split_name("query", query_limit),
    )
    count = 0
    for row in queries_dataset:
        query_id = str(row["id"])
        relevant_doc_ids = positives_by_query.get(query_id)
        if not relevant_doc_ids:
            continue
        if not _filter_relevant_doc_ids(relevant_doc_ids, available_doc_ids):
            continue
        count += 1
    return count


def _expected_document_count(
    *,
    dataset_id: str,
    document_limit: int | None,
) -> int:
    documents_dataset = load_dataset(
        dataset_id,
        "corpus",
        split=_split_name("corpus", document_limit),
    )
    return len(documents_dataset)


def _available_document_ids(
    *,
    dataset_id: str,
    document_limit: int | None,
) -> set[str] | None:
    if document_limit is None:
        return None

    documents_dataset = load_dataset(
        dataset_id,
        "corpus",
        split=_split_name("corpus", document_limit),
    )
    return {str(row["id"]) for row in documents_dataset}


def _filter_relevant_doc_ids(
    relevant_doc_ids: tuple[str, ...],
    available_doc_ids: set[str] | None,
) -> tuple[str, ...]:
    if available_doc_ids is None:
        return relevant_doc_ids
    return tuple(doc_id for doc_id in relevant_doc_ids if doc_id in available_doc_ids)


def _mean_query_vector_count(queries: tuple[EncodedJudgedQuery, ...]) -> int:
    if not queries:
        return 0
    total = 0
    for query in queries:
        total += query.vector_count
    return round(total / len(queries))


def _build_encoded_query_cache(
    *,
    dataset_id: str,
    model_name: str,
    query_limit: int | None,
    available_doc_ids: set[str] | None = None,
) -> EncodedJudgedQueryCache:
    positives_by_query = _load_positive_doc_ids_by_query(dataset_id=dataset_id)
    queries_dataset = load_dataset(
        dataset_id,
        "query",
        split=_split_name("query", query_limit),
    )

    encoded_queries: list[EncodedJudgedQuery] = []
    vector_dim = 0
    for row in queries_dataset:
        query_id = str(row["id"])
        relevant_doc_ids = positives_by_query.get(query_id)
        if not relevant_doc_ids:
            continue
        relevant_doc_ids = _filter_relevant_doc_ids(
            relevant_doc_ids,
            available_doc_ids,
        )
        if not relevant_doc_ids:
            continue

        vectors = encode_query_text(str(row["text"]), model_name)
        if vectors:
            vector_dim = len(vectors[0])
        encoded_queries.append(
            EncodedJudgedQuery(
                query_id=query_id,
                text=str(row["text"]),
                relevant_doc_ids=relevant_doc_ids,
                vector_count=len(vectors),
                vectors=vectors,
            )
        )

    return EncodedJudgedQueryCache(
        dataset_id=dataset_id,
        model_name=model_name,
        vector_dim=vector_dim,
        query_count=len(encoded_queries),
        nominal_query_vector_count=_mean_query_vector_count(tuple(encoded_queries)),
        queries=tuple(encoded_queries),
    )


def ensure_encoded_r2med_queries(
    *,
    path: Path = DEFAULT_QUERY_CACHE_PATH,
    dataset_id: str = DEFAULT_DATASET_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    query_limit: int | None = None,
    document_limit: int | None = None,
    force_rebuild: bool = False,
) -> tuple[EncodedJudgedQueryCache, float, bool]:
    available_doc_ids = _available_document_ids(
        dataset_id=dataset_id,
        document_limit=document_limit,
    )
    if path.exists() and not force_rebuild:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected_query_count = _expected_query_count(
            dataset_id=dataset_id,
            query_limit=query_limit,
            available_doc_ids=available_doc_ids,
        )
        if (
            str(payload["dataset_id"]) == dataset_id
            and str(payload["model_name"]) == model_name
            and int(payload["query_count"]) == expected_query_count
        ):
            queries = tuple(
                EncodedJudgedQuery(
                    query_id=str(row["query_id"]),
                    text=str(row["text"]),
                    relevant_doc_ids=tuple(
                        str(doc_id) for doc_id in row["relevant_doc_ids"]
                    ),
                    vector_count=int(row["vector_count"]),
                    vectors=row["vectors"],
                )
                for row in payload["queries"]
            )
            return (
                EncodedJudgedQueryCache(
                    dataset_id=str(payload["dataset_id"]),
                    model_name=str(payload["model_name"]),
                    vector_dim=int(payload["vector_dim"]),
                    query_count=int(payload["query_count"]),
                    nominal_query_vector_count=int(
                        payload["nominal_query_vector_count"]
                    ),
                    queries=queries,
                ),
                0.0,
                True,
            )

    start = time.perf_counter()
    cache = _build_encoded_query_cache(
        dataset_id=dataset_id,
        model_name=model_name,
        query_limit=query_limit,
        available_doc_ids=available_doc_ids,
    )
    elapsed_seconds = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return cache, elapsed_seconds, False


def _stream_encoded_r2med_documents(
    *,
    dataset_id: str,
    model_name: str,
    document_limit: int | None,
    document_batch_size: int = DEFAULT_DOCUMENT_BATCH_SIZE,
):
    documents_dataset = load_dataset(
        dataset_id,
        "corpus",
        split=_split_name("corpus", document_limit),
    )
    batch_doc_ids: list[str] = []
    batch_texts: list[str] = []

    for row in documents_dataset:
        batch_doc_ids.append(str(row["id"]))
        batch_texts.append(str(row["text"]))
        if len(batch_doc_ids) < document_batch_size:
            continue

        encoded_batch = encode_document_texts(
            batch_texts,
            model_name,
            batch_size=document_batch_size,
        )
        for doc_id, token_matrix in zip(batch_doc_ids, encoded_batch, strict=True):
            yield StreamedPackedSnapshotDocument(
                doc_id=doc_id,
                token_matrix=token_matrix,
            )
        batch_doc_ids.clear()
        batch_texts.clear()

    if batch_doc_ids:
        encoded_batch = encode_document_texts(
            batch_texts,
            model_name,
            batch_size=document_batch_size,
        )
        for doc_id, token_matrix in zip(batch_doc_ids, encoded_batch, strict=True):
            yield StreamedPackedSnapshotDocument(
                doc_id=doc_id,
                token_matrix=token_matrix,
            )


def ensure_r2med_biology_snapshot(
    *,
    root: Path = DEFAULT_SNAPSHOT_ROOT,
    dataset_id: str = DEFAULT_DATASET_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    document_limit: int | None = None,
    document_batch_size: int = DEFAULT_DOCUMENT_BATCH_SIZE,
    force_rebuild: bool = False,
) -> tuple[StreamedPackedSnapshotSummary, float, bool]:
    if root.exists() and not force_rebuild:
        store = DirectoryLateStore(root)
        stats = store.stats()
        expected_document_count = _expected_document_count(
            dataset_id=dataset_id,
            document_limit=document_limit,
        )
        expected_vector_dim = int(get_checkpoint(model_name).colbert_config.dim)
        if (
            stats.document_count <= 0
            or stats.vector_dim is None
            or stats.document_count != expected_document_count
            or int(stats.vector_dim) != expected_vector_dim
        ):
            pass
        else:
            return (
                StreamedPackedSnapshotSummary(
                    document_count=stats.document_count,
                    total_vector_count=stats.total_vector_count,
                    reserved_vector_capacity=(
                        stats.document_count
                        * int(get_checkpoint(model_name).colbert_config.doc_maxlen)
                    ),
                    vector_dim=int(stats.vector_dim),
                    storage_byte_size=stats.storage_byte_size or 0,
                ),
                0.0,
                True,
            )

    documents_dataset = load_dataset(
        dataset_id,
        "corpus",
        split=_split_name("corpus", document_limit),
    )
    checkpoint = get_checkpoint(model_name)
    start = time.perf_counter()
    summary = write_streamed_packed_directory_snapshot(
        root,
        _stream_encoded_r2med_documents(
            dataset_id=dataset_id,
            model_name=model_name,
            document_limit=document_limit,
            document_batch_size=document_batch_size,
        ),
        document_count=len(documents_dataset),
        vector_dim=int(checkpoint.colbert_config.dim),
        max_document_vector_count=int(checkpoint.colbert_config.doc_maxlen),
    )
    return summary, time.perf_counter() - start, False


def _build_query_batch(
    query_cache: EncodedJudgedQueryCache,
) -> tuple[kayak.LateQueryBatch, tuple[str, ...]]:
    queries = tuple(
        kayak.query(query.vectors, text=query.text)
        for query in query_cache.queries
    )
    return (
        kayak.query_batch(queries),
        tuple(query.query_id for query in query_cache.queries),
    )


def benchmark_r2med_biology_full_exact(
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    query_cache_path: Path = DEFAULT_QUERY_CACHE_PATH,
    dataset_id: str = DEFAULT_DATASET_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    document_limit: int | None = None,
    query_limit: int | None = None,
    document_batch_size: int = DEFAULT_DOCUMENT_BATCH_SIZE,
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
    k: int = DEFAULT_K,
    backend: str = kayak.MOJO_EXACT_CPU_BACKEND,
    force_rebuild_snapshot: bool = False,
    force_rebuild_queries: bool = False,
) -> tuple[R2MEDFullExactBenchmarkSummary, tuple[tuple[kayak.SearchHit, ...], ...]]:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    snapshot_summary, snapshot_build_seconds, snapshot_reused = (
        ensure_r2med_biology_snapshot(
            root=snapshot_root,
            dataset_id=dataset_id,
            model_name=model_name,
            document_limit=document_limit,
            document_batch_size=document_batch_size,
            force_rebuild=force_rebuild_snapshot,
        )
    )
    query_cache, query_encode_seconds, query_cache_reused = (
        ensure_encoded_r2med_queries(
            path=query_cache_path,
            dataset_id=dataset_id,
            model_name=model_name,
            query_limit=query_limit,
            document_limit=document_limit,
            force_rebuild=force_rebuild_queries,
        )
    )

    store = DirectoryLateStore(snapshot_root)
    index = store.load_index(include_text=False)
    if query_cache.query_count == 0:
        raise ValueError(
            "R2MED query set is empty after filtering to the available corpus; "
            "increase document_limit or remove it"
        )
    query_batch, _ = _build_query_batch(query_cache)

    clear_prepared_packed_index_cache()

    cold_start = time.perf_counter()
    kayak.search_batch(query_batch, index, k=k, backend=backend)
    cold_first_search_seconds = time.perf_counter() - cold_start

    for _ in range(warmup_iterations):
        kayak.search_batch(query_batch, index, k=k, backend=backend)

    elapsed_seconds: list[float] = []
    ranked_hits: tuple[tuple[kayak.SearchHit, ...], ...] = ()
    ranked_doc_ids_by_query: list[tuple[str, ...]] = []
    for measurement_iteration in range(measurement_iterations):
        start = time.perf_counter()
        hits_by_query = kayak.search_batch(query_batch, index, k=k, backend=backend)
        elapsed_seconds.append(
            (time.perf_counter() - start) / float(query_batch.batch_size)
        )
        if measurement_iteration == 0:
            ranked_hits = tuple(tuple(hits) for hits in hits_by_query)
            ranked_doc_ids_by_query = [
                tuple(hit.doc_id for hit in hits)
                for hits in hits_by_query
            ]

    task_metrics = summarize_ranked_task(
        task={
            "primary_metric": "ndcg",
            "k": k,
            "documents": [{}] * index.document_count,
            "queries": [
                {
                    "relevant_doc_ids": list(query.relevant_doc_ids),
                }
                for query in query_cache.queries
            ],
        },
        ranked_doc_ids_by_query=ranked_doc_ids_by_query,
    )

    clear_prepared_packed_index_cache()

    summary = R2MEDFullExactBenchmarkSummary(
        dataset_id=dataset_id,
        model_name=model_name,
        family="r2med",
        slice_name=(
            "r2med_biology_full"
            if document_limit is None and query_limit is None
            else "r2med_biology_partial"
        ),
        primary_metric=task_metrics.primary_metric,
        primary_value=task_metrics.primary_value,
        mean_ndcg_at_k=task_metrics.mean_ndcg_at_k,
        mean_reciprocal_rank=task_metrics.mean_reciprocal_rank,
        mean_recall_at_k=task_metrics.mean_recall_at_k,
        success_rate_at_k=task_metrics.success_rate_at_k,
        cold_first_search_seconds=cold_first_search_seconds,
        cold_first_search_seconds_per_query=(
            cold_first_search_seconds / float(query_cache.query_count)
        ),
        mean_search_seconds=float(sum(elapsed_seconds) / len(elapsed_seconds)),
        query_encode_seconds=query_encode_seconds,
        snapshot_build_seconds=snapshot_build_seconds,
        snapshot_reused=snapshot_reused,
        query_cache_reused=query_cache_reused,
        k=k,
        query_count=query_cache.query_count,
        document_count=index.document_count,
        nominal_query_vector_count=query_cache.nominal_query_vector_count,
        nominal_document_vector_count=round(
            snapshot_summary.total_vector_count / float(index.document_count)
        ),
        document_vector_count_total=snapshot_summary.total_vector_count,
        reserved_vector_capacity=snapshot_summary.reserved_vector_capacity,
        vector_dim=index.vector_dim,
        backend=backend,
        query_warmup_iterations=warmup_iterations,
        query_measurement_iterations=measurement_iterations,
        snapshot_storage_byte_size=snapshot_summary.storage_byte_size,
    )
    return summary, ranked_hits
