"""Owns one-vector raw-text chunk baselines with pluggable text embedders.

This module owns the benchmark path for dense chunk retrieval where each chunk
and query is represented by one vector. It does not own any specific embedding
provider; callers inject the text embedder explicitly.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import time
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence

import numpy as np

import kayak

from .judged_metrics import summarize_ranked_task
from .kayak_task_benchmark import rank_task_with_kayak_exact
from .source_text_chunking import chunk_text_by_source_tokens
from .source_text_chunking import effective_model_max_length
from .source_text_chunking import RawTextChunkSpec
from .source_text_chunking import source_token_ids
from .source_text_chunking import SourceTokenizer


EmbedTextsFn = Callable[[list[str], str, int], Sequence[Sequence[float]]]


@dataclass(slots=True)
class SingleVectorChunkArtifact:
    index: kayak.LateIndex
    parent_by_chunk: dict[str, str]
    spec: RawTextChunkSpec
    build_seconds: float
    embedding_model_name: str
    vector_dim: int
    normalize_embeddings: bool
    generated_chunk_count: int
    mean_chunks_per_document: float
    max_chunks_per_document: int
    mean_source_chunk_token_count: float
    max_source_chunk_token_count: int


@dataclass(frozen=True, slots=True)
class SingleVectorChunkTaskBenchmarkSummary:
    dataset_id: str
    family: str
    slice_name: str
    embedding_model_name: str
    source_tokenizer_name: str
    primary_metric: str
    primary_value: float
    mean_ndcg_at_k: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    success_rate_at_k: float
    mean_search_seconds: float
    build_seconds: float
    query_count: int
    document_count: int
    k: int
    source_chunk_tokens: int
    source_overlap_tokens: int
    generated_chunk_count: int
    mean_chunks_per_document: float
    max_chunks_per_document: int
    mean_source_chunk_token_count: float
    max_source_chunk_token_count: int
    vector_dim: int
    embedding_batch_size: int
    normalize_embeddings: bool
    parent_aggregation: str
    backend: str

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SingleVectorChunkSweepBundle:
    dataset_id: str
    family: str
    slice_name: str
    exact_model_name: str
    primary_metric: str
    k: int
    exact_primary_value: float
    exact_mean_recall_at_k: float
    rows: tuple[dict[str, object], ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _as_row_matrix(
    vector: Sequence[float],
    *,
    normalize: bool,
) -> np.ndarray:
    matrix = np.asarray(vector, dtype=np.float32)
    if matrix.ndim != 1:
        raise ValueError("single-vector embeddings must be 1D")
    if matrix.shape[0] <= 0:
        raise ValueError("single-vector embeddings must be non-empty")
    row = matrix.reshape(1, matrix.shape[0])
    if not normalize:
        return row
    norm = float(np.linalg.norm(row))
    if norm <= 0.0:
        return row
    return row / np.float32(norm)


def _dedup_parent_docs(
    chunk_hits,
    parent_by_chunk: Mapping[str, str],
    *,
    k: int,
) -> tuple[str, ...]:
    ranked_doc_ids: list[str] = []
    seen: set[str] = set()
    for hit in chunk_hits:
        doc_id = parent_by_chunk[hit.doc_id]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranked_doc_ids.append(doc_id)
        if len(ranked_doc_ids) >= k:
            break
    return tuple(ranked_doc_ids)


def _tokenizer_name(tokenizer: SourceTokenizer) -> str:
    encoding_name = getattr(tokenizer, "encoding_name", None)
    if isinstance(encoding_name, str) and encoding_name:
        return encoding_name
    model_name = getattr(tokenizer, "model_name", None)
    if isinstance(model_name, str) and model_name:
        return model_name
    return type(tokenizer).__name__


def build_single_vector_chunk_artifact(
    task: Mapping[str, Any],
    *,
    spec: RawTextChunkSpec,
    tokenizer: SourceTokenizer,
    embed_texts_fn: EmbedTextsFn,
    embedding_model_name: str,
    embedding_batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> SingleVectorChunkArtifact:
    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive")

    tokenizer_max_length = effective_model_max_length(tokenizer)
    if (
        tokenizer_max_length is not None
        and spec.max_chunk_tokens > tokenizer_max_length
    ):
        raise ValueError(
            "source chunk size exceeds tokenizer model_max_length; "
            f"chunk={spec.max_chunk_tokens}, tokenizer_max_length={tokenizer_max_length}"
        )

    start = time.perf_counter()
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    parent_by_chunk: dict[str, str] = {}
    chunk_counts: list[int] = []
    source_chunk_token_counts: list[int] = []

    for row in task["documents"]:
        raw_chunks = chunk_text_by_source_tokens(
            str(row["text"]),
            tokenizer=tokenizer,
            spec=spec,
        )
        chunk_counts.append(len(raw_chunks))
        for chunk_index, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{row['doc_id']}::densechunk{chunk_index}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk_text)
            parent_by_chunk[chunk_id] = str(row["doc_id"])
            source_chunk_token_counts.append(
                len(source_token_ids(chunk_text, tokenizer=tokenizer))
            )

    raw_vectors = embed_texts_fn(
        chunk_texts,
        embedding_model_name,
        embedding_batch_size,
    )
    chunk_matrices = [
        _as_row_matrix(vector, normalize=normalize_embeddings)
        for vector in raw_vectors
    ]
    vector_dims = {int(matrix.shape[1]) for matrix in chunk_matrices}
    if len(vector_dims) != 1:
        raise ValueError("all chunk embeddings must share one vector dimension")

    index = kayak.documents(
        chunk_ids,
        chunk_matrices,
        texts=chunk_texts,
    ).pack()
    build_seconds = time.perf_counter() - start
    vector_dim = next(iter(vector_dims), 0)
    return SingleVectorChunkArtifact(
        index=index,
        parent_by_chunk=parent_by_chunk,
        spec=spec,
        build_seconds=build_seconds,
        embedding_model_name=embedding_model_name,
        vector_dim=vector_dim,
        normalize_embeddings=normalize_embeddings,
        generated_chunk_count=len(chunk_ids),
        mean_chunks_per_document=(
            float(sum(chunk_counts)) / float(len(chunk_counts)) if chunk_counts else 0.0
        ),
        max_chunks_per_document=max(chunk_counts) if chunk_counts else 0,
        mean_source_chunk_token_count=(
            float(sum(source_chunk_token_counts)) / float(len(source_chunk_token_counts))
            if source_chunk_token_counts
            else 0.0
        ),
        max_source_chunk_token_count=(
            max(source_chunk_token_counts) if source_chunk_token_counts else 0
        ),
    )


def benchmark_task_with_single_vector_chunks(
    task: Mapping[str, Any],
    *,
    spec: RawTextChunkSpec,
    tokenizer: SourceTokenizer,
    embed_texts_fn: EmbedTextsFn,
    embedding_model_name: str,
    embedding_batch_size: int = 64,
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
    normalize_embeddings: bool = True,
    backend: str = kayak.NUMPY_REFERENCE_BACKEND,
) -> SingleVectorChunkTaskBenchmarkSummary:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    artifact = build_single_vector_chunk_artifact(
        task,
        spec=spec,
        tokenizer=tokenizer,
        embed_texts_fn=embed_texts_fn,
        embedding_model_name=embedding_model_name,
        embedding_batch_size=embedding_batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    query_texts = [str(query["text"]) for query in task["queries"]]
    raw_query_vectors = embed_texts_fn(
        query_texts,
        embedding_model_name,
        embedding_batch_size,
    )
    queries = tuple(
        kayak.query(
            _as_row_matrix(vector, normalize=normalize_embeddings),
            text=query_text,
        )
        for query_text, vector in zip(query_texts, raw_query_vectors, strict=True)
    )
    query_batch = kayak.query_batch(queries)
    hit_k = artifact.index.document_count
    final_k = int(task["k"])

    for _ in range(warmup_iterations):
        kayak.search_batch(query_batch, artifact.index, k=hit_k, backend=backend)

    elapsed_seconds: list[float] = []
    ranked_doc_ids_by_query: list[tuple[str, ...]] = []
    for measurement_iteration in range(measurement_iterations):
        start = time.perf_counter()
        chunk_hits_by_query = kayak.search_batch(
            query_batch,
            artifact.index,
            k=hit_k,
            backend=backend,
        )
        elapsed_seconds.append(
            (time.perf_counter() - start) / float(query_batch.batch_size)
        )
        if measurement_iteration == 0:
            ranked_doc_ids_by_query = [
                _dedup_parent_docs(
                    chunk_hits,
                    artifact.parent_by_chunk,
                    k=final_k,
                )
                for chunk_hits in chunk_hits_by_query
            ]

    metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=ranked_doc_ids_by_query,
    )
    return SingleVectorChunkTaskBenchmarkSummary(
        dataset_id=str(task["dataset_id"]),
        family=str(task["family"]),
        slice_name=str(task["slice_name"]),
        embedding_model_name=artifact.embedding_model_name,
        source_tokenizer_name=_tokenizer_name(tokenizer),
        primary_metric=metrics.primary_metric,
        primary_value=metrics.primary_value,
        mean_ndcg_at_k=metrics.mean_ndcg_at_k,
        mean_reciprocal_rank=metrics.mean_reciprocal_rank,
        mean_recall_at_k=metrics.mean_recall_at_k,
        success_rate_at_k=metrics.success_rate_at_k,
        mean_search_seconds=float(sum(elapsed_seconds) / len(elapsed_seconds)),
        build_seconds=artifact.build_seconds,
        query_count=metrics.query_count,
        document_count=metrics.document_count,
        k=metrics.k,
        source_chunk_tokens=spec.max_chunk_tokens,
        source_overlap_tokens=spec.overlap_tokens,
        generated_chunk_count=artifact.generated_chunk_count,
        mean_chunks_per_document=artifact.mean_chunks_per_document,
        max_chunks_per_document=artifact.max_chunks_per_document,
        mean_source_chunk_token_count=artifact.mean_source_chunk_token_count,
        max_source_chunk_token_count=artifact.max_source_chunk_token_count,
        vector_dim=artifact.vector_dim,
        embedding_batch_size=embedding_batch_size,
        normalize_embeddings=normalize_embeddings,
        parent_aggregation="dedup_max_chunk_score",
        backend=backend,
    )


def build_single_vector_chunk_sweep_bundle(
    task: Mapping[str, Any],
    *,
    specs: Sequence[RawTextChunkSpec],
    tokenizer: SourceTokenizer,
    embed_texts_fn: EmbedTextsFn,
    embedding_model_name: str,
    embedding_batch_size: int = 64,
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
    normalize_embeddings: bool = True,
    backend: str = kayak.NUMPY_REFERENCE_BACKEND,
) -> SingleVectorChunkSweepBundle:
    exact_summary = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=rank_task_with_kayak_exact(
            task,
            backend=backend,
        ),
    )
    rows = tuple(
        benchmark_task_with_single_vector_chunks(
            task,
            spec=spec,
            tokenizer=tokenizer,
            embed_texts_fn=embed_texts_fn,
            embedding_model_name=embedding_model_name,
            embedding_batch_size=embedding_batch_size,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            normalize_embeddings=normalize_embeddings,
            backend=backend,
        ).to_json_ready()
        for spec in specs
    )
    return SingleVectorChunkSweepBundle(
        dataset_id=str(task["dataset_id"]),
        family=str(task["family"]),
        slice_name=str(task["slice_name"]),
        exact_model_name=str(task["model_name"]),
        primary_metric=exact_summary.primary_metric,
        k=exact_summary.k,
        exact_primary_value=exact_summary.primary_value,
        exact_mean_recall_at_k=exact_summary.mean_recall_at_k,
        rows=rows,
    )
