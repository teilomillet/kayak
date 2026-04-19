"""Owns raw-text token chunking sweeps for chunked one-vector baselines.

This module starts from judged task JSONs that already carry raw text plus
encoded late-interaction vectors. It does not try to infer token chunking from
stored vector rows. Instead, it re-chunks raw text at explicit source-token
sizes, re-encodes each chunk, and benchmarks the resulting one-vector-per-chunk
baseline against the task's judged metrics.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Protocol
from typing import Sequence

import numpy as np

import kayak

from .colbert_encoder import encode_document_texts
from .colbert_encoder import get_checkpoint
from .judged_metrics import summarize_ranked_task
from .kayak_task_benchmark import rank_task_with_kayak_exact
from .source_text_chunking import chunk_text_by_source_tokens
from .source_text_chunking import effective_model_max_length
from .source_text_chunking import load_hf_source_tokenizer
from .source_text_chunking import RawTextChunkSpec
from .source_text_chunking import source_token_ids
from .source_text_chunking import SourceTokenizer


EncodeDocumentTextsFn = Callable[
    [list[str], str, int],
    Sequence[Sequence[Sequence[float]]],
]

_AUTO_DOCUMENT_VECTOR_CAP = object()


def _default_encode_document_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> Sequence[Sequence[Sequence[float]]]:
    return encode_document_texts(
        texts,
        model_name,
        batch_size=batch_size,
    )


@dataclass(slots=True)
class RawTextChunkedOneVecArtifact:
    index: kayak.LateIndex
    parent_by_chunk: dict[str, str]
    spec: RawTextChunkSpec
    build_seconds: float
    encoder_doc_maxlen: int | None
    generated_chunk_count: int
    mean_chunks_per_document: float
    max_chunks_per_document: int
    mean_source_chunk_token_count: float
    max_source_chunk_token_count: int
    mean_chunk_vector_count: float
    max_chunk_vector_count: int
    at_doc_maxlen_chunk_count: int
    at_doc_maxlen_chunk_fraction: float


@dataclass(frozen=True, slots=True)
class RawTextChunkedOneVecTaskBenchmarkSummary:
    dataset_id: str
    family: str
    slice_name: str
    model_name: str
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
    encoder_doc_maxlen: int | None
    source_chunk_tokens: int
    source_overlap_tokens: int
    generated_chunk_count: int
    mean_chunks_per_document: float
    max_chunks_per_document: int
    mean_source_chunk_token_count: float
    max_source_chunk_token_count: int
    mean_chunk_vector_count: float
    max_chunk_vector_count: int
    at_doc_maxlen_chunk_count: int
    at_doc_maxlen_chunk_fraction: float
    backend: str

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RawTextChunkSweepBundle:
    dataset_id: str
    family: str
    slice_name: str
    model_name: str
    primary_metric: str
    k: int
    exact_primary_value: float
    exact_mean_recall_at_k: float
    onevec_primary_value: float
    onevec_mean_recall_at_k: float
    rows: tuple[dict[str, object], ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=4)
def _load_source_tokenizer(model_name: str) -> SourceTokenizer:
    return load_hf_source_tokenizer(model_name)


def _mean_vec(matrix: np.ndarray) -> np.ndarray:
    return np.mean(
        matrix,
        axis=0,
        dtype=np.float32,
        keepdims=True,
    ).astype(np.float32)


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


def _load_document_vector_cap(model_name: str) -> int | None:
    checkpoint = get_checkpoint(model_name)
    config = getattr(checkpoint, "colbert_config", None)
    raw_value = getattr(config, "doc_maxlen", None)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _build_onevec_queries(
    task: Mapping[str, Any],
) -> tuple[kayak.LateQuery, ...]:
    queries = []
    for query in task["queries"]:
        matrix = np.asarray(query["vectors"], dtype=np.float32)
        queries.append(
            kayak.query(
                _mean_vec(matrix),
                text=str(query["text"]),
            )
        )
    return tuple(queries)


def _rank_task_with_onevec_full_document(
    task: Mapping[str, Any],
    *,
    backend: str,
) -> tuple[tuple[str, ...], ...]:
    index = kayak.documents(
        [document["doc_id"] for document in task["documents"]],
        [
            _mean_vec(np.asarray(document["vectors"], dtype=np.float32))
            for document in task["documents"]
        ],
        texts=[document["text"] for document in task["documents"]],
    ).pack()
    queries = _build_onevec_queries(task)
    hits_by_query = kayak.search_batch(
        kayak.query_batch(queries),
        index,
        k=int(task["k"]),
        backend=backend,
    )
    return tuple(tuple(hit.doc_id for hit in hits) for hits in hits_by_query)


def build_raw_text_chunked_onevec_artifact(
    task: Mapping[str, Any],
    *,
    spec: RawTextChunkSpec,
    tokenizer: SourceTokenizer | None = None,
    encode_document_texts_fn: EncodeDocumentTextsFn = _default_encode_document_texts,
    encode_batch_size: int = 8,
    model_name: str | None = None,
    document_vector_cap: int | None | object = _AUTO_DOCUMENT_VECTOR_CAP,
) -> RawTextChunkedOneVecArtifact:
    if encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")

    resolved_model_name = model_name or str(task["model_name"])
    source_tokenizer = tokenizer or _load_source_tokenizer(resolved_model_name)
    resolved_document_vector_cap = (
        _load_document_vector_cap(resolved_model_name)
        if document_vector_cap is _AUTO_DOCUMENT_VECTOR_CAP
        else document_vector_cap
    )
    tokenizer_max_length = effective_model_max_length(source_tokenizer)
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
            tokenizer=source_tokenizer,
            spec=spec,
        )
        chunk_counts.append(len(raw_chunks))
        for chunk_index, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{row['doc_id']}::rawchunk{chunk_index}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk_text)
            parent_by_chunk[chunk_id] = str(row["doc_id"])
            source_chunk_token_counts.append(
                len(source_token_ids(chunk_text, tokenizer=source_tokenizer))
            )

    encoded_chunks = encode_document_texts_fn(
        chunk_texts,
        resolved_model_name,
        encode_batch_size,
    )
    chunk_matrices = [
        np.asarray(encoded, dtype=np.float32)
        for encoded in encoded_chunks
    ]
    index = kayak.documents(
        chunk_ids,
        [_mean_vec(matrix) for matrix in chunk_matrices],
        texts=chunk_texts,
    ).pack()
    build_seconds = time.perf_counter() - start

    chunk_vector_counts = [int(matrix.shape[0]) for matrix in chunk_matrices]
    at_doc_maxlen_chunk_count = (
        sum(
            1
            for count in chunk_vector_counts
            if resolved_document_vector_cap is not None
            and count >= resolved_document_vector_cap
        )
        if chunk_vector_counts
        else 0
    )
    return RawTextChunkedOneVecArtifact(
        index=index,
        parent_by_chunk=parent_by_chunk,
        spec=spec,
        build_seconds=build_seconds,
        encoder_doc_maxlen=(
            int(resolved_document_vector_cap)
            if resolved_document_vector_cap is not None
            else None
        ),
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
        mean_chunk_vector_count=(
            float(sum(chunk_vector_counts)) / float(len(chunk_vector_counts))
            if chunk_vector_counts
            else 0.0
        ),
        max_chunk_vector_count=max(chunk_vector_counts) if chunk_vector_counts else 0,
        at_doc_maxlen_chunk_count=at_doc_maxlen_chunk_count,
        at_doc_maxlen_chunk_fraction=(
            float(at_doc_maxlen_chunk_count) / float(len(chunk_vector_counts))
            if chunk_vector_counts
            else 0.0
        ),
    )


def benchmark_task_with_raw_text_chunked_onevec(
    task: Mapping[str, Any],
    *,
    spec: RawTextChunkSpec,
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
    backend: str = kayak.NUMPY_REFERENCE_BACKEND,
    tokenizer: SourceTokenizer | None = None,
    encode_document_texts_fn: EncodeDocumentTextsFn = _default_encode_document_texts,
    encode_batch_size: int = 8,
    model_name: str | None = None,
    document_vector_cap: int | None | object = _AUTO_DOCUMENT_VECTOR_CAP,
) -> RawTextChunkedOneVecTaskBenchmarkSummary:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    artifact = build_raw_text_chunked_onevec_artifact(
        task,
        spec=spec,
        tokenizer=tokenizer,
        encode_document_texts_fn=encode_document_texts_fn,
        encode_batch_size=encode_batch_size,
        model_name=model_name,
        document_vector_cap=document_vector_cap,
    )
    queries = _build_onevec_queries(task)
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
    return RawTextChunkedOneVecTaskBenchmarkSummary(
        dataset_id=str(task["dataset_id"]),
        family=str(task["family"]),
        slice_name=str(task["slice_name"]),
        model_name=str(model_name or task["model_name"]),
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
        encoder_doc_maxlen=artifact.encoder_doc_maxlen,
        source_chunk_tokens=spec.max_chunk_tokens,
        source_overlap_tokens=spec.overlap_tokens,
        generated_chunk_count=artifact.generated_chunk_count,
        mean_chunks_per_document=artifact.mean_chunks_per_document,
        max_chunks_per_document=artifact.max_chunks_per_document,
        mean_source_chunk_token_count=artifact.mean_source_chunk_token_count,
        max_source_chunk_token_count=artifact.max_source_chunk_token_count,
        mean_chunk_vector_count=artifact.mean_chunk_vector_count,
        max_chunk_vector_count=artifact.max_chunk_vector_count,
        at_doc_maxlen_chunk_count=artifact.at_doc_maxlen_chunk_count,
        at_doc_maxlen_chunk_fraction=artifact.at_doc_maxlen_chunk_fraction,
        backend=backend,
    )


def build_raw_text_chunk_sweep_bundle(
    task: Mapping[str, Any],
    *,
    specs: Sequence[RawTextChunkSpec],
    backend: str = kayak.NUMPY_REFERENCE_BACKEND,
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
    tokenizer: SourceTokenizer | None = None,
    encode_document_texts_fn: EncodeDocumentTextsFn = _default_encode_document_texts,
    encode_batch_size: int = 8,
    model_name: str | None = None,
    document_vector_cap: int | None | object = _AUTO_DOCUMENT_VECTOR_CAP,
) -> RawTextChunkSweepBundle:
    exact_summary = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=rank_task_with_kayak_exact(
            task,
            backend=backend,
        ),
    )
    onevec_summary = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=_rank_task_with_onevec_full_document(
            task,
            backend=backend,
        ),
    )
    rows = tuple(
        benchmark_task_with_raw_text_chunked_onevec(
            task,
            spec=spec,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            backend=backend,
            tokenizer=tokenizer,
            encode_document_texts_fn=encode_document_texts_fn,
            encode_batch_size=encode_batch_size,
            model_name=model_name,
            document_vector_cap=document_vector_cap,
        ).to_json_ready()
        for spec in specs
    )
    return RawTextChunkSweepBundle(
        dataset_id=str(task["dataset_id"]),
        family=str(task["family"]),
        slice_name=str(task["slice_name"]),
        model_name=str(model_name or task["model_name"]),
        primary_metric=str(task["primary_metric"]),
        k=int(task["k"]),
        exact_primary_value=exact_summary.primary_value,
        exact_mean_recall_at_k=exact_summary.mean_recall_at_k,
        onevec_primary_value=onevec_summary.primary_value,
        onevec_mean_recall_at_k=onevec_summary.mean_recall_at_k,
        rows=rows,
    )
