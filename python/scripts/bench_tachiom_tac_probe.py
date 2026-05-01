from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex
from kayak_bridge.tachiom_probe import (
    TachiomHnswConfig,
    TachiomResidualPqConfig,
    TachiomResidualPqIndex,
    TachiomResidualPqMojoIndex,
    TachiomTacConfig,
    TachiomTacHnswIndex,
    TachiomTacHnswMojoIndex,
    TachiomTacI8MojoIndex,
    TachiomTacIndex,
    TachiomTacMojoIndex,
    exact_search_positions,
    mean_candidate_set_recall_at_k,
    mean_recall_at_k,
)


TACHIOM_PAPER_URL = "https://arxiv.org/pdf/2604.28142"


@dataclass(frozen=True, slots=True)
class TokenStructuredShape:
    document_count: int
    document_vector_count: int
    query_count: int
    query_vector_count: int
    vector_dim: int
    top_k: int
    vocabulary_size: int

    @property
    def total_document_vector_count(self) -> int:
        return self.document_count * self.document_vector_count

    @property
    def total_query_vector_count(self) -> int:
        return self.query_count * self.query_vector_count

    def validate(self) -> None:
        for name, value in self.to_json_ready().items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.top_k > self.document_count:
            raise ValueError("top_k must not exceed document_count")

    def to_json_ready(self) -> dict[str, int]:
        return {
            "document_count": self.document_count,
            "document_vector_count": self.document_vector_count,
            "document_vector_count_total": self.total_document_vector_count,
            "query_count": self.query_count,
            "query_vector_count": self.query_vector_count,
            "query_vector_count_total": self.total_query_vector_count,
            "vector_dim": self.vector_dim,
            "top_k": self.top_k,
            "vocabulary_size": self.vocabulary_size,
        }


@dataclass(frozen=True, slots=True)
class TokenStructuredInputs:
    documents: np.ndarray
    token_ids: np.ndarray
    queries: np.ndarray
    query_token_ids: np.ndarray
    doc_ids: tuple[str, ...]
    token_probabilities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a Tachiom-style token-aware centroid probe on a "
            "synthetic token-structured late-interaction workload."
        )
    )
    parser.add_argument(
        "--engines",
        type=_parse_engines,
        default=_parse_engines("exact,tachiom_tac,kayak_plaid"),
        help=(
            "Comma-separated engines: exact,tachiom_tac,tachiom_tac_mojo,"
            "tachiom_tac_hnsw,tachiom_tac_hnsw_mojo,tachiom_tac_i8_mojo,tachiom_tac_pq,"
            "tachiom_tac_pq_mojo,kayak_plaid."
        ),
    )
    parser.add_argument("--document-count", type=int, default=256)
    parser.add_argument("--document-vector-count", type=int, default=96)
    parser.add_argument("--query-count", type=int, default=8)
    parser.add_argument("--query-vector-count", type=int, default=24)
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--vocabulary-size", type=int, default=512)
    parser.add_argument("--zipf-skew", type=float, default=1.1)
    parser.add_argument("--document-noise", type=float, default=0.08)
    parser.add_argument("--query-noise", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--tac-centroid-count", type=int, default=512)
    parser.add_argument("--tac-centroids-per-query-vector", type=int, default=16)
    parser.add_argument("--tac-candidate-k", type=int, default=80)
    parser.add_argument("--tac-micro-token-threshold", type=int, default=4)
    parser.add_argument("--tac-small-token-threshold", type=int, default=16)
    parser.add_argument("--tac-min-vectors-per-centroid", type=int, default=4)
    parser.add_argument("--tac-kmeans-iterations", type=int, default=6)
    parser.add_argument(
        "--tac-candidate-pruning-alpha",
        type=float,
        default=0.0,
        help="Optional Tachiom Candidates Pruning alpha; 0 disables pruning.",
    )
    parser.add_argument("--tac-pq-subspace-count", type=int, default=32)
    parser.add_argument("--tac-pq-codebook-size", type=int, default=256)
    parser.add_argument("--tac-pq-kmeans-iterations", type=int, default=6)
    parser.add_argument(
        "--tac-pq-training-sample-count",
        type=int,
        default=0,
        help="Optional residual-PQ training sample count; 0 uses all tokens.",
    )
    parser.add_argument("--tac-hnsw-max-neighbors", type=int, default=16)
    parser.add_argument("--tac-hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--tac-hnsw-ef-search", type=int, default=64)
    parser.add_argument("--tac-hnsw-level-probability", type=float, default=0.0625)
    parser.add_argument("--tac-hnsw-seed", type=int, default=7)
    parser.add_argument("--kayak-plaid-centroid-count", type=int, default=128)
    parser.add_argument("--kayak-plaid-centroids-per-query-vector", type=int, default=32)
    parser.add_argument("--kayak-plaid-candidate-k", type=int, default=80)
    parser.add_argument(
        "--kayak-plaid-payload",
        choices=("exact", "i8"),
        default="i8",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/tachiom_tac_probe/summary.json"),
    )
    parser.add_argument("--emit-quiet-mean", action="store_true")
    return parser.parse_args()


def _parse_engines(value: str) -> tuple[str, ...]:
    engines = tuple(part.strip() for part in value.split(",") if part.strip())
    supported = {
        "exact",
        "tachiom_tac",
        "tachiom_tac_mojo",
        "tachiom_tac_hnsw",
        "tachiom_tac_hnsw_mojo",
        "tachiom_tac_i8_mojo",
        "tachiom_tac_pq",
        "tachiom_tac_pq_mojo",
        "kayak_plaid",
    }
    unknown = tuple(engine for engine in engines if engine not in supported)
    if unknown:
        raise argparse.ArgumentTypeError(
            "unsupported engine(s): "
            + ", ".join(unknown)
            + "; supported engines: "
            + ", ".join(sorted(supported))
        )
    if not engines:
        raise argparse.ArgumentTypeError("at least one engine is required")
    if "exact" not in engines:
        raise argparse.ArgumentTypeError("exact reference is required")
    return engines


def _tachiom_byte_fields(index: TachiomTacIndex) -> dict[str, int | str]:
    return {
        "index_bytes": index.index_bytes,
        "index_bytes_kind": "exact_vectors_plus_tac_sidecar",
        "tac_sidecar_index_bytes": index.sidecar_index_bytes,
        "exact_rerank_vector_bytes": index.exact_rerank_vector_bytes,
        "total_bytes_with_exact_rerank": index.index_bytes,
    }


def build_token_structured_inputs(
    shape: TokenStructuredShape,
    *,
    seed: int,
    zipf_skew: float,
    document_noise: float,
    query_noise: float,
) -> TokenStructuredInputs:
    shape.validate()
    rng = np.random.default_rng(seed)
    token_centers = rng.standard_normal(
        (shape.vocabulary_size, shape.vector_dim),
    ).astype(np.float32)
    token_centers = _l2_normalize_last_axis(token_centers)
    ranks = np.arange(1, shape.vocabulary_size + 1, dtype=np.float64)
    token_probabilities = 1.0 / np.power(ranks, zipf_skew)
    token_probabilities = token_probabilities / token_probabilities.sum()

    token_ids = rng.choice(
        shape.vocabulary_size,
        size=(shape.document_count, shape.document_vector_count),
        p=token_probabilities,
    ).astype(np.int64)
    documents = token_centers[token_ids] + rng.normal(
        0.0,
        document_noise,
        size=(shape.document_count, shape.document_vector_count, shape.vector_dim),
    ).astype(np.float32)
    documents = _l2_normalize_last_axis(documents)

    queries = np.empty(
        (shape.query_count, shape.query_vector_count, shape.vector_dim),
        dtype=np.float32,
    )
    query_token_ids = np.empty(
        (shape.query_count, shape.query_vector_count),
        dtype=np.int64,
    )
    inverse_probability = 1.0 / token_probabilities
    for query_index in range(shape.query_count):
        source_document = int(rng.integers(0, shape.document_count))
        source_token_ids = token_ids[source_document]
        source_weights = inverse_probability[source_token_ids]
        source_weights = source_weights / source_weights.sum()
        source_positions = rng.choice(
            shape.document_vector_count,
            size=shape.query_vector_count,
            replace=True,
            p=source_weights,
        )
        selected_token_ids = source_token_ids[source_positions]
        query_token_ids[query_index] = selected_token_ids
        queries[query_index] = token_centers[selected_token_ids] + rng.normal(
            0.0,
            query_noise,
            size=(shape.query_vector_count, shape.vector_dim),
        ).astype(np.float32)
    queries = _l2_normalize_last_axis(queries)
    return TokenStructuredInputs(
        documents=documents,
        token_ids=token_ids,
        queries=queries,
        query_token_ids=query_token_ids,
        doc_ids=tuple(f"doc-{index:08d}" for index in range(shape.document_count)),
        token_probabilities=token_probabilities,
    )


def benchmark_exact(
    *,
    shape: TokenStructuredShape,
    inputs: TokenStructuredInputs,
    warmup_iterations: int,
    measurement_iterations: int,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...]]:
    _validate_measurement_counts(warmup_iterations, measurement_iterations)
    for _ in range(warmup_iterations):
        exact_search_positions(
            documents=inputs.documents,
            queries=inputs.queries,
            final_k=shape.top_k,
        )

    durations: list[float] = []
    first_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        positions = exact_search_positions(
            documents=inputs.documents,
            queries=inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_positions = positions
    if first_positions is None:
        raise RuntimeError("exact benchmark produced no positions")
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return (
        {
            "system_name": "numpy_exact_maxsim_reference",
            "engine": "exact",
            "status": "ok",
            "index_kind": "full_scan_token_vectors",
            "vector_metric": "dot_product",
            "build_seconds": 0.0,
            "index_bytes": int(inputs.documents.nbytes),
            "query_batch_min_seconds": stats["min_seconds"],
            "query_batch_median_seconds": stats["median_seconds"],
            "query_batch_mean_seconds": batch_mean_seconds,
            "query_batch_max_seconds": stats["max_seconds"],
            "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
            "query_qps": shape.query_count / batch_mean_seconds,
            "candidate_recall_at_k_vs_exact": 1.0,
            "final_recall_at_k_vs_exact": 1.0,
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
        },
        first_positions,
    )


def benchmark_tachiom_tac(
    *,
    shape: TokenStructuredShape,
    inputs: TokenStructuredInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: TachiomTacConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    _validate_measurement_counts(warmup_iterations, measurement_iterations)
    config.validate(final_k=shape.top_k)
    index = TachiomTacIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        token_ids=inputs.token_ids,
        config=config,
        final_k=shape.top_k,
    )

    for _ in range(warmup_iterations):
        index.search_batch_positions(inputs.queries, final_k=shape.top_k)

    durations: list[float] = []
    first_final_positions = None
    first_candidate_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = index.search_batch_positions(
            inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = index.candidate_positions_batch(
                inputs.queries,
                final_k=shape.top_k,
            )

    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("TAC benchmark produced no positions")
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "tachiom_tac_exact_centroid_probe",
        "engine": "tachiom_tac",
        "status": "ok",
        "paper_status": "partial_first_gate_no_hnsw_no_pq",
        "index_kind": "token_aware_centroid_postings_exact_rerank",
        "vector_metric": "dot_product",
        "build_seconds": index.build_seconds,
        **_tachiom_byte_fields(index),
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "candidate_recall_at_k_vs_exact": mean_candidate_set_recall_at_k(
            candidate_positions_by_query=first_candidate_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "final_recall_at_k_vs_exact": mean_recall_at_k(
            candidate_positions_by_query=first_final_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "candidate_pruning_alpha": config.candidate_pruning_alpha,
        **_candidate_window_fields(first_candidate_positions),
        "posting_count": index.posting_count,
        "allocation": index.allocation_summary.to_json_ready(),
    }


def benchmark_tachiom_tac_mojo(
    *,
    shape: TokenStructuredShape,
    inputs: TokenStructuredInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: TachiomTacConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    _validate_measurement_counts(warmup_iterations, measurement_iterations)
    if shape.vector_dim != 128:
        return {
            "system_name": "tachiom_tac_mojo_exact_centroid_probe",
            "engine": "tachiom_tac_mojo",
            "status": "skipped",
            "reason": "Mojo TAC currently requires vector_dim=128",
        }
    config.validate(final_k=shape.top_k)
    python_index = TachiomTacIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        token_ids=inputs.token_ids,
        config=config,
        final_k=shape.top_k,
    )
    started_at = time.perf_counter()
    index = TachiomTacMojoIndex.from_tac_index(python_index)
    mojo_prepare_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        index.search_batch_positions(inputs.queries, final_k=shape.top_k)

    durations: list[float] = []
    first_final_positions = None
    first_candidate_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = index.search_batch_positions(
            inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = index.candidate_positions_batch(
                inputs.queries,
                final_k=shape.top_k,
            )

    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("Mojo TAC benchmark produced no positions")
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "tachiom_tac_mojo_exact_centroid_probe",
        "engine": "tachiom_tac_mojo",
        "status": "ok",
        "paper_status": "partial_first_gate_no_hnsw_no_pq",
        "index_kind": "token_aware_centroid_postings_exact_rerank",
        "backend": "mojo_tachiom_tac_exact_centroid",
        "vector_metric": "dot_product",
        "build_seconds": python_index.build_seconds + mojo_prepare_seconds,
        "python_tac_build_seconds": python_index.build_seconds,
        "mojo_prepare_seconds": mojo_prepare_seconds,
        **_tachiom_byte_fields(python_index),
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "candidate_recall_at_k_vs_exact": mean_candidate_set_recall_at_k(
            candidate_positions_by_query=first_candidate_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "final_recall_at_k_vs_exact": mean_recall_at_k(
            candidate_positions_by_query=first_final_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": python_index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "candidate_pruning_alpha": config.candidate_pruning_alpha,
        **_candidate_window_fields(first_candidate_positions),
        "posting_count": python_index.posting_count,
        "allocation": python_index.allocation_summary.to_json_ready(),
    }


def benchmark_tachiom_tac_hnsw(
    *,
    shape: TokenStructuredShape,
    inputs: TokenStructuredInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: TachiomTacConfig,
    hnsw_config: TachiomHnswConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    _validate_measurement_counts(warmup_iterations, measurement_iterations)
    config.validate(final_k=shape.top_k)
    started_at = time.perf_counter()
    tac_index = TachiomTacIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        token_ids=inputs.token_ids,
        config=config,
        final_k=shape.top_k,
    )
    index = TachiomTacHnswIndex.from_tac_index(
        tac_index,
        config=hnsw_config,
    )
    build_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        index.search_batch_positions(inputs.queries, final_k=shape.top_k)

    durations: list[float] = []
    first_final_positions = None
    first_candidate_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = index.search_batch_positions(
            inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = index.candidate_positions_batch(
                inputs.queries,
                final_k=shape.top_k,
            )

    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("TAC HNSW benchmark produced no positions")
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "tachiom_tac_hnsw_centroid_probe",
        "engine": "tachiom_tac_hnsw",
        "status": "ok",
        "paper_status": "hnsw_centroid_layer_exact_rerank_no_pq",
        "index_kind": "token_aware_hnsw_centroid_postings_exact_rerank",
        "backend": "python_tachiom_centroid_hnsw",
        "vector_metric": "dot_product",
        "build_seconds": build_seconds,
        "python_tac_build_seconds": tac_index.build_seconds,
        "python_hnsw_build_seconds": index.graph.build_seconds,
        "index_bytes": index.index_bytes,
        "index_bytes_kind": "exact_vectors_plus_tac_sidecar_plus_hnsw_graph",
        "tac_sidecar_index_bytes": tac_index.sidecar_index_bytes,
        "hnsw_graph_bytes": index.graph.graph_bytes,
        "exact_rerank_vector_bytes": tac_index.exact_rerank_vector_bytes,
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "candidate_recall_at_k_vs_exact": mean_candidate_set_recall_at_k(
            candidate_positions_by_query=first_candidate_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "final_recall_at_k_vs_exact": mean_recall_at_k(
            candidate_positions_by_query=first_final_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "candidate_pruning_alpha": config.candidate_pruning_alpha,
        **_candidate_window_fields(first_candidate_positions),
        "posting_count": index.posting_count,
        "hnsw_max_neighbors": hnsw_config.max_neighbors,
        "hnsw_ef_construction": hnsw_config.ef_construction,
        "hnsw_ef_search": hnsw_config.ef_search,
        "hnsw_level_probability": hnsw_config.level_probability,
        "hnsw_seed": hnsw_config.seed,
        "allocation": tac_index.allocation_summary.to_json_ready(),
    }


def benchmark_tachiom_tac_hnsw_mojo(
    *,
    shape: TokenStructuredShape,
    inputs: TokenStructuredInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: TachiomTacConfig,
    hnsw_config: TachiomHnswConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    _validate_measurement_counts(warmup_iterations, measurement_iterations)
    if shape.vector_dim != 128:
        return {
            "system_name": "tachiom_tac_hnsw_mojo_centroid_probe",
            "engine": "tachiom_tac_hnsw_mojo",
            "status": "skipped",
            "reason": "Mojo TAC HNSW currently requires vector_dim=128",
        }
    config.validate(final_k=shape.top_k)
    started_at = time.perf_counter()
    tac_index = TachiomTacIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        token_ids=inputs.token_ids,
        config=config,
        final_k=shape.top_k,
    )
    hnsw_index = TachiomTacHnswIndex.from_tac_index(
        tac_index,
        config=hnsw_config,
    )
    prepared_index = TachiomTacHnswMojoIndex.from_hnsw_index(hnsw_index)
    build_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        prepared_index.search_batch_positions(inputs.queries, final_k=shape.top_k)

    durations: list[float] = []
    first_final_positions = None
    first_candidate_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = prepared_index.search_batch_positions(
            inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = prepared_index.candidate_positions_batch(
                inputs.queries,
                final_k=shape.top_k,
            )

    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("Mojo TAC HNSW benchmark produced no positions")
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "tachiom_tac_hnsw_mojo_centroid_probe",
        "engine": "tachiom_tac_hnsw_mojo",
        "status": "ok",
        "paper_status": "hnsw_centroid_layer_native_search_exact_rerank_no_pq",
        "index_kind": "token_aware_hnsw_centroid_postings_exact_rerank",
        "backend": "mojo_tachiom_centroid_hnsw",
        "vector_metric": "dot_product",
        "build_seconds": build_seconds,
        "python_tac_build_seconds": tac_index.build_seconds,
        "python_hnsw_build_seconds": hnsw_index.graph.build_seconds,
        "index_bytes": prepared_index.index_bytes,
        "index_bytes_kind": "exact_vectors_plus_tac_sidecar_plus_hnsw_graph",
        "tac_sidecar_index_bytes": tac_index.sidecar_index_bytes,
        "hnsw_graph_bytes": hnsw_index.graph.graph_bytes,
        "exact_rerank_vector_bytes": tac_index.exact_rerank_vector_bytes,
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "candidate_recall_at_k_vs_exact": mean_candidate_set_recall_at_k(
            candidate_positions_by_query=first_candidate_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "final_recall_at_k_vs_exact": mean_recall_at_k(
            candidate_positions_by_query=first_final_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": prepared_index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "candidate_pruning_alpha": config.candidate_pruning_alpha,
        **_candidate_window_fields(first_candidate_positions),
        "posting_count": prepared_index.posting_count,
        "hnsw_graph_edge_count": prepared_index.graph_edge_count,
        "hnsw_max_neighbors": hnsw_config.max_neighbors,
        "hnsw_ef_construction": hnsw_config.ef_construction,
        "hnsw_ef_search": hnsw_config.ef_search,
        "hnsw_level_probability": hnsw_config.level_probability,
        "hnsw_seed": hnsw_config.seed,
        "allocation": tac_index.allocation_summary.to_json_ready(),
    }


def benchmark_tachiom_tac_i8_mojo(
    *,
    shape: TokenStructuredShape,
    inputs: TokenStructuredInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: TachiomTacConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    _validate_measurement_counts(warmup_iterations, measurement_iterations)
    if shape.vector_dim != 128:
        return {
            "system_name": "tachiom_tac_i8_mojo_probe",
            "engine": "tachiom_tac_i8_mojo",
            "status": "skipped",
            "reason": "Mojo TAC i8 currently requires vector_dim=128",
        }
    config.validate(final_k=shape.top_k)
    python_index = TachiomTacIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        token_ids=inputs.token_ids,
        config=config,
        final_k=shape.top_k,
    )
    started_at = time.perf_counter()
    index = TachiomTacI8MojoIndex.from_tac_index(python_index)
    mojo_prepare_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        index.search_batch_positions(inputs.queries, final_k=shape.top_k)

    durations: list[float] = []
    first_final_positions = None
    first_candidate_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = index.search_batch_positions(
            inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = index.candidate_positions_batch(
                inputs.queries,
                final_k=shape.top_k,
            )

    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("Mojo TAC i8 benchmark produced no positions")
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "tachiom_tac_i8_mojo_probe",
        "engine": "tachiom_tac_i8_mojo",
        "status": "ok",
        "paper_status": "compressed_rerank_probe_no_hnsw_no_pq",
        "index_kind": index.index_kind,
        "backend": "mojo_tachiom_tac_i8_rerank",
        "vector_metric": "dot_product",
        "build_seconds": python_index.build_seconds + mojo_prepare_seconds,
        "python_tac_build_seconds": python_index.build_seconds,
        "mojo_prepare_seconds": mojo_prepare_seconds,
        "index_bytes": index.index_bytes,
        "index_bytes_kind": "tac_sidecar_plus_i8_token_payload",
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "candidate_recall_at_k_vs_exact": mean_candidate_set_recall_at_k(
            candidate_positions_by_query=first_candidate_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "final_recall_at_k_vs_exact": mean_recall_at_k(
            candidate_positions_by_query=first_final_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "candidate_pruning_alpha": config.candidate_pruning_alpha,
        **_candidate_window_fields(first_candidate_positions),
        "posting_count": index.posting_count,
        "payload": "i8",
        "rerank": index.rerank_kind,
        "allocation": python_index.allocation_summary.to_json_ready(),
    }


def benchmark_tachiom_tac_pq(
    *,
    shape: TokenStructuredShape,
    inputs: TokenStructuredInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: TachiomTacConfig,
    pq_config: TachiomResidualPqConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    _validate_measurement_counts(warmup_iterations, measurement_iterations)
    config.validate(final_k=shape.top_k)
    started_at = time.perf_counter()
    tac_index = TachiomTacIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        token_ids=inputs.token_ids,
        config=config,
        final_k=shape.top_k,
    )
    pq_index = TachiomResidualPqIndex.from_tac_index(
        tac_index,
        config=pq_config,
    )
    build_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        pq_index.search_batch_positions(inputs.queries, final_k=shape.top_k)

    durations: list[float] = []
    first_final_positions = None
    first_candidate_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = pq_index.search_batch_positions(
            inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = pq_index.candidate_positions_batch(
                inputs.queries,
                final_k=shape.top_k,
            )

    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("TAC residual-PQ benchmark produced no positions")
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "tachiom_tac_residual_pq_probe",
        "engine": "tachiom_tac_pq",
        "status": "ok",
        "paper_status": "residual_pq_refine_reference_no_hnsw",
        "index_kind": pq_index.index_kind,
        "backend": "python_tachiom_tac_residual_pq",
        "vector_metric": "dot_product",
        "build_seconds": build_seconds,
        "python_tac_build_seconds": tac_index.build_seconds,
        "python_pq_build_seconds": pq_index.build_seconds,
        "index_bytes": pq_index.index_bytes,
        "index_bytes_kind": "tac_sidecar_plus_centroid_ids_norms_residual_pq_codes",
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "candidate_recall_at_k_vs_exact": mean_candidate_set_recall_at_k(
            candidate_positions_by_query=first_candidate_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "final_recall_at_k_vs_exact": mean_recall_at_k(
            candidate_positions_by_query=first_final_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": pq_index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "candidate_pruning_alpha": config.candidate_pruning_alpha,
        **_candidate_window_fields(first_candidate_positions),
        "posting_count": pq_index.posting_count,
        "payload": "normalized_residual_pq",
        "rerank": pq_index.rerank_kind,
        "pq_subspace_count": pq_index.subspace_count,
        "pq_codebook_size": pq_config.codebook_size,
        "pq_effective_codebook_size": pq_index.effective_codebook_size,
        "pq_training_sample_count": pq_config.training_sample_count,
        "pq_code_bits": 8,
        "allocation": tac_index.allocation_summary.to_json_ready(),
    }


def benchmark_tachiom_tac_pq_mojo(
    *,
    shape: TokenStructuredShape,
    inputs: TokenStructuredInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: TachiomTacConfig,
    pq_config: TachiomResidualPqConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    _validate_measurement_counts(warmup_iterations, measurement_iterations)
    if shape.vector_dim != 128:
        return {
            "system_name": "tachiom_tac_residual_pq_mojo_probe",
            "engine": "tachiom_tac_pq_mojo",
            "status": "skipped",
            "reason": "Mojo TAC residual-PQ currently requires vector_dim=128",
        }
    config.validate(final_k=shape.top_k)
    started_at = time.perf_counter()
    tac_index = TachiomTacIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        token_ids=inputs.token_ids,
        config=config,
        final_k=shape.top_k,
    )
    pq_index = TachiomResidualPqIndex.from_tac_index(
        tac_index,
        config=pq_config,
    )
    prepared_index = TachiomResidualPqMojoIndex.from_pq_index(pq_index)
    build_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        prepared_index.search_batch_positions(inputs.queries, final_k=shape.top_k)

    durations: list[float] = []
    first_final_positions = None
    first_candidate_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = prepared_index.search_batch_positions(
            inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = prepared_index.candidate_positions_batch(
                inputs.queries,
                final_k=shape.top_k,
            )

    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("Mojo TAC residual-PQ benchmark produced no positions")
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "tachiom_tac_residual_pq_mojo_probe",
        "engine": "tachiom_tac_pq_mojo",
        "status": "ok",
        "paper_status": "residual_pq_refine_native_no_hnsw",
        "index_kind": prepared_index.index_kind,
        "backend": "mojo_tachiom_tac_residual_pq",
        "vector_metric": "dot_product",
        "build_seconds": build_seconds,
        "python_tac_build_seconds": tac_index.build_seconds,
        "python_pq_build_seconds": pq_index.build_seconds,
        "index_bytes": prepared_index.index_bytes,
        "index_bytes_kind": "tac_sidecar_plus_centroid_ids_norms_residual_pq_codes",
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "candidate_recall_at_k_vs_exact": mean_candidate_set_recall_at_k(
            candidate_positions_by_query=first_candidate_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "final_recall_at_k_vs_exact": mean_recall_at_k(
            candidate_positions_by_query=first_final_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": prepared_index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "candidate_pruning_alpha": config.candidate_pruning_alpha,
        **_candidate_window_fields(first_candidate_positions),
        "posting_count": prepared_index.posting_count,
        "payload": "normalized_residual_pq",
        "rerank": prepared_index.rerank_kind,
        "pq_subspace_count": prepared_index.subspace_count,
        "pq_codebook_size": pq_config.codebook_size,
        "pq_effective_codebook_size": prepared_index.effective_codebook_size,
        "pq_training_sample_count": pq_config.training_sample_count,
        "pq_code_bits": 8,
        "allocation": tac_index.allocation_summary.to_json_ready(),
    }


def benchmark_kayak_plaid(
    *,
    shape: TokenStructuredShape,
    inputs: TokenStructuredInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: KayakPlaidApproxConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    _validate_measurement_counts(warmup_iterations, measurement_iterations)
    if shape.vector_dim != 128:
        return {
            "system_name": "kayak_plaid_mojo_probe",
            "engine": "kayak_plaid",
            "status": "skipped",
            "reason": "Kayak PLAID probe currently requires vector_dim=128",
        }
    config.validate(final_k=shape.top_k)
    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=config,
        final_k=shape.top_k,
    )
    build_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        index.search_batch_positions(inputs.queries, final_k=shape.top_k)

    durations: list[float] = []
    first_positions = None
    first_candidate_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        positions = index.search_batch_positions(
            inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_positions = positions
            if config.payload == "i8":
                first_candidate_positions = index.i8_candidate_positions_batch(
                    inputs.queries
                )
    if first_positions is None:
        raise RuntimeError("Kayak PLAID benchmark produced no positions")

    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "kayak_plaid_mojo_probe",
        "engine": "kayak_plaid",
        "status": "ok",
        "index_kind": index.index_kind,
        "backend": "mojo_centroid_postings",
        "vector_metric": "dot_product",
        "build_seconds": build_seconds,
        "index_bytes": index.index_bytes,
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "candidate_recall_at_k_vs_exact": (
            mean_candidate_set_recall_at_k(
                candidate_positions_by_query=first_candidate_positions,
                reference_positions_by_query=reference_positions_by_query,
                k=shape.top_k,
            )
            if first_candidate_positions is not None
            else None
        ),
        "final_recall_at_k_vs_exact": mean_recall_at_k(
            candidate_positions_by_query=first_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "payload": config.payload,
        "rerank": index.rerank_kind,
    }


def build_pairwise_rows(systems: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(
        (system for system in systems if system["engine"] == "exact" and system["status"] == "ok"),
        None,
    )
    if baseline is None:
        return []
    rows: list[dict[str, Any]] = []
    for system in systems:
        if system is baseline or system["status"] != "ok":
            continue
        rows.append(
            {
                "baseline": baseline["system_name"],
                "candidate": system["system_name"],
                "query_batch_seconds_ratio_vs_exact": (
                    float(system["query_batch_mean_seconds"])
                    / float(baseline["query_batch_mean_seconds"])
                ),
                "query_qps_ratio_vs_exact": (
                    float(system["query_qps"]) / float(baseline["query_qps"])
                ),
                "index_bytes_ratio_vs_exact_vectors": _optional_ratio(
                    system.get("index_bytes"),
                    baseline.get("index_bytes"),
                ),
                "candidate_recall_at_k_vs_exact": system.get(
                    "candidate_recall_at_k_vs_exact"
                ),
                "final_recall_at_k_vs_exact": system.get(
                    "final_recall_at_k_vs_exact"
                ),
            }
        )
    return rows


def emit_quiet_mean_sections(
    systems: Sequence[dict[str, Any]],
    *,
    label_prefix: str = "",
) -> None:
    for system in systems:
        if system.get("status") != "ok":
            continue
        mean_seconds = system.get("query_batch_mean_seconds")
        if mean_seconds is None:
            continue
        label_parts = [
            part
            for part in (
                label_prefix,
                str(system.get("config_name") or system.get("engine")),
                str(system.get("system_name")),
            )
            if part
        ]
        print("== " + " / ".join(label_parts) + " ==")
        print("Mean:", mean_seconds)


def main() -> None:
    args = parse_args()
    shape = TokenStructuredShape(
        document_count=args.document_count,
        document_vector_count=args.document_vector_count,
        query_count=args.query_count,
        query_vector_count=args.query_vector_count,
        vector_dim=args.vector_dim,
        top_k=args.top_k,
        vocabulary_size=args.vocabulary_size,
    )
    inputs = build_token_structured_inputs(
        shape,
        seed=args.seed,
        zipf_skew=args.zipf_skew,
        document_noise=args.document_noise,
        query_noise=args.query_noise,
    )
    exact_row, reference_positions = benchmark_exact(
        shape=shape,
        inputs=inputs,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    systems: list[dict[str, Any]] = [exact_row]
    if "tachiom_tac" in args.engines:
        systems.append(
            benchmark_tachiom_tac(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=TachiomTacConfig(
                    centroid_count=args.tac_centroid_count,
                    micro_token_threshold=args.tac_micro_token_threshold,
                    small_token_threshold=args.tac_small_token_threshold,
                    min_vectors_per_centroid=args.tac_min_vectors_per_centroid,
                    kmeans_iterations=args.tac_kmeans_iterations,
                    centroids_per_query_vector=args.tac_centroids_per_query_vector,
                    candidate_k=args.tac_candidate_k,
                    candidate_pruning_alpha=_optional_pruning_alpha(
                        args.tac_candidate_pruning_alpha
                    ),
                ),
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )
    if "tachiom_tac_mojo" in args.engines:
        systems.append(
            benchmark_tachiom_tac_mojo(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=TachiomTacConfig(
                    centroid_count=args.tac_centroid_count,
                    micro_token_threshold=args.tac_micro_token_threshold,
                    small_token_threshold=args.tac_small_token_threshold,
                    min_vectors_per_centroid=args.tac_min_vectors_per_centroid,
                    kmeans_iterations=args.tac_kmeans_iterations,
                    centroids_per_query_vector=args.tac_centroids_per_query_vector,
                    candidate_k=args.tac_candidate_k,
                    candidate_pruning_alpha=_optional_pruning_alpha(
                        args.tac_candidate_pruning_alpha
                    ),
                ),
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )
    if "tachiom_tac_hnsw" in args.engines:
        systems.append(
            benchmark_tachiom_tac_hnsw(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=TachiomTacConfig(
                    centroid_count=args.tac_centroid_count,
                    micro_token_threshold=args.tac_micro_token_threshold,
                    small_token_threshold=args.tac_small_token_threshold,
                    min_vectors_per_centroid=args.tac_min_vectors_per_centroid,
                    kmeans_iterations=args.tac_kmeans_iterations,
                    centroids_per_query_vector=args.tac_centroids_per_query_vector,
                    candidate_k=args.tac_candidate_k,
                    candidate_pruning_alpha=_optional_pruning_alpha(
                        args.tac_candidate_pruning_alpha
                    ),
                ),
                hnsw_config=TachiomHnswConfig(
                    max_neighbors=args.tac_hnsw_max_neighbors,
                    ef_construction=args.tac_hnsw_ef_construction,
                    ef_search=args.tac_hnsw_ef_search,
                    level_probability=args.tac_hnsw_level_probability,
                    seed=args.tac_hnsw_seed,
                ),
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )
    if "tachiom_tac_hnsw_mojo" in args.engines:
        systems.append(
            benchmark_tachiom_tac_hnsw_mojo(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=TachiomTacConfig(
                    centroid_count=args.tac_centroid_count,
                    micro_token_threshold=args.tac_micro_token_threshold,
                    small_token_threshold=args.tac_small_token_threshold,
                    min_vectors_per_centroid=args.tac_min_vectors_per_centroid,
                    kmeans_iterations=args.tac_kmeans_iterations,
                    centroids_per_query_vector=args.tac_centroids_per_query_vector,
                    candidate_k=args.tac_candidate_k,
                    candidate_pruning_alpha=_optional_pruning_alpha(
                        args.tac_candidate_pruning_alpha
                    ),
                ),
                hnsw_config=TachiomHnswConfig(
                    max_neighbors=args.tac_hnsw_max_neighbors,
                    ef_construction=args.tac_hnsw_ef_construction,
                    ef_search=args.tac_hnsw_ef_search,
                    level_probability=args.tac_hnsw_level_probability,
                    seed=args.tac_hnsw_seed,
                ),
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )
    if "tachiom_tac_i8_mojo" in args.engines:
        systems.append(
            benchmark_tachiom_tac_i8_mojo(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=TachiomTacConfig(
                    centroid_count=args.tac_centroid_count,
                    micro_token_threshold=args.tac_micro_token_threshold,
                    small_token_threshold=args.tac_small_token_threshold,
                    min_vectors_per_centroid=args.tac_min_vectors_per_centroid,
                    kmeans_iterations=args.tac_kmeans_iterations,
                    centroids_per_query_vector=args.tac_centroids_per_query_vector,
                    candidate_k=args.tac_candidate_k,
                    candidate_pruning_alpha=_optional_pruning_alpha(
                        args.tac_candidate_pruning_alpha
                    ),
                ),
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )
    if "tachiom_tac_pq" in args.engines:
        systems.append(
            benchmark_tachiom_tac_pq(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=TachiomTacConfig(
                    centroid_count=args.tac_centroid_count,
                    micro_token_threshold=args.tac_micro_token_threshold,
                    small_token_threshold=args.tac_small_token_threshold,
                    min_vectors_per_centroid=args.tac_min_vectors_per_centroid,
                    kmeans_iterations=args.tac_kmeans_iterations,
                    centroids_per_query_vector=args.tac_centroids_per_query_vector,
                    candidate_k=args.tac_candidate_k,
                    candidate_pruning_alpha=_optional_pruning_alpha(
                        args.tac_candidate_pruning_alpha
                    ),
                ),
                pq_config=TachiomResidualPqConfig(
                    subspace_count=args.tac_pq_subspace_count,
                    codebook_size=args.tac_pq_codebook_size,
                    kmeans_iterations=args.tac_pq_kmeans_iterations,
                    training_sample_count=_optional_sample_count(
                        args.tac_pq_training_sample_count
                    ),
                ),
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )
    if "tachiom_tac_pq_mojo" in args.engines:
        systems.append(
            benchmark_tachiom_tac_pq_mojo(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=TachiomTacConfig(
                    centroid_count=args.tac_centroid_count,
                    micro_token_threshold=args.tac_micro_token_threshold,
                    small_token_threshold=args.tac_small_token_threshold,
                    min_vectors_per_centroid=args.tac_min_vectors_per_centroid,
                    kmeans_iterations=args.tac_kmeans_iterations,
                    centroids_per_query_vector=args.tac_centroids_per_query_vector,
                    candidate_k=args.tac_candidate_k,
                    candidate_pruning_alpha=_optional_pruning_alpha(
                        args.tac_candidate_pruning_alpha
                    ),
                ),
                pq_config=TachiomResidualPqConfig(
                    subspace_count=args.tac_pq_subspace_count,
                    codebook_size=args.tac_pq_codebook_size,
                    kmeans_iterations=args.tac_pq_kmeans_iterations,
                    training_sample_count=_optional_sample_count(
                        args.tac_pq_training_sample_count
                    ),
                ),
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )
    if "kayak_plaid" in args.engines:
        systems.append(
            benchmark_kayak_plaid(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=KayakPlaidApproxConfig(
                    centroid_count=args.kayak_plaid_centroid_count,
                    centroids_per_query_vector=args.kayak_plaid_centroids_per_query_vector,
                    candidate_k=args.kayak_plaid_candidate_k,
                    payload=args.kayak_plaid_payload,
                ),
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )

    token_counts = np.bincount(
        inputs.token_ids.reshape(-1),
        minlength=shape.vocabulary_size,
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "paper": {
            "name": "Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing",
            "url": TACHIOM_PAPER_URL,
        },
        "epistemic_status": {
            "claim": (
                "This is a first-gate TAC probe: token-aware centroid allocation "
                "and exact centroid scan, not the full Tachiom HNSW/PQ system."
            ),
            "promote_to_engine_only_if": (
                "candidate recall per cost beats current Kayak centroid/posting "
                "or i8 PLAID-style lanes on explicit vector-count workloads."
            ),
        },
        "shape": shape.to_json_ready(),
        "token_distribution": {
            "zipf_skew": args.zipf_skew,
            "observed_token_type_count": int(np.count_nonzero(token_counts)),
            "top_100_token_vector_fraction": float(
                token_counts[np.argsort(token_counts)[-100:]].sum()
                / max(1, token_counts.sum())
            ),
            "max_token_frequency": int(token_counts.max()),
            "median_nonzero_token_frequency": float(
                np.median(token_counts[token_counts > 0])
            ),
        },
        "systems": systems,
        "pairwise_vs_exact": build_pairwise_rows(systems),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        emit_quiet_mean_sections(systems)


def _validate_measurement_counts(
    warmup_iterations: int,
    measurement_iterations: int,
) -> None:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")


def _measurement_stats(durations: Sequence[float]) -> dict[str, float]:
    if not durations:
        raise ValueError("durations must not be empty")
    ordered = sorted(float(duration) for duration in durations)
    return {
        "min_seconds": ordered[0],
        "median_seconds": ordered[len(ordered) // 2],
        "mean_seconds": float(sum(ordered) / len(ordered)),
        "max_seconds": ordered[-1],
    }


def _optional_ratio(value: object, baseline: object) -> float | None:
    if value is None or baseline is None:
        return None
    denominator = float(baseline)
    if denominator == 0.0:
        return None
    return float(value) / denominator


def _optional_sample_count(value: int) -> int | None:
    if value <= 0:
        return None
    return value


def _optional_pruning_alpha(value: float) -> float | None:
    if value <= 0.0:
        return None
    return value


def _candidate_window_fields(
    candidate_positions_by_query: Sequence[Sequence[int]],
) -> dict[str, float | int]:
    counts = [len(row) for row in candidate_positions_by_query]
    if not counts:
        return {
            "candidate_window_min_count": 0,
            "candidate_window_mean_count": 0.0,
            "candidate_window_max_count": 0,
        }
    return {
        "candidate_window_min_count": min(counts),
        "candidate_window_mean_count": float(sum(counts) / len(counts)),
        "candidate_window_max_count": max(counts),
    }


def _l2_normalize_last_axis(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-12))


if __name__ == "__main__":
    main()
