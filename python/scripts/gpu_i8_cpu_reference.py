from __future__ import annotations

import time
from typing import Any, Sequence

from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex

from bench_fastplaid_speed_track import (
    SpeedTrackShape,
    _measurement_stats,
    benchmark_kayak_exact,
    benchmark_kayak_plaid,
    build_synthetic_inputs,
    mean_recall_at_k,
)


def cpu_reference_rows(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    centroid_count: int,
    centroids_per_query_vector: int,
    seed: int,
    normalize_vectors: bool,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    inputs = build_synthetic_inputs(
        shape,
        seed=seed,
        normalize_vectors=normalize_vectors,
    )
    exact_row, reference_positions = benchmark_kayak_exact(
        shape=shape,
        inputs=inputs,
        backend="mojo_exact_cpu",
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    )
    i8_config = KayakPlaidApproxConfig(
        centroid_count=centroid_count,
        centroids_per_query_vector=centroids_per_query_vector,
        candidate_k=candidate_k,
        payload="i8",
    )
    i8_row = benchmark_kayak_plaid(
        shape=shape,
        inputs=inputs,
        reference_positions_by_query=reference_positions,
        config=i8_config,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    )
    same_candidate_i8_row = benchmark_cpu_i8_same_candidate_rerank(
        shape=shape,
        inputs=inputs,
        reference_positions_by_query=reference_positions,
        config=i8_config,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    )
    return {
        "exact_reference": exact_row,
        "cpu_i8_reference": i8_row,
        "cpu_i8_same_candidate_reference": same_candidate_i8_row,
        "reference_scope": (
            "same-candidate CPU i8 rerank is the direct reference for the "
            "future GPU primitive; full CPU i8 and exact rows remain context"
        ),
    }


def benchmark_cpu_i8_same_candidate_rerank(
    *,
    shape: SpeedTrackShape,
    inputs: Any,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: KayakPlaidApproxConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    config.validate(final_k=shape.top_k)
    if config.payload != "i8":
        raise ValueError("same-candidate rerank reference requires payload='i8'")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=config,
        final_k=shape.top_k,
    )
    build_seconds = time.perf_counter() - started_at

    started_at = time.perf_counter()
    candidate_positions = index.i8_candidate_positions_batch(inputs.queries)
    candidate_generation_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        index.i8_score_candidate_positions_batch(inputs.queries, candidate_positions)

    durations: list[float] = []
    first_scores = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        current_scores = index.i8_score_candidate_positions_batch(
            inputs.queries,
            candidate_positions,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_scores = current_scores

    if first_scores is None:  # pragma: no cover - guarded above.
        raise RuntimeError("CPU i8 same-candidate reference produced no scores")

    ranked_positions = _rank_candidate_positions_by_score(
        candidate_positions,
        first_scores,
        final_k=shape.top_k,
    )
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    candidate_score_count_total = sum(len(row) for row in candidate_positions)
    return {
        "system_name": "kayak_plaid_i8_same_candidate_cpu_reference",
        "engine": "kayak",
        "status": "ok",
        "engine_version": "mojo_bridge",
        "index_kind": index.index_kind,
        "backend": "mojo_i8_same_candidate_rerank",
        "vector_metric": "dot_product",
        "benchmark_scope": "same_candidate_i8_rerank_only",
        "build_seconds": build_seconds,
        "candidate_generation_seconds": candidate_generation_seconds,
        "index_bytes": index.index_bytes,
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "candidate_score_qps": candidate_score_count_total / batch_mean_seconds,
        "recall_at_k_vs_kayak_exact": mean_recall_at_k(
            candidate_positions_by_query=ranked_positions,
            reference_positions_by_query=reference_positions_by_query,
            k=shape.top_k,
        ),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "candidate_position_count_total": candidate_score_count_total,
        "candidate_score_count_total": candidate_score_count_total,
        "payload": config.payload,
        "rerank": "i8_maxsim_same_candidate_window",
        "candidate_positions_sample": [
            list(row[: min(8, len(row))])
            for row in candidate_positions[: min(2, len(candidate_positions))]
        ],
        "candidate_scores_sample": [
            list(row[: min(8, len(row))])
            for row in first_scores[: min(2, len(first_scores))]
        ],
    }


def _rank_candidate_positions_by_score(
    candidate_positions_by_query: Sequence[Sequence[int]],
    scores_by_query: Sequence[Sequence[float]],
    *,
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    if len(candidate_positions_by_query) != len(scores_by_query):
        raise ValueError("candidate and score query counts must match")
    ranked_rows: list[tuple[int, ...]] = []
    for candidate_positions, scores in zip(
        candidate_positions_by_query,
        scores_by_query,
        strict=True,
    ):
        if len(candidate_positions) != len(scores):
            raise ValueError("candidate and score row lengths must match")
        ranked_offsets = sorted(
            range(len(candidate_positions)),
            key=lambda offset: (float(scores[offset]), -offset),
            reverse=True,
        )
        ranked_rows.append(
            tuple(
                int(candidate_positions[offset])
                for offset in ranked_offsets[:final_k]
            )
        )
    return tuple(ranked_rows)
