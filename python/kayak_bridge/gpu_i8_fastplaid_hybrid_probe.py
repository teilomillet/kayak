"""Hybrid fused-shortlist plus exact-rerank GPU probe.

This module owns the benchmark-only two-stage primitive: fused GPU
centroid-posting scores generate a shortlist, then the existing GPU exact i8
address scorer reranks that shortlist. It does not expose public search.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from kayak_bridge.gpu_device_capability import MojoGpuCapability
from kayak_bridge.mojo_gpu_i8_rerank import (
    prepare_i8_address_session_handle,
    prepare_i8_fused_centroid_posting_session_handle,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxIndex
from kayak_bridge.gpu_i8_fastplaid_hybrid_reference import (
    position_rows,
    rank_candidate_positions_by_score,
    topk_score_delta_max_abs,
)

from bench_fastplaid_speed_track import SpeedTrackShape  # noqa: E402


STATUS_OK = "ok"


def run_hybrid_shortlist_exact_rerank_probe(
    *,
    capability: MojoGpuCapability,
    shape: SpeedTrackShape,
    query_windows: np.ndarray,
    payload: Any,
    index: KayakPlaidApproxIndex,
    fused_reference_score_windows: Sequence[np.ndarray],
    shortlist_k: int,
    centroids_per_query_vector: int,
    warmup_iterations: int,
) -> dict[str, object] | None:
    if not capability.available:
        return None
    if shortlist_k < shape.top_k:
        raise ValueError("shortlist_k must be greater than or equal to top_k")
    fused_handle = None
    exact_handle = None
    release_seconds = 0.0
    try:
        fused_handle = prepare_i8_fused_centroid_posting_session_handle(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            payload=payload,
            centroids_per_query_vector=centroids_per_query_vector,
            top_k=shortlist_k,
        )
        exact_handle = prepare_i8_address_session_handle(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            candidate_k=shortlist_k,
            payload=payload,
        )
        for _ in range(warmup_iterations):
            shortlist = fused_handle.score_topk_device_without_reference(
                queries=query_windows[0],
                reference_document_scores=fused_reference_score_windows[0],
            )
            exact_handle.score_topk_without_reference(
                queries=query_windows[0],
                candidate_positions_by_query=_position_rows(
                    shortlist.positions,
                    query_count=shape.query_count,
                    row_width=shortlist_k,
                ),
                top_k=shape.top_k,
            )
        results = []
        for window_index in range(len(fused_reference_score_windows)):
            results.append(
                _run_hybrid_window(
                    fused_handle=fused_handle,
                    exact_handle=exact_handle,
                    index=index,
                    queries=query_windows[window_index],
                    fused_reference_scores=fused_reference_score_windows[
                        window_index
                    ],
                    shape=shape,
                    shortlist_k=shortlist_k,
                )
            )
        release_seconds += fused_handle.close()
        release_seconds += exact_handle.close()
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        for handle in (fused_handle, exact_handle):
            if handle is not None:
                try:
                    release_seconds += handle.close()
                except Exception:
                    pass
        return {"status": "error", "parsed": {}, "error": str(exc)}

    parsed = _parsed_payload(
        shape=shape,
        results=results,
        shortlist_k=shortlist_k,
        centroids_per_query_vector=centroids_per_query_vector,
        fused_prepare_seconds=fused_handle.prepare_extension_call_seconds,
        exact_prepare_seconds=exact_handle.prepare_extension_call_seconds,
        release_seconds=release_seconds,
        warmup_iterations=warmup_iterations,
    )
    return {
        "status": (
            STATUS_OK if parsed["final_topk_position_agreement"] == 1.0 else "error"
        ),
        "parsed": parsed,
        "measurements": [_window_to_json_ready(result) for result in results],
    }


def _run_hybrid_window(
    *,
    fused_handle: Any,
    exact_handle: Any,
    index: KayakPlaidApproxIndex,
    queries: np.ndarray,
    fused_reference_scores: np.ndarray,
    shape: SpeedTrackShape,
    shortlist_k: int,
) -> dict[str, Any]:
    fused_result = fused_handle.score_topk_device_without_reference(
        queries=queries,
        reference_document_scores=fused_reference_scores,
    )
    shortlist_positions = _position_rows(
        fused_result.positions,
        query_count=shape.query_count,
        row_width=shortlist_k,
    )
    exact_result = exact_handle.score_topk_without_reference(
        queries=queries,
        candidate_positions_by_query=shortlist_positions,
        top_k=shape.top_k,
    )
    reference_scores = index.i8_score_candidate_positions_batch(
        queries,
        shortlist_positions,
    )
    expected_positions = rank_candidate_positions_by_score(
        shortlist_positions,
        reference_scores,
        final_k=shape.top_k,
    )
    expected_flat = tuple(
        int(position) for row in expected_positions for position in row
    )
    final_matches = sum(
        1
        for actual, expected in zip(exact_result.positions, expected_flat)
        if actual == expected
    )
    return {
        "fused_result": fused_result,
        "exact_result": exact_result,
        "shortlist_positions": shortlist_positions,
        "expected_positions": expected_positions,
        "final_topk_position_match_count": final_matches,
        "final_topk_position_count": len(expected_flat),
        "exact_score_delta_max_abs": topk_score_delta_max_abs(
            exact_result.scores,
            reference_scores,
            exact_result.positions,
            shortlist_positions,
            top_k=shape.top_k,
        ),
    }


def _parsed_payload(
    *,
    shape: SpeedTrackShape,
    results: Sequence[dict[str, Any]],
    shortlist_k: int,
    centroids_per_query_vector: int,
    fused_prepare_seconds: float,
    exact_prepare_seconds: float,
    release_seconds: float,
    warmup_iterations: int,
) -> dict[str, object]:
    window_count = len(results)
    fused_total = sum(
        result["fused_result"].extension_call_seconds for result in results
    )
    exact_total = sum(
        result["exact_result"].extension_call_seconds for result in results
    )
    final_match_count = sum(
        int(result["final_topk_position_match_count"]) for result in results
    )
    final_position_count = sum(
        int(result["final_topk_position_count"]) for result in results
    )
    first_positions = (
        _position_rows(
            results[0]["exact_result"].positions,
            query_count=shape.query_count,
            row_width=shape.top_k,
        )
        if results
        else ()
    )
    return {
        "bridge_scope": "fused_shortlist_exact_rerank_fastplaid_scope",
        "candidate_score_count_per_window": shape.query_count * shortlist_k,
        "candidate_score_count_total": window_count * shape.query_count * shortlist_k,
        "centroids_per_query_vector": centroids_per_query_vector,
        "exact_prepare_extension_call_seconds": exact_prepare_seconds,
        "exact_rerank_topk_seconds_per_window": exact_total / float(window_count),
        "exact_rerank_topk_seconds_total": exact_total,
        "exact_score_delta_max_abs": max(
            (float(result["exact_score_delta_max_abs"]) for result in results),
            default=0.0,
        ),
        "final_topk_position_agreement": (
            final_match_count / float(final_position_count)
            if final_position_count
            else 0.0
        ),
        "final_topk_position_count": final_position_count,
        "final_topk_position_match_count": final_match_count,
        "first_window_positions_by_query": [list(row) for row in first_positions],
        "fused_device_topk_seconds_per_window": fused_total / float(window_count),
        "fused_device_topk_seconds_total": fused_total,
        "fused_prepare_extension_call_seconds": fused_prepare_seconds,
        "hybrid_extension_seconds_per_window": (
            fused_total + exact_total
        ) / float(window_count),
        "hybrid_extension_seconds_total": fused_total + exact_total,
        "query_count_per_window": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "release_extension_call_seconds": release_seconds,
        "shortlist_k": shortlist_k,
        "top_k": shape.top_k,
        "topk_return_count_per_window": shape.query_count * shape.top_k,
        "validation_reference_scores_sent_to_extension": False,
        "vector_dim": shape.vector_dim,
        "warmup_iterations": warmup_iterations,
        "window_count": window_count,
    }


def _position_rows(
    positions: Sequence[int],
    *,
    query_count: int,
    row_width: int,
) -> tuple[tuple[int, ...], ...]:
    return position_rows(
        positions,
        query_count=query_count,
        row_width=row_width,
    )


def _window_to_json_ready(result: dict[str, Any]) -> dict[str, object]:
    return {
        "exact_rerank": result["exact_result"].to_json_ready(),
        "exact_score_delta_max_abs": result["exact_score_delta_max_abs"],
        "final_topk_position_count": result["final_topk_position_count"],
        "final_topk_position_match_count": result[
            "final_topk_position_match_count"
        ],
        "fused_shortlist": result["fused_result"].to_json_ready(),
    }
