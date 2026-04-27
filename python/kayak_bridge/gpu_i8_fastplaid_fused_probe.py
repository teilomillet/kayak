"""Fused GPU i8 probe mechanics for FastPlaid scope reports.

This module owns query-window preparation, CPU fused-score references, and the
prepared GPU handle loop. Report assembly lives in
gpu_i8_fastplaid_fused_scope.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from kayak_bridge.gpu_device_capability import MojoGpuCapability
from kayak_bridge.mojo_gpu_i8_rerank import (
    prepare_i8_fused_centroid_posting_session_handle,
    selected_posting_accumulation_reference_scores,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxIndex

from bench_fastplaid_speed_track import SpeedTrackShape  # noqa: E402


STATUS_OK = "ok"


def run_fused_centroid_posting_scope_probe(
    *,
    capability: MojoGpuCapability,
    shape: SpeedTrackShape,
    query_windows: np.ndarray,
    payload: Any,
    reference_score_windows: Sequence[np.ndarray],
    centroids_per_query_vector: int,
    configured_candidate_k: int,
    warmup_iterations: int,
) -> dict[str, object] | None:
    if not capability.available:
        return None
    handle = None
    release_seconds = 0.0
    try:
        handle = prepare_i8_fused_centroid_posting_session_handle(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            payload=payload,
            centroids_per_query_vector=centroids_per_query_vector,
            top_k=shape.top_k,
        )
        for _ in range(warmup_iterations):
            handle.score_topk_without_reference(
                queries=query_windows[0],
                reference_document_scores=reference_score_windows[0],
            )
            handle.score_topk_device_without_reference(
                queries=query_windows[0],
                reference_document_scores=reference_score_windows[0],
            )
        host_results = []
        device_results = []
        for window_index in range(len(reference_score_windows)):
            host_results.append(
                handle.score_topk_without_reference(
                    queries=query_windows[window_index],
                    reference_document_scores=reference_score_windows[window_index],
                )
            )
        for window_index in range(len(reference_score_windows)):
            device_results.append(
                handle.score_topk_device_without_reference(
                    queries=query_windows[window_index],
                    reference_document_scores=reference_score_windows[window_index],
                )
            )
        release_seconds = handle.close()
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        if handle is not None:
            try:
                release_seconds = handle.close()
            except Exception:
                pass
        return {"status": "error", "parsed": {}, "error": str(exc)}

    parsed = _parsed_payload(
        shape=shape,
        host_results=host_results,
        device_results=device_results,
        handle=handle,
        release_seconds=release_seconds,
        centroids_per_query_vector=centroids_per_query_vector,
        configured_candidate_k=configured_candidate_k,
        warmup_iterations=warmup_iterations,
    )
    status = (
        STATUS_OK
        if parsed["topk_position_agreement"] == 1.0
        and parsed["device_topk_position_agreement"] == 1.0
        else "error"
    )
    return {
        "status": status,
        "parsed": parsed,
        "measurements": [
            handle.to_json_ready(),
            {"host_topk": [result.to_json_ready() for result in host_results]},
            {"device_topk": [result.to_json_ready() for result in device_results]},
            {"release_extension_call_seconds": release_seconds},
        ],
    }


def fused_query_windows(
    *,
    base_queries: np.ndarray,
    shape: SpeedTrackShape,
    window_count: int,
    seed: int,
) -> np.ndarray:
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    query_windows = np.empty(
        (
            window_count,
            shape.query_count,
            shape.query_vector_count,
            shape.vector_dim,
        ),
        dtype=np.float32,
    )
    query_windows[0] = np.ascontiguousarray(base_queries, dtype=np.float32)
    if window_count == 1:
        return query_windows
    rng = np.random.default_rng(seed)
    query_windows[1:] = rng.standard_normal(query_windows[1:].shape).astype(np.float32)
    return query_windows


def fused_reference_score_windows(
    *,
    index: KayakPlaidApproxIndex,
    payload: Any,
    query_windows: np.ndarray,
    centroids_per_query_vector: int,
) -> tuple[np.ndarray, ...]:
    reference_scores: list[np.ndarray] = []
    for queries in query_windows:
        selected = index.i8_selected_centroids_batch(
            queries,
            centroids_per_query_vector=centroids_per_query_vector,
        )
        reference_scores.append(
            selected_posting_accumulation_reference_scores(
                payload=payload,
                selected=selected,
            )
        )
    return tuple(reference_scores)


def rank_document_scores_by_query(
    reference_scores: np.ndarray,
    *,
    shape: SpeedTrackShape,
) -> tuple[tuple[int, ...], ...]:
    positions: list[tuple[int, ...]] = []
    for query_index in range(shape.query_count):
        query_base = query_index * shape.document_count
        ranked = sorted(
            range(shape.document_count),
            key=lambda position: (
                -float(reference_scores[query_base + position]),
                position,
            ),
        )
        positions.append(tuple(ranked[: shape.top_k]))
    return tuple(positions)


def _parsed_payload(
    *,
    shape: SpeedTrackShape,
    host_results: Sequence[Any],
    device_results: Sequence[Any],
    handle: Any,
    release_seconds: float,
    centroids_per_query_vector: int,
    configured_candidate_k: int,
    warmup_iterations: int,
) -> dict[str, object]:
    window_count = len(host_results)
    host_extension_total = sum(result.extension_call_seconds for result in host_results)
    device_extension_total = sum(
        result.extension_call_seconds for result in device_results
    )
    host_marshalling_total = sum(result.host_marshalling_seconds for result in host_results)
    device_marshalling_total = sum(
        result.host_marshalling_seconds for result in device_results
    )
    host_topk_count = sum(result.topk_position_count for result in host_results)
    device_topk_count = sum(result.topk_position_count for result in device_results)
    host_topk_matches = sum(
        result.topk_position_match_count for result in host_results
    )
    device_topk_matches = sum(
        result.topk_position_match_count for result in device_results
    )
    host_score_delta = max(
        (result.topk_score_delta_max_abs for result in host_results),
        default=0.0,
    )
    device_score_delta = max(
        (result.topk_score_delta_max_abs for result in device_results),
        default=0.0,
    )
    return {
        "bridge_scope": "fused_centroid_posting_explicit_handle_fastplaid_scope",
        "candidate_score_count_per_window": shape.query_count * shape.document_count,
        "candidate_score_count_total": (
            window_count * shape.query_count * shape.document_count
        ),
        "centroids_per_query_vector": centroids_per_query_vector,
        "configured_candidate_k": configured_candidate_k,
        "device_topk_position_agreement": _agreement(
            device_topk_matches,
            device_topk_count,
        ),
        "device_topk_position_count": device_topk_count,
        "device_topk_position_match_count": device_topk_matches,
        "device_topk_score_delta_max_abs": device_score_delta,
        "device_topk_score_extension_call_seconds_per_window": (
            device_extension_total / float(window_count)
        ),
        "device_topk_score_extension_call_seconds_total": device_extension_total,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "host_marshalling_seconds_per_window": (
            host_marshalling_total / float(window_count)
        ),
        "device_host_marshalling_seconds_per_window": (
            device_marshalling_total / float(window_count)
        ),
        "payload_source": "real_kayak_i8_snapshot",
        "prepare_extension_call_seconds": handle.prepare_extension_call_seconds,
        "prepare_host_marshalling_seconds": handle.prepare_host_marshalling_seconds,
        "query_count_per_window": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "release_extension_call_seconds": release_seconds,
        "score_extension_call_seconds_per_window": (
            host_extension_total / float(window_count)
        ),
        "score_extension_call_seconds_total": host_extension_total,
        "top_k": shape.top_k,
        "topk_position_agreement": _agreement(host_topk_matches, host_topk_count),
        "topk_position_count": host_topk_count,
        "topk_position_match_count": host_topk_matches,
        "topk_return_count_per_window": shape.query_count * shape.top_k,
        "topk_score_delta_max_abs": host_score_delta,
        "total_document_vector_count": shape.document_count * shape.document_vector_count,
        "validation_reference_scores_sent_to_extension": False,
        "vector_dim": shape.vector_dim,
        "warmup_iterations": warmup_iterations,
        "window_count": window_count,
    }


def _agreement(matches: int, count: int) -> float:
    return matches / float(count) if count else 0.0
