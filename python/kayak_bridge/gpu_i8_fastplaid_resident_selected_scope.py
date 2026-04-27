"""Resident selected-posting rows for FastPlaid scope comparisons.

This module assembles a benchmark-only two-stage primitive: CPU selected
centroids feed a resident GPU selected-posting candidate window, then the
existing GPU exact i8 address scorer reranks that candidate window. It does
not expose a public GPU search backend.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Sequence

import numpy as np

from kayak_bridge.gpu_device_capability import MojoGpuCapability
from kayak_bridge.gpu_i8_fastplaid_fused_probe import fused_query_windows
from kayak_bridge.gpu_i8_fastplaid_hybrid_reference import (
    position_rows,
    rank_candidate_positions_by_score,
    topk_score_delta_max_abs,
)
from kayak_bridge.gpu_i8_fastplaid_topk_metrics import (
    STATUS_PARTIAL_GPU_UNAVAILABLE,
)
from kayak_bridge.mojo_gpu_i8_rerank import (
    prepare_i8_address_session_handle,
    score_i8_selected_posting_resident_dense_candidate_positions_addresses,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex

from bench_fastplaid_speed_track import (  # noqa: E402
    SpeedTrackShape,
    SyntheticInputs,
    mean_recall_at_k,
)


STATUS_OK = "ok"


def build_resident_selected_posting_exact_rerank_scope_row(
    *,
    shape: SpeedTrackShape,
    inputs: SyntheticInputs,
    reference_positions: Sequence[Sequence[int]],
    capability: MojoGpuCapability,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=KayakPlaidApproxConfig(
            centroid_count=args.kayak_plaid_centroid_count,
            centroids_per_query_vector=args.kayak_plaid_centroids_per_query_vector,
            candidate_k=args.candidate_k,
            payload="i8",
        ),
        final_k=shape.top_k,
    )
    build_seconds = time.perf_counter() - started_at
    payload = index.i8_payload_snapshot()
    query_windows = fused_query_windows(
        base_queries=inputs.queries,
        shape=shape,
        window_count=args.gpu_topk_session_iterations,
        seed=args.seed + 500_000,
    )
    probe = run_resident_selected_posting_exact_rerank_probe(
        capability=capability,
        shape=shape,
        query_windows=query_windows,
        payload=payload,
        index=index,
        candidate_k=args.candidate_k,
        centroids_per_query_vector=args.kayak_plaid_centroids_per_query_vector,
        warmup_iterations=args.warmup_iterations,
    )
    parsed = _parsed_payload(probe)
    return {
        "system_name": (
            "kayak_gpu_i8_resident_selected_posting_exact_rerank_probe"
        ),
        "engine": "kayak_gpu_probe",
        "status": (
            probe.get("status")
            if isinstance(probe, dict)
            else STATUS_PARTIAL_GPU_UNAVAILABLE
        ),
        "backend": (
            f"mojo:{capability.target_accelerator}"
            if capability.target_accelerator is not None
            else None
        ),
        "index_kind": index.index_kind,
        "rerank_kind": index.rerank_kind,
        "scope": "real_kayak_i8_resident_selected_posting_exact_rerank_topk",
        "shape": _shape_payload(shape, candidate_k=args.candidate_k),
        "candidate_k": args.candidate_k,
        "cpu_i8_build_seconds": build_seconds,
        "recall_at_k_vs_kayak_exact": (
            mean_recall_at_k(
                candidate_positions_by_query=parsed.get(
                    "first_window_positions_by_query",
                    (),
                ),
                reference_positions_by_query=reference_positions,
                k=shape.top_k,
            )
            if parsed
            else None
        ),
        "target_accelerator": capability.target_accelerator,
        "parsed": parsed,
        "error": probe.get("error") if isinstance(probe, dict) else None,
        "measurements": probe.get("measurements") if isinstance(probe, dict) else None,
    }


def build_missing_resident_selected_posting_exact_rerank_scope_row(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    capability: MojoGpuCapability,
) -> dict[str, Any]:
    return {
        "system_name": (
            "kayak_gpu_i8_resident_selected_posting_exact_rerank_probe"
        ),
        "engine": "kayak_gpu_probe",
        "status": STATUS_PARTIAL_GPU_UNAVAILABLE,
        "backend": (
            f"mojo:{capability.target_accelerator}"
            if capability.target_accelerator is not None
            else None
        ),
        "index_kind": "sampled_centroid_postings_i8_proxy",
        "scope": "real_kayak_i8_resident_selected_posting_exact_rerank_topk",
        "shape": _shape_payload(shape, candidate_k=candidate_k),
        "parsed": {},
    }


def run_resident_selected_posting_exact_rerank_probe(
    *,
    capability: MojoGpuCapability,
    shape: SpeedTrackShape,
    query_windows: np.ndarray,
    payload: Any,
    index: KayakPlaidApproxIndex,
    candidate_k: int,
    centroids_per_query_vector: int,
    warmup_iterations: int,
) -> dict[str, object] | None:
    if not capability.available:
        return None
    if candidate_k < shape.top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")

    exact_handle = None
    release_seconds = 0.0
    try:
        exact_handle = prepare_i8_address_session_handle(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            candidate_k=candidate_k,
            payload=payload,
        )
        for _ in range(warmup_iterations):
            _run_resident_selected_window(
                exact_handle=exact_handle,
                index=index,
                payload=payload,
                queries=query_windows[0],
                shape=shape,
                candidate_k=candidate_k,
                centroids_per_query_vector=centroids_per_query_vector,
                target_accelerator=capability.target_accelerator or "",
            )
        results = [
            _run_resident_selected_window(
                exact_handle=exact_handle,
                index=index,
                payload=payload,
                queries=query_windows[window_index],
                shape=shape,
                candidate_k=candidate_k,
                centroids_per_query_vector=centroids_per_query_vector,
                target_accelerator=capability.target_accelerator or "",
            )
            for window_index in range(len(query_windows))
        ]
        release_seconds += exact_handle.close()
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        if exact_handle is not None:
            try:
                release_seconds += exact_handle.close()
            except Exception:
                pass
        return {"status": "error", "parsed": {}, "error": str(exc)}

    parsed = _parsed_payload_from_results(
        shape=shape,
        results=results,
        candidate_k=candidate_k,
        centroids_per_query_vector=centroids_per_query_vector,
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


def _run_resident_selected_window(
    *,
    exact_handle: Any,
    index: KayakPlaidApproxIndex,
    payload: Any,
    queries: np.ndarray,
    shape: SpeedTrackShape,
    candidate_k: int,
    centroids_per_query_vector: int,
    target_accelerator: str,
) -> dict[str, Any]:
    selected_started_at = time.perf_counter()
    selected = index.i8_selected_centroids_batch(
        queries,
        centroids_per_query_vector=centroids_per_query_vector,
    )
    cpu_selected_centroids_seconds = time.perf_counter() - selected_started_at
    candidate_result = (
        score_i8_selected_posting_resident_dense_candidate_positions_addresses(
            target_accelerator=target_accelerator,
            shape=shape,
            payload=payload,
            selected=selected,
            candidate_k=candidate_k,
        )
    )
    candidate_positions = position_rows(
        candidate_result.positions,
        query_count=shape.query_count,
        row_width=candidate_k,
    )
    exact_result = exact_handle.score_topk_without_reference(
        queries=queries,
        candidate_positions_by_query=candidate_positions,
        top_k=shape.top_k,
    )
    reference_scores = index.i8_score_candidate_positions_batch(
        queries,
        candidate_positions,
    )
    expected_positions = rank_candidate_positions_by_score(
        candidate_positions,
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
        "candidate_result": candidate_result,
        "exact_result": exact_result,
        "candidate_positions": candidate_positions,
        "cpu_selected_centroids_seconds": cpu_selected_centroids_seconds,
        "expected_positions": expected_positions,
        "final_topk_position_match_count": final_matches,
        "final_topk_position_count": len(expected_flat),
        "exact_score_delta_max_abs": topk_score_delta_max_abs(
            exact_result.scores,
            reference_scores,
            exact_result.positions,
            candidate_positions,
            top_k=shape.top_k,
        ),
    }


def _parsed_payload_from_results(
    *,
    shape: SpeedTrackShape,
    results: Sequence[dict[str, Any]],
    candidate_k: int,
    centroids_per_query_vector: int,
    exact_prepare_seconds: float,
    release_seconds: float,
    warmup_iterations: int,
) -> dict[str, object]:
    window_count = len(results)
    resident_total = sum(
        result["candidate_result"].score_plus_candidate_selection_seconds
        for result in results
    )
    resident_cold_total = sum(
        result["candidate_result"].prepare_call_seconds
        + result["candidate_result"].score_call_seconds
        + result["candidate_result"].candidate_selection_seconds
        + result["candidate_result"].release_call_seconds
        for result in results
    )
    selected_total = sum(
        float(result["cpu_selected_centroids_seconds"]) for result in results
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
        position_rows(
            results[0]["exact_result"].positions,
            query_count=shape.query_count,
            row_width=shape.top_k,
        )
        if results
        else ()
    )
    return {
        "bridge_scope": (
            "resident_selected_posting_exact_rerank_fastplaid_scope"
        ),
        "candidate_k": candidate_k,
        "candidate_score_count_per_window": shape.query_count * candidate_k,
        "candidate_score_count_total": window_count * shape.query_count * candidate_k,
        "candidate_position_agreement_min": min(
            (
                float(result["candidate_result"].candidate_position_agreement)
                for result in results
            ),
            default=0.0,
        ),
        "centroids_per_query_vector": centroids_per_query_vector,
        "cpu_selected_centroids_seconds_per_window": selected_total
        / float(window_count),
        "cpu_selected_centroids_seconds_total": selected_total,
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
        "query_count_per_window": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "release_extension_call_seconds": release_seconds,
        "resident_candidate_cold_seconds_per_window": (
            resident_cold_total / float(window_count)
        ),
        "resident_candidate_seconds_per_window": resident_total / float(window_count),
        "resident_selected_posting_exact_rerank_cold_seconds_per_window": (
            selected_total + resident_cold_total + exact_total
        )
        / float(window_count),
        "resident_selected_posting_exact_rerank_seconds_per_window": (
            selected_total + resident_total + exact_total
        )
        / float(window_count),
        "top_k": shape.top_k,
        "topk_return_count_per_window": shape.query_count * shape.top_k,
        "validation_reference_scores_sent_to_extension": False,
        "vector_dim": shape.vector_dim,
        "warmup_iterations": warmup_iterations,
        "window_count": window_count,
    }


def _shape_payload(shape: SpeedTrackShape, *, candidate_k: int) -> dict[str, int]:
    return shape.to_json_ready() | {
        "candidate_k": candidate_k,
        "candidate_score_count_total": shape.query_count * candidate_k,
    }


def _parsed_payload(row: dict[str, object] | None) -> dict[str, object]:
    parsed = None if row is None else row.get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def _window_to_json_ready(result: dict[str, Any]) -> dict[str, object]:
    return {
        "candidate_generation": result["candidate_result"].to_json_ready(),
        "cpu_selected_centroids_seconds": result[
            "cpu_selected_centroids_seconds"
        ],
        "exact_rerank": result["exact_result"].to_json_ready(),
        "exact_score_delta_max_abs": result["exact_score_delta_max_abs"],
        "final_topk_position_count": result["final_topk_position_count"],
        "final_topk_position_match_count": result[
            "final_topk_position_match_count"
        ],
    }
