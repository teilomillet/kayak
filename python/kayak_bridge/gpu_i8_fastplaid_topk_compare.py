"""Prepared-handle top-k rows for GPU i8/FastPlaid comparison.

This module owns the real-payload GPU top-k comparison boundary. It does not
run FastPlaid full search or write reports; the script-level harness owns that
orchestration.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Sequence

from kayak_bridge.gpu_device_capability import MojoGpuCapability
from kayak_bridge.gpu_i8_address_resident_windows import (
    build_query_windows,
    reshape_windows,
    run_cross_call_prepared_handle_topk_no_reference_probe,
    run_cross_call_prepared_handle_topk_probe,
    timing_payload_per_window,
)
from kayak_bridge.gpu_i8_address_serve_sweep import (
    AddressServeSweepCase,
    AddressServeSweepControls,
)
from kayak_bridge.gpu_i8_fastplaid_topk_metrics import (
    STATUS_PARTIAL_GPU_UNAVAILABLE,
    prepared_handle_topk_derived_metrics,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex

from bench_fastplaid_speed_track import (  # noqa: E402
    SpeedTrackShape,
    SyntheticInputs,
    mean_recall_at_k,
)
from profile_gpu_i8_real_payload_rerank import (  # noqa: E402
    rank_candidate_positions_by_score,
    time_candidate_generation,
    time_same_candidate_scores,
)


def build_prepared_handle_topk_scope_row(
    *,
    shape: SpeedTrackShape,
    inputs: SyntheticInputs,
    reference_positions: Sequence[Sequence[int]],
    capability: MojoGpuCapability,
    args: argparse.Namespace,
) -> dict[str, Any]:
    controls = _address_controls(shape=shape, args=args)
    case = AddressServeSweepCase(
        name="fastplaid_compare",
        document_count=shape.document_count,
        document_vector_count=shape.document_vector_count,
        query_count=shape.query_count,
        query_vector_count=shape.query_vector_count,
        candidate_k=args.candidate_k,
    )
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

    base_candidate_positions = index.i8_candidate_positions_batch(inputs.queries)
    base_reference_scores = index.i8_score_candidate_positions_batch(
        inputs.queries,
        base_candidate_positions,
    )
    ranked_positions = rank_candidate_positions_by_score(
        base_candidate_positions,
        base_reference_scores,
        final_k=shape.top_k,
    )
    query_windows = build_query_windows(
        base_queries=inputs.queries,
        shape=shape,
        window_count=controls.resident_session_iterations,
        seed=args.seed + 200_000,
    )
    multi_queries = query_windows.reshape(
        controls.resident_session_iterations * shape.query_count,
        shape.query_vector_count,
        shape.vector_dim,
    )
    multi_candidate_positions = index.i8_candidate_positions_batch(multi_queries)
    multi_reference_scores = index.i8_score_candidate_positions_batch(
        multi_queries,
        multi_candidate_positions,
    )
    candidate_timing = time_candidate_generation(
        index,
        multi_queries,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    score_timing = time_same_candidate_scores(
        index,
        multi_queries,
        multi_candidate_positions,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    candidate_position_windows = reshape_windows(
        multi_candidate_positions,
        window_count=controls.resident_session_iterations,
        query_count=shape.query_count,
    )
    reference_score_windows = reshape_windows(
        multi_reference_scores,
        window_count=controls.resident_session_iterations,
        query_count=shape.query_count,
    )
    probe = run_cross_call_prepared_handle_topk_probe(
        case=case,
        shape=shape,
        controls=controls,
        capability=capability,
        query_windows=query_windows,
        payload=index.i8_payload_snapshot(),
        candidate_positions=candidate_position_windows,
        reference_scores=reference_score_windows,
    )
    no_reference_probe = run_cross_call_prepared_handle_topk_no_reference_probe(
        case=case,
        shape=shape,
        controls=controls,
        capability=capability,
        query_windows=query_windows,
        payload=index.i8_payload_snapshot(),
        candidate_positions=candidate_position_windows,
        reference_scores=reference_score_windows,
    )
    parsed = _parsed_payload(probe)
    no_reference_parsed = _parsed_payload(no_reference_probe)
    candidate_per_window = (
        candidate_timing.mean_seconds / float(controls.resident_session_iterations)
    )
    score_per_window = (
        score_timing.mean_seconds / float(controls.resident_session_iterations)
    )
    return {
        "system_name": "kayak_gpu_i8_prepared_handle_topk_probe",
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
        "scope": "real_kayak_i8_candidate_window_prepared_handle_topk",
        "shape": _shape_payload(shape, candidate_k=args.candidate_k),
        "cpu_i8_build_seconds": build_seconds,
        "cpu_i8_multi_window_candidate_generation": timing_payload_per_window(
            candidate_timing,
            window_count=controls.resident_session_iterations,
        ),
        "cpu_i8_multi_window_same_candidate_reference": (
            timing_payload_per_window(
                score_timing,
                window_count=controls.resident_session_iterations,
            )
            | {
                "candidate_score_count_per_window": (
                    shape.query_count * args.candidate_k
                ),
                "candidate_score_count_total": (
                    controls.resident_session_iterations
                    * shape.query_count
                    * args.candidate_k
                ),
            }
        ),
        "recall_at_k_vs_kayak_exact": mean_recall_at_k(
            candidate_positions_by_query=ranked_positions,
            reference_positions_by_query=reference_positions,
            k=shape.top_k,
        ),
        "target_accelerator": capability.target_accelerator,
        "parsed": parsed,
        "derived": prepared_handle_topk_derived_metrics(
            parsed=parsed,
            candidate_generation_per_window=candidate_per_window,
            cpu_score_per_window=score_per_window,
        ),
        "no_reference_status": (
            no_reference_probe.get("status")
            if isinstance(no_reference_probe, dict)
            else STATUS_PARTIAL_GPU_UNAVAILABLE
        ),
        "no_reference_parsed": no_reference_parsed,
        "no_reference_derived": prepared_handle_topk_derived_metrics(
            parsed=no_reference_parsed,
            candidate_generation_per_window=candidate_per_window,
            cpu_score_per_window=score_per_window,
        ),
        "error": probe.get("error") if isinstance(probe, dict) else None,
        "no_reference_error": (
            no_reference_probe.get("error")
            if isinstance(no_reference_probe, dict)
            else None
        ),
        "measurements": probe.get("measurements") if isinstance(probe, dict) else None,
        "no_reference_measurements": (
            no_reference_probe.get("measurements")
            if isinstance(no_reference_probe, dict)
            else None
        ),
    }


def build_missing_prepared_handle_topk_scope_row(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    capability: MojoGpuCapability,
) -> dict[str, Any]:
    return {
        "system_name": "kayak_gpu_i8_prepared_handle_topk_probe",
        "engine": "kayak_gpu_probe",
        "status": STATUS_PARTIAL_GPU_UNAVAILABLE,
        "backend": (
            f"mojo:{capability.target_accelerator}"
            if capability.target_accelerator is not None
            else None
        ),
        "index_kind": "sampled_centroid_postings_i8_proxy",
        "scope": "real_kayak_i8_candidate_window_prepared_handle_topk",
        "shape": _shape_payload(shape, candidate_k=candidate_k),
        "parsed": {},
        "derived": prepared_handle_topk_derived_metrics(
            parsed={},
            candidate_generation_per_window=None,
            cpu_score_per_window=None,
        ),
        "no_reference_status": STATUS_PARTIAL_GPU_UNAVAILABLE,
        "no_reference_parsed": {},
        "no_reference_derived": prepared_handle_topk_derived_metrics(
            parsed={},
            candidate_generation_per_window=None,
            cpu_score_per_window=None,
        ),
    }


def _address_controls(
    *,
    shape: SpeedTrackShape,
    args: argparse.Namespace,
) -> AddressServeSweepControls:
    controls = AddressServeSweepControls(
        vector_dim=shape.vector_dim,
        top_k=shape.top_k,
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        resident_session_iterations=args.gpu_topk_session_iterations,
        kayak_plaid_centroid_count=args.kayak_plaid_centroid_count,
        kayak_plaid_centroids_per_query_vector=(
            args.kayak_plaid_centroids_per_query_vector
        ),
        gpu_query_command=args.gpu_query_command,
    )
    controls.validate()
    return controls


def _shape_payload(shape: SpeedTrackShape, *, candidate_k: int) -> dict[str, int]:
    return shape.to_json_ready() | {
        "candidate_k": candidate_k,
        "candidate_score_count_total": shape.query_count * candidate_k,
    }


def _parsed_payload(row: dict[str, object] | None) -> dict[str, object]:
    parsed = None if row is None else row.get("parsed")
    return parsed if isinstance(parsed, dict) else {}
