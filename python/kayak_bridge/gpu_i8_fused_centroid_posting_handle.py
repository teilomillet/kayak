"""Benchmark-only prepared handle for the fused GPU i8 primitive.

This module owns report assembly for an explicit prepare/score/release GPU
payload handle. It does not add public GPU search dispatch.
"""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any, Iterable, Sequence

from kayak_bridge.gpu_device_capability import MojoGpuCapability, probe_mojo_gpu
from kayak_bridge.gpu_i8_address_serve_sweep import (
    STATUS_OK,
    STATUS_PARTIAL_GPU_UNAVAILABLE,
    AddressServeSweepCase,
    AddressServeSweepControls,
)
from kayak_bridge.gpu_i8_candidate_generation_payload import (
    candidate_window_kind,
    time_cpu_candidate_generation,
)
from kayak_bridge.gpu_i8_candidate_posting_traversal import (
    max_float,
    min_float,
    optional_float,
    profile_cpu_candidate_generation,
    ratio,
    selected_posting_summary,
    sum_optional,
)
from kayak_bridge.gpu_i8_centroid_budget_policy import choose_policy_budget
from kayak_bridge.mojo_gpu_i8_rerank import (
    prepare_i8_fused_centroid_posting_session_handle,
    selected_posting_accumulation_reference_scores,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex

from bench_fastplaid_speed_track import SpeedTrackShape, build_synthetic_inputs


STATUS_BLOCKED_GPU_FUSED_HANDLE_FAILED = "blocked_gpu_i8_fused_handle_failed"


def build_report(
    *,
    cases: Sequence[AddressServeSweepCase],
    controls: AddressServeSweepControls,
    centroid_budget_policy: str | None,
) -> tuple[dict[str, Any], MojoGpuCapability]:
    controls.validate()
    capability = probe_mojo_gpu(controls.gpu_query_command)
    rows = [
        run_case(
            case,
            case_index=case_index,
            controls=controls,
            centroid_budget_policy=centroid_budget_policy,
            capability=capability,
        )
        for case_index, case in enumerate(cases)
    ]
    return (
        {
            "schema_version": 1,
            "benchmark": "gpu_i8_fused_centroid_posting_handle",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": report_status(capability=capability, rows=rows),
            "controls": controls.to_json_ready(),
            "centroid_budget_policy": centroid_budget_policy,
            "mojo_gpu_capability": capability.to_json_ready(),
            "cases": rows,
            "summary": summary_payload(rows),
            "measurement_note": (
                "This benchmark prepares a fused GPU i8 centroid-posting "
                "payload once, then measures no-reference top-k score calls "
                "through an explicit handle. It is an internal primitive, not "
                "a public GPU backend."
            ),
        },
        capability,
    )


def run_case(
    case: AddressServeSweepCase,
    *,
    case_index: int,
    controls: AddressServeSweepControls,
    centroid_budget_policy: str | None,
    capability: MojoGpuCapability,
) -> dict[str, Any]:
    shape = case.shape(vector_dim=controls.vector_dim, top_k=controls.top_k)
    centroids_per_query_vector = _centroids_per_query_vector(
        case=case,
        controls=controls,
        centroid_budget_policy=centroid_budget_policy,
    )
    inputs = build_synthetic_inputs(
        shape,
        seed=controls.seed + case_index,
        normalize_vectors=False,
    )
    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=KayakPlaidApproxConfig(
            centroid_count=controls.kayak_plaid_centroid_count,
            centroids_per_query_vector=centroids_per_query_vector,
            candidate_k=case.candidate_k,
            payload="i8",
        ),
        final_k=shape.top_k,
    )
    build_seconds = time.perf_counter() - started_at
    cpu_candidate_timing = time_cpu_candidate_generation(
        index,
        inputs.queries,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    cpu_candidate_profile = profile_cpu_candidate_generation(
        index,
        queries=inputs.queries,
        shape=shape,
        centroids_per_query_vector=centroids_per_query_vector,
        candidate_k=case.candidate_k,
        measurement_iterations=controls.measurement_iterations,
    )
    payload = index.i8_payload_snapshot()
    selected = index.i8_selected_centroids_batch(
        inputs.queries,
        centroids_per_query_vector=centroids_per_query_vector,
    )
    reference_scores = selected_posting_accumulation_reference_scores(
        payload=payload,
        selected=selected,
    )
    gpu_probe = run_gpu_handle_probe(
        capability=capability,
        shape=shape,
        queries=inputs.queries,
        payload=payload,
        reference_scores=reference_scores,
        centroids_per_query_vector=centroids_per_query_vector,
        top_k=shape.top_k,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    gpu_status = _gpu_status(gpu_probe)
    gpu_parsed = _parsed_payload(gpu_probe)
    aggregate = cpu_candidate_profile.aggregate
    cpu_slice = sum_optional(
        _cpu_centroid_scoring_plus_selection(aggregate),
        optional_float(aggregate.get("posting_accumulation_mean_seconds_batch_sum")),
        optional_float(aggregate.get("final_topk_mean_seconds_batch_sum")),
    )
    return {
        "name": case.name,
        "status": STATUS_OK if gpu_status == STATUS_OK else gpu_status,
        "candidate_window_kind": candidate_window_kind(case),
        "shape": case.to_json_ready(
            vector_dim=controls.vector_dim,
            top_k=controls.top_k,
        ),
        "seed": controls.seed + case_index,
        "centroids_per_query_vector": centroids_per_query_vector,
        "cpu_i8_build_seconds": build_seconds,
        "cpu_i8_candidate_generation": cpu_candidate_timing.to_json_ready(),
        "cpu_i8_candidate_generation_profile": (
            cpu_candidate_profile.to_json_ready()
        ),
        "selected_postings": selected_posting_summary(
            payload=payload,
            selected=selected,
        ),
        "gpu_i8_fused_centroid_posting_handle": {
            "status": gpu_status,
            "target_accelerator": capability.target_accelerator,
            "parsed": gpu_parsed,
            "error": gpu_probe.get("error") if isinstance(gpu_probe, dict) else None,
        },
        "comparison": comparison_payload(
            cpu_candidate_generation_mean_seconds=(
                cpu_candidate_timing.mean_seconds
            ),
            cpu_centroid_selection_posting_topk_seconds=cpu_slice,
            gpu_parsed=gpu_parsed,
        ),
    }


def run_gpu_handle_probe(
    *,
    capability: MojoGpuCapability,
    shape: SpeedTrackShape,
    queries: Any,
    payload: Any,
    reference_scores: Any,
    centroids_per_query_vector: int,
    top_k: int,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, object] | None:
    if not capability.available:
        return None
    handle = None
    try:
        handle = prepare_i8_fused_centroid_posting_session_handle(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            payload=payload,
            centroids_per_query_vector=centroids_per_query_vector,
            top_k=top_k,
        )
        for _ in range(warmup_iterations):
            handle.score_topk_without_reference(
                queries=queries,
                reference_document_scores=reference_scores,
            )
        results = [
            handle.score_topk_without_reference(
                queries=queries,
                reference_document_scores=reference_scores,
            )
            for _ in range(measurement_iterations)
        ]
        for _ in range(warmup_iterations):
            handle.score_topk_device_without_reference(
                queries=queries,
                reference_document_scores=reference_scores,
            )
        device_results = [
            handle.score_topk_device_without_reference(
                queries=queries,
                reference_document_scores=reference_scores,
            )
            for _ in range(measurement_iterations)
        ]
        profile_result = handle.profile_topk_without_reference(
            queries=queries,
            reference_document_scores=reference_scores,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
        release_seconds = handle.close()
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        if handle is not None:
            handle.close()
        return {"status": "error", "parsed": {}, "error": str(exc)}
    parsed = {
        "bridge_scope": "fused_centroid_posting_explicit_handle",
        "payload_source": "real_kayak_i8_snapshot",
        "vector_dim": shape.vector_dim,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "query_count": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "centroids_per_query_vector": centroids_per_query_vector,
        "top_k": top_k,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "prepare_host_marshalling_seconds": (
            handle.prepare_host_marshalling_seconds
        ),
        "prepare_extension_call_seconds": handle.prepare_extension_call_seconds,
        "release_extension_call_seconds": release_seconds,
        "score_host_marshalling_mean_seconds": _mean(
            result.host_marshalling_seconds for result in results
        ),
        "score_extension_call_mean_seconds": _mean(
            result.extension_call_seconds for result in results
        ),
        "score_extension_call_min_seconds": min(
            result.extension_call_seconds for result in results
        ),
        "score_extension_call_max_seconds": max(
            result.extension_call_seconds for result in results
        ),
        "topk_position_count": results[-1].topk_position_count,
        "topk_position_agreement_min": min(
            result.topk_position_agreement for result in results
        ),
        "topk_position_mismatch_max": max(
            result.topk_position_count - result.topk_position_match_count
            for result in results
        ),
        "topk_score_delta_max_abs": max(
            result.topk_score_delta_max_abs for result in results
        ),
        "document_score_count": results[-1].document_score_count,
        "device_topk_score_extension_call_mean_seconds": _mean(
            result.extension_call_seconds for result in device_results
        ),
        "device_topk_score_extension_call_min_seconds": min(
            result.extension_call_seconds for result in device_results
        ),
        "device_topk_score_extension_call_max_seconds": max(
            result.extension_call_seconds for result in device_results
        ),
        "device_topk_position_agreement_min": min(
            result.topk_position_agreement for result in device_results
        ),
        "device_topk_position_mismatch_max": max(
            result.topk_position_count - result.topk_position_match_count
            for result in device_results
        ),
        "device_topk_score_delta_max_abs": max(
            result.topk_score_delta_max_abs for result in device_results
        ),
        "profile": profile_result.to_json_ready(),
    }
    status = (
        STATUS_OK
        if parsed["topk_position_mismatch_max"] == 0
        and parsed["device_topk_position_mismatch_max"] == 0
        else "error"
    )
    return {
        "status": status,
        "parsed": parsed,
        "measurements": [result.to_json_ready() for result in results],
    }


def comparison_payload(
    *,
    cpu_candidate_generation_mean_seconds: float,
    cpu_centroid_selection_posting_topk_seconds: float | None,
    gpu_parsed: dict[str, object],
) -> dict[str, float | None]:
    prepare_extension = optional_float(
        gpu_parsed.get("prepare_extension_call_seconds")
    )
    score_extension = optional_float(
        gpu_parsed.get("score_extension_call_mean_seconds")
    )
    device_topk_score_extension = optional_float(
        gpu_parsed.get("device_topk_score_extension_call_mean_seconds")
    )
    score_host_marshalling = optional_float(
        gpu_parsed.get("score_host_marshalling_mean_seconds")
    )
    score_host_plus_extension = sum_optional(
        score_host_marshalling,
        score_extension,
    )
    prepare_plus_score_extension = sum_optional(
        prepare_extension,
        score_extension,
    )
    profile = gpu_parsed.get("profile")
    if not isinstance(profile, dict):
        profile = {}
    profile_kernel_chain = sum_optional(
        optional_float(profile.get("centroid_score_kernel_mean_seconds")),
        optional_float(profile.get("centroid_selection_kernel_mean_seconds")),
        optional_float(profile.get("accumulation_kernel_mean_seconds")),
        optional_float(profile.get("reduction_kernel_mean_seconds")),
    )
    profile_transfer_topk = sum_optional(
        optional_float(profile.get("query_host_to_device_mean_seconds")),
        profile_kernel_chain,
        optional_float(profile.get("device_to_host_mean_seconds")),
        optional_float(
            profile.get("host_topk_destructive_estimated_mean_seconds")
        ),
    )
    return {
        "cpu_i8_candidate_generation_mean_seconds": (
            cpu_candidate_generation_mean_seconds
        ),
        "cpu_i8_centroid_selection_posting_topk_seconds": (
            cpu_centroid_selection_posting_topk_seconds
        ),
        "gpu_fused_handle_prepare_extension_call_seconds": prepare_extension,
        "gpu_fused_handle_score_host_marshalling_mean_seconds": (
            score_host_marshalling
        ),
        "gpu_fused_handle_score_extension_call_mean_seconds": score_extension,
        "gpu_fused_handle_device_topk_score_extension_call_mean_seconds": (
            device_topk_score_extension
        ),
        "gpu_fused_handle_score_host_plus_extension_mean_seconds": (
            score_host_plus_extension
        ),
        "gpu_fused_handle_prepare_plus_score_extension_seconds": (
            prepare_plus_score_extension
        ),
        "gpu_fused_handle_score_extension_seconds_per_cpu_candidate_generation_second": ratio(
            score_extension,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_fused_handle_score_host_plus_extension_seconds_per_cpu_candidate_generation_second": ratio(
            score_host_plus_extension,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_fused_handle_score_extension_seconds_per_cpu_centroid_selection_posting_topk_second": ratio(
            score_extension,
            cpu_centroid_selection_posting_topk_seconds,
        ),
        "gpu_fused_handle_device_topk_score_extension_seconds_per_cpu_candidate_generation_second": ratio(
            device_topk_score_extension,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_fused_handle_device_topk_score_extension_seconds_per_host_topk_score_extension_second": ratio(
            device_topk_score_extension,
            score_extension,
        ),
        "gpu_fused_handle_profile_kernel_chain_mean_seconds": (
            profile_kernel_chain
        ),
        "gpu_fused_handle_profile_query_h2d_kernel_d2h_topk_mean_seconds": (
            profile_transfer_topk
        ),
    }


def summary_payload(rows: Sequence[dict[str, Any]]) -> dict[str, object]:
    ok_rows = _ok_rows(rows)
    non_full_rows = [
        row
        for row in ok_rows
        if row.get("candidate_window_kind") == "non_full"
    ]
    score_vs_candidate = [
        row["comparison"].get(
            "gpu_fused_handle_score_extension_seconds_per_cpu_candidate_generation_second"
        )
        for row in ok_rows
    ]
    non_full_score_vs_candidate = [
        row["comparison"].get(
            "gpu_fused_handle_score_extension_seconds_per_cpu_candidate_generation_second"
        )
        for row in non_full_rows
    ]
    non_full_score_vs_slice = [
        row["comparison"].get(
            "gpu_fused_handle_score_extension_seconds_per_cpu_centroid_selection_posting_topk_second"
        )
        for row in non_full_rows
    ]
    non_full_device_score_vs_candidate = [
        row["comparison"].get(
            "gpu_fused_handle_device_topk_score_extension_seconds_per_cpu_candidate_generation_second"
        )
        for row in non_full_rows
    ]
    non_full_device_score_vs_host_score = [
        row["comparison"].get(
            "gpu_fused_handle_device_topk_score_extension_seconds_per_host_topk_score_extension_second"
        )
        for row in non_full_rows
    ]
    return {
        "case_count": len(rows),
        "ok_case_count": len(ok_rows),
        "ok_non_full_case_count": len(non_full_rows),
        "full_window_case_count": sum(
            1 for row in ok_rows if row.get("candidate_window_kind") == "full"
        ),
        "best_score_extension_vs_cpu_candidate_generation_ratio": min_float(
            score_vs_candidate
        ),
        "worst_score_extension_vs_cpu_candidate_generation_ratio": max_float(
            score_vs_candidate
        ),
        "best_non_full_score_extension_vs_cpu_candidate_generation_ratio": min_float(
            non_full_score_vs_candidate
        ),
        "worst_non_full_score_extension_vs_cpu_candidate_generation_ratio": max_float(
            non_full_score_vs_candidate
        ),
        "best_non_full_score_extension_vs_cpu_centroid_selection_posting_topk_ratio": min_float(
            non_full_score_vs_slice
        ),
        "worst_non_full_score_extension_vs_cpu_centroid_selection_posting_topk_ratio": max_float(
            non_full_score_vs_slice
        ),
        "best_non_full_device_topk_score_extension_vs_cpu_candidate_generation_ratio": min_float(
            non_full_device_score_vs_candidate
        ),
        "worst_non_full_device_topk_score_extension_vs_cpu_candidate_generation_ratio": max_float(
            non_full_device_score_vs_candidate
        ),
        "best_non_full_device_topk_score_extension_vs_host_topk_score_extension_ratio": min_float(
            non_full_device_score_vs_host_score
        ),
        "worst_non_full_device_topk_score_extension_vs_host_topk_score_extension_ratio": max_float(
            non_full_device_score_vs_host_score
        ),
    }


def report_status(
    *,
    capability: MojoGpuCapability,
    rows: Sequence[dict[str, Any]],
) -> str:
    if not capability.available:
        return STATUS_PARTIAL_GPU_UNAVAILABLE
    if all(row.get("status") == STATUS_OK for row in rows):
        return STATUS_OK
    return STATUS_BLOCKED_GPU_FUSED_HANDLE_FAILED


def _cpu_centroid_scoring_plus_selection(
    aggregate: dict[str, object],
) -> float | None:
    return sum_optional(
        optional_float(aggregate.get("centroid_scoring_mean_seconds_batch_sum")),
        optional_float(aggregate.get("centroid_selection_mean_seconds_batch_sum")),
    )


def _centroids_per_query_vector(
    *,
    case: AddressServeSweepCase,
    controls: AddressServeSweepControls,
    centroid_budget_policy: str | None,
) -> int:
    if centroid_budget_policy is None:
        return controls.kayak_plaid_centroids_per_query_vector
    return choose_policy_budget(
        centroid_budget_policy, case
    ).centroids_per_query_vector


def _gpu_status(gpu_probe: dict[str, object] | None) -> object:
    if isinstance(gpu_probe, dict):
        return gpu_probe.get("status")
    return STATUS_PARTIAL_GPU_UNAVAILABLE


def _parsed_payload(gpu_probe: dict[str, object] | None) -> dict[str, object]:
    if isinstance(gpu_probe, dict) and isinstance(gpu_probe.get("parsed"), dict):
        return dict(gpu_probe["parsed"])
    return {}


def _ok_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == STATUS_OK and isinstance(row.get("comparison"), dict)
    ]


def _mean(values: Iterable[float]) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)
