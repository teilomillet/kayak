"""Benchmark-only fused GPU probe for i8 candidate generation substeps.

This module scores sampled centroids on GPU, selects centroids on device, and
accumulates selected posting scores into per-document scores. It does not own a
public search API or a production backend.
"""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any, Sequence

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
    profile_i8_fused_centroid_posting_accumulation_addresses,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex

from bench_fastplaid_speed_track import SpeedTrackShape, build_synthetic_inputs


STATUS_BLOCKED_GPU_FUSED_CENTROID_POSTING_ACCUMULATION_FAILED = (
    "blocked_gpu_i8_fused_centroid_posting_accumulation_failed"
)


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
            "benchmark": "gpu_i8_fused_centroid_posting_accumulation",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": report_status(capability=capability, rows=rows),
            "controls": controls.to_json_ready(),
            "centroid_budget_policy": centroid_budget_policy,
            "mojo_gpu_capability": capability.to_json_ready(),
            "cases": rows,
            "summary": summary_payload(rows),
            "measurement_note": (
                "This benchmark profiles a fused internal primitive: GPU "
                "centroid scoring, device centroid selection, GPU posting "
                "accumulation, score readback, and host document top-k. It is "
                "not a public GPU search backend."
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
    gpu_probe = run_gpu_fused_probe(
        capability=capability,
        shape=shape,
        queries=inputs.queries,
        payload=payload,
        selected=selected,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    gpu_status = _gpu_status(gpu_probe)
    gpu_parsed = _parsed_payload(gpu_probe)
    aggregate = cpu_candidate_profile.aggregate
    cpu_centroid_scoring_plus_selection = _cpu_centroid_scoring_plus_selection(
        aggregate
    )
    cpu_posting_accumulation = optional_float(
        aggregate.get("posting_accumulation_mean_seconds_batch_sum")
    )
    cpu_final_topk = optional_float(
        aggregate.get("final_topk_mean_seconds_batch_sum")
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
        "gpu_i8_fused_centroid_posting_accumulation": {
            "status": gpu_status,
            "target_accelerator": capability.target_accelerator,
            "parsed": gpu_parsed,
            "error": gpu_probe.get("error") if isinstance(gpu_probe, dict) else None,
        },
        "comparison": comparison_payload(
            cpu_candidate_generation_mean_seconds=(
                cpu_candidate_timing.mean_seconds
            ),
            cpu_centroid_scoring_plus_selection_seconds=(
                cpu_centroid_scoring_plus_selection
            ),
            cpu_posting_accumulation_seconds=cpu_posting_accumulation,
            cpu_final_topk_seconds=cpu_final_topk,
            gpu_parsed=gpu_parsed,
        ),
    }


def run_gpu_fused_probe(
    *,
    capability: MojoGpuCapability,
    shape: SpeedTrackShape,
    queries: Any,
    payload: Any,
    selected: Any,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, object] | None:
    if not capability.available:
        return None
    try:
        result = profile_i8_fused_centroid_posting_accumulation_addresses(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            queries=queries,
            payload=payload,
            selected=selected,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        return {"status": "error", "parsed": {}, "error": str(exc)}
    parsed = {
        "bridge_scope": "fused_centroid_posting_accumulation_addresses",
        "payload_source": "real_kayak_i8_snapshot",
        "vector_dim": shape.vector_dim,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "query_count": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
    } | result.to_json_ready()
    status = STATUS_OK if result.fused_agreement_ok else "error"
    return {
        "status": status,
        "parsed": parsed,
        "measurements": [result.to_json_ready()],
    }


def comparison_payload(
    *,
    cpu_candidate_generation_mean_seconds: float,
    cpu_centroid_scoring_plus_selection_seconds: float | None,
    cpu_posting_accumulation_seconds: float | None,
    cpu_final_topk_seconds: float | None,
    gpu_parsed: dict[str, object],
) -> dict[str, float | None]:
    extension_call = optional_float(gpu_parsed.get("extension_call_seconds"))
    mojo_ingest = optional_float(gpu_parsed.get("mojo_host_ingest_mean_seconds"))
    payload_h2d = optional_float(
        gpu_parsed.get("payload_host_to_device_mean_seconds")
    )
    query_h2d = optional_float(
        gpu_parsed.get("query_host_to_device_mean_seconds")
    )
    centroid_score_kernel = optional_float(
        gpu_parsed.get("centroid_score_kernel_mean_seconds")
    )
    centroid_selection_kernel = optional_float(
        gpu_parsed.get("centroid_selection_kernel_mean_seconds")
    )
    accumulation_kernel = optional_float(
        gpu_parsed.get("accumulation_kernel_mean_seconds")
    )
    d2h = optional_float(gpu_parsed.get("device_to_host_mean_seconds"))
    host_topk = optional_float(gpu_parsed.get("host_topk_mean_seconds"))
    selected_validation_d2h = optional_float(
        gpu_parsed.get("selected_validation_device_to_host_mean_seconds")
    )
    cpu_scoring_selection_posting_topk = sum_optional(
        cpu_centroid_scoring_plus_selection_seconds,
        cpu_posting_accumulation_seconds,
        cpu_final_topk_seconds,
    )
    kernel_chain = sum_optional(
        centroid_score_kernel,
        centroid_selection_kernel,
        accumulation_kernel,
    )
    resident_without_host_topk = sum_optional(
        query_h2d,
        kernel_chain,
        d2h,
    )
    resident = sum_optional(resident_without_host_topk, host_topk)
    cold = sum_optional(payload_h2d, resident)
    return {
        "cpu_i8_candidate_generation_mean_seconds": (
            cpu_candidate_generation_mean_seconds
        ),
        "cpu_i8_centroid_scoring_plus_selection_seconds": (
            cpu_centroid_scoring_plus_selection_seconds
        ),
        "cpu_i8_posting_accumulation_seconds": cpu_posting_accumulation_seconds,
        "cpu_i8_final_topk_seconds": cpu_final_topk_seconds,
        "cpu_i8_centroid_selection_posting_topk_seconds": (
            cpu_scoring_selection_posting_topk
        ),
        "gpu_fused_extension_call_seconds": extension_call,
        "gpu_fused_mojo_host_ingest_mean_seconds": mojo_ingest,
        "gpu_fused_payload_h2d_mean_seconds": payload_h2d,
        "gpu_fused_query_h2d_mean_seconds": query_h2d,
        "gpu_fused_centroid_score_kernel_mean_seconds": (
            centroid_score_kernel
        ),
        "gpu_fused_centroid_selection_kernel_mean_seconds": (
            centroid_selection_kernel
        ),
        "gpu_fused_accumulation_kernel_mean_seconds": accumulation_kernel,
        "gpu_fused_kernel_chain_mean_seconds": kernel_chain,
        "gpu_fused_device_to_host_mean_seconds": d2h,
        "gpu_fused_host_topk_mean_seconds": host_topk,
        "gpu_fused_selected_validation_device_to_host_mean_seconds": (
            selected_validation_d2h
        ),
        "gpu_fused_resident_without_host_topk_mean_seconds": (
            resident_without_host_topk
        ),
        "gpu_fused_resident_payload_mean_seconds": resident,
        "gpu_fused_cold_payload_mean_seconds": cold,
        "gpu_fused_resident_payload_seconds_per_cpu_candidate_generation_second": ratio(
            resident,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_fused_cold_payload_seconds_per_cpu_candidate_generation_second": ratio(
            cold,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_fused_resident_payload_seconds_per_cpu_centroid_selection_posting_topk_second": ratio(
            resident,
            cpu_scoring_selection_posting_topk,
        ),
        "gpu_fused_kernel_chain_seconds_per_cpu_centroid_selection_posting_second": ratio(
            kernel_chain,
            sum_optional(
                cpu_centroid_scoring_plus_selection_seconds,
                cpu_posting_accumulation_seconds,
            ),
        ),
    }


def summary_payload(rows: Sequence[dict[str, Any]]) -> dict[str, object]:
    ok_rows = _ok_rows(rows)
    non_full_rows = [
        row
        for row in ok_rows
        if row.get("candidate_window_kind") == "non_full"
    ]
    resident_vs_candidate = [
        row["comparison"].get(
            "gpu_fused_resident_payload_seconds_per_cpu_candidate_generation_second"
        )
        for row in ok_rows
    ]
    non_full_resident_vs_candidate = [
        row["comparison"].get(
            "gpu_fused_resident_payload_seconds_per_cpu_candidate_generation_second"
        )
        for row in non_full_rows
    ]
    non_full_cold_vs_candidate = [
        row["comparison"].get(
            "gpu_fused_cold_payload_seconds_per_cpu_candidate_generation_second"
        )
        for row in non_full_rows
    ]
    non_full_resident_vs_cpu_slice = [
        row["comparison"].get(
            "gpu_fused_resident_payload_seconds_per_cpu_centroid_selection_posting_topk_second"
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
        "best_resident_payload_vs_cpu_candidate_generation_ratio": min_float(
            resident_vs_candidate
        ),
        "worst_resident_payload_vs_cpu_candidate_generation_ratio": max_float(
            resident_vs_candidate
        ),
        "best_non_full_resident_payload_vs_cpu_candidate_generation_ratio": min_float(
            non_full_resident_vs_candidate
        ),
        "worst_non_full_resident_payload_vs_cpu_candidate_generation_ratio": max_float(
            non_full_resident_vs_candidate
        ),
        "best_non_full_cold_payload_vs_cpu_candidate_generation_ratio": min_float(
            non_full_cold_vs_candidate
        ),
        "worst_non_full_cold_payload_vs_cpu_candidate_generation_ratio": max_float(
            non_full_cold_vs_candidate
        ),
        "best_non_full_resident_payload_vs_cpu_centroid_selection_posting_topk_ratio": min_float(
            non_full_resident_vs_cpu_slice
        ),
        "worst_non_full_resident_payload_vs_cpu_centroid_selection_posting_topk_ratio": max_float(
            non_full_resident_vs_cpu_slice
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
    return STATUS_BLOCKED_GPU_FUSED_CENTROID_POSTING_ACCUMULATION_FAILED


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
