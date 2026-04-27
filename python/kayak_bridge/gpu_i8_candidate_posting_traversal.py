"""Benchmark-only GPU probe for i8 selected-centroid posting traversal.

This module owns report assembly for expanding selected centroid posting lists
on the GPU. It does not accumulate per-document scores or produce candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import time
from typing import Any, Sequence

import numpy as np

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
from kayak_bridge.gpu_i8_centroid_budget_policy import choose_policy_budget
from kayak_bridge.mojo_exact_cpu import load_module
from kayak_bridge.mojo_gpu_i8_rerank import (
    profile_i8_selected_posting_traversal_addresses,
)
from kayak_bridge.plaid_approx import (
    KayakPlaidApproxConfig,
    KayakPlaidApproxIndex,
    KayakPlaidI8PayloadSnapshot,
    KayakPlaidI8SelectedCentroids,
)

from bench_fastplaid_speed_track import SpeedTrackShape, build_synthetic_inputs


STATUS_BLOCKED_GPU_POSTING_TRAVERSAL_FAILED = (
    "blocked_gpu_candidate_posting_traversal_failed"
)


@dataclass(frozen=True, slots=True)
class CpuCandidateGenerationProfile:
    profiles_by_query: tuple[dict[str, Any], ...]
    aggregate: dict[str, Any]

    def to_json_ready(self) -> dict[str, object]:
        return {
            "query_profiles": self.profiles_by_query,
            "aggregate": self.aggregate,
        }


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
            "benchmark": "gpu_i8_candidate_posting_traversal",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": report_status(capability=capability, rows=rows),
            "controls": controls.to_json_ready(),
            "centroid_budget_policy": centroid_budget_policy,
            "mojo_gpu_capability": capability.to_json_ready(),
            "cases": rows,
            "summary": summary_payload(rows),
            "measurement_note": (
                "This benchmark profiles only selected-centroid posting "
                "traversal on the GPU. It expands selected centroid posting "
                "lists into doc-index visits and score tags, then compares the "
                "expanded stream with a deterministic CPU reference. It does "
                "not perform per-document max/reduce accumulation or candidate "
                "top-k on the GPU."
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
    selected_summary = selected_posting_summary(
        payload=payload,
        selected=selected,
    )
    gpu_probe = run_gpu_posting_traversal_probe(
        capability=capability,
        shape=shape,
        payload=payload,
        selected=selected,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    gpu_status = _gpu_status(gpu_probe)
    gpu_parsed = _parsed_payload(gpu_probe)
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
        "selected_postings": selected_summary,
        "gpu_candidate_posting_traversal": {
            "status": gpu_status,
            "target_accelerator": capability.target_accelerator,
            "parsed": gpu_parsed,
            "error": gpu_probe.get("error") if isinstance(gpu_probe, dict) else None,
        },
        "comparison": comparison_payload(
            cpu_candidate_generation_mean_seconds=(
                cpu_candidate_timing.mean_seconds
            ),
            cpu_posting_accumulation_seconds=optional_float(
                cpu_candidate_profile.aggregate.get(
                    "posting_accumulation_mean_seconds_batch_sum"
                )
            ),
            gpu_parsed=gpu_parsed,
        ),
    }


def profile_cpu_candidate_generation(
    index: KayakPlaidApproxIndex,
    *,
    queries: np.ndarray,
    shape: SpeedTrackShape,
    centroids_per_query_vector: int,
    candidate_k: int,
    measurement_iterations: int,
) -> CpuCandidateGenerationProfile:
    module = load_module()
    raw_profiles = (
        module.plaid_i8_candidate_generation_profile_prepared_batch_address(
            [
                int(queries.ctypes.data),
                int(shape.query_count),
                int(shape.query_vector_count),
                int(centroids_per_query_vector),
                int(candidate_k),
                int(measurement_iterations),
                index._prepared_index,
            ]
        )
    )
    profiles = tuple(_profile_pairs_to_dict(row) for row in raw_profiles)
    return CpuCandidateGenerationProfile(
        profiles_by_query=profiles,
        aggregate=_aggregate_cpu_profiles(profiles),
    )


def selected_posting_summary(
    *,
    payload: KayakPlaidI8PayloadSnapshot,
    selected: KayakPlaidI8SelectedCentroids,
) -> dict[str, int | float]:
    payload.validate()
    selected.validate()
    positions = np.asarray(selected.positions_array(), dtype=np.int64).reshape(-1)
    offsets = np.asarray(payload.centroid_doc_offsets, dtype=np.int64).reshape(-1)
    visit_count = 0
    for centroid_position in positions:
        centroid = int(centroid_position)
        if centroid < 0 or centroid >= payload.centroid_count:
            raise ValueError("selected centroid position out of range")
        visit_count += int(offsets[centroid + 1] - offsets[centroid])
    selected_count = int(positions.size)
    return {
        "centroid_count": int(payload.centroid_count),
        "posting_count": int(payload.posting_count),
        "selected_centroid_count": selected_count,
        "selected_centroid_count_per_query": (
            selected.selected_centroid_count_per_query
        ),
        "selected_centroid_count_total": selected.selected_centroid_count_total,
        "expanded_posting_count": visit_count,
        "expanded_postings_per_selected_centroid": ratio(
            float(visit_count),
            float(selected_count),
        ),
    }


def run_gpu_posting_traversal_probe(
    *,
    capability: MojoGpuCapability,
    shape: SpeedTrackShape,
    payload: KayakPlaidI8PayloadSnapshot,
    selected: KayakPlaidI8SelectedCentroids,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, object] | None:
    if not capability.available:
        return None
    try:
        result = profile_i8_selected_posting_traversal_addresses(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            payload=payload,
            selected=selected,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        return {"status": "error", "parsed": {}, "error": str(exc)}
    parsed = {
        "bridge_scope": "selected_centroid_posting_traversal_addresses",
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
    status = STATUS_OK if result.traversal_agreement_ok else "error"
    return {
        "status": status,
        "parsed": parsed,
        "measurements": [result.to_json_ready()],
    }


def comparison_payload(
    *,
    cpu_candidate_generation_mean_seconds: float,
    cpu_posting_accumulation_seconds: float | None,
    gpu_parsed: dict[str, object],
) -> dict[str, float | None]:
    extension_call = optional_float(gpu_parsed.get("extension_call_seconds"))
    mojo_ingest = optional_float(gpu_parsed.get("mojo_host_ingest_mean_seconds"))
    payload_h2d = optional_float(
        gpu_parsed.get("payload_host_to_device_mean_seconds")
    )
    selected_h2d = optional_float(
        gpu_parsed.get("selected_host_to_device_mean_seconds")
    )
    kernel = optional_float(gpu_parsed.get("kernel_mean_seconds"))
    d2h = optional_float(gpu_parsed.get("device_to_host_mean_seconds"))
    selected_h2d_kernel_d2h = sum_optional(selected_h2d, kernel, d2h)
    all_gpu_measured = sum_optional(payload_h2d, selected_h2d, kernel, d2h)
    return {
        "cpu_i8_candidate_generation_mean_seconds": (
            cpu_candidate_generation_mean_seconds
        ),
        "cpu_i8_posting_accumulation_seconds": cpu_posting_accumulation_seconds,
        "gpu_posting_traversal_extension_call_seconds": extension_call,
        "gpu_posting_traversal_mojo_host_ingest_mean_seconds": mojo_ingest,
        "gpu_posting_payload_h2d_mean_seconds": payload_h2d,
        "gpu_posting_selected_h2d_mean_seconds": selected_h2d,
        "gpu_posting_kernel_mean_seconds": kernel,
        "gpu_posting_device_to_host_mean_seconds": d2h,
        "gpu_posting_selected_h2d_kernel_d2h_mean_seconds": (
            selected_h2d_kernel_d2h
        ),
        "gpu_posting_all_measured_mean_seconds": all_gpu_measured,
        "gpu_posting_kernel_seconds_per_cpu_candidate_generation_second": ratio(
            kernel,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_posting_selected_h2d_kernel_d2h_seconds_per_cpu_candidate_generation_second": ratio(
            selected_h2d_kernel_d2h,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_posting_all_measured_seconds_per_cpu_candidate_generation_second": ratio(
            all_gpu_measured,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_posting_kernel_seconds_per_cpu_posting_accumulation_second": ratio(
            kernel,
            cpu_posting_accumulation_seconds,
        ),
        "gpu_posting_selected_h2d_kernel_d2h_seconds_per_cpu_posting_accumulation_second": ratio(
            selected_h2d_kernel_d2h,
            cpu_posting_accumulation_seconds,
        ),
        "gpu_posting_all_measured_seconds_per_cpu_posting_accumulation_second": ratio(
            all_gpu_measured,
            cpu_posting_accumulation_seconds,
        ),
    }


def summary_payload(rows: Sequence[dict[str, Any]]) -> dict[str, object]:
    ok_rows = _ok_rows(rows)
    non_full_rows = [
        row for row in ok_rows if row.get("candidate_window_kind") == "non_full"
    ]
    kernel_vs_candidate = [
        row["comparison"].get(
            "gpu_posting_kernel_seconds_per_cpu_candidate_generation_second"
        )
        for row in ok_rows
    ]
    selected_path_vs_candidate = [
        row["comparison"].get(
            "gpu_posting_selected_h2d_kernel_d2h_seconds_per_cpu_candidate_generation_second"
        )
        for row in ok_rows
    ]
    all_measured_vs_candidate = [
        row["comparison"].get(
            "gpu_posting_all_measured_seconds_per_cpu_candidate_generation_second"
        )
        for row in ok_rows
    ]
    selected_path_vs_posting = [
        row["comparison"].get(
            "gpu_posting_selected_h2d_kernel_d2h_seconds_per_cpu_posting_accumulation_second"
        )
        for row in ok_rows
    ]
    non_full_all_measured_vs_candidate = [
        row["comparison"].get(
            "gpu_posting_all_measured_seconds_per_cpu_candidate_generation_second"
        )
        for row in non_full_rows
    ]
    expanded_posting_counts = [
        row["selected_postings"].get("expanded_posting_count")
        for row in ok_rows
        if isinstance(row.get("selected_postings"), dict)
    ]
    return {
        "case_count": len(rows),
        "ok_case_count": len(ok_rows),
        "ok_non_full_case_count": len(non_full_rows),
        "full_window_case_count": sum(
            1 for row in ok_rows if row.get("candidate_window_kind") == "full"
        ),
        "best_kernel_vs_candidate_generation_ratio": min_float(
            kernel_vs_candidate
        ),
        "worst_kernel_vs_candidate_generation_ratio": max_float(
            kernel_vs_candidate
        ),
        "best_selected_path_vs_candidate_generation_ratio": min_float(
            selected_path_vs_candidate
        ),
        "worst_selected_path_vs_candidate_generation_ratio": max_float(
            selected_path_vs_candidate
        ),
        "best_all_measured_vs_candidate_generation_ratio": min_float(
            all_measured_vs_candidate
        ),
        "worst_all_measured_vs_candidate_generation_ratio": max_float(
            all_measured_vs_candidate
        ),
        "best_selected_path_vs_posting_accumulation_ratio": min_float(
            selected_path_vs_posting
        ),
        "worst_selected_path_vs_posting_accumulation_ratio": max_float(
            selected_path_vs_posting
        ),
        "best_non_full_all_measured_vs_candidate_generation_ratio": min_float(
            non_full_all_measured_vs_candidate
        ),
        "worst_non_full_all_measured_vs_candidate_generation_ratio": max_float(
            non_full_all_measured_vs_candidate
        ),
        "max_expanded_posting_count": max_int(expanded_posting_counts),
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
    return STATUS_BLOCKED_GPU_POSTING_TRAVERSAL_FAILED


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


def _profile_pairs_to_dict(row: Sequence[Sequence[Any]]) -> dict[str, Any]:
    return {str(key): value for key, value in row}


def _aggregate_cpu_profiles(profiles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not profiles:
        return {}
    float_fields = (
        "full_candidate_mean_seconds",
        "posting_accumulation_mean_seconds",
        "final_topk_mean_seconds",
    )
    int_fields = (
        "selected_centroid_count",
        "posting_visit_count",
        "touched_document_count",
        "output_candidate_count",
    )
    aggregate: dict[str, Any] = {
        f"{field}_batch_sum": sum(float(profile[field]) for profile in profiles)
        for field in float_fields
    }
    for field in int_fields:
        aggregate[f"{field}_total"] = sum(int(profile[field]) for profile in profiles)
    first = profiles[0]
    aggregate["query_count"] = len(profiles)
    aggregate["query_vector_count"] = int(first["query_vector_count"])
    aggregate["document_count"] = int(first["document_count"])
    aggregate["document_vector_count"] = int(first["document_vector_count"])
    aggregate["total_document_vector_count"] = int(
        first["total_document_vector_count"]
    )
    aggregate["centroid_count"] = int(first["centroid_count"])
    aggregate["centroids_per_query_vector"] = int(
        first["centroids_per_query_vector"]
    )
    aggregate["candidate_k"] = int(first["candidate_k"])
    return aggregate


def _gpu_status(gpu_probe: dict[str, object] | None) -> object:
    if isinstance(gpu_probe, dict):
        return gpu_probe.get("status")
    return STATUS_PARTIAL_GPU_UNAVAILABLE


def _parsed_payload(gpu_probe: dict[str, object] | None) -> dict[str, object]:
    if isinstance(gpu_probe, dict) and isinstance(gpu_probe.get("parsed"), dict):
        return dict(gpu_probe["parsed"])
    return {}


def optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def sum_optional(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values)


def _ok_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == STATUS_OK and isinstance(row.get("comparison"), dict)
    ]


def min_float(values: Sequence[object]) -> float | None:
    floats = [float(value) for value in values if isinstance(value, (float, int))]
    return min(floats) if floats else None


def max_float(values: Sequence[object]) -> float | None:
    floats = [float(value) for value in values if isinstance(value, (float, int))]
    return max(floats) if floats else None


def max_int(values: Sequence[object]) -> int | None:
    ints = [int(value) for value in values if isinstance(value, int)]
    return max(ints) if ints else None
