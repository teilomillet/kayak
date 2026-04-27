"""Benchmark-only GPU preparation probe for i8 candidate-generation payloads.

This module owns report assembly for the centroid-posting tensors needed by a
future GPU candidate generator. It does not implement candidate generation.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from kayak_bridge.gpu_i8_centroid_budget_policy import choose_policy_budget
from kayak_bridge.mojo_gpu_i8_rerank import (
    profile_i8_candidate_generation_payload_addresses,
)
from kayak_bridge.plaid_approx import (
    KayakPlaidApproxConfig,
    KayakPlaidApproxIndex,
)

from bench_fastplaid_speed_track import SpeedTrackShape, build_synthetic_inputs
from profile_gpu_i8_real_payload_rerank import ratio


STATUS_BLOCKED_GPU_CANDIDATE_PAYLOAD_FAILED = (
    "blocked_gpu_candidate_generation_payload_failed"
)


@dataclass(frozen=True, slots=True)
class CpuTiming:
    mean_seconds: float
    measurement_iterations: int

    def to_json_ready(self) -> dict[str, object]:
        return {
            "mean_seconds": self.mean_seconds,
            "measurement_iterations": self.measurement_iterations,
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
            "benchmark": "gpu_i8_candidate_generation_payload",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": report_status(capability=capability, rows=rows),
            "controls": controls.to_json_ready(),
            "centroid_budget_policy": centroid_budget_policy,
            "mojo_gpu_capability": capability.to_json_ready(),
            "cases": rows,
            "summary": summary_payload(rows),
            "measurement_note": (
                "This benchmark profiles only the centroid-posting payload "
                "preparation boundary needed by a future GPU candidate "
                "generator. It copies centroid token indices, centroid "
                "document offsets, and centroid document indices to the GPU "
                "and reads them back for agreement checks. It does not "
                "perform posting accumulation or candidate top-k on the GPU."
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
    payload = index.i8_payload_snapshot()
    gpu_probe = run_gpu_payload_probe(
        capability=capability,
        shape=shape,
        payload=payload,
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
        "candidate_generation_payload": {
            "centroid_count": payload.centroid_count,
            "posting_count": payload.posting_count,
            "byte_counts": payload.candidate_generation_byte_counts(),
            "total_payload_bytes": sum(
                payload.candidate_generation_byte_counts().values()
            ),
        },
        "gpu_candidate_generation_payload": {
            "status": gpu_status,
            "target_accelerator": capability.target_accelerator,
            "parsed": gpu_parsed,
            "error": gpu_probe.get("error") if isinstance(gpu_probe, dict) else None,
        },
        "comparison": comparison_payload(
            cpu_candidate_generation_mean_seconds=(
                cpu_candidate_timing.mean_seconds
            ),
            gpu_parsed=gpu_parsed,
        ),
    }


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


def time_cpu_candidate_generation(
    index: KayakPlaidApproxIndex,
    queries: Any,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
) -> CpuTiming:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")
    for _ in range(warmup_iterations):
        index.i8_candidate_positions_batch(queries)
    elapsed = 0.0
    for _ in range(measurement_iterations):
        started_at = time.perf_counter()
        index.i8_candidate_positions_batch(queries)
        elapsed += time.perf_counter() - started_at
    return CpuTiming(
        mean_seconds=elapsed / float(measurement_iterations),
        measurement_iterations=measurement_iterations,
    )


def run_gpu_payload_probe(
    *,
    capability: MojoGpuCapability,
    shape: SpeedTrackShape,
    payload: Any,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, object] | None:
    if not capability.available:
        return None
    try:
        result = profile_i8_candidate_generation_payload_addresses(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            payload=payload,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        return {"status": "error", "parsed": {}, "error": str(exc)}
    parsed = {
        "bridge_scope": "candidate_generation_payload_prepare_addresses",
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
    status = (
        STATUS_OK
        if result.payload_agreement_ok and result.posting_invariants_ok
        else "error"
    )
    return {
        "status": status,
        "parsed": parsed,
        "measurements": [result.to_json_ready()],
    }


def comparison_payload(
    *,
    cpu_candidate_generation_mean_seconds: float,
    gpu_parsed: dict[str, object],
) -> dict[str, float | None]:
    extension_call = optional_float(gpu_parsed.get("extension_call_seconds"))
    mojo_ingest = optional_float(gpu_parsed.get("mojo_host_ingest_mean_seconds"))
    h2d = optional_float(gpu_parsed.get("host_to_device_mean_seconds"))
    d2h = optional_float(gpu_parsed.get("device_to_host_mean_seconds"))
    h2d_d2h = _sum_optional(h2d, d2h)
    return {
        "cpu_i8_candidate_generation_mean_seconds": (
            cpu_candidate_generation_mean_seconds
        ),
        "gpu_payload_extension_call_seconds": extension_call,
        "gpu_payload_mojo_host_ingest_mean_seconds": mojo_ingest,
        "gpu_payload_host_to_device_mean_seconds": h2d,
        "gpu_payload_device_to_host_mean_seconds": d2h,
        "gpu_payload_h2d_plus_d2h_mean_seconds": h2d_d2h,
        "gpu_payload_extension_call_seconds_per_cpu_candidate_generation_second": ratio(
            extension_call,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_payload_h2d_seconds_per_cpu_candidate_generation_second": ratio(
            h2d,
            cpu_candidate_generation_mean_seconds,
        ),
        "gpu_payload_h2d_plus_d2h_seconds_per_cpu_candidate_generation_second": ratio(
            h2d_d2h,
            cpu_candidate_generation_mean_seconds,
        ),
    }


def summary_payload(rows: Sequence[dict[str, Any]]) -> dict[str, object]:
    ok_rows = _ok_rows(rows)
    non_full_rows = [
        row for row in ok_rows if row.get("candidate_window_kind") == "non_full"
    ]
    h2d_ratios = [
        row["comparison"].get(
            "gpu_payload_h2d_seconds_per_cpu_candidate_generation_second"
        )
        for row in ok_rows
    ]
    h2d_d2h_ratios = [
        row["comparison"].get(
            "gpu_payload_h2d_plus_d2h_seconds_per_cpu_candidate_generation_second"
        )
        for row in ok_rows
    ]
    extension_ratios = [
        row["comparison"].get(
            "gpu_payload_extension_call_seconds_per_cpu_candidate_generation_second"
        )
        for row in ok_rows
    ]
    payload_bytes = [
        row["candidate_generation_payload"].get("total_payload_bytes")
        for row in ok_rows
        if isinstance(row.get("candidate_generation_payload"), dict)
    ]
    non_full_h2d_ratios = [
        row["comparison"].get(
            "gpu_payload_h2d_seconds_per_cpu_candidate_generation_second"
        )
        for row in non_full_rows
    ]
    non_full_h2d_d2h_ratios = [
        row["comparison"].get(
            "gpu_payload_h2d_plus_d2h_seconds_per_cpu_candidate_generation_second"
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
        "best_payload_h2d_ratio": min_float(h2d_ratios),
        "worst_payload_h2d_ratio": max_float(h2d_ratios),
        "best_payload_h2d_plus_d2h_ratio": min_float(h2d_d2h_ratios),
        "worst_payload_h2d_plus_d2h_ratio": max_float(h2d_d2h_ratios),
        "best_payload_extension_call_ratio": min_float(extension_ratios),
        "worst_payload_extension_call_ratio": max_float(extension_ratios),
        "best_non_full_payload_h2d_ratio": min_float(non_full_h2d_ratios),
        "worst_non_full_payload_h2d_ratio": max_float(non_full_h2d_ratios),
        "best_non_full_payload_h2d_plus_d2h_ratio": min_float(
            non_full_h2d_d2h_ratios
        ),
        "worst_non_full_payload_h2d_plus_d2h_ratio": max_float(
            non_full_h2d_d2h_ratios
        ),
        "max_payload_bytes": max_int(payload_bytes),
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
    return STATUS_BLOCKED_GPU_CANDIDATE_PAYLOAD_FAILED


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


def candidate_window_kind(case: AddressServeSweepCase) -> str:
    if case.candidate_k >= case.document_count:
        return "full"
    return "non_full"


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


def _sum_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left + right
