from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kayak_bridge.gpu_device_capability import (
    CommandResult,
    GPU_STATUS_AVAILABLE,
    GPU_STATUS_ERROR,
    GPU_STATUS_TOOL_MISSING,
    GPU_STATUS_UNAVAILABLE,
    MojoGpuCapability,
    gpu_query_memory_bytes,
    gpu_query_value,
    host_gpu_inventory,
    probe_mojo_gpu,
    run_command,
)


BENCHMARK_NAME = "gpu_i8_rerank_contract"
GPU_RERANK_STATUS_BACKEND_INTEGRATION_MISSING = (
    "blocked_gpu_backend_integration_missing"
)
GPU_RERANK_STATUS_BATCH_KERNEL_MISSING = GPU_RERANK_STATUS_BACKEND_INTEGRATION_MISSING
GPU_RERANK_STATUS_KERNEL_MISSING = GPU_RERANK_STATUS_BACKEND_INTEGRATION_MISSING
GPU_RERANK_STATUS_RUNTIME_UNAVAILABLE = "blocked_gpu_runtime_unavailable"
GPU_RERANK_STATUS_NO_HARDWARE = "blocked_gpu_hardware_not_detected"
GPU_RERANK_STATUS_COPY_PROBE_FAILED = "blocked_gpu_copy_probe_failed"
GPU_RERANK_STATUS_SINGLE_SCORE_FAILED = "blocked_gpu_single_score_failed"
GPU_RERANK_STATUS_CANDIDATE_SCORE_FAILED = "blocked_gpu_candidate_score_failed"


def shape_contract(shape: Any, *, candidate_k: int) -> dict[str, object]:
    payload = shape.to_json_ready()
    payload["candidate_k"] = candidate_k
    payload["candidate_position_count_total"] = shape.query_count * candidate_k
    return payload


def tensor_contract(shape: Any, *, candidate_k: int) -> dict[str, object]:
    return {
        "query_values": {
            "shape": [
                shape.query_count,
                shape.query_vector_count,
                shape.vector_dim,
            ],
            "dtype": "VectorScalar/Float32",
        },
        "token_codes": {
            "shape": [
                shape.document_count * shape.document_vector_count,
                shape.vector_dim,
            ],
            "dtype": "Int8",
        },
        "token_scales": {
            "shape": [shape.document_count * shape.document_vector_count],
            "dtype": "ScoreScalar/Float32",
        },
        "doc_offsets": {
            "shape": [shape.document_count + 1],
            "dtype": "Int64 in Python bridge, Int in Mojo lists today",
        },
        "candidate_positions": {
            "shape": [shape.query_count, candidate_k],
            "dtype": "Int64 host contract, narrower device type unresolved",
        },
        "candidate_scores": {
            "shape": [shape.query_count, candidate_k],
            "dtype": "ScoreScalar/Float32",
        },
    }


def gpu_timing_contract() -> dict[str, None]:
    return {
        "build_or_prepare_seconds": None,
        "host_to_device_seconds": None,
        "kernel_seconds": None,
        "device_to_host_seconds": None,
        "end_to_end_seconds": None,
    }


def report_status(
    capability: MojoGpuCapability,
    inventory: dict[str, object],
    *,
    copy_probe_status: str | None = None,
    single_score_probe_status: str | None = None,
    candidate_score_probe_status: str | None = None,
) -> str:
    if capability.available:
        if copy_probe_status is not None and copy_probe_status != "ok":
            return GPU_RERANK_STATUS_COPY_PROBE_FAILED
        if (
            single_score_probe_status is not None
            and single_score_probe_status != "ok"
        ):
            return GPU_RERANK_STATUS_SINGLE_SCORE_FAILED
        if (
            candidate_score_probe_status is not None
            and candidate_score_probe_status != "ok"
        ):
            return GPU_RERANK_STATUS_CANDIDATE_SCORE_FAILED
        return GPU_RERANK_STATUS_BACKEND_INTEGRATION_MISSING
    if inventory.get("hardware_present") is True:
        return GPU_RERANK_STATUS_RUNTIME_UNAVAILABLE
    return GPU_RERANK_STATUS_NO_HARDWARE


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_quiet_wrapper_section(report: dict[str, Any]) -> None:
    cpu_reference = report.get("cpu_reference")
    if not isinstance(cpu_reference, dict):
        return
    cpu_i8_reference = cpu_reference.get("cpu_i8_same_candidate_reference")
    if not isinstance(cpu_i8_reference, dict):
        cpu_i8_reference = cpu_reference.get("cpu_i8_reference")
    if not isinstance(cpu_i8_reference, dict):
        return
    mean_seconds = cpu_i8_reference.get("query_batch_mean_seconds")
    if not isinstance(mean_seconds, (float, int)):
        return

    print("== cpu_i8_same_candidate_reference_query_batch ==")
    print("Mean:", mean_seconds)


def exit_code_for_report(
    *,
    capability: MojoGpuCapability,
    allow_missing_gpu: bool,
    allow_missing_kernel: bool,
    copy_probe_status: str | None = None,
    single_score_probe_status: str | None = None,
    candidate_score_probe_status: str | None = None,
) -> int:
    if capability.status == GPU_STATUS_ERROR:
        return 4
    if capability.status == GPU_STATUS_TOOL_MISSING:
        return 5
    if capability.status == GPU_STATUS_UNAVAILABLE and not allow_missing_gpu:
        return 2
    if capability.available and copy_probe_status not in {None, "ok"}:
        return 6
    if capability.available and single_score_probe_status not in {None, "ok"}:
        return 7
    if capability.available and candidate_score_probe_status not in {None, "ok"}:
        return 8
    if not allow_missing_kernel:
        return 3
    return 0
