from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from kayak_bridge.cache_paths import REPO_ROOT
from kayak_bridge.gpu_device_capability import run_command
from kayak_bridge.gpu_probe_output import parse_key_value_probe_output


GPU_COPY_PROBE_STATUS_OK = "ok"
GPU_COPY_PROBE_STATUS_TARGET_MISSING = "skipped_target_accelerator_missing"
GPU_COPY_PROBE_STATUS_ERROR = "error"
GPU_COPY_PROBE_STATUS_ROUNDTRIP_FAILED = "roundtrip_failed"

COPY_PROBE_PATH = REPO_ROOT / "benchmarks" / "gpu_i8_copy_roundtrip_probe.mojo"


def gpu_copy_probe_counts(shape: Any, *, candidate_k: int) -> dict[str, int]:
    total_doc_vectors = shape.document_count * shape.document_vector_count
    query_values = shape.query_count * shape.query_vector_count * shape.vector_dim
    token_scales = total_doc_vectors
    candidate_scores = shape.query_count * candidate_k
    return {
        "float32_count": query_values + token_scales + candidate_scores,
        "int8_count": total_doc_vectors * shape.vector_dim,
        "int64_count": (shape.document_count + 1) + (shape.query_count * candidate_k),
    }


def run_gpu_copy_roundtrip_probe(
    shape: Any,
    *,
    candidate_k: int,
    target_accelerator: str | None,
    mojo_command: Sequence[str] = ("mojo",),
) -> dict[str, object]:
    counts = gpu_copy_probe_counts(shape, candidate_k=candidate_k)
    if target_accelerator is None:
        return {
            "status": GPU_COPY_PROBE_STATUS_TARGET_MISSING,
            "counts": counts,
            "probe": None,
        }

    command = (
        *mojo_command,
        "run",
        "--target-accelerator",
        target_accelerator,
        "-D",
        f"float32_count={counts['float32_count']}",
        "-D",
        f"int8_count={counts['int8_count']}",
        "-D",
        f"int64_count={counts['int64_count']}",
        str(COPY_PROBE_PATH),
    )
    result = run_command(command, timeout_seconds=45.0)
    parsed = parse_gpu_copy_probe_output(result.stdout)
    if result.returncode != 0:
        status = GPU_COPY_PROBE_STATUS_ERROR
    elif _all_roundtrips_ok(parsed):
        status = GPU_COPY_PROBE_STATUS_OK
    else:
        status = GPU_COPY_PROBE_STATUS_ROUNDTRIP_FAILED

    return {
        "status": status,
        "counts": counts,
        "target_accelerator": target_accelerator,
        "probe": result.to_json_ready(),
        "parsed": parsed,
    }


def parse_gpu_copy_probe_output(stdout: str) -> dict[str, object]:
    return parse_key_value_probe_output(stdout)


def _all_roundtrips_ok(parsed: dict[str, object]) -> bool:
    return (
        parsed.get("status") == GPU_COPY_PROBE_STATUS_OK
        and parsed.get("float32_roundtrip_ok") is True
        and parsed.get("int8_roundtrip_ok") is True
        and parsed.get("int64_roundtrip_ok") is True
    )
