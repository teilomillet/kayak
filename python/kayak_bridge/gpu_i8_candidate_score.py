from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from kayak_bridge.cache_paths import REPO_ROOT
from kayak_bridge.gpu_device_capability import run_command
from kayak_bridge.gpu_probe_output import parse_key_value_probe_output


GPU_CANDIDATE_SCORE_STATUS_OK = "ok"
GPU_CANDIDATE_SCORE_STATUS_TARGET_MISSING = "skipped_target_accelerator_missing"
GPU_CANDIDATE_SCORE_STATUS_ERROR = "error"
GPU_CANDIDATE_SCORE_STATUS_AGREEMENT_FAILED = "score_agreement_failed"

CANDIDATE_SCORE_PROBE_PATH = (
    REPO_ROOT / "benchmarks" / "gpu_i8_candidate_score_probe.mojo"
)


def gpu_candidate_score_probe_counts(
    shape: Any,
    *,
    candidate_k: int,
) -> dict[str, int]:
    total_doc_vectors = shape.document_count * shape.document_vector_count
    candidate_score_count = shape.query_count * candidate_k
    return {
        "query_count": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "total_document_vector_count": total_doc_vectors,
        "candidate_k": candidate_k,
        "candidate_position_count": candidate_score_count,
        "candidate_score_count": candidate_score_count,
        "vector_dim": shape.vector_dim,
        "query_value_count": (
            shape.query_count * shape.query_vector_count * shape.vector_dim
        ),
        "token_code_count": total_doc_vectors * shape.vector_dim,
        "token_scale_count": total_doc_vectors,
        "doc_offset_count": shape.document_count + 1,
    }


def run_gpu_i8_candidate_score_probe(
    shape: Any,
    *,
    candidate_k: int,
    target_accelerator: str | None,
    mojo_command: Sequence[str] = ("mojo",),
) -> dict[str, object]:
    counts = gpu_candidate_score_probe_counts(shape, candidate_k=candidate_k)
    if target_accelerator is None:
        return {
            "status": GPU_CANDIDATE_SCORE_STATUS_TARGET_MISSING,
            "counts": counts,
            "probe": None,
        }

    command = (
        *mojo_command,
        "run",
        "--target-accelerator",
        target_accelerator,
        "-D",
        f"query_count={counts['query_count']}",
        "-D",
        f"query_vector_count={counts['query_vector_count']}",
        "-D",
        f"document_count={counts['document_count']}",
        "-D",
        f"document_vector_count={counts['document_vector_count']}",
        "-D",
        f"candidate_k={counts['candidate_k']}",
        str(CANDIDATE_SCORE_PROBE_PATH),
    )
    result = run_command(command, timeout_seconds=45.0)
    parsed = parse_key_value_probe_output(result.stdout)
    if result.returncode != 0:
        status = GPU_CANDIDATE_SCORE_STATUS_ERROR
    elif (
        parsed.get("score_agreement_ok") is True
        and parsed.get("twopass_score_agreement_ok") is True
    ):
        status = GPU_CANDIDATE_SCORE_STATUS_OK
    else:
        status = GPU_CANDIDATE_SCORE_STATUS_AGREEMENT_FAILED

    return {
        "status": status,
        "counts": counts,
        "target_accelerator": target_accelerator,
        "probe": result.to_json_ready(),
        "parsed": parsed,
    }
