from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kayak_bridge.cache_paths import REPO_ROOT
from kayak_bridge.gpu_device_capability import run_command
from kayak_bridge.gpu_probe_output import parse_key_value_probe_output
from kayak_bridge.plaid_approx import KayakPlaidI8PayloadSnapshot


GPU_REAL_PAYLOAD_STATUS_OK = "ok"
GPU_REAL_PAYLOAD_STATUS_TARGET_MISSING = "skipped_target_accelerator_missing"
GPU_REAL_PAYLOAD_STATUS_ERROR = "error"
GPU_REAL_PAYLOAD_STATUS_AGREEMENT_FAILED = "score_agreement_failed"

REAL_PAYLOAD_SCORE_PROBE_PATH = (
    REPO_ROOT / "benchmarks" / "gpu_i8_real_payload_score_probe.mojo"
)


def gpu_real_payload_probe_counts(
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


def write_gpu_i8_real_payload_probe_inputs(
    root: Path,
    *,
    shape: Any,
    candidate_k: int,
    queries: np.ndarray,
    payload: KayakPlaidI8PayloadSnapshot,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
) -> dict[str, object]:
    counts = gpu_real_payload_probe_counts(shape, candidate_k=candidate_k)
    root.mkdir(parents=True, exist_ok=True)

    query_values = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1)
    token_codes = np.ascontiguousarray(payload.token_codes, dtype=np.int8)
    token_scales = np.ascontiguousarray(payload.token_scales, dtype=np.float32)
    doc_offsets = np.ascontiguousarray(payload.doc_offsets, dtype=np.uint64)
    candidate_positions = _flatten_u64_rows(
        candidate_positions_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="candidate_positions_by_query",
    )
    reference_scores = _flatten_f32_rows(
        reference_scores_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="reference_scores_by_query",
    )

    _require_size("query_values", query_values, counts["query_value_count"])
    _require_size("token_codes", token_codes, counts["token_code_count"])
    _require_size("token_scales", token_scales, counts["token_scale_count"])
    _require_size("doc_offsets", doc_offsets, counts["doc_offset_count"])
    _require_size(
        "candidate_positions",
        candidate_positions,
        counts["candidate_position_count"],
    )
    _require_size("reference_scores", reference_scores, counts["candidate_score_count"])

    (root / "query_values.bin").write_bytes(query_values.tobytes(order="C"))
    (root / "token_codes.bin").write_bytes(token_codes.tobytes(order="C"))
    (root / "token_scales.bin").write_bytes(token_scales.tobytes(order="C"))
    (root / "doc_offsets.bin").write_bytes(doc_offsets.tobytes(order="C"))
    (root / "candidate_positions.bin").write_bytes(
        candidate_positions.tobytes(order="C")
    )
    (root / "reference_scores.bin").write_bytes(reference_scores.tobytes(order="C"))

    manifest = {
        "schema_version": 1,
        "payload_source": "real_kayak_i8_snapshot",
        "counts": counts,
        "files": {
            "query_values": "query_values.bin",
            "token_codes": "token_codes.bin",
            "token_scales": "token_scales.bin",
            "doc_offsets": "doc_offsets.bin",
            "candidate_positions": "candidate_positions.bin",
            "reference_scores": "reference_scores.bin",
        },
        "byte_counts": {
            "query_values": int(query_values.nbytes),
            "token_codes": int(token_codes.nbytes),
            "token_scales": int(token_scales.nbytes),
            "doc_offsets": int(doc_offsets.nbytes),
            "candidate_positions": int(candidate_positions.nbytes),
            "reference_scores": int(reference_scores.nbytes),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_gpu_i8_real_payload_score_probe(
    shape: Any,
    *,
    candidate_k: int,
    data_root: Path,
    target_accelerator: str | None,
    mojo_command: Sequence[str] = ("mojo",),
) -> dict[str, object]:
    counts = gpu_real_payload_probe_counts(shape, candidate_k=candidate_k)
    if target_accelerator is None:
        return {
            "status": GPU_REAL_PAYLOAD_STATUS_TARGET_MISSING,
            "counts": counts,
            "probe": None,
        }

    command = (
        *mojo_command,
        "run",
        "--target-accelerator",
        target_accelerator,
        "-D",
        f"data_root={data_root}",
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
        str(REAL_PAYLOAD_SCORE_PROBE_PATH),
    )
    result = run_command(command, timeout_seconds=60.0)
    parsed = parse_key_value_probe_output(result.stdout)
    if result.returncode != 0:
        status = GPU_REAL_PAYLOAD_STATUS_ERROR
    elif (
        parsed.get("score_agreement_ok") is True
        and parsed.get("twopass_score_agreement_ok") is True
    ):
        status = GPU_REAL_PAYLOAD_STATUS_OK
    else:
        status = GPU_REAL_PAYLOAD_STATUS_AGREEMENT_FAILED

    return {
        "status": status,
        "counts": counts,
        "target_accelerator": target_accelerator,
        "data_root": str(data_root),
        "probe": result.to_json_ready(),
        "parsed": parsed,
    }


def _flatten_u64_rows(
    rows: Sequence[Sequence[int]],
    *,
    expected_rows: int,
    expected_cols: int,
    name: str,
) -> np.ndarray:
    if len(rows) != expected_rows:
        raise ValueError(f"{name} row count must match query_count")
    values: list[int] = []
    for row in rows:
        if len(row) != expected_cols:
            raise ValueError(f"{name} rows must match candidate_k")
        values.extend(int(value) for value in row)
    return np.asarray(values, dtype=np.uint64)


def _flatten_f32_rows(
    rows: Sequence[Sequence[float]],
    *,
    expected_rows: int,
    expected_cols: int,
    name: str,
) -> np.ndarray:
    if len(rows) != expected_rows:
        raise ValueError(f"{name} row count must match query_count")
    values: list[float] = []
    for row in rows:
        if len(row) != expected_cols:
            raise ValueError(f"{name} rows must match candidate_k")
        values.extend(float(value) for value in row)
    return np.asarray(values, dtype=np.float32)


def _require_size(name: str, values: np.ndarray, expected: int) -> None:
    if int(values.size) != expected:
        raise ValueError(f"{name} size must be {expected}, got {values.size}")
