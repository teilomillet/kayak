"""Defines the GPU i8 address serving sweep contract and summaries.

This module owns sweep case parsing, explicit shape validation, and ratio
semantics. It does not build indexes, run GPU probes, or print CLI output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Sequence

from kayak_bridge.gpu_device_capability import MojoGpuCapability

from bench_fastplaid_speed_track import SpeedTrackShape
from profile_gpu_i8_real_payload_rerank import (
    STATUS_OK,
    STATUS_PARTIAL_GPU_UNAVAILABLE,
    ratio,
    shape_payload,
    sum_optional,
)


STATUS_BLOCKED_GPU_ADDRESS_SERVE_FAILED = "blocked_gpu_address_serve_failed"


@dataclass(frozen=True, slots=True)
class AddressServeSweepCase:
    name: str
    document_count: int
    document_vector_count: int
    query_count: int
    query_vector_count: int
    candidate_k: int

    def shape(self, *, vector_dim: int, top_k: int) -> SpeedTrackShape:
        shape = SpeedTrackShape(
            document_count=self.document_count,
            document_vector_count=self.document_vector_count,
            query_count=self.query_count,
            query_vector_count=self.query_vector_count,
            vector_dim=vector_dim,
            top_k=top_k,
            update_document_count=0,
        )
        shape.validate()
        if self.candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if self.candidate_k > self.document_count:
            raise ValueError("candidate_k must not exceed document_count")
        return shape

    def to_json_ready(self, *, vector_dim: int, top_k: int) -> dict[str, int | str]:
        shape = self.shape(vector_dim=vector_dim, top_k=top_k)
        return {"name": self.name} | shape_payload(shape, candidate_k=self.candidate_k)


@dataclass(frozen=True, slots=True)
class AddressServeSweepControls:
    vector_dim: int = 128
    top_k: int = 10
    seed: int = 7
    warmup_iterations: int = 1
    measurement_iterations: int = 3
    resident_session_iterations: int = 4
    kayak_plaid_centroid_count: int = 128
    kayak_plaid_centroids_per_query_vector: int = 32
    gpu_query_command: str = "gpu-query"

    def validate(self) -> None:
        if self.vector_dim != 128:
            raise ValueError("GPU i8 address serving sweep requires vector_dim=128")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.measurement_iterations <= 0:
            raise ValueError("measurement_iterations must be positive")
        if self.resident_session_iterations <= 0:
            raise ValueError("resident_session_iterations must be positive")

    def to_json_ready(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "kayak_plaid_centroid_count": self.kayak_plaid_centroid_count,
            "kayak_plaid_centroids_per_query_vector": (
                self.kayak_plaid_centroids_per_query_vector
            ),
            "kayak_plaid_payload": "i8",
            "warmup_iterations": self.warmup_iterations,
            "measurement_iterations": self.measurement_iterations,
            "resident_session_iterations": self.resident_session_iterations,
        }


DEFAULT_CASES: tuple[AddressServeSweepCase, ...] = (
    AddressServeSweepCase("baseline", 256, 16, 2, 8, 128),
    AddressServeSweepCase("candidate32", 256, 16, 2, 8, 32),
    AddressServeSweepCase("candidate256", 256, 16, 2, 8, 256),
    AddressServeSweepCase("query_vectors16", 256, 16, 2, 16, 128),
    AddressServeSweepCase("doc_vectors32", 256, 32, 2, 8, 128),
    AddressServeSweepCase("documents512", 512, 16, 2, 8, 128),
)


def parse_sweep_case(value: str) -> AddressServeSweepCase:
    name, separator, assignments_text = value.partition(":")
    if not separator or not name.strip() or not assignments_text.strip():
        raise argparse.ArgumentTypeError(
            "case must use name:key=value,key=value syntax"
        )
    return AddressServeSweepCase(
        name=_quiet_safe_name(name.strip()),
        **_parse_case_assignments(assignments_text),
    )


def _parse_case_assignments(value: str) -> dict[str, int]:
    aliases = {
        "documents": "document_count",
        "document_count": "document_count",
        "document_vectors": "document_vector_count",
        "document_vector_count": "document_vector_count",
        "queries": "query_count",
        "query_count": "query_count",
        "query_vectors": "query_vector_count",
        "query_vector_count": "query_vector_count",
        "candidate_k": "candidate_k",
    }
    assignments: dict[str, int] = {}
    for item in value.split(","):
        key, separator, raw_value = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError("case assignments must use key=value")
        canonical_key = aliases.get(key.strip())
        if canonical_key is None:
            raise argparse.ArgumentTypeError(f"unknown case key: {key.strip()}")
        if canonical_key in assignments:
            raise argparse.ArgumentTypeError(f"repeated case key: {canonical_key}")
        assignments[canonical_key] = _parse_int(raw_value, canonical_key)
    _require_case_keys(assignments)
    return assignments


def _parse_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"case value for {key} must be an integer"
        ) from exc


def _require_case_keys(assignments: dict[str, int]) -> None:
    required = {
        "document_count",
        "document_vector_count",
        "query_count",
        "query_vector_count",
        "candidate_k",
    }
    missing = required - set(assignments)
    if missing:
        raise argparse.ArgumentTypeError(
            "case missing required keys: " + ", ".join(sorted(missing))
        )


def _quiet_safe_name(value: str) -> str:
    if any(not (character.isalnum() or character in {"_", "-"}) for character in value):
        raise argparse.ArgumentTypeError(
            "case name may contain only letters, digits, '_' or '-'"
        )
    return value


def comparison_payload(
    *,
    cpu_candidate_generation_mean_seconds: float,
    cpu_score_mean_seconds: float,
    gpu_parsed: dict[str, object],
    resident_parsed: dict[str, object] | None = None,
) -> dict[str, float | None]:
    gpu_extension_call = optional_float(gpu_parsed.get("extension_call_seconds"))
    resident = resident_parsed if resident_parsed is not None else {}
    resident_per_iteration = optional_float(
        resident.get("extension_call_seconds_per_iteration")
    )
    cpu_candidate_plus_score = (
        cpu_candidate_generation_mean_seconds + cpu_score_mean_seconds
    )
    cpu_candidate_plus_gpu_score = sum_optional(
        cpu_candidate_generation_mean_seconds,
        gpu_extension_call,
    )
    cpu_candidate_plus_resident_score = sum_optional(
        cpu_candidate_generation_mean_seconds,
        resident_per_iteration,
    )
    return {
        "cpu_i8_candidate_generation_mean_seconds": (
            cpu_candidate_generation_mean_seconds
        ),
        "cpu_i8_same_candidate_score_mean_seconds": cpu_score_mean_seconds,
        "cpu_i8_candidate_generation_plus_score_mean_seconds": (
            cpu_candidate_plus_score
        ),
        "gpu_address_serve_extension_call_seconds": gpu_extension_call,
        "gpu_address_resident_session_extension_call_seconds_per_iteration": (
            resident_per_iteration
        ),
        "cpu_candidate_generation_plus_gpu_address_serve_seconds": (
            cpu_candidate_plus_gpu_score
        ),
        "cpu_candidate_generation_plus_gpu_resident_iteration_seconds": (
            cpu_candidate_plus_resident_score
        ),
        "gpu_address_serve_extension_call_seconds_per_cpu_score_second": ratio(
            gpu_extension_call,
            cpu_score_mean_seconds,
        ),
        "gpu_address_resident_iteration_seconds_per_cpu_score_second": ratio(
            resident_per_iteration,
            cpu_score_mean_seconds,
        ),
        "cpu_candidate_plus_gpu_address_serve_seconds_per_cpu_candidate_plus_score_second": ratio(
            cpu_candidate_plus_gpu_score,
            cpu_candidate_plus_score,
        ),
        "cpu_candidate_plus_gpu_resident_iteration_seconds_per_cpu_candidate_plus_score_second": ratio(
            cpu_candidate_plus_resident_score,
            cpu_candidate_plus_score,
        ),
    }


def optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def summary_payload(rows: Sequence[dict[str, Any]]) -> dict[str, object]:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == STATUS_OK and isinstance(row.get("comparison"), dict)
    ]
    isolated_ratios = [
        row["comparison"].get(
            "gpu_address_serve_extension_call_seconds_per_cpu_score_second"
        )
        for row in ok_rows
    ]
    envelope_ratios = [
        row["comparison"].get(
            "cpu_candidate_plus_gpu_address_serve_seconds_per_cpu_candidate_plus_score_second"
        )
        for row in ok_rows
    ]
    resident_ratios = [
        row["comparison"].get(
            "gpu_address_resident_iteration_seconds_per_cpu_score_second"
        )
        for row in ok_rows
    ]
    resident_envelope_ratios = [
        row["comparison"].get(
            "cpu_candidate_plus_gpu_resident_iteration_seconds_per_cpu_candidate_plus_score_second"
        )
        for row in ok_rows
    ]
    return {
        "case_count": len(rows),
        "ok_case_count": len(ok_rows),
        "best_isolated_gpu_ratio": min_float(isolated_ratios),
        "worst_isolated_gpu_ratio": max_float(isolated_ratios),
        "best_candidate_plus_gpu_ratio": min_float(envelope_ratios),
        "worst_candidate_plus_gpu_ratio": max_float(envelope_ratios),
        "best_resident_iteration_gpu_ratio": min_float(resident_ratios),
        "worst_resident_iteration_gpu_ratio": max_float(resident_ratios),
        "best_candidate_plus_resident_iteration_ratio": min_float(
            resident_envelope_ratios
        ),
        "worst_candidate_plus_resident_iteration_ratio": max_float(
            resident_envelope_ratios
        ),
    }


def min_float(values: Sequence[object]) -> float | None:
    floats = [float(value) for value in values if isinstance(value, (float, int))]
    return min(floats) if floats else None


def max_float(values: Sequence[object]) -> float | None:
    floats = [float(value) for value in values if isinstance(value, (float, int))]
    return max(floats) if floats else None


def report_status(
    *,
    capability: MojoGpuCapability,
    rows: Sequence[dict[str, Any]],
) -> str:
    if not capability.available:
        return STATUS_PARTIAL_GPU_UNAVAILABLE
    if any(row.get("status") != STATUS_OK for row in rows):
        return STATUS_BLOCKED_GPU_ADDRESS_SERVE_FAILED
    return STATUS_OK
