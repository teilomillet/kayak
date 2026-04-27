"""Policy-driven FastPlaid comparison helpers for GPU i8 profiling.

This module owns argument construction and summary extraction for comparing a
shape-only centroid-budget policy against FastPlaid. It does not run FastPlaid
or write report files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from kayak_bridge.gpu_i8_address_serve_sweep import AddressServeSweepCase
from kayak_bridge.gpu_i8_centroid_budget_policy import choose_policy_budget


STATUS_OK = "ok"


@dataclass(frozen=True, slots=True)
class FastPlaidPolicyCompareControls:
    vector_dim: int = 128
    top_k: int = 10
    seed: int = 7
    warmup_iterations: int = 1
    measurement_iterations: int = 1
    gpu_topk_session_iterations: int = 4
    kayak_plaid_centroid_count: int = 128
    policy_name: str = "shape_rule_v0"
    fastplaid_devices: tuple[str, ...] = ("cpu", "cuda")

    def validate(self) -> None:
        if self.vector_dim != 128:
            raise ValueError("GPU i8 FastPlaid policy compare requires dim128")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.measurement_iterations <= 0:
            raise ValueError("measurement_iterations must be positive")
        if self.gpu_topk_session_iterations <= 0:
            raise ValueError("gpu_topk_session_iterations must be positive")
        if not self.fastplaid_devices:
            raise ValueError("at least one FastPlaid device is required")

    def to_json_ready(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "kayak_plaid_centroid_count": self.kayak_plaid_centroid_count,
            "policy_name": self.policy_name,
            "fastplaid_devices": list(self.fastplaid_devices),
            "warmup_iterations": self.warmup_iterations,
            "measurement_iterations": self.measurement_iterations,
            "gpu_topk_session_iterations": self.gpu_topk_session_iterations,
        }


def parse_fastplaid_devices(value: str) -> tuple[str, ...]:
    devices: list[str] = []
    for raw_device in value.split(","):
        device = raw_device.strip()
        if not device:
            continue
        if device not in devices:
            devices.append(device)
    if not devices:
        raise ValueError("at least one FastPlaid device is required")
    return tuple(devices)


def compare_argv_for_case(
    *,
    case: AddressServeSweepCase,
    case_index: int,
    controls: FastPlaidPolicyCompareControls,
    fastplaid_device: str,
    index_root: Path,
    output: Path,
    allow_missing_gpu: bool,
    require_fastplaid: bool,
    overwrite_index_root: bool,
) -> list[str]:
    controls.validate()
    choice = choose_policy_budget(controls.policy_name, case)
    argv = [
        "--document-count",
        str(case.document_count),
        "--document-vector-count",
        str(case.document_vector_count),
        "--query-count",
        str(case.query_count),
        "--query-vector-count",
        str(case.query_vector_count),
        "--vector-dim",
        str(controls.vector_dim),
        "--candidate-k",
        str(case.candidate_k),
        "--top-k",
        str(controls.top_k),
        "--seed",
        str(controls.seed + case_index),
        "--warmup-iterations",
        str(controls.warmup_iterations),
        "--measurement-iterations",
        str(controls.measurement_iterations),
        "--gpu-topk-session-iterations",
        str(controls.gpu_topk_session_iterations),
        "--kayak-plaid-centroid-count",
        str(controls.kayak_plaid_centroid_count),
        "--kayak-plaid-centroids-per-query-vector",
        str(choice.centroids_per_query_vector),
        "--fastplaid-device",
        fastplaid_device,
        "--index-root",
        str(index_root),
        "--output",
        str(output),
    ]
    if allow_missing_gpu:
        argv.append("--allow-missing-gpu")
    if require_fastplaid:
        argv.append("--require-fastplaid")
    if overwrite_index_root:
        argv.append("--overwrite-index-root")
    return argv


def summarize_case_device_report(
    *,
    case: AddressServeSweepCase,
    case_index: int,
    controls: FastPlaidPolicyCompareControls,
    fastplaid_device: str,
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    choice = choose_policy_budget(controls.policy_name, case)
    kayak_i8 = _system_by_name(report.get("systems", []), "kayak_plaid_mojo_probe")
    fastplaid = _system_by_name(report.get("systems", []), "fastplaid")
    comparison = report.get(
        "gpu_prepared_handle_topk_no_reference_vs_fastplaid_scope_comparison",
        {},
    )
    return {
        "name": case.name,
        "status": report.get("status"),
        "fastplaid_device": fastplaid_device,
        "seed": controls.seed + case_index,
        "shape": report.get("shape"),
        "policy": choice.to_json_ready(),
        "report_path": str(report_path),
        "kayak_i8_recall_at_k_vs_kayak_exact": _optional_float(
            kayak_i8.get("recall_at_k_vs_kayak_exact") if kayak_i8 else None
        ),
        "fastplaid_recall_at_k_vs_kayak_exact": _optional_float(
            fastplaid.get("recall_at_k_vs_kayak_exact") if fastplaid else None
        ),
        "fastplaid_query_batch_mean_seconds": _optional_float(
            fastplaid.get("query_batch_mean_seconds") if fastplaid else None
        ),
        "cpu_candidate_generation_plus_gpu_topk_no_reference_seconds_per_window": (
            _optional_float(
                comparison.get(
                    "cpu_candidate_generation_plus_gpu_topk_seconds_per_window"
                )
            )
        ),
        "cpu_candidate_generation_plus_gpu_topk_no_reference_seconds_per_fastplaid_batch_second": (
            _optional_float(
                comparison.get(
                    "cpu_candidate_generation_plus_gpu_topk_seconds_per_fastplaid_batch_second"
                )
            )
        ),
        "topk_position_agreement": _optional_float(
            comparison.get("topk_position_agreement")
        ),
    }


def summary_payload(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == STATUS_OK]
    ratios = [
        float(
            row[
                "cpu_candidate_generation_plus_gpu_topk_no_reference_seconds_per_fastplaid_batch_second"
            ]
        )
        for row in ok_rows
        if row[
            "cpu_candidate_generation_plus_gpu_topk_no_reference_seconds_per_fastplaid_batch_second"
        ]
        is not None
    ]
    recall_deltas = [
        float(row["kayak_i8_recall_at_k_vs_kayak_exact"])
        - float(row["fastplaid_recall_at_k_vs_kayak_exact"])
        for row in ok_rows
        if row["kayak_i8_recall_at_k_vs_kayak_exact"] is not None
        and row["fastplaid_recall_at_k_vs_kayak_exact"] is not None
    ]
    return {
        "row_count": len(rows),
        "ok_row_count": len(ok_rows),
        "mean_envelope_seconds_per_fastplaid_batch_second": _mean(ratios),
        "max_envelope_seconds_per_fastplaid_batch_second": (
            max(ratios) if ratios else None
        ),
        "min_recall_delta_vs_fastplaid": (
            min(recall_deltas) if recall_deltas else None
        ),
        "topk_position_agreement_min": _min_optional(
            row.get("topk_position_agreement") for row in ok_rows
        ),
    }


def report_status(rows: Sequence[dict[str, Any]]) -> str:
    if all(row.get("status") == STATUS_OK for row in rows):
        return STATUS_OK
    return "error"


def _system_by_name(
    systems: Sequence[dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    return next(
        (system for system in systems if system.get("system_name") == name),
        None,
    )


def _optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _min_optional(values: Sequence[object]) -> float | None:
    floats = [
        float(value)
        for value in values
        if isinstance(value, (float, int))
    ]
    return min(floats) if floats else None
