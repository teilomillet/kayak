"""Policy-driven FastPlaid comparison helpers for GPU i8 profiling.

This module owns argument construction and summary extraction for comparing a
shape-only centroid-budget policy against FastPlaid. It does not run FastPlaid
or write report files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from kayak_bridge.gpu_i8_address_serve_sweep import AddressServeSweepCase
from kayak_bridge.gpu_i8_candidate_window_policy import (
    INPUT_CANDIDATE_K_POLICY,
    choose_candidate_window,
    validate_candidate_window_policy,
)
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
    candidate_window_policy: str = INPUT_CANDIDATE_K_POLICY
    kayak_i8_candidate_order: str = "unordered"
    kayak_i8_positive_centroids_only: bool = False
    fastplaid_devices: tuple[str, ...] = ("cpu", "cuda")
    gpu_hybrid_shortlist_k: int | None = None

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
        if (
            self.gpu_hybrid_shortlist_k is not None
            and self.gpu_hybrid_shortlist_k < self.top_k
        ):
            raise ValueError(
                "gpu_hybrid_shortlist_k must be greater than or equal to top_k"
            )
        if self.kayak_i8_candidate_order not in {"ordered", "unordered"}:
            raise ValueError("kayak_i8_candidate_order must be ordered or unordered")
        validate_candidate_window_policy(self.candidate_window_policy)
        if (
            self.kayak_i8_positive_centroids_only
            and self.kayak_i8_candidate_order != "unordered"
        ):
            raise ValueError("positive centroid candidates require unordered order")
        if not self.fastplaid_devices:
            raise ValueError("at least one FastPlaid device is required")

    def to_json_ready(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "kayak_plaid_centroid_count": self.kayak_plaid_centroid_count,
            "policy_name": self.policy_name,
            "candidate_window_policy": self.candidate_window_policy,
            "fastplaid_devices": list(self.fastplaid_devices),
            "warmup_iterations": self.warmup_iterations,
            "measurement_iterations": self.measurement_iterations,
            "gpu_topk_session_iterations": self.gpu_topk_session_iterations,
            "gpu_hybrid_shortlist_k": self.gpu_hybrid_shortlist_k,
            "kayak_i8_candidate_order": self.kayak_i8_candidate_order,
            "kayak_i8_positive_centroids_only": (
                self.kayak_i8_positive_centroids_only
            ),
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
    if (
        controls.gpu_hybrid_shortlist_k is not None
        and controls.gpu_hybrid_shortlist_k > case.document_count
    ):
        raise ValueError(
            "gpu_hybrid_shortlist_k must be no larger than document_count"
        )
    choice = choose_policy_budget(controls.policy_name, case)
    candidate_choice = choose_candidate_window(
        controls.candidate_window_policy,
        case,
    )
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
        str(candidate_choice.candidate_k),
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
        "--kayak-i8-candidate-order",
        controls.kayak_i8_candidate_order,
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
    if controls.kayak_i8_positive_centroids_only:
        argv.append("--kayak-i8-positive-centroids-only")
    if controls.gpu_hybrid_shortlist_k is not None:
        argv.extend(
            [
                "--gpu-hybrid-shortlist-k",
                str(controls.gpu_hybrid_shortlist_k),
            ]
        )
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
    candidate_choice = choose_candidate_window(
        controls.candidate_window_policy,
        case,
    )
    kayak_i8 = _system_by_name(report.get("systems", []), "kayak_plaid_mojo_probe")
    fastplaid = _system_by_name(report.get("systems", []), "fastplaid")
    comparison = report.get(
        "gpu_prepared_handle_topk_no_reference_vs_fastplaid_scope_comparison",
        {},
    )
    fused_comparison = report.get(
        "gpu_fused_centroid_posting_vs_fastplaid_scope_comparison",
        {},
    )
    hybrid_comparison = report.get(
        "gpu_hybrid_shortlist_exact_rerank_vs_fastplaid_scope_comparison",
        {},
    )
    resident_selected_comparison = report.get(
        "gpu_resident_selected_posting_exact_rerank_vs_fastplaid_scope_comparison",
        {},
    )
    candidate_seconds = _optional_float(
        comparison.get("cpu_candidate_generation_seconds_per_window")
    )
    gpu_topk_seconds = _optional_float(
        comparison.get("gpu_prepared_handle_topk_seconds_per_window")
    )
    envelope_seconds = _optional_float(
        comparison.get(
            "cpu_candidate_generation_plus_gpu_topk_seconds_per_window"
        )
    )
    fused_device_seconds = _optional_float(
        fused_comparison.get("gpu_fused_device_topk_seconds_per_window")
    )
    fused_device_ratio = _optional_float(
        fused_comparison.get(
            "gpu_fused_device_topk_seconds_per_fastplaid_batch_second"
        )
    )
    fused_recall = _optional_float(
        fused_comparison.get("recall_at_k_vs_kayak_exact")
    )
    hybrid_seconds = _optional_float(
        hybrid_comparison.get("gpu_hybrid_seconds_per_window")
    )
    hybrid_ratio = _optional_float(
        hybrid_comparison.get(
            "gpu_hybrid_seconds_per_fastplaid_batch_second"
        )
    )
    hybrid_recall = _optional_float(
        hybrid_comparison.get("recall_at_k_vs_kayak_exact")
    )
    resident_selected_seconds = _optional_float(
        resident_selected_comparison.get(
            "gpu_resident_selected_exact_rerank_seconds_per_window_total"
        )
    )
    resident_selected_ratio = _optional_float(
        resident_selected_comparison.get(
            "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
        )
    )
    resident_selected_recall = _optional_float(
        resident_selected_comparison.get("recall_at_k_vs_kayak_exact")
    )
    fastplaid_recall = _optional_float(
        fastplaid.get("recall_at_k_vs_kayak_exact") if fastplaid else None
    )
    return {
        "name": case.name,
        "status": report.get("status"),
        "fastplaid_device": fastplaid_device,
        "seed": controls.seed + case_index,
        "shape": report.get("shape"),
        "policy": choice.to_json_ready(),
        "candidate_window_policy": candidate_choice.to_json_ready(),
        "input_candidate_k": candidate_choice.input_candidate_k,
        "effective_candidate_k": candidate_choice.candidate_k,
        "kayak_i8_candidate_order": controls.kayak_i8_candidate_order,
        "kayak_i8_positive_centroids_only": (
            controls.kayak_i8_positive_centroids_only
        ),
        "report_path": str(report_path),
        "kayak_i8_recall_at_k_vs_kayak_exact": _optional_float(
            kayak_i8.get("recall_at_k_vs_kayak_exact") if kayak_i8 else None
        ),
        "fastplaid_recall_at_k_vs_kayak_exact": fastplaid_recall,
        "fastplaid_query_batch_mean_seconds": _optional_float(
            fastplaid.get("query_batch_mean_seconds") if fastplaid else None
        ),
        "cpu_candidate_generation_seconds_per_window": candidate_seconds,
        "gpu_topk_no_reference_seconds_per_window": gpu_topk_seconds,
        "cpu_candidate_generation_plus_gpu_topk_no_reference_seconds_per_window": (
            envelope_seconds
        ),
        "cpu_candidate_generation_plus_gpu_topk_no_reference_seconds_per_fastplaid_batch_second": (
            _optional_float(
                comparison.get(
                    "cpu_candidate_generation_plus_gpu_topk_seconds_per_fastplaid_batch_second"
                )
            )
        ),
        "cpu_candidate_generation_share_of_envelope": _ratio(
            candidate_seconds,
            envelope_seconds,
        ),
        "gpu_topk_no_reference_share_of_envelope": _ratio(
            gpu_topk_seconds,
            envelope_seconds,
        ),
        "topk_position_agreement": _optional_float(
            comparison.get("topk_position_agreement")
        ),
        "gpu_fused_device_topk_seconds_per_window": fused_device_seconds,
        "gpu_fused_device_topk_seconds_per_fastplaid_batch_second": (
            fused_device_ratio
        ),
        "gpu_fused_device_topk_seconds_per_host_topk_second": _optional_float(
            fused_comparison.get(
                "gpu_fused_device_topk_seconds_per_host_topk_second"
            )
        ),
        "gpu_fused_recall_at_k_vs_kayak_exact": fused_recall,
        "gpu_fused_recall_delta_vs_fastplaid": (
            fused_recall - fastplaid_recall
            if fused_recall is not None and fastplaid_recall is not None
            else None
        ),
        "gpu_fused_device_topk_position_agreement": _optional_float(
            fused_comparison.get("device_topk_position_agreement")
        ),
        "gpu_hybrid_seconds_per_window": hybrid_seconds,
        "gpu_hybrid_seconds_per_fastplaid_batch_second": hybrid_ratio,
        "gpu_hybrid_exact_rerank_share": _optional_float(
            hybrid_comparison.get("gpu_hybrid_exact_rerank_share")
        ),
        "gpu_hybrid_recall_at_k_vs_kayak_exact": hybrid_recall,
        "gpu_hybrid_recall_delta_vs_fastplaid": (
            hybrid_recall - fastplaid_recall
            if hybrid_recall is not None and fastplaid_recall is not None
            else None
        ),
        "gpu_hybrid_final_topk_position_agreement": _optional_float(
            hybrid_comparison.get("final_topk_position_agreement")
        ),
        "gpu_hybrid_shortlist_k": _optional_int(
            hybrid_comparison.get("shortlist_k")
        ),
        "gpu_resident_selected_exact_rerank_seconds_per_window": (
            resident_selected_seconds
        ),
        "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second": (
            resident_selected_ratio
        ),
        "gpu_resident_selected_exact_rerank_cold_seconds_per_fastplaid_batch_second": (
            _optional_float(
                resident_selected_comparison.get(
                    "gpu_resident_selected_exact_rerank_cold_seconds_per_fastplaid_batch_second"
                )
            )
        ),
        "gpu_resident_selected_exact_rerank_exact_share": _optional_float(
            resident_selected_comparison.get(
                "gpu_resident_selected_exact_rerank_exact_share"
            )
        ),
        "gpu_resident_selected_cpu_selection_share": _optional_float(
            resident_selected_comparison.get(
                "gpu_resident_selected_cpu_selection_share"
            )
        ),
        "gpu_resident_selected_candidate_share": _optional_float(
            resident_selected_comparison.get(
                "gpu_resident_selected_candidate_share"
            )
        ),
        "gpu_resident_selected_recall_at_k_vs_kayak_exact": (
            resident_selected_recall
        ),
        "gpu_resident_selected_recall_delta_vs_fastplaid": (
            resident_selected_recall - fastplaid_recall
            if resident_selected_recall is not None
            and fastplaid_recall is not None
            else None
        ),
        "gpu_resident_selected_final_topk_position_agreement": _optional_float(
            resident_selected_comparison.get("final_topk_position_agreement")
        ),
        "gpu_resident_selected_candidate_position_agreement_min": _optional_float(
            resident_selected_comparison.get("candidate_position_agreement_min")
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
    fused_ratios = [
        float(row["gpu_fused_device_topk_seconds_per_fastplaid_batch_second"])
        for row in ok_rows
        if row["gpu_fused_device_topk_seconds_per_fastplaid_batch_second"]
        is not None
    ]
    fused_recall_deltas = [
        float(row["gpu_fused_recall_delta_vs_fastplaid"])
        for row in ok_rows
        if row["gpu_fused_recall_delta_vs_fastplaid"] is not None
    ]
    hybrid_ratios = [
        float(row["gpu_hybrid_seconds_per_fastplaid_batch_second"])
        for row in ok_rows
        if row["gpu_hybrid_seconds_per_fastplaid_batch_second"] is not None
    ]
    hybrid_recall_deltas = [
        float(row["gpu_hybrid_recall_delta_vs_fastplaid"])
        for row in ok_rows
        if row["gpu_hybrid_recall_delta_vs_fastplaid"] is not None
    ]
    resident_selected_ratios = [
        float(
            row[
                "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
            ]
        )
        for row in ok_rows
        if row[
            "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
        ]
        is not None
    ]
    resident_selected_cold_ratios = [
        float(
            row[
                "gpu_resident_selected_exact_rerank_cold_seconds_per_fastplaid_batch_second"
            ]
        )
        for row in ok_rows
        if row[
            "gpu_resident_selected_exact_rerank_cold_seconds_per_fastplaid_batch_second"
        ]
        is not None
    ]
    resident_selected_recall_deltas = [
        float(row["gpu_resident_selected_recall_delta_vs_fastplaid"])
        for row in ok_rows
        if row["gpu_resident_selected_recall_delta_vs_fastplaid"] is not None
    ]
    return {
        "row_count": len(rows),
        "ok_row_count": len(ok_rows),
        "mean_cpu_candidate_generation_share_of_envelope": _mean_optional(
            row.get("cpu_candidate_generation_share_of_envelope")
            for row in ok_rows
        ),
        "mean_gpu_topk_no_reference_share_of_envelope": _mean_optional(
            row.get("gpu_topk_no_reference_share_of_envelope")
            for row in ok_rows
        ),
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
        "mean_gpu_fused_device_topk_seconds_per_fastplaid_batch_second": (
            _mean(fused_ratios)
        ),
        "max_gpu_fused_device_topk_seconds_per_fastplaid_batch_second": (
            max(fused_ratios) if fused_ratios else None
        ),
        "min_gpu_fused_recall_delta_vs_fastplaid": (
            min(fused_recall_deltas) if fused_recall_deltas else None
        ),
        "gpu_fused_device_topk_position_agreement_min": _min_optional(
            row.get("gpu_fused_device_topk_position_agreement")
            for row in ok_rows
        ),
        "mean_gpu_hybrid_seconds_per_fastplaid_batch_second": (
            _mean(hybrid_ratios)
        ),
        "max_gpu_hybrid_seconds_per_fastplaid_batch_second": (
            max(hybrid_ratios) if hybrid_ratios else None
        ),
        "min_gpu_hybrid_recall_delta_vs_fastplaid": (
            min(hybrid_recall_deltas) if hybrid_recall_deltas else None
        ),
        "mean_gpu_hybrid_exact_rerank_share": _mean_optional(
            row.get("gpu_hybrid_exact_rerank_share") for row in ok_rows
        ),
        "gpu_hybrid_final_topk_position_agreement_min": _min_optional(
            row.get("gpu_hybrid_final_topk_position_agreement")
            for row in ok_rows
        ),
        "mean_gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second": (
            _mean(resident_selected_ratios)
        ),
        "max_gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second": (
            max(resident_selected_ratios) if resident_selected_ratios else None
        ),
        "max_gpu_resident_selected_exact_rerank_cold_seconds_per_fastplaid_batch_second": (
            max(resident_selected_cold_ratios)
            if resident_selected_cold_ratios
            else None
        ),
        "min_gpu_resident_selected_recall_delta_vs_fastplaid": (
            min(resident_selected_recall_deltas)
            if resident_selected_recall_deltas
            else None
        ),
        "mean_gpu_resident_selected_exact_rerank_exact_share": _mean_optional(
            row.get("gpu_resident_selected_exact_rerank_exact_share")
            for row in ok_rows
        ),
        "mean_gpu_resident_selected_cpu_selection_share": _mean_optional(
            row.get("gpu_resident_selected_cpu_selection_share")
            for row in ok_rows
        ),
        "mean_gpu_resident_selected_candidate_share": _mean_optional(
            row.get("gpu_resident_selected_candidate_share")
            for row in ok_rows
        ),
        "gpu_resident_selected_final_topk_position_agreement_min": _min_optional(
            row.get("gpu_resident_selected_final_topk_position_agreement")
            for row in ok_rows
        ),
        "gpu_resident_selected_candidate_position_agreement_min": _min_optional(
            row.get("gpu_resident_selected_candidate_position_agreement_min")
            for row in ok_rows
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


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _mean_optional(values: Iterable[object]) -> float | None:
    floats = [
        float(value)
        for value in values
        if isinstance(value, (float, int))
    ]
    return _mean(floats)


def _min_optional(values: Iterable[object]) -> float | None:
    floats = [
        float(value)
        for value in values
        if isinstance(value, (float, int))
    ]
    return min(floats) if floats else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator
