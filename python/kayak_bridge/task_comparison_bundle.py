"""Owns a compact bundle for one task-level Kayak versus LanceDB comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _system_by_name(scorecard: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for system in scorecard["systems"]:
        if str(system["name"]) == name:
            return system
    raise ValueError(f"scorecard system not found: {name}")


def _pairwise_by_candidate(
    scorecard: Mapping[str, Any], candidate: str
) -> Mapping[str, Any]:
    for pairwise in scorecard["pairwise"]:
        if str(pairwise["candidate"]) == candidate:
            return pairwise
    raise ValueError(f"scorecard pairwise comparison not found: {candidate}")


def _largest_row(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = summary.get("rows", [])
    if not rows:
        raise ValueError("summary rows must not be empty")
    return rows[-1]


def _base_row(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = summary.get("rows", [])
    if not rows:
        raise ValueError("summary rows must not be empty")
    return rows[0]


@dataclass(frozen=True, slots=True)
class TaskComparisonBundle:
    dataset_id: str
    family: str
    slice_name: str
    primary_metric: str
    task_path: str
    kayak_exact_path: str
    lancedb_scan_path: str
    lancedb_indexed_variance_path: str
    lancedb_indexed_frozen_path: str
    scorecard_path: str
    scale_sweep_path: str | None
    storage_compare_path: str | None
    storage_scale_path: str | None
    kayak_exact_primary_value: float
    kayak_exact_mean_search_seconds: float
    lancedb_scan_primary_value: float
    lancedb_scan_mean_search_seconds: float
    lancedb_scan_latency_ratio_vs_kayak: float
    lancedb_indexed_frozen_primary_value: float
    lancedb_indexed_frozen_mean_search_seconds: float
    lancedb_indexed_frozen_latency_ratio_vs_kayak: float
    indexed_freeze_policy: str
    indexed_rebuild_count: int
    indexed_index_num_partitions: int | None
    indexed_index_num_sub_vectors: int | None
    indexed_index_target_partition_size: int | None
    indexed_nprobes: int | None
    indexed_refine_factor: int | None
    indexed_primary_value_min: float
    indexed_primary_value_max: float
    indexed_mean_search_seconds_min: float
    indexed_mean_search_seconds_max: float
    storage_compare_kayak_load_from_lancedb_seconds: float | None
    storage_compare_lancedb_scan_mean_search_seconds: float | None
    storage_compare_kayak_exact_from_lancedb_mean_search_seconds: float | None
    storage_compare_lancedb_scan_latency_ratio_vs_kayak_from_lancedb: float | None
    base_storage_byte_ratio_vs_kayak: float | None
    base_storage_build_seconds_ratio_vs_kayak: float | None
    base_storage_search_seconds_ratio_vs_kayak: float | None
    largest_storage_byte_ratio_vs_kayak: float | None
    largest_storage_build_seconds_ratio_vs_kayak: float | None
    largest_storage_search_seconds_ratio_vs_kayak: float | None
    largest_scale_document_count: int | None
    largest_scale_lancedb_scan_latency_ratio_vs_kayak: float | None

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def build_task_comparison_bundle(
    *,
    task_path: str,
    kayak_exact_path: str,
    lancedb_scan_path: str,
    lancedb_indexed_variance_path: str,
    lancedb_indexed_frozen_path: str,
    scorecard_path: str,
    scorecard: Mapping[str, Any],
    indexed_variance: Mapping[str, Any],
    indexed_frozen: Mapping[str, Any],
    scale_sweep_path: str | None = None,
    scale_sweep: Mapping[str, Any] | None = None,
    storage_compare_path: str | None = None,
    storage_compare: Mapping[str, Any] | None = None,
    storage_scale_path: str | None = None,
    storage_scale: Mapping[str, Any] | None = None,
) -> TaskComparisonBundle:
    kayak_exact = _system_by_name(scorecard, "kayak_exact")
    lancedb_scan = _system_by_name(scorecard, "lancedb_scan")
    lancedb_indexed = _system_by_name(scorecard, "lancedb_ivf_pq_frozen")
    scan_pairwise = _pairwise_by_candidate(scorecard, "lancedb_scan")
    indexed_pairwise = _pairwise_by_candidate(scorecard, "lancedb_ivf_pq_frozen")

    largest_scale_document_count = None
    largest_scale_latency_ratio = None
    if scale_sweep is not None:
        largest_scale_row = _largest_row(scale_sweep)
        largest_scale_document_count = int(largest_scale_row["actual_document_count"])
        largest_scale_latency_ratio = float(
            largest_scale_row["lancedb_scan_latency_ratio_vs_kayak"]
        )

    base_storage_byte_ratio = None
    base_storage_build_ratio = None
    base_storage_search_ratio = None
    largest_storage_byte_ratio = None
    largest_storage_build_ratio = None
    largest_storage_search_ratio = None
    if storage_scale is not None:
        base_storage_row = _base_row(storage_scale)
        largest_storage_row = _largest_row(storage_scale)
        base_storage_byte_ratio = base_storage_row["lancedb_storage_byte_ratio_vs_kayak"]
        base_storage_build_ratio = base_storage_row[
            "lancedb_build_seconds_ratio_vs_kayak"
        ]
        base_storage_search_ratio = base_storage_row[
            "lancedb_search_seconds_ratio_vs_kayak"
        ]
        largest_storage_byte_ratio = largest_storage_row[
            "lancedb_storage_byte_ratio_vs_kayak"
        ]
        largest_storage_build_ratio = largest_storage_row[
            "lancedb_build_seconds_ratio_vs_kayak"
        ]
        largest_storage_search_ratio = largest_storage_row[
            "lancedb_search_seconds_ratio_vs_kayak"
        ]
        if base_storage_byte_ratio is not None:
            base_storage_byte_ratio = float(base_storage_byte_ratio)
        if base_storage_build_ratio is not None:
            base_storage_build_ratio = float(base_storage_build_ratio)
        if base_storage_search_ratio is not None:
            base_storage_search_ratio = float(base_storage_search_ratio)
        if largest_storage_byte_ratio is not None:
            largest_storage_byte_ratio = float(largest_storage_byte_ratio)
        if largest_storage_build_ratio is not None:
            largest_storage_build_ratio = float(largest_storage_build_ratio)
        if largest_storage_search_ratio is not None:
            largest_storage_search_ratio = float(largest_storage_search_ratio)

    storage_compare_latency_ratio = None
    storage_compare_load_seconds = None
    storage_compare_lancedb_search_seconds = None
    storage_compare_kayak_search_seconds = None
    if storage_compare is not None:
        storage_compare_latency_ratio = float(
            storage_compare["lancedb_scan_latency_ratio_vs_kayak_from_lancedb"]
        )
        storage_compare_load_seconds = float(
            storage_compare["kayak_load_from_lancedb_seconds"]
        )
        storage_compare_lancedb_search_seconds = float(
            storage_compare["lancedb_scan"]["mean_search_seconds"]
        )
        storage_compare_kayak_search_seconds = float(
            storage_compare["kayak_exact_from_lancedb"]["mean_search_seconds"]
        )

    return TaskComparisonBundle(
        dataset_id=str(scorecard["dataset_id"]),
        family=str(scorecard["family"]),
        slice_name=str(scorecard["slice_name"]),
        primary_metric=str(scorecard["primary_metric"]),
        task_path=task_path,
        kayak_exact_path=kayak_exact_path,
        lancedb_scan_path=lancedb_scan_path,
        lancedb_indexed_variance_path=lancedb_indexed_variance_path,
        lancedb_indexed_frozen_path=lancedb_indexed_frozen_path,
        scorecard_path=scorecard_path,
        scale_sweep_path=scale_sweep_path,
        storage_compare_path=storage_compare_path,
        storage_scale_path=storage_scale_path,
        kayak_exact_primary_value=float(kayak_exact["primary_value"]),
        kayak_exact_mean_search_seconds=float(kayak_exact["mean_search_seconds"]),
        lancedb_scan_primary_value=float(lancedb_scan["primary_value"]),
        lancedb_scan_mean_search_seconds=float(lancedb_scan["mean_search_seconds"]),
        lancedb_scan_latency_ratio_vs_kayak=float(
            scan_pairwise["mean_search_seconds_ratio_vs_baseline"]
        ),
        lancedb_indexed_frozen_primary_value=float(lancedb_indexed["primary_value"]),
        lancedb_indexed_frozen_mean_search_seconds=float(
            lancedb_indexed["mean_search_seconds"]
        ),
        lancedb_indexed_frozen_latency_ratio_vs_kayak=float(
            indexed_pairwise["mean_search_seconds_ratio_vs_baseline"]
        ),
        indexed_freeze_policy=str(indexed_frozen["freeze_policy"]),
        indexed_rebuild_count=int(indexed_frozen["rebuild_count"]),
        indexed_index_num_partitions=(
            None
            if indexed_frozen.get("index_num_partitions") is None
            else int(indexed_frozen["index_num_partitions"])
        ),
        indexed_index_num_sub_vectors=(
            None
            if indexed_frozen.get("index_num_sub_vectors") is None
            else int(indexed_frozen["index_num_sub_vectors"])
        ),
        indexed_index_target_partition_size=(
            None
            if indexed_frozen.get("index_target_partition_size") is None
            else int(indexed_frozen["index_target_partition_size"])
        ),
        indexed_nprobes=(
            None
            if indexed_frozen.get("indexed_nprobes") is None
            else int(indexed_frozen["indexed_nprobes"])
        ),
        indexed_refine_factor=(
            None
            if indexed_frozen.get("indexed_refine_factor") is None
            else int(indexed_frozen["indexed_refine_factor"])
        ),
        indexed_primary_value_min=float(indexed_variance["primary_value_min"]),
        indexed_primary_value_max=float(indexed_variance["primary_value_max"]),
        indexed_mean_search_seconds_min=float(
            indexed_variance["mean_search_seconds_min"]
        ),
        indexed_mean_search_seconds_max=float(
            indexed_variance["mean_search_seconds_max"]
        ),
        storage_compare_kayak_load_from_lancedb_seconds=storage_compare_load_seconds,
        storage_compare_lancedb_scan_mean_search_seconds=(
            storage_compare_lancedb_search_seconds
        ),
        storage_compare_kayak_exact_from_lancedb_mean_search_seconds=(
            storage_compare_kayak_search_seconds
        ),
        storage_compare_lancedb_scan_latency_ratio_vs_kayak_from_lancedb=(
            storage_compare_latency_ratio
        ),
        base_storage_byte_ratio_vs_kayak=base_storage_byte_ratio,
        base_storage_build_seconds_ratio_vs_kayak=base_storage_build_ratio,
        base_storage_search_seconds_ratio_vs_kayak=base_storage_search_ratio,
        largest_storage_byte_ratio_vs_kayak=largest_storage_byte_ratio,
        largest_storage_build_seconds_ratio_vs_kayak=largest_storage_build_ratio,
        largest_storage_search_seconds_ratio_vs_kayak=largest_storage_search_ratio,
        largest_scale_document_count=largest_scale_document_count,
        largest_scale_lancedb_scan_latency_ratio_vs_kayak=largest_scale_latency_ratio,
    )
