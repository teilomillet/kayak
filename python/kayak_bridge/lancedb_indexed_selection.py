"""Owns reproducible indexed-candidate selection from LanceDB sweep summaries.

This module turns a broad indexed sweep into a small selected-candidate plan.
The policy is intentionally explicit and auditable:
- include `default` when present
- include the highest-quality indexed config
- include the fastest indexed config that strictly improves on scan quality
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .lancedb_index_sweep import LanceDbIndexedSweepSpec
from .lancedb_index_controls import (
    LanceDbIndexBuildControls,
    LanceDbIndexedQueryControls,
)


def _require_rows(summary: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    rows = summary.get("rows", [])
    if not rows:
        raise ValueError("sweep summary must include at least one row")
    return rows


def _optional_int(summary: Mapping[str, Any], key: str) -> int | None:
    value = summary.get(key)
    if value is None:
        return None
    return int(value)


def _spec_from_row(row: Mapping[str, Any]) -> LanceDbIndexedSweepSpec:
    return LanceDbIndexedSweepSpec(
        name=str(row["config_name"]),
        index_build_controls=LanceDbIndexBuildControls(
            num_partitions=_optional_int(row, "index_num_partitions"),
            num_sub_vectors=_optional_int(row, "index_num_sub_vectors"),
            target_partition_size=_optional_int(row, "index_target_partition_size"),
        ).validated(),
        indexed_query_controls=LanceDbIndexedQueryControls(
            nprobes=_optional_int(row, "indexed_nprobes"),
            refine_factor=_optional_int(row, "indexed_refine_factor"),
        ).validated(),
    )


@dataclass(frozen=True, slots=True)
class LanceDbIndexedSelectionRow:
    role: str
    config_name: str
    primary_value: float
    mean_search_seconds: float
    primary_value_delta_vs_scan: float | None
    index_num_partitions: int | None
    index_num_sub_vectors: int | None
    index_target_partition_size: int | None
    indexed_nprobes: int | None
    indexed_refine_factor: int | None


@dataclass(frozen=True, slots=True)
class LanceDbIndexedSelectionSummary:
    dataset_id: str
    family: str
    slice_name: str
    primary_metric: str
    selection_policy: str
    selected_count: int
    rows: tuple[LanceDbIndexedSelectionRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _row_to_selection(role: str, row: Mapping[str, Any]) -> LanceDbIndexedSelectionRow:
    return LanceDbIndexedSelectionRow(
        role=role,
        config_name=str(row["config_name"]),
        primary_value=float(row["primary_value"]),
        mean_search_seconds=float(row["mean_search_seconds"]),
        primary_value_delta_vs_scan=(
            None
            if row.get("primary_value_delta_vs_scan") is None
            else float(row["primary_value_delta_vs_scan"])
        ),
        index_num_partitions=_optional_int(row, "index_num_partitions"),
        index_num_sub_vectors=_optional_int(row, "index_num_sub_vectors"),
        index_target_partition_size=_optional_int(
            row, "index_target_partition_size"
        ),
        indexed_nprobes=_optional_int(row, "indexed_nprobes"),
        indexed_refine_factor=_optional_int(row, "indexed_refine_factor"),
    )


def select_lancedb_indexed_candidates_from_sweep_summary(
    sweep_summary: Mapping[str, Any],
) -> tuple[LanceDbIndexedSelectionSummary, tuple[LanceDbIndexedSweepSpec, ...]]:
    rows = list(_require_rows(sweep_summary))
    selected_rows: list[LanceDbIndexedSelectionRow] = []
    selected_specs: list[LanceDbIndexedSweepSpec] = []
    selected_names: set[str] = set()

    def include(role: str, row: Mapping[str, Any]) -> None:
        config_name = str(row["config_name"])
        if config_name in selected_names:
            return
        selected_names.add(config_name)
        selected_rows.append(_row_to_selection(role, row))
        selected_specs.append(_spec_from_row(row))

    default_row = None
    for row in rows:
        if str(row["config_name"]) == "default":
            default_row = row
            break
    if default_row is not None:
        include("default", default_row)

    best_quality_row = min(
        rows,
        key=lambda row: (
            -float(row["primary_value"]),
            float(row["mean_search_seconds"]),
            str(row["config_name"]),
        ),
    )
    include("best_quality", best_quality_row)

    quality_improving_rows = [
        row
        for row in rows
        if row.get("primary_value_delta_vs_scan") is not None
        and float(row["primary_value_delta_vs_scan"]) > 0.0
    ]
    if quality_improving_rows:
        fastest_improving_row = min(
            quality_improving_rows,
            key=lambda row: (
                float(row["mean_search_seconds"]),
                -float(row["primary_value"]),
                str(row["config_name"]),
            ),
        )
        include("fastest_quality_improving", fastest_improving_row)

    summary = LanceDbIndexedSelectionSummary(
        dataset_id=str(sweep_summary["dataset_id"]),
        family=str(sweep_summary["family"]),
        slice_name=str(sweep_summary["slice_name"]),
        primary_metric=str(sweep_summary["primary_metric"]),
        selection_policy=(
            "include_default_plus_best_quality_plus_fastest_quality_improving"
        ),
        selected_count=len(selected_rows),
        rows=tuple(selected_rows),
    )
    return summary, tuple(selected_specs)
