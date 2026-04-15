"""Owns reusable LanceDB indexed-configuration sweeps for one task.

This module does not execute LanceDB itself. It only:
- parses explicit sweep configuration strings into validated controls
- builds a compact machine-readable summary over already-produced artifacts

Execution stays in scripts so benchmarks remain easy to inspect and rerun.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .lancedb_index_controls import (
    LanceDbIndexBuildControls,
    LanceDbIndexedQueryControls,
)


def _optional_float(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def _optional_int(summary: Mapping[str, Any], key: str) -> int | None:
    value = summary.get(key)
    if value is None:
        return None
    return int(value)


def _required_float(summary: Mapping[str, Any], key: str) -> float:
    if key not in summary:
        raise ValueError(f"summary is missing required key: {key}")
    return float(summary[key])


def _require_same_task_shape(
    *,
    summary: Mapping[str, Any],
    dataset_id: str,
    family: str,
    slice_name: str,
    primary_metric: str,
    k: int,
) -> None:
    if str(summary["dataset_id"]) != dataset_id:
        raise ValueError("all sweep summaries must share dataset_id")
    if str(summary["family"]) != family:
        raise ValueError("all sweep summaries must share family")
    if str(summary["slice_name"]) != slice_name:
        raise ValueError("all sweep summaries must share slice_name")
    if str(summary["primary_metric"]) != primary_metric:
        raise ValueError("all sweep summaries must share primary_metric")
    if int(summary["k"]) != k:
        raise ValueError("all sweep summaries must share k")


@dataclass(frozen=True, slots=True)
class LanceDbIndexedSweepSpec:
    name: str
    index_build_controls: LanceDbIndexBuildControls
    indexed_query_controls: LanceDbIndexedQueryControls


@dataclass(frozen=True, slots=True)
class LanceDbIndexedSweepRow:
    config_name: str
    index_num_partitions: int | None
    index_num_sub_vectors: int | None
    index_target_partition_size: int | None
    indexed_nprobes: int | None
    indexed_refine_factor: int | None
    freeze_policy: str
    rebuild_count: int
    primary_value: float
    mean_search_seconds: float
    primary_value_min: float
    primary_value_max: float
    mean_search_seconds_min: float
    mean_search_seconds_max: float
    primary_value_delta_vs_scan: float | None
    mean_search_seconds_ratio_vs_scan: float | None
    primary_value_delta_vs_kayak: float | None
    mean_search_seconds_ratio_vs_kayak: float | None
    variance_path: str
    frozen_path: str


@dataclass(frozen=True, slots=True)
class LanceDbIndexedSweepSummary:
    dataset_id: str
    family: str
    slice_name: str
    primary_metric: str
    k: int
    task_path: str
    kayak_exact_path: str | None
    lancedb_scan_path: str | None
    kayak_exact_primary_value: float | None
    kayak_exact_mean_search_seconds: float | None
    lancedb_scan_primary_value: float | None
    lancedb_scan_mean_search_seconds: float | None
    config_count: int
    rows: tuple[LanceDbIndexedSweepRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def parse_lancedb_indexed_sweep_spec(value: str) -> LanceDbIndexedSweepSpec:
    raw = value.strip()
    if not raw:
        raise ValueError("sweep config must not be empty")

    if ":" in raw:
        name, raw_controls = raw.split(":", 1)
    else:
        name, raw_controls = raw, ""

    name = name.strip()
    if not name:
        raise ValueError("sweep config name must not be empty")

    controls: dict[str, int] = {}
    if raw_controls.strip():
        for assignment in raw_controls.split(","):
            item = assignment.strip()
            if not item:
                raise ValueError("sweep config contains an empty assignment")
            if "=" not in item:
                raise ValueError(
                    "sweep config assignments must use key=value syntax"
                )
            key, raw_number = item.split("=", 1)
            key = key.strip()
            raw_number = raw_number.strip()
            if key in controls:
                raise ValueError(f"sweep config repeats key: {key}")
            if not raw_number:
                raise ValueError(f"sweep config is missing a value for {key}")
            try:
                controls[key] = int(raw_number)
            except ValueError as exc:
                raise ValueError(
                    f"sweep config value for {key} must be an integer"
                ) from exc

    allowed_keys = {
        "index_num_partitions",
        "index_num_sub_vectors",
        "index_target_partition_size",
        "indexed_nprobes",
        "indexed_refine_factor",
    }
    unknown_keys = sorted(set(controls) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            "sweep config contains unknown keys: " + ", ".join(unknown_keys)
        )

    return LanceDbIndexedSweepSpec(
        name=name,
        index_build_controls=LanceDbIndexBuildControls(
            num_partitions=controls.get("index_num_partitions"),
            num_sub_vectors=controls.get("index_num_sub_vectors"),
            target_partition_size=controls.get("index_target_partition_size"),
        ).validated(),
        indexed_query_controls=LanceDbIndexedQueryControls(
            nprobes=controls.get("indexed_nprobes"),
            refine_factor=controls.get("indexed_refine_factor"),
        ).validated(),
    )


def build_lancedb_indexed_sweep_summary(
    *,
    task_path: str,
    kayak_exact_path: str | None,
    kayak_exact: Mapping[str, Any] | None,
    lancedb_scan_path: str | None,
    lancedb_scan: Mapping[str, Any] | None,
    configs: Sequence[tuple[str, str, Mapping[str, Any], str, Mapping[str, Any]]],
) -> LanceDbIndexedSweepSummary:
    if not configs:
        raise ValueError("indexed sweep requires at least one config")

    first_frozen = configs[0][4]
    dataset_id = str(first_frozen["dataset_id"])
    family = str(first_frozen["family"])
    slice_name = str(first_frozen["slice_name"])
    primary_metric = str(first_frozen["primary_metric"])
    k = int(first_frozen["k"])

    if kayak_exact is not None:
        _require_same_task_shape(
            summary=kayak_exact,
            dataset_id=dataset_id,
            family=family,
            slice_name=slice_name,
            primary_metric=primary_metric,
            k=k,
        )
    if lancedb_scan is not None:
        _require_same_task_shape(
            summary=lancedb_scan,
            dataset_id=dataset_id,
            family=family,
            slice_name=slice_name,
            primary_metric=primary_metric,
            k=k,
        )

    kayak_exact_primary_value = (
        None if kayak_exact is None else _required_float(kayak_exact, "primary_value")
    )
    kayak_exact_mean_search_seconds = (
        None
        if kayak_exact is None
        else _required_float(kayak_exact, "mean_search_seconds")
    )
    lancedb_scan_primary_value = (
        None
        if lancedb_scan is None
        else _required_float(lancedb_scan, "primary_value")
    )
    lancedb_scan_mean_search_seconds = (
        None
        if lancedb_scan is None
        else _required_float(lancedb_scan, "mean_search_seconds")
    )

    rows: list[LanceDbIndexedSweepRow] = []
    for config_name, variance_path, variance, frozen_path, frozen in configs:
        _require_same_task_shape(
            summary=variance,
            dataset_id=dataset_id,
            family=family,
            slice_name=slice_name,
            primary_metric=primary_metric,
            k=k,
        )
        _require_same_task_shape(
            summary=frozen,
            dataset_id=dataset_id,
            family=family,
            slice_name=slice_name,
            primary_metric=primary_metric,
            k=k,
        )

        mean_search_seconds = _required_float(frozen, "mean_search_seconds")
        primary_value = _required_float(frozen, "primary_value")

        primary_delta_vs_scan = None
        latency_ratio_vs_scan = None
        if lancedb_scan_primary_value is not None:
            primary_delta_vs_scan = primary_value - lancedb_scan_primary_value
        if (
            lancedb_scan_mean_search_seconds is not None
            and lancedb_scan_mean_search_seconds != 0.0
        ):
            latency_ratio_vs_scan = (
                mean_search_seconds / lancedb_scan_mean_search_seconds
            )

        primary_delta_vs_kayak = None
        latency_ratio_vs_kayak = None
        if kayak_exact_primary_value is not None:
            primary_delta_vs_kayak = primary_value - kayak_exact_primary_value
        if (
            kayak_exact_mean_search_seconds is not None
            and kayak_exact_mean_search_seconds != 0.0
        ):
            latency_ratio_vs_kayak = (
                mean_search_seconds / kayak_exact_mean_search_seconds
            )

        rows.append(
            LanceDbIndexedSweepRow(
                config_name=config_name,
                index_num_partitions=_optional_int(frozen, "index_num_partitions"),
                index_num_sub_vectors=_optional_int(frozen, "index_num_sub_vectors"),
                index_target_partition_size=_optional_int(
                    frozen, "index_target_partition_size"
                ),
                indexed_nprobes=_optional_int(frozen, "indexed_nprobes"),
                indexed_refine_factor=_optional_int(frozen, "indexed_refine_factor"),
                freeze_policy=str(frozen["freeze_policy"]),
                rebuild_count=int(frozen["rebuild_count"]),
                primary_value=primary_value,
                mean_search_seconds=mean_search_seconds,
                primary_value_min=_required_float(variance, "primary_value_min"),
                primary_value_max=_required_float(variance, "primary_value_max"),
                mean_search_seconds_min=_required_float(
                    variance, "mean_search_seconds_min"
                ),
                mean_search_seconds_max=_required_float(
                    variance, "mean_search_seconds_max"
                ),
                primary_value_delta_vs_scan=primary_delta_vs_scan,
                mean_search_seconds_ratio_vs_scan=latency_ratio_vs_scan,
                primary_value_delta_vs_kayak=primary_delta_vs_kayak,
                mean_search_seconds_ratio_vs_kayak=latency_ratio_vs_kayak,
                variance_path=variance_path,
                frozen_path=frozen_path,
            )
        )

    return LanceDbIndexedSweepSummary(
        dataset_id=dataset_id,
        family=family,
        slice_name=slice_name,
        primary_metric=primary_metric,
        k=k,
        task_path=task_path,
        kayak_exact_path=kayak_exact_path,
        lancedb_scan_path=lancedb_scan_path,
        kayak_exact_primary_value=kayak_exact_primary_value,
        kayak_exact_mean_search_seconds=kayak_exact_mean_search_seconds,
        lancedb_scan_primary_value=lancedb_scan_primary_value,
        lancedb_scan_mean_search_seconds=lancedb_scan_mean_search_seconds,
        config_count=len(rows),
        rows=tuple(rows),
    )
