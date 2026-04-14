"""Owns fixed-document-count vector-density sweeps for Kayak versus LanceDB."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .kayak_task_benchmark import benchmark_task_with_kayak_exact
from .lancedb_benchmark import (
    _filter_zero_vectors,
    benchmark_task_with_lancedb,
)
from .task_vector_scale import build_vector_scaled_task


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class VectorDensityComparisonRow:
    document_vector_multiplier: int
    document_count: int
    stored_document_vector_count_total: int
    stored_document_vector_count_mean: int
    kayak_exact_primary_value: float
    kayak_exact_mean_search_seconds: float
    lancedb_scan_primary_value: float
    lancedb_scan_mean_search_seconds: float
    lancedb_scan_latency_ratio_vs_kayak: float | None
    lancedb_scan_primary_ratio_vs_kayak: float | None


@dataclass(frozen=True, slots=True)
class VectorDensityComparisonSummary:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    primary_metric: str
    k: int
    document_count: int
    base_nominal_document_vector_count: int
    base_zero_document_vector_count_filtered: int
    base_zero_query_vector_count_filtered: int
    inflation_policy: str
    kayak_exact_engine: str
    kayak_exact_backend: str
    lancedb_engine: str
    lancedb_engine_version: str
    rows: tuple[VectorDensityComparisonRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_vector_density_sweep(
    task: Mapping[str, Any],
    *,
    database_root: Path,
    table_prefix: str,
    document_vector_multipliers: Sequence[int],
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
) -> VectorDensityComparisonSummary:
    if not document_vector_multipliers:
        raise ValueError("document_vector_multipliers must not be empty")

    (
        filtered_task,
        zero_document_vector_count,
        zero_query_vector_count,
        _stored_document_vector_count_total,
        _stored_query_vector_count_total,
    ) = _filter_zero_vectors(task)

    rows: list[VectorDensityComparisonRow] = []
    lancedb_engine_version = ""
    inflation_policy = "repeat_document_vectors_in_place"
    for multiplier in sorted({int(value) for value in document_vector_multipliers}):
        scaled = build_vector_scaled_task(
            filtered_task,
            document_vector_multiplier=multiplier,
            inflation_policy=inflation_policy,
        )
        scaled_task = scaled.task
        exact_summary = benchmark_task_with_kayak_exact(
            scaled_task,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
        lancedb_summary = benchmark_task_with_lancedb(
            task=scaled_task,
            database_root=database_root / f"vectors_x{multiplier}",
            table_name=f"{table_prefix}_vectors_x{multiplier}",
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            build_index=False,
        )
        lancedb_engine_version = lancedb_summary.engine_version
        rows.append(
            VectorDensityComparisonRow(
                document_vector_multiplier=multiplier,
                document_count=exact_summary.document_count,
                stored_document_vector_count_total=(
                    exact_summary.document_vector_count_total
                ),
                stored_document_vector_count_mean=(
                    exact_summary.document_vector_count_mean
                ),
                kayak_exact_primary_value=exact_summary.primary_value,
                kayak_exact_mean_search_seconds=exact_summary.mean_search_seconds,
                lancedb_scan_primary_value=lancedb_summary.primary_value,
                lancedb_scan_mean_search_seconds=lancedb_summary.mean_search_seconds,
                lancedb_scan_latency_ratio_vs_kayak=_safe_ratio(
                    lancedb_summary.mean_search_seconds,
                    exact_summary.mean_search_seconds,
                ),
                lancedb_scan_primary_ratio_vs_kayak=_safe_ratio(
                    lancedb_summary.primary_value,
                    exact_summary.primary_value,
                ),
            )
        )

    return VectorDensityComparisonSummary(
        dataset_id=str(filtered_task["dataset_id"]),
        model_name=str(filtered_task["model_name"]),
        family=str(filtered_task["family"]),
        slice_name=str(filtered_task["slice_name"]),
        primary_metric=str(filtered_task["primary_metric"]),
        k=int(filtered_task["k"]),
        document_count=len(filtered_task["documents"]),
        base_nominal_document_vector_count=int(
            filtered_task["nominal_document_vector_count"]
        ),
        base_zero_document_vector_count_filtered=zero_document_vector_count,
        base_zero_query_vector_count_filtered=zero_query_vector_count,
        inflation_policy=inflation_policy,
        kayak_exact_engine="kayak",
        kayak_exact_backend="mojo_exact_cpu",
        lancedb_engine="lancedb",
        lancedb_engine_version=lancedb_engine_version,
        rows=tuple(rows),
    )
