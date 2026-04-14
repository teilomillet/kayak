"""Owns same-task scale sweeps for Kayak exact versus LanceDB scan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .kayak_task_benchmark import (
    benchmark_task_with_kayak_exact,
    rank_task_with_kayak_exact,
)
from .lancedb_benchmark import benchmark_task_with_lancedb
from .task_scale import (
    build_scaled_task_with_document_copies,
    choose_repeatable_distractor_doc_ids,
    protected_doc_ids_for_scale_sweep,
)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class ScaleComparisonRow:
    target_document_count: int
    actual_document_count: int
    scale_factor_vs_base: float
    duplicated_document_count: int
    kayak_exact_primary_value: float
    kayak_exact_mean_search_seconds: float
    lancedb_scan_primary_value: float
    lancedb_scan_mean_search_seconds: float
    lancedb_scan_latency_ratio_vs_kayak: float | None
    lancedb_scan_primary_ratio_vs_kayak: float | None


@dataclass(frozen=True, slots=True)
class ScaleComparisonSummary:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    primary_metric: str
    k: int
    base_document_count: int
    base_nominal_document_vector_count: int
    protected_document_count: int
    repeatable_distractor_source_count: int
    inflation_policy: str
    kayak_exact_engine: str
    kayak_exact_backend: str
    lancedb_engine: str
    lancedb_engine_version: str
    rows: tuple[ScaleComparisonRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_same_task_scale_sweep(
    task: Mapping[str, Any],
    *,
    database_root: Path,
    table_prefix: str,
    target_document_counts: Sequence[int],
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
) -> ScaleComparisonSummary:
    if not target_document_counts:
        raise ValueError("target_document_counts must not be empty")

    base_ranked_doc_ids_by_query = rank_task_with_kayak_exact(task)
    protected_doc_ids = protected_doc_ids_for_scale_sweep(
        task,
        ranked_doc_ids_by_query=base_ranked_doc_ids_by_query,
    )
    repeatable_doc_ids = choose_repeatable_distractor_doc_ids(
        task,
        protected_doc_ids=protected_doc_ids,
    )

    rows: list[ScaleComparisonRow] = []
    sorted_target_document_counts = sorted(
        {int(target_document_count) for target_document_count in target_document_counts}
    )
    base_document_count = len(task["documents"])
    inflation_policy = "repeat_nonprotected_documents_with_unique_doc_ids"
    lancedb_engine_version = ""

    for target_document_count in sorted_target_document_counts:
        scaled = build_scaled_task_with_document_copies(
            task,
            target_document_count=target_document_count,
            repeatable_doc_ids=repeatable_doc_ids,
            inflation_policy=inflation_policy,
        )

        exact_summary = benchmark_task_with_kayak_exact(
            scaled.task,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
        lancedb_summary = benchmark_task_with_lancedb(
            task=scaled.task,
            database_root=database_root / f"scan_docs_{target_document_count}",
            table_name=f"{table_prefix}_{target_document_count}",
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            build_index=False,
        )
        lancedb_engine_version = lancedb_summary.engine_version

        rows.append(
            ScaleComparisonRow(
                target_document_count=target_document_count,
                actual_document_count=exact_summary.document_count,
                scale_factor_vs_base=float(exact_summary.document_count)
                / float(base_document_count),
                duplicated_document_count=scaled.duplicated_document_count,
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

    return ScaleComparisonSummary(
        dataset_id=str(task["dataset_id"]),
        model_name=str(task["model_name"]),
        family=str(task["family"]),
        slice_name=str(task["slice_name"]),
        primary_metric=str(task["primary_metric"]),
        k=int(task["k"]),
        base_document_count=base_document_count,
        base_nominal_document_vector_count=int(task["nominal_document_vector_count"]),
        protected_document_count=len(protected_doc_ids),
        repeatable_distractor_source_count=len(repeatable_doc_ids),
        inflation_policy=inflation_policy,
        kayak_exact_engine="kayak",
        kayak_exact_backend="mojo_exact_cpu",
        lancedb_engine="lancedb",
        lancedb_engine_version=lancedb_engine_version,
        rows=tuple(rows),
    )
