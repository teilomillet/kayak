"""Owns compact aggregate summaries for multi-slice LanceDB hard-matrix runs.

This module only aggregates already-produced per-slice artifacts.
It does not execute benchmarks itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _candidate_row(
    bundle: Mapping[str, Any], candidate_name: str
) -> Mapping[str, Any]:
    for row in bundle["rows"]:
        if str(row["name"]) == candidate_name:
            return row
    raise ValueError(f"candidate row not found: {candidate_name}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass(frozen=True, slots=True)
class LanceDbIndexedMatrixSliceRow:
    dataset_key: str
    dataset_id: str
    family: str
    slice_name: str
    primary_metric: str
    task_path: str
    sweep_summary_path: str
    candidate_selection_path: str
    candidate_bundle_path: str
    query_count: int
    document_count: int
    k: int
    nominal_query_vector_count: int
    nominal_document_vector_count: int
    sweep_config_count: int
    selected_candidate_count: int
    kayak_exact_primary_value: float
    kayak_exact_mean_search_seconds: float
    lancedb_scan_primary_value: float
    lancedb_scan_mean_search_seconds: float
    best_quality_candidate_name: str
    best_quality_candidate_primary_value: float
    best_quality_candidate_mean_search_seconds: float
    best_quality_primary_value_delta_vs_scan: float
    best_quality_mean_search_seconds_ratio_vs_scan: float
    fastest_quality_improving_candidate_name: str | None
    fastest_quality_improving_candidate_primary_value: float | None
    fastest_quality_improving_candidate_mean_search_seconds: float | None
    fastest_quality_improving_primary_value_delta_vs_scan: float | None
    fastest_quality_improving_mean_search_seconds_ratio_vs_scan: float | None


@dataclass(frozen=True, slots=True)
class LanceDbIndexedMatrixSummary:
    matrix_id: str
    selection_policy: str
    sweep_config_names: tuple[str, ...]
    dataset_keys: tuple[str, ...]
    slice_count: int
    total_queries: int
    total_documents: int
    rows: tuple[LanceDbIndexedMatrixSliceRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def build_lancedb_indexed_matrix_summary(
    *,
    matrix_id: str,
    selection_policy: str,
    sweep_config_names: Sequence[str],
    slices: Sequence[
        tuple[
            str,
            str,
            Mapping[str, Any],
            str,
            Mapping[str, Any],
            str,
            Mapping[str, Any],
            str,
            Mapping[str, Any],
        ]
    ],
) -> LanceDbIndexedMatrixSummary:
    if not slices:
        raise ValueError("matrix summary requires at least one slice")

    rows: list[LanceDbIndexedMatrixSliceRow] = []
    dataset_keys: list[str] = []
    total_queries = 0
    total_documents = 0

    for (
        dataset_key,
        task_path,
        task,
        sweep_summary_path,
        sweep_summary,
        candidate_selection_path,
        candidate_selection,
        candidate_bundle_path,
        candidate_bundle,
    ) in slices:
        if str(sweep_summary["dataset_id"]) != str(candidate_bundle["dataset_id"]):
            raise ValueError("sweep summary and candidate bundle dataset_id must match")
        if str(sweep_summary["slice_name"]) != str(candidate_bundle["slice_name"]):
            raise ValueError("sweep summary and candidate bundle slice_name must match")
        if str(sweep_summary["primary_metric"]) != str(
            candidate_bundle["primary_metric"]
        ):
            raise ValueError(
                "sweep summary and candidate bundle primary_metric must match"
            )

        best_quality_name = str(candidate_bundle["best_quality_candidate_name"])
        best_quality_row = _candidate_row(candidate_bundle, best_quality_name)

        fastest_quality_improving_name = candidate_bundle.get(
            "fastest_quality_improving_candidate_name"
        )
        fastest_quality_improving_row = None
        if fastest_quality_improving_name is not None:
            fastest_quality_improving_row = _candidate_row(
                candidate_bundle,
                str(fastest_quality_improving_name),
            )

        query_count = len(task["queries"])
        document_count = len(task["documents"])
        total_queries += query_count
        total_documents += document_count
        dataset_keys.append(dataset_key)

        rows.append(
            LanceDbIndexedMatrixSliceRow(
                dataset_key=dataset_key,
                dataset_id=str(candidate_bundle["dataset_id"]),
                family=str(candidate_bundle["family"]),
                slice_name=str(candidate_bundle["slice_name"]),
                primary_metric=str(candidate_bundle["primary_metric"]),
                task_path=task_path,
                sweep_summary_path=sweep_summary_path,
                candidate_selection_path=candidate_selection_path,
                candidate_bundle_path=candidate_bundle_path,
                query_count=query_count,
                document_count=document_count,
                k=int(task["k"]),
                nominal_query_vector_count=int(task["nominal_query_vector_count"]),
                nominal_document_vector_count=int(
                    task["nominal_document_vector_count"]
                ),
                sweep_config_count=int(sweep_summary["config_count"]),
                selected_candidate_count=int(candidate_selection["selected_count"]),
                kayak_exact_primary_value=float(
                    candidate_bundle["kayak_exact_primary_value"]
                ),
                kayak_exact_mean_search_seconds=float(
                    candidate_bundle["kayak_exact_mean_search_seconds"]
                ),
                lancedb_scan_primary_value=float(
                    candidate_bundle["lancedb_scan_primary_value"]
                ),
                lancedb_scan_mean_search_seconds=float(
                    candidate_bundle["lancedb_scan_mean_search_seconds"]
                ),
                best_quality_candidate_name=best_quality_name,
                best_quality_candidate_primary_value=float(
                    best_quality_row["primary_value"]
                ),
                best_quality_candidate_mean_search_seconds=float(
                    best_quality_row["mean_search_seconds"]
                ),
                best_quality_primary_value_delta_vs_scan=float(
                    best_quality_row["primary_value_delta_vs_scan"]
                ),
                best_quality_mean_search_seconds_ratio_vs_scan=float(
                    best_quality_row["mean_search_seconds_ratio_vs_scan"]
                ),
                fastest_quality_improving_candidate_name=(
                    None
                    if fastest_quality_improving_name is None
                    else str(fastest_quality_improving_name)
                ),
                fastest_quality_improving_candidate_primary_value=(
                    None
                    if fastest_quality_improving_row is None
                    else float(fastest_quality_improving_row["primary_value"])
                ),
                fastest_quality_improving_candidate_mean_search_seconds=(
                    None
                    if fastest_quality_improving_row is None
                    else float(fastest_quality_improving_row["mean_search_seconds"])
                ),
                fastest_quality_improving_primary_value_delta_vs_scan=(
                    None
                    if fastest_quality_improving_row is None
                    else float(fastest_quality_improving_row["primary_value_delta_vs_scan"])
                ),
                fastest_quality_improving_mean_search_seconds_ratio_vs_scan=(
                    None
                    if fastest_quality_improving_row is None
                    else float(
                        fastest_quality_improving_row[
                            "mean_search_seconds_ratio_vs_scan"
                        ]
                    )
                ),
            )
        )

    return LanceDbIndexedMatrixSummary(
        matrix_id=matrix_id,
        selection_policy=selection_policy,
        sweep_config_names=tuple(str(name) for name in sweep_config_names),
        dataset_keys=tuple(dataset_keys),
        slice_count=len(rows),
        total_queries=total_queries,
        total_documents=total_documents,
        rows=tuple(rows),
    )
