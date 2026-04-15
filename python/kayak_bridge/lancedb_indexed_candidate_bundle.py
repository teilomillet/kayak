"""Owns compact multi-candidate LanceDB indexed comparison bundles.

This bundle is for the "selected indexed candidates" workflow:
- keep Kayak exact and LanceDB scan as stable anchors
- compare several explicit LanceDB indexed configurations side by side
- surface the mechanically best indexed candidates by quality and by latency
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


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


@dataclass(frozen=True, slots=True)
class LanceDbIndexedCandidateRow:
    name: str
    artifact_path: str
    freeze_policy: str
    rebuild_count: int
    primary_value: float
    mean_search_seconds: float
    primary_value_delta_vs_scan: float
    mean_search_seconds_ratio_vs_scan: float
    mean_search_seconds_ratio_vs_kayak: float
    index_num_partitions: int | None
    index_num_sub_vectors: int | None
    index_target_partition_size: int | None
    indexed_nprobes: int | None
    indexed_refine_factor: int | None


@dataclass(frozen=True, slots=True)
class LanceDbIndexedCandidateBundle:
    dataset_id: str
    family: str
    slice_name: str
    primary_metric: str
    task_path: str
    kayak_exact_path: str
    lancedb_scan_path: str
    scorecard_path: str
    candidate_count: int
    kayak_exact_primary_value: float
    kayak_exact_mean_search_seconds: float
    lancedb_scan_primary_value: float
    lancedb_scan_mean_search_seconds: float
    best_quality_candidate_name: str
    best_quality_candidate_primary_value: float
    best_quality_candidate_mean_search_seconds: float
    fastest_non_regressing_candidate_name: str | None
    fastest_non_regressing_candidate_primary_value: float | None
    fastest_non_regressing_candidate_mean_search_seconds: float | None
    fastest_quality_improving_candidate_name: str | None
    fastest_quality_improving_candidate_primary_value: float | None
    fastest_quality_improving_candidate_mean_search_seconds: float | None
    rows: tuple[LanceDbIndexedCandidateRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _optional_int(summary: Mapping[str, Any], key: str) -> int | None:
    value = summary.get(key)
    if value is None:
        return None
    return int(value)


def _best_quality_row(
    rows: Sequence[LanceDbIndexedCandidateRow],
) -> LanceDbIndexedCandidateRow:
    return min(rows, key=lambda row: (-row.primary_value, row.mean_search_seconds, row.name))


def _fastest_matching_row(
    rows: Sequence[LanceDbIndexedCandidateRow],
    *,
    scan_primary_value: float,
    strict_improvement: bool,
) -> LanceDbIndexedCandidateRow | None:
    eligible = []
    for row in rows:
        if strict_improvement:
            if row.primary_value > scan_primary_value:
                eligible.append(row)
        else:
            if row.primary_value >= scan_primary_value:
                eligible.append(row)
    if not eligible:
        return None
    return min(eligible, key=lambda row: (row.mean_search_seconds, -row.primary_value, row.name))


def build_lancedb_indexed_candidate_bundle(
    *,
    task_path: str,
    kayak_exact_path: str,
    lancedb_scan_path: str,
    scorecard_path: str,
    scorecard: Mapping[str, Any],
    candidates: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> LanceDbIndexedCandidateBundle:
    if not candidates:
        raise ValueError("candidate bundle requires at least one indexed candidate")

    kayak_exact = _system_by_name(scorecard, "kayak_exact")
    lancedb_scan = _system_by_name(scorecard, "lancedb_scan")
    scan_primary_value = float(lancedb_scan["primary_value"])

    rows: list[LanceDbIndexedCandidateRow] = []
    for system_name, artifact_path, frozen_summary in candidates:
        system = _system_by_name(scorecard, system_name)
        pairwise = _pairwise_by_candidate(scorecard, system_name)
        rows.append(
            LanceDbIndexedCandidateRow(
                name=system_name,
                artifact_path=artifact_path,
                freeze_policy=str(frozen_summary["freeze_policy"]),
                rebuild_count=int(frozen_summary["rebuild_count"]),
                primary_value=float(system["primary_value"]),
                mean_search_seconds=float(system["mean_search_seconds"]),
                primary_value_delta_vs_scan=float(system["primary_value"])
                - scan_primary_value,
                mean_search_seconds_ratio_vs_scan=float(system["mean_search_seconds"])
                / float(lancedb_scan["mean_search_seconds"]),
                mean_search_seconds_ratio_vs_kayak=float(
                    pairwise["mean_search_seconds_ratio_vs_baseline"]
                ),
                index_num_partitions=_optional_int(
                    frozen_summary, "index_num_partitions"
                ),
                index_num_sub_vectors=_optional_int(
                    frozen_summary, "index_num_sub_vectors"
                ),
                index_target_partition_size=_optional_int(
                    frozen_summary, "index_target_partition_size"
                ),
                indexed_nprobes=_optional_int(frozen_summary, "indexed_nprobes"),
                indexed_refine_factor=_optional_int(
                    frozen_summary, "indexed_refine_factor"
                ),
            )
        )

    best_quality = _best_quality_row(rows)
    fastest_non_regressing = _fastest_matching_row(
        rows,
        scan_primary_value=scan_primary_value,
        strict_improvement=False,
    )
    fastest_quality_improving = _fastest_matching_row(
        rows,
        scan_primary_value=scan_primary_value,
        strict_improvement=True,
    )

    return LanceDbIndexedCandidateBundle(
        dataset_id=str(scorecard["dataset_id"]),
        family=str(scorecard["family"]),
        slice_name=str(scorecard["slice_name"]),
        primary_metric=str(scorecard["primary_metric"]),
        task_path=task_path,
        kayak_exact_path=kayak_exact_path,
        lancedb_scan_path=lancedb_scan_path,
        scorecard_path=scorecard_path,
        candidate_count=len(rows),
        kayak_exact_primary_value=float(kayak_exact["primary_value"]),
        kayak_exact_mean_search_seconds=float(kayak_exact["mean_search_seconds"]),
        lancedb_scan_primary_value=scan_primary_value,
        lancedb_scan_mean_search_seconds=float(lancedb_scan["mean_search_seconds"]),
        best_quality_candidate_name=best_quality.name,
        best_quality_candidate_primary_value=best_quality.primary_value,
        best_quality_candidate_mean_search_seconds=best_quality.mean_search_seconds,
        fastest_non_regressing_candidate_name=(
            None if fastest_non_regressing is None else fastest_non_regressing.name
        ),
        fastest_non_regressing_candidate_primary_value=(
            None
            if fastest_non_regressing is None
            else fastest_non_regressing.primary_value
        ),
        fastest_non_regressing_candidate_mean_search_seconds=(
            None
            if fastest_non_regressing is None
            else fastest_non_regressing.mean_search_seconds
        ),
        fastest_quality_improving_candidate_name=(
            None
            if fastest_quality_improving is None
            else fastest_quality_improving.name
        ),
        fastest_quality_improving_candidate_primary_value=(
            None
            if fastest_quality_improving is None
            else fastest_quality_improving.primary_value
        ),
        fastest_quality_improving_candidate_mean_search_seconds=(
            None
            if fastest_quality_improving is None
            else fastest_quality_improving.mean_search_seconds
        ),
        rows=tuple(rows),
    )
