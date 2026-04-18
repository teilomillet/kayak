"""Owns compact candidate bundles over reference LEMUR sweep artifacts.

This bundle is for the "selected LEMUR candidates" workflow:
- keep the exact baseline as a stable anchor
- compare explicit `(latent_dim, candidate_k)` settings side by side
- surface the mechanically best quality and speed tradeoffs from one sweep
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _candidate_name(row: Mapping[str, Any]) -> str:
    return (
        "lemur"
        f"_latent{int(row['latent_dim'])}"
        f"_k{int(row['candidate_k'])}"
        f"_landmarks{int(row['landmark_count'])}"
    )


@dataclass(frozen=True, slots=True)
class LemurCandidateBundleRow:
    name: str
    latent_dim: int
    candidate_k: int
    landmark_count: int
    fit_seconds: float
    primary_value: float
    mean_search_seconds: float
    mean_exact_topk_shortlist_recall: float
    primary_ratio_vs_exact: float | None
    mean_search_seconds_ratio_vs_exact: float | None


@dataclass(frozen=True, slots=True)
class LemurCandidateBundle:
    dataset_id: str
    family: str
    slice_name: str
    primary_metric: str
    exact_path: str
    sweep_path: str
    candidate_count: int
    exact_primary_value: float
    exact_mean_search_seconds: float
    best_quality_candidate_name: str
    best_quality_candidate_primary_value: float
    best_quality_candidate_mean_search_seconds: float
    fastest_full_shortlist_recall_candidate_name: str | None
    fastest_full_shortlist_recall_candidate_primary_value: float | None
    fastest_full_shortlist_recall_candidate_mean_search_seconds: float | None
    fastest_no_primary_regression_candidate_name: str | None
    fastest_no_primary_regression_candidate_primary_value: float | None
    fastest_no_primary_regression_candidate_mean_search_seconds: float | None
    rows: tuple[LemurCandidateBundleRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _best_quality_row(
    rows: Sequence[LemurCandidateBundleRow],
) -> LemurCandidateBundleRow:
    return min(
        rows,
        key=lambda row: (
            -row.primary_value,
            row.mean_search_seconds,
            row.candidate_k,
            row.latent_dim,
            row.name,
        ),
    )


def _fastest_matching_row(
    rows: Sequence[LemurCandidateBundleRow],
    *,
    minimum_primary_value: float | None = None,
    minimum_shortlist_recall: float | None = None,
) -> LemurCandidateBundleRow | None:
    eligible = []
    for row in rows:
        if (
            minimum_primary_value is not None
            and row.primary_value < minimum_primary_value
        ):
            continue
        if (
            minimum_shortlist_recall is not None
            and row.mean_exact_topk_shortlist_recall < minimum_shortlist_recall
        ):
            continue
        eligible.append(row)

    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            row.mean_search_seconds,
            -row.primary_value,
            -row.mean_exact_topk_shortlist_recall,
            row.candidate_k,
            row.latent_dim,
            row.name,
        ),
    )


def build_lemur_candidate_bundle(
    *,
    exact_path: str,
    sweep_path: str,
    sweep_summary: Mapping[str, Any],
) -> LemurCandidateBundle:
    raw_rows = sweep_summary.get("rows", [])
    if not raw_rows:
        raise ValueError("sweep summary rows must not be empty")

    rows = tuple(
        LemurCandidateBundleRow(
            name=_candidate_name(row),
            latent_dim=int(row["latent_dim"]),
            candidate_k=int(row["candidate_k"]),
            landmark_count=int(row["landmark_count"]),
            fit_seconds=float(row["fit_seconds"]),
            primary_value=float(row["primary_value"]),
            mean_search_seconds=float(row["mean_search_seconds"]),
            mean_exact_topk_shortlist_recall=float(
                row["mean_exact_topk_shortlist_recall"]
            ),
            primary_ratio_vs_exact=(
                None
                if row.get("primary_ratio_vs_exact") is None
                else float(row["primary_ratio_vs_exact"])
            ),
            mean_search_seconds_ratio_vs_exact=(
                None
                if row.get("mean_search_seconds_ratio_vs_exact") is None
                else float(row["mean_search_seconds_ratio_vs_exact"])
            ),
        )
        for row in raw_rows
    )

    best_quality = _best_quality_row(rows)
    fastest_full_shortlist_recall = _fastest_matching_row(
        rows,
        minimum_shortlist_recall=1.0,
    )
    fastest_no_primary_regression = _fastest_matching_row(
        rows,
        minimum_primary_value=float(sweep_summary["exact_primary_value"]),
    )

    return LemurCandidateBundle(
        dataset_id=str(sweep_summary["dataset_id"]),
        family=str(sweep_summary["family"]),
        slice_name=str(sweep_summary["slice_name"]),
        primary_metric=str(sweep_summary["primary_metric"]),
        exact_path=exact_path,
        sweep_path=sweep_path,
        candidate_count=len(rows),
        exact_primary_value=float(sweep_summary["exact_primary_value"]),
        exact_mean_search_seconds=float(sweep_summary["exact_mean_search_seconds"]),
        best_quality_candidate_name=best_quality.name,
        best_quality_candidate_primary_value=best_quality.primary_value,
        best_quality_candidate_mean_search_seconds=best_quality.mean_search_seconds,
        fastest_full_shortlist_recall_candidate_name=(
            None
            if fastest_full_shortlist_recall is None
            else fastest_full_shortlist_recall.name
        ),
        fastest_full_shortlist_recall_candidate_primary_value=(
            None
            if fastest_full_shortlist_recall is None
            else fastest_full_shortlist_recall.primary_value
        ),
        fastest_full_shortlist_recall_candidate_mean_search_seconds=(
            None
            if fastest_full_shortlist_recall is None
            else fastest_full_shortlist_recall.mean_search_seconds
        ),
        fastest_no_primary_regression_candidate_name=(
            None
            if fastest_no_primary_regression is None
            else fastest_no_primary_regression.name
        ),
        fastest_no_primary_regression_candidate_primary_value=(
            None
            if fastest_no_primary_regression is None
            else fastest_no_primary_regression.primary_value
        ),
        fastest_no_primary_regression_candidate_mean_search_seconds=(
            None
            if fastest_no_primary_regression is None
            else fastest_no_primary_regression.mean_search_seconds
        ),
        rows=rows,
    )
