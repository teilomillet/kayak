"""Owns narrow comparison scorecards over benchmark summary artifacts.

This module does not run benchmarks. It only normalizes already-produced
artifact summaries into one explicit scorecard so external comparisons are
versionable and auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _required_value(summary: Mapping[str, Any], key: str) -> Any:
    if key not in summary:
        raise ValueError(f"summary is missing required key: {key}")
    return summary[key]


@dataclass(frozen=True, slots=True)
class ScorecardSystem:
    name: str
    engine: str
    index_kind: str
    primary_value: float
    mean_search_seconds: float
    bytes_per_document: float | None
    bytes_per_vector: float | None
    artifact_path: str


@dataclass(frozen=True, slots=True)
class ScorecardPairwiseComparison:
    baseline: str
    candidate: str
    primary_value_delta: float
    primary_value_ratio: float | None
    mean_search_seconds_ratio_vs_baseline: float
    bytes_per_document_ratio_vs_baseline: float | None


@dataclass(frozen=True, slots=True)
class ComparisonScorecard:
    dataset_id: str
    family: str
    slice_name: str
    primary_metric: str
    k: int
    systems: tuple[ScorecardSystem, ...]
    pairwise: tuple[ScorecardPairwiseComparison, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _optional_float(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def _scorecard_system(
    *,
    name: str,
    artifact_path: str,
    summary: Mapping[str, Any],
) -> ScorecardSystem:
    engine = str(summary.get("engine", "kayak"))
    index_kind = str(summary.get("index_kind", "exact"))
    return ScorecardSystem(
        name=name,
        engine=engine,
        index_kind=index_kind,
        primary_value=float(_required_value(summary, "primary_value")),
        mean_search_seconds=float(_required_value(summary, "mean_search_seconds")),
        bytes_per_document=_optional_float(summary, "bytes_per_document"),
        bytes_per_vector=_optional_float(summary, "bytes_per_vector"),
        artifact_path=artifact_path,
    )


def _pairwise_comparison(
    *,
    baseline: ScorecardSystem,
    candidate: ScorecardSystem,
) -> ScorecardPairwiseComparison:
    primary_value_ratio = None
    if baseline.primary_value != 0.0:
        primary_value_ratio = candidate.primary_value / baseline.primary_value

    bytes_per_document_ratio = None
    if (
        baseline.bytes_per_document is not None
        and candidate.bytes_per_document is not None
        and baseline.bytes_per_document != 0.0
    ):
        bytes_per_document_ratio = (
            candidate.bytes_per_document / baseline.bytes_per_document
        )

    return ScorecardPairwiseComparison(
        baseline=baseline.name,
        candidate=candidate.name,
        primary_value_delta=candidate.primary_value - baseline.primary_value,
        primary_value_ratio=primary_value_ratio,
        mean_search_seconds_ratio_vs_baseline=(
            candidate.mean_search_seconds / baseline.mean_search_seconds
        ),
        bytes_per_document_ratio_vs_baseline=bytes_per_document_ratio,
    )


def build_comparison_scorecard(
    *,
    systems: Sequence[tuple[str, str, Mapping[str, Any]]],
    baseline_name: str,
) -> ComparisonScorecard:
    if not systems:
        raise ValueError("scorecard requires at least one system")

    first_summary = systems[0][2]
    dataset_id = str(_required_value(first_summary, "dataset_id"))
    family = str(_required_value(first_summary, "family"))
    slice_name = str(_required_value(first_summary, "slice_name"))
    primary_metric = str(_required_value(first_summary, "primary_metric"))
    k = int(_required_value(first_summary, "k"))

    scorecard_systems = []
    for name, artifact_path, summary in systems:
        if str(_required_value(summary, "dataset_id")) != dataset_id:
            raise ValueError("all scorecard summaries must share dataset_id")
        if str(_required_value(summary, "family")) != family:
            raise ValueError("all scorecard summaries must share family")
        if str(_required_value(summary, "slice_name")) != slice_name:
            raise ValueError("all scorecard summaries must share slice_name")
        if str(_required_value(summary, "primary_metric")) != primary_metric:
            raise ValueError("all scorecard summaries must share primary_metric")
        if int(_required_value(summary, "k")) != k:
            raise ValueError("all scorecard summaries must share k")

        scorecard_systems.append(
            _scorecard_system(
                name=name,
                artifact_path=artifact_path,
                summary=summary,
            )
        )

    baseline = None
    for system in scorecard_systems:
        if system.name == baseline_name:
            baseline = system
            break
    if baseline is None:
        raise ValueError(f"baseline system not found: {baseline_name}")

    pairwise = []
    for system in scorecard_systems:
        if system.name == baseline.name:
            continue
        pairwise.append(
            _pairwise_comparison(
                baseline=baseline,
                candidate=system,
            )
        )

    return ComparisonScorecard(
        dataset_id=dataset_id,
        family=family,
        slice_name=slice_name,
        primary_metric=primary_metric,
        k=k,
        systems=tuple(scorecard_systems),
        pairwise=tuple(pairwise),
    )
