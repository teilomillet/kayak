"""Owns a compact bundle for the frozen LanceDB Lane A comparison surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class LaneSliceBundle:
    dataset_id: str
    slice_name: str
    primary_metric: str
    kayak_exact_primary_value: float
    lancedb_scan_primary_value: float
    lancedb_scan_latency_ratio_vs_kayak: float
    lancedb_indexed_frozen_primary_value: float
    lancedb_indexed_frozen_latency_ratio_vs_kayak: float
    indexed_freeze_policy: str
    indexed_rebuild_count: int
    indexed_primary_value_min: float
    indexed_primary_value_max: float
    indexed_mean_search_seconds_min: float
    indexed_mean_search_seconds_max: float
    scorecard_path: str
    indexed_variance_path: str
    indexed_frozen_path: str


@dataclass(frozen=True, slots=True)
class LanceDbLaneABundle:
    lane_id: str
    indexed_policy: str
    gold: LaneSliceBundle
    evidence: LaneSliceBundle

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _build_slice_bundle(
    *,
    scorecard_path: str,
    scorecard: Mapping[str, Any],
    indexed_variance_path: str,
    indexed_variance: Mapping[str, Any],
    indexed_frozen_path: str,
    indexed_frozen: Mapping[str, Any],
) -> LaneSliceBundle:
    systems_by_name = {system["name"]: system for system in scorecard["systems"]}
    pairwise_by_candidate = {
        pair["candidate"]: pair for pair in scorecard["pairwise"]
    }

    kayak_exact = systems_by_name["kayak_exact"]
    lancedb_scan = systems_by_name["lancedb_scan"]
    lancedb_indexed = systems_by_name["lancedb_ivf_pq_frozen"]
    indexed_pairwise = pairwise_by_candidate["lancedb_ivf_pq_frozen"]
    scan_pairwise = pairwise_by_candidate["lancedb_scan"]

    return LaneSliceBundle(
        dataset_id=str(scorecard["dataset_id"]),
        slice_name=str(scorecard["slice_name"]),
        primary_metric=str(scorecard["primary_metric"]),
        kayak_exact_primary_value=float(kayak_exact["primary_value"]),
        lancedb_scan_primary_value=float(lancedb_scan["primary_value"]),
        lancedb_scan_latency_ratio_vs_kayak=float(
            scan_pairwise["mean_search_seconds_ratio_vs_baseline"]
        ),
        lancedb_indexed_frozen_primary_value=float(lancedb_indexed["primary_value"]),
        lancedb_indexed_frozen_latency_ratio_vs_kayak=float(
            indexed_pairwise["mean_search_seconds_ratio_vs_baseline"]
        ),
        indexed_freeze_policy=str(indexed_frozen["freeze_policy"]),
        indexed_rebuild_count=int(indexed_frozen["rebuild_count"]),
        indexed_primary_value_min=float(indexed_variance["primary_value_min"]),
        indexed_primary_value_max=float(indexed_variance["primary_value_max"]),
        indexed_mean_search_seconds_min=float(
            indexed_variance["mean_search_seconds_min"]
        ),
        indexed_mean_search_seconds_max=float(
            indexed_variance["mean_search_seconds_max"]
        ),
        scorecard_path=scorecard_path,
        indexed_variance_path=indexed_variance_path,
        indexed_frozen_path=indexed_frozen_path,
    )


def build_lancedb_lane_a_bundle(
    *,
    gold_scorecard_path: str,
    gold_scorecard: Mapping[str, Any],
    gold_indexed_variance_path: str,
    gold_indexed_variance: Mapping[str, Any],
    gold_indexed_frozen_path: str,
    gold_indexed_frozen: Mapping[str, Any],
    evidence_scorecard_path: str,
    evidence_scorecard: Mapping[str, Any],
    evidence_indexed_variance_path: str,
    evidence_indexed_variance: Mapping[str, Any],
    evidence_indexed_frozen_path: str,
    evidence_indexed_frozen: Mapping[str, Any],
) -> LanceDbLaneABundle:
    gold = _build_slice_bundle(
        scorecard_path=gold_scorecard_path,
        scorecard=gold_scorecard,
        indexed_variance_path=gold_indexed_variance_path,
        indexed_variance=gold_indexed_variance,
        indexed_frozen_path=gold_indexed_frozen_path,
        indexed_frozen=gold_indexed_frozen,
    )
    evidence = _build_slice_bundle(
        scorecard_path=evidence_scorecard_path,
        scorecard=evidence_scorecard,
        indexed_variance_path=evidence_indexed_variance_path,
        indexed_variance=evidence_indexed_variance,
        indexed_frozen_path=evidence_indexed_frozen_path,
        indexed_frozen=evidence_indexed_frozen,
    )

    if gold.indexed_freeze_policy != evidence.indexed_freeze_policy:
        raise ValueError("gold and evidence indexed freeze policies must match")

    return LanceDbLaneABundle(
        lane_id="browsecomp_plus_lane_a",
        indexed_policy=gold.indexed_freeze_policy,
        gold=gold,
        evidence=evidence,
    )
