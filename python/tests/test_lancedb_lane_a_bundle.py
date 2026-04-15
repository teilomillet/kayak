from __future__ import annotations

from kayak_bridge.lancedb_lane_a_bundle import build_lancedb_lane_a_bundle


def _scorecard(slice_name: str) -> dict[str, object]:
    return {
        "dataset_id": "dataset://a",
        "slice_name": slice_name,
        "primary_metric": "ndcg",
        "systems": [
            {"name": "kayak_exact", "primary_value": 0.8},
            {"name": "lancedb_scan", "primary_value": 0.75},
            {"name": "lancedb_ivf_pq_frozen", "primary_value": 0.82},
        ],
        "pairwise": [
            {"candidate": "lancedb_scan", "mean_search_seconds_ratio_vs_baseline": 2.0},
            {
                "candidate": "lancedb_ivf_pq_frozen",
                "mean_search_seconds_ratio_vs_baseline": 1.5,
            },
        ],
    }


def _variance() -> dict[str, object]:
    return {
        "primary_value_min": 0.8,
        "primary_value_max": 0.84,
        "mean_search_seconds_min": 0.012,
        "mean_search_seconds_max": 0.018,
    }


def _frozen() -> dict[str, object]:
    return {
        "freeze_policy": "mean_across_5_rebuilds",
        "rebuild_count": 5,
        "index_num_partitions": 8,
        "index_num_sub_vectors": 16,
        "index_target_partition_size": 256,
        "indexed_nprobes": 32,
        "indexed_refine_factor": 2,
    }


def test_build_lancedb_lane_a_bundle_surfaces_indexed_controls() -> None:
    bundle = build_lancedb_lane_a_bundle(
        gold_scorecard_path="/tmp/gold_scorecard.json",
        gold_scorecard=_scorecard("gold"),
        gold_indexed_variance_path="/tmp/gold_variance.json",
        gold_indexed_variance=_variance(),
        gold_indexed_frozen_path="/tmp/gold_frozen.json",
        gold_indexed_frozen=_frozen(),
        evidence_scorecard_path="/tmp/evidence_scorecard.json",
        evidence_scorecard=_scorecard("evidence"),
        evidence_indexed_variance_path="/tmp/evidence_variance.json",
        evidence_indexed_variance=_variance(),
        evidence_indexed_frozen_path="/tmp/evidence_frozen.json",
        evidence_indexed_frozen=_frozen(),
    )

    assert bundle.gold.indexed_index_num_partitions == 8
    assert bundle.gold.indexed_index_num_sub_vectors == 16
    assert bundle.gold.indexed_index_target_partition_size == 256
    assert bundle.gold.indexed_nprobes == 32
    assert bundle.gold.indexed_refine_factor == 2
