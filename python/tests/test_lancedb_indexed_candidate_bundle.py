from __future__ import annotations

from kayak_bridge.lancedb_indexed_candidate_bundle import (
    build_lancedb_indexed_candidate_bundle,
)


def test_build_lancedb_indexed_candidate_bundle_picks_quality_and_speed_winners() -> None:
    bundle = build_lancedb_indexed_candidate_bundle(
        task_path="/tmp/task.json",
        kayak_exact_path="/tmp/kayak.json",
        lancedb_scan_path="/tmp/scan.json",
        scorecard_path="/tmp/scorecard.json",
        scorecard={
            "dataset_id": "dataset://a",
            "family": "mock",
            "slice_name": "slice",
            "primary_metric": "ndcg",
            "systems": [
                {
                    "name": "kayak_exact",
                    "primary_value": 0.80,
                    "mean_search_seconds": 0.01,
                },
                {
                    "name": "lancedb_scan",
                    "primary_value": 0.78,
                    "mean_search_seconds": 0.03,
                },
                {
                    "name": "lancedb_indexed_default",
                    "primary_value": 0.77,
                    "mean_search_seconds": 0.02,
                },
                {
                    "name": "lancedb_indexed_tradeoff",
                    "primary_value": 0.81,
                    "mean_search_seconds": 0.022,
                },
                {
                    "name": "lancedb_indexed_quality",
                    "primary_value": 0.83,
                    "mean_search_seconds": 0.028,
                },
            ],
            "pairwise": [
                {
                    "candidate": "lancedb_scan",
                    "mean_search_seconds_ratio_vs_baseline": 3.0,
                },
                {
                    "candidate": "lancedb_indexed_default",
                    "mean_search_seconds_ratio_vs_baseline": 2.0,
                },
                {
                    "candidate": "lancedb_indexed_tradeoff",
                    "mean_search_seconds_ratio_vs_baseline": 2.2,
                },
                {
                    "candidate": "lancedb_indexed_quality",
                    "mean_search_seconds_ratio_vs_baseline": 2.8,
                },
            ],
        },
        candidates=[
            (
                "lancedb_indexed_default",
                "/tmp/default.json",
                {
                    "freeze_policy": "mean_across_5_rebuilds",
                    "rebuild_count": 5,
                },
            ),
            (
                "lancedb_indexed_tradeoff",
                "/tmp/tradeoff.json",
                {
                    "freeze_policy": "mean_across_5_rebuilds",
                    "rebuild_count": 5,
                    "index_num_partitions": 8,
                    "indexed_refine_factor": 2,
                },
            ),
            (
                "lancedb_indexed_quality",
                "/tmp/quality.json",
                {
                    "freeze_policy": "mean_across_5_rebuilds",
                    "rebuild_count": 5,
                    "indexed_refine_factor": 1,
                },
            ),
        ],
    )

    assert bundle.candidate_count == 3
    assert bundle.best_quality_candidate_name == "lancedb_indexed_quality"
    assert bundle.fastest_non_regressing_candidate_name == "lancedb_indexed_tradeoff"
    assert bundle.fastest_quality_improving_candidate_name == "lancedb_indexed_tradeoff"
    assert bundle.rows[1].index_num_partitions == 8
    assert bundle.rows[1].indexed_refine_factor == 2
