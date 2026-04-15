from __future__ import annotations

from kayak_bridge.task_comparison_bundle import build_task_comparison_bundle


def test_build_task_comparison_bundle_extracts_key_ratios() -> None:
    bundle = build_task_comparison_bundle(
        task_path="/tmp/task.json",
        kayak_exact_path="/tmp/kayak.json",
        lancedb_scan_path="/tmp/lancedb_scan.json",
        lancedb_indexed_variance_path="/tmp/lancedb_var.json",
        lancedb_indexed_frozen_path="/tmp/lancedb_frozen.json",
        scorecard_path="/tmp/scorecard.json",
        scorecard={
            "dataset_id": "mock://dataset",
            "family": "mock",
            "slice_name": "mock_slice",
            "primary_metric": "ndcg",
            "systems": [
                {
                    "name": "kayak_exact",
                    "primary_value": 0.8,
                    "mean_search_seconds": 0.01,
                },
                {
                    "name": "lancedb_scan",
                    "primary_value": 0.7,
                    "mean_search_seconds": 0.02,
                },
                {
                    "name": "lancedb_ivf_pq_frozen",
                    "primary_value": 0.6,
                    "mean_search_seconds": 0.005,
                },
            ],
            "pairwise": [
                {
                    "candidate": "lancedb_scan",
                    "mean_search_seconds_ratio_vs_baseline": 2.0,
                },
                {
                    "candidate": "lancedb_ivf_pq_frozen",
                    "mean_search_seconds_ratio_vs_baseline": 0.5,
                },
            ],
        },
        indexed_variance={
            "primary_value_min": 0.55,
            "primary_value_max": 0.65,
            "mean_search_seconds_min": 0.004,
            "mean_search_seconds_max": 0.006,
        },
        indexed_frozen={
            "freeze_policy": "mean_across_5_rebuilds",
            "rebuild_count": 5,
            "index_num_partitions": 8,
            "index_num_sub_vectors": 16,
            "index_target_partition_size": 256,
            "indexed_nprobes": 32,
            "indexed_refine_factor": 2,
        },
        scale_sweep_path="/tmp/scale.json",
        scale_sweep={
            "rows": [
                {"actual_document_count": 100, "lancedb_scan_latency_ratio_vs_kayak": 2.0},
                {"actual_document_count": 800, "lancedb_scan_latency_ratio_vs_kayak": 6.0},
            ]
        },
        storage_compare_path="/tmp/storage_compare.json",
        storage_compare={
            "kayak_load_from_lancedb_seconds": 0.25,
            "lancedb_scan": {"mean_search_seconds": 0.03},
            "kayak_exact_from_lancedb": {"mean_search_seconds": 0.01},
            "lancedb_scan_latency_ratio_vs_kayak_from_lancedb": 3.0,
        },
        storage_scale_path="/tmp/storage_scale.json",
        storage_scale={
            "rows": [
                {
                    "lancedb_storage_byte_ratio_vs_kayak": 1.2,
                    "lancedb_build_seconds_ratio_vs_kayak": 2.4,
                    "lancedb_search_seconds_ratio_vs_kayak": 3.2,
                },
                {
                    "lancedb_storage_byte_ratio_vs_kayak": 1.8,
                    "lancedb_build_seconds_ratio_vs_kayak": 1.9,
                    "lancedb_search_seconds_ratio_vs_kayak": 2.7,
                },
            ]
        },
    )

    assert bundle.dataset_id == "mock://dataset"
    assert bundle.kayak_exact_mean_search_seconds == 0.01
    assert bundle.lancedb_scan_mean_search_seconds == 0.02
    assert bundle.lancedb_scan_latency_ratio_vs_kayak == 2.0
    assert bundle.lancedb_indexed_frozen_mean_search_seconds == 0.005
    assert bundle.lancedb_indexed_frozen_latency_ratio_vs_kayak == 0.5
    assert bundle.indexed_index_num_partitions == 8
    assert bundle.indexed_index_num_sub_vectors == 16
    assert bundle.indexed_index_target_partition_size == 256
    assert bundle.indexed_nprobes == 32
    assert bundle.indexed_refine_factor == 2
    assert bundle.largest_scale_document_count == 800
    assert bundle.storage_compare_kayak_load_from_lancedb_seconds == 0.25
    assert bundle.storage_compare_lancedb_scan_mean_search_seconds == 0.03
    assert (
        bundle.storage_compare_kayak_exact_from_lancedb_mean_search_seconds == 0.01
    )
    assert bundle.base_storage_build_seconds_ratio_vs_kayak == 2.4
    assert bundle.base_storage_search_seconds_ratio_vs_kayak == 3.2
    assert bundle.largest_storage_byte_ratio_vs_kayak == 1.8
    assert bundle.largest_storage_build_seconds_ratio_vs_kayak == 1.9
    assert bundle.largest_storage_search_seconds_ratio_vs_kayak == 2.7
