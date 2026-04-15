from __future__ import annotations

import unittest

from kayak_bridge.lancedb_index_sweep import (
    build_lancedb_indexed_sweep_summary,
    parse_lancedb_indexed_sweep_spec,
)


class LanceDbIndexSweepTests(unittest.TestCase):
    def test_parse_sweep_spec_extracts_build_and_query_controls(self) -> None:
        spec = parse_lancedb_indexed_sweep_spec(
            "p8_refine2:index_num_partitions=8,indexed_refine_factor=2"
        )

        self.assertEqual(spec.name, "p8_refine2")
        self.assertEqual(spec.index_build_controls.num_partitions, 8)
        self.assertIsNone(spec.index_build_controls.num_sub_vectors)
        self.assertEqual(spec.indexed_query_controls.refine_factor, 2)
        self.assertIsNone(spec.indexed_query_controls.nprobes)

    def test_parse_sweep_spec_rejects_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            parse_lancedb_indexed_sweep_spec("bad:foo=1")

    def test_build_sweep_summary_computes_baseline_deltas(self) -> None:
        summary = build_lancedb_indexed_sweep_summary(
            task_path="/tmp/task.json",
            kayak_exact_path="/tmp/kayak.json",
            kayak_exact={
                "dataset_id": "dataset://a",
                "family": "mock",
                "slice_name": "slice",
                "primary_metric": "ndcg",
                "k": 10,
                "primary_value": 0.8,
                "mean_search_seconds": 0.01,
            },
            lancedb_scan_path="/tmp/scan.json",
            lancedb_scan={
                "dataset_id": "dataset://a",
                "family": "mock",
                "slice_name": "slice",
                "primary_metric": "ndcg",
                "k": 10,
                "primary_value": 0.75,
                "mean_search_seconds": 0.02,
            },
            configs=[
                (
                    "refine2",
                    "/tmp/refine2_variance.json",
                    {
                        "dataset_id": "dataset://a",
                        "family": "mock",
                        "slice_name": "slice",
                        "primary_metric": "ndcg",
                        "k": 10,
                        "primary_value_min": 0.79,
                        "primary_value_max": 0.83,
                        "mean_search_seconds_min": 0.014,
                        "mean_search_seconds_max": 0.016,
                    },
                    "/tmp/refine2_frozen.json",
                    {
                        "dataset_id": "dataset://a",
                        "family": "mock",
                        "slice_name": "slice",
                        "primary_metric": "ndcg",
                        "k": 10,
                        "freeze_policy": "mean_across_5_rebuilds",
                        "rebuild_count": 5,
                        "index_num_partitions": 8,
                        "indexed_refine_factor": 2,
                        "primary_value": 0.81,
                        "mean_search_seconds": 0.015,
                    },
                )
            ],
        )

        self.assertEqual(summary.config_count, 1)
        row = summary.rows[0]
        self.assertEqual(row.config_name, "refine2")
        self.assertEqual(row.index_num_partitions, 8)
        self.assertEqual(row.indexed_refine_factor, 2)
        self.assertAlmostEqual(row.primary_value_delta_vs_scan, 0.06)
        self.assertAlmostEqual(row.primary_value_delta_vs_kayak, 0.01)
        self.assertAlmostEqual(row.mean_search_seconds_ratio_vs_scan, 0.75)
        self.assertAlmostEqual(row.mean_search_seconds_ratio_vs_kayak, 1.5)


if __name__ == "__main__":
    unittest.main()
