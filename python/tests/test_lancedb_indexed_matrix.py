from __future__ import annotations

import unittest

from kayak_bridge.lancedb_indexed_matrix import (
    build_lancedb_indexed_matrix_summary,
)


class LanceDbIndexedMatrixTests(unittest.TestCase):
    def test_builds_matrix_summary_from_slice_artifacts(self) -> None:
        summary = build_lancedb_indexed_matrix_summary(
            matrix_id="hard_matrix",
            selection_policy="include_default_plus_best_quality_plus_fastest_quality_improving",
            sweep_config_names=("default", "refine2"),
            slices=(
                (
                    "legal_rag_bench_real_subset",
                    "/tmp/legal_task.json",
                    {
                        "k": 10,
                        "nominal_query_vector_count": 32,
                        "nominal_document_vector_count": 157,
                        "queries": [{}, {}],
                        "documents": [{}, {}, {}],
                    },
                    "/tmp/legal_sweep.json",
                    {
                        "dataset_id": "isaacus/legal-rag-bench/qa",
                        "family": "legal_rag_bench",
                        "slice_name": "legal_rag_bench_real_subset",
                        "primary_metric": "mrr",
                        "config_count": 2,
                    },
                    "/tmp/legal_selection.json",
                    {
                        "selected_count": 2,
                        "selection_policy": "include_default_plus_best_quality_plus_fastest_quality_improving",
                    },
                    "/tmp/legal_bundle.json",
                    {
                        "dataset_id": "isaacus/legal-rag-bench/qa",
                        "family": "legal_rag_bench",
                        "slice_name": "legal_rag_bench_real_subset",
                        "primary_metric": "mrr",
                        "kayak_exact_primary_value": 0.16,
                        "kayak_exact_mean_search_seconds": 0.001,
                        "lancedb_scan_primary_value": 0.16,
                        "lancedb_scan_mean_search_seconds": 0.028,
                        "best_quality_candidate_name": "lancedb_indexed_refine2",
                        "fastest_quality_improving_candidate_name": "lancedb_indexed_refine2",
                        "rows": [
                            {
                                "name": "lancedb_indexed_refine2",
                                "primary_value": 0.18,
                                "mean_search_seconds": 0.026,
                                "primary_value_delta_vs_scan": 0.02,
                                "mean_search_seconds_ratio_vs_scan": 0.93,
                            }
                        ],
                    },
                ),
            ),
        ).to_json_ready()

        self.assertEqual(summary["slice_count"], 1)
        self.assertEqual(summary["total_queries"], 2)
        self.assertEqual(summary["total_documents"], 3)
        self.assertEqual(summary["rows"][0]["dataset_key"], "legal_rag_bench_real_subset")
        self.assertEqual(
            summary["rows"][0]["best_quality_candidate_name"],
            "lancedb_indexed_refine2",
        )
        self.assertEqual(
            summary["rows"][0]["fastest_quality_improving_candidate_name"],
            "lancedb_indexed_refine2",
        )
        self.assertEqual(summary["rows"][0]["selected_candidate_count"], 2)

    def test_rejects_mismatched_slice_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset_id"):
            build_lancedb_indexed_matrix_summary(
                matrix_id="hard_matrix",
                selection_policy="policy",
                sweep_config_names=("default",),
                slices=(
                    (
                        "key",
                        "/tmp/task.json",
                        {
                            "k": 10,
                            "nominal_query_vector_count": 32,
                            "nominal_document_vector_count": 64,
                            "queries": [{}],
                            "documents": [{}],
                        },
                        "/tmp/sweep.json",
                        {
                            "dataset_id": "dataset://a",
                            "family": "family",
                            "slice_name": "slice",
                            "primary_metric": "mrr",
                            "config_count": 1,
                        },
                        "/tmp/selection.json",
                        {
                            "selected_count": 1,
                            "selection_policy": "policy",
                        },
                        "/tmp/bundle.json",
                        {
                            "dataset_id": "dataset://b",
                            "family": "family",
                            "slice_name": "slice",
                            "primary_metric": "mrr",
                            "kayak_exact_primary_value": 0.1,
                            "kayak_exact_mean_search_seconds": 0.001,
                            "lancedb_scan_primary_value": 0.1,
                            "lancedb_scan_mean_search_seconds": 0.01,
                            "best_quality_candidate_name": "lancedb_indexed_default",
                            "fastest_quality_improving_candidate_name": None,
                            "rows": [
                                {
                                    "name": "lancedb_indexed_default",
                                    "primary_value": 0.1,
                                    "mean_search_seconds": 0.01,
                                    "primary_value_delta_vs_scan": 0.0,
                                    "mean_search_seconds_ratio_vs_scan": 1.0,
                                }
                            ],
                        },
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
