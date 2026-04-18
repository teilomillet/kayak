from __future__ import annotations

import unittest

import numpy as np

from kayak_bridge.lemur_task_benchmark import (
    benchmark_task_with_reference_lemur,
    rank_task_with_reference_lemur,
)


class LemurTaskBenchmarkTests(unittest.TestCase):
    def test_rank_task_requires_candidate_k_to_cover_final_k(self) -> None:
        task = {
            "dataset_id": "dataset://tiny",
            "model_name": "unit-test-model",
            "family": "tiny_family",
            "slice_name": "tiny_slice",
            "primary_metric": "ndcg",
            "k": 2,
            "nominal_query_vector_count": 1,
            "nominal_document_vector_count": 2,
            "vector_dim": 2,
            "documents": [
                {"doc_id": "doc-a", "text": "alpha", "vector_count": 2, "vectors": [[1.0, 0.0], [0.0, 1.0]]},
                {"doc_id": "doc-b", "text": "beta", "vector_count": 2, "vectors": [[1.0, 0.0], [1.0, 0.0]]},
            ],
            "queries": [
                {
                    "query_id": "q-1",
                    "text": "alpha beta",
                    "relevant_doc_ids": ["doc-a"],
                    "vector_count": 2,
                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "candidate_k must be greater than or equal to final_k"):
            rank_task_with_reference_lemur(
                task,
                latent_dim=2,
                candidate_k=1,
            )

    def test_benchmark_tiny_task_with_identity_feature_map(self) -> None:
        task = {
            "dataset_id": "dataset://tiny",
            "model_name": "unit-test-model",
            "family": "tiny_family",
            "slice_name": "tiny_slice",
            "primary_metric": "ndcg",
            "k": 1,
            "nominal_query_vector_count": 2,
            "nominal_document_vector_count": 2,
            "vector_dim": 2,
            "documents": [
                {
                    "doc_id": "doc-a",
                    "text": "alpha beta",
                    "vector_count": 2,
                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                },
                {
                    "doc_id": "doc-b",
                    "text": "alpha alpha",
                    "vector_count": 2,
                    "vectors": [[1.0, 0.0], [1.0, 0.0]],
                },
            ],
            "queries": [
                {
                    "query_id": "q-1",
                    "text": "alpha beta",
                    "relevant_doc_ids": ["doc-a"],
                    "vector_count": 2,
                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                }
            ],
        }

        summary = benchmark_task_with_reference_lemur(
            task,
            latent_dim=2,
            candidate_k=2,
            warmup_iterations=0,
            measurement_iterations=1,
            feature_weights=np.eye(2, dtype=np.float32),
            landmark_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            activation="relu",
            query_divisor=1.0,
            apply_layer_norm=False,
        )

        self.assertEqual(summary.engine, "lemur_reference")
        self.assertEqual(summary.index_kind, "latent_single_vector_reference")
        self.assertEqual(summary.candidate_k, 2)
        self.assertEqual(summary.latent_dim, 2)
        self.assertEqual(summary.landmark_count, 2)
        self.assertAlmostEqual(summary.primary_value, 1.0)
        self.assertAlmostEqual(summary.mean_ndcg_at_k, 1.0)
        self.assertAlmostEqual(summary.mean_recall_at_k, 1.0)
        self.assertAlmostEqual(summary.success_rate_at_k, 1.0)


if __name__ == "__main__":
    unittest.main()
