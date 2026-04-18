from __future__ import annotations

import unittest

import numpy as np

import kayak
from kayak_bridge.lemur_candidate_sweep import (
    benchmark_reference_lemur_candidate_sweep,
)
from kayak_bridge.lemur_task_benchmark import benchmark_task_with_reference_lemur


class LemurCandidateSweepTests(unittest.TestCase):
    def test_reference_benchmark_reports_fit_seconds(self) -> None:
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

        self.assertGreaterEqual(summary.fit_seconds, 0.0)

    def test_candidate_sweep_serializes_rows_and_exact_baseline(self) -> None:
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

        summary = benchmark_reference_lemur_candidate_sweep(
            task,
            latent_dims=[2],
            candidate_ks=[1, 2],
            feature_weights=np.eye(2, dtype=np.float32),
            landmark_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            activation="relu",
            query_divisor=1.0,
            apply_layer_norm=False,
            exact_backend=kayak.NUMPY_REFERENCE_BACKEND,
            rerank_backend=kayak.NUMPY_REFERENCE_BACKEND,
            warmup_iterations=0,
            measurement_iterations=1,
        )

        payload = summary.to_json_ready()

        self.assertEqual(summary.exact_primary_value, 1.0)
        self.assertEqual(len(summary.rows), 2)
        self.assertEqual(summary.rows[0].candidate_k, 1)
        self.assertEqual(summary.rows[1].candidate_k, 2)
        self.assertGreaterEqual(summary.rows[0].fit_seconds, 0.0)
        self.assertGreaterEqual(summary.rows[0].mean_exact_topk_shortlist_recall, 0.0)
        self.assertLessEqual(summary.rows[0].mean_exact_topk_shortlist_recall, 1.0)
        self.assertEqual(payload["rows"][1]["candidate_k"], 2)


if __name__ == "__main__":
    unittest.main()
