from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from kayak_bridge.latent_proxy_artifact import (
    export_trained_reference_lemur_artifact,
)
from kayak_bridge.latent_proxy_task_benchmark import (
    benchmark_task_with_latent_proxy_artifact,
    rank_task_with_latent_proxy_artifact,
)
from kayak_bridge.trained_reference_lemur import _UpstreamLemurMlp


class LatentProxyTaskBenchmarkTests(unittest.TestCase):
    def test_benchmark_tiny_task_with_exported_artifact(self) -> None:
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

        model = _UpstreamLemurMlp(
            input_dim=2,
            output_dim=1,
            hidden_dim=2,
            final_hidden_dim=2,
            num_layers=1,
            activation="relu",
        )
        with torch.no_grad():
            linear = model.feature_extractor[0]
            linear.weight.copy_(torch.eye(2, dtype=torch.float32))
            linear.bias.zero_()
            norm = model.feature_extractor[1]
            norm.weight.copy_(torch.ones(2, dtype=torch.float32))
            norm.bias.zero_()
            model.output_layer.weight.zero_()
        weights = torch.tensor(
            [[1.0, 0.0], [0.25, 0.75]],
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            mlp_path = tmp_root / "mlp.pt"
            w_path = tmp_root / "w.pt"
            artifact_root = tmp_root / "latent_proxy"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": dict(model.config),
                    "output_mean": 0.0,
                    "output_std": 1.0,
                },
                mlp_path,
            )
            torch.save({"W": weights}, w_path)

            import kayak

            index = kayak.documents(
                [document["doc_id"] for document in task["documents"]],
                [document["vectors"] for document in task["documents"]],
                texts=[document["text"] for document in task["documents"]],
            ).pack()
            export_trained_reference_lemur_artifact(
                index,
                artifact_root,
                mlp_path=mlp_path,
                w_path=w_path,
                query_divisor=1.0,
                model_name="unit-test-model",
            )

            rankings = rank_task_with_latent_proxy_artifact(
                task,
                artifact_root=artifact_root,
                candidate_k=2,
            )
            summary = benchmark_task_with_latent_proxy_artifact(
                task,
                artifact_root=artifact_root,
                candidate_k=2,
                warmup_iterations=0,
                measurement_iterations=1,
            )

        self.assertEqual(rankings, (("doc-a",),))
        self.assertEqual(summary.engine, "latent_proxy_artifact_reference")
        self.assertEqual(summary.index_kind, "latent_proxy_artifact")
        self.assertEqual(summary.stage1_backend, "latent_proxy_artifact_reference")
        self.assertEqual(summary.candidate_k, 2)
        self.assertEqual(summary.latent_dim, 2)
        self.assertEqual(summary.landmark_count, 0)
        self.assertAlmostEqual(summary.primary_value, 1.0)
        self.assertAlmostEqual(summary.mean_ndcg_at_k, 1.0)
        self.assertAlmostEqual(summary.mean_recall_at_k, 1.0)
        self.assertAlmostEqual(summary.success_rate_at_k, 1.0)


if __name__ == "__main__":
    unittest.main()
