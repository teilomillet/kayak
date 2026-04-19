from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from kayak_bridge.latent_proxy_artifact import (
    export_trained_reference_lemur_artifact,
)
from kayak_bridge.native_latent_proxy_task_benchmark import (
    benchmark_materialized_collection_search,
    materialize_native_latent_proxy_collection,
)
from kayak_bridge.trained_reference_lemur import _UpstreamLemurMlp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _mojo_backend_available() -> bool:
    if shutil.which("mojo") is not None:
        return True
    return (REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "mojo").exists()


@unittest.skipUnless(
    _mojo_backend_available(), "native latent-proxy benchmark requires Mojo"
)
class NativeLatentProxyTaskBenchmarkTests(unittest.TestCase):
    def test_materialized_native_latent_proxy_search_matches_tiny_fixture(self) -> None:
        task = {
            "dataset_id": "dataset://tiny",
            "model_name": "unit-test-model",
            "family": "tiny_family",
            "slice_name": "tiny_slice",
            "why": "unit test native latent proxy bridge",
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
            task_path = tmp_root / "task.json"
            task_path.write_text(json.dumps(task), encoding="utf-8")
            mlp_path = tmp_root / "mlp.pt"
            w_path = tmp_root / "w.pt"
            artifact_root = tmp_root / "latent_proxy"
            collection_root = tmp_root / "collection"
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

            materialization = materialize_native_latent_proxy_collection(
                task_path=task_path,
                artifact_root=artifact_root,
                collection_root=collection_root,
            )
            latent_summary = benchmark_materialized_collection_search(
                task_path=task_path,
                collection_root=collection_root,
                candidate_k=2,
            )
            exact_summary = benchmark_materialized_collection_search(
                task_path=task_path,
                collection_root=collection_root,
                candidate_k=2,
                candidate_generator_kind="exact_full_scan",
            )

        self.assertEqual(materialization["dataset_id"], "dataset://tiny")
        self.assertEqual(materialization["document_count"], 2)
        self.assertEqual(materialization["vector_dim"], 2)
        self.assertEqual(materialization["latent_dim"], 2)
        self.assertGreater(materialization["stage1_byte_size"], 0)
        self.assertTrue(materialization["text_corpus_loaded"])

        self.assertEqual(latent_summary["candidate_generator_kind"], "latent_proxy")
        self.assertEqual(latent_summary["candidate_k"], 2)
        self.assertAlmostEqual(latent_summary["mean_ndcg_at_k"], 1.0)
        self.assertAlmostEqual(latent_summary["mean_recall_at_k"], 1.0)
        self.assertAlmostEqual(latent_summary["success_rate_at_k"], 1.0)

        self.assertEqual(exact_summary["candidate_generator_kind"], "exact_full_scan")
        self.assertAlmostEqual(exact_summary["mean_ndcg_at_k"], 1.0)
        self.assertAlmostEqual(exact_summary["mean_recall_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
