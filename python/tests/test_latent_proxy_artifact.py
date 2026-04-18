from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import kayak
from kayak_bridge.latent_proxy_artifact import (
    export_trained_reference_lemur_artifact,
)
from kayak_bridge.trained_reference_lemur import _UpstreamLemurMlp


def _matrix(*rows: tuple[float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


class LatentProxyArtifactTests(unittest.TestCase):
    def test_exports_trained_checkpoint_to_native_latent_proxy_layout(self) -> None:
        index = kayak.documents(
            ["doc-a", "doc-b"],
            [
                _matrix((1.0, 0.0), (0.0, 1.0)),
                _matrix((1.0, 0.0), (1.0, 0.0)),
            ],
        ).pack()

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
            output_root = tmp_root / "latent_proxy"
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

            artifact = export_trained_reference_lemur_artifact(
                index,
                output_root,
                mlp_path=mlp_path,
                w_path=w_path,
                query_divisor=1.0,
                model_name="synthetic",
            )

            manifest = (output_root / "manifest.tsv").read_text(encoding="utf-8")
            self.assertTrue((output_root / "doc_ids.tsv").exists())
            self.assertTrue((output_root / "proxy_vectors.bin").exists())
            self.assertIn("artifact_kind\tlatent_proxy_index", manifest)
            self.assertIn(
                "projection_block_0_order_kind\tlinear_norm_activation",
                manifest,
            )
            self.assertIn(
                "projection_block_0_activation_output_scale\t1.0",
                manifest,
            )

        self.assertEqual(artifact.input_vector_dim, 2)
        self.assertEqual(artifact.query_divisor, 1.0)
        self.assertEqual(artifact.doc_ids, ("doc-a", "doc-b"))
        self.assertEqual(artifact.proxy_vectors.shape, (2, 2))
        self.assertEqual(len(artifact.blocks), 1)
        self.assertEqual(artifact.blocks[0].order_kind, "linear_norm_activation")
        self.assertEqual(artifact.blocks[0].activation_kind, "relu")
        self.assertEqual(artifact.blocks[0].activation_output_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
