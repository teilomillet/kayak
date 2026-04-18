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
from kayak_bridge.latent_proxy_reference import (
    compute_query_latent_proxy_features,
    latent_proxy_similarity_scores,
    load_latent_proxy_artifact,
)
from kayak_bridge.trained_reference_lemur import (
    _UpstreamLemurElm,
    _UpstreamLemurMlp,
    load_trained_reference_lemur,
)


def _matrix(*rows: tuple[float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def _write_checkpoint_pair(
    *,
    tmp_root: Path,
    model: torch.nn.Module,
    weights: torch.Tensor,
) -> tuple[Path, Path]:
    mlp_path = tmp_root / "mlp.pt"
    w_path = tmp_root / "w.pt"
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
    return mlp_path, w_path


class LatentProxyReferenceTests(unittest.TestCase):
    def test_exported_mlp_artifact_reproduces_trained_reference_scores(self) -> None:
        index = kayak.documents(
            ["doc-a", "doc-b"],
            [
                _matrix((1.0, 0.0), (0.0, 1.0)),
                _matrix((1.0, 0.0), (1.0, 0.0)),
            ],
        ).pack()
        query = kayak.query(_matrix((1.0, 0.0), (0.0, 1.0)))

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
            norm.weight.copy_(torch.tensor([1.25, 0.75], dtype=torch.float32))
            norm.bias.copy_(torch.tensor([0.0, 0.125], dtype=torch.float32))
            model.output_layer.weight.zero_()
        weights = torch.tensor(
            [[1.0, 0.0], [0.25, 0.75]],
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            mlp_path, w_path = _write_checkpoint_pair(
                tmp_root=tmp_root,
                model=model,
                weights=weights,
            )
            output_root = tmp_root / "latent_proxy"
            export_trained_reference_lemur_artifact(
                index,
                output_root,
                mlp_path=mlp_path,
                w_path=w_path,
                query_divisor=1.0,
                model_name="synthetic-mlp",
            )

            reference = load_trained_reference_lemur(
                index,
                mlp_path=mlp_path,
                w_path=w_path,
                query_divisor=1.0,
            )
            artifact = load_latent_proxy_artifact(output_root)

        np.testing.assert_allclose(
            compute_query_latent_proxy_features(query, artifact),
            reference.compute_query_features(query),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            latent_proxy_similarity_scores(query, artifact).values,
            reference.similarity_scores(query).values,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_exported_elm_artifact_reproduces_trained_reference_scores(self) -> None:
        index = kayak.documents(
            ["doc-a", "doc-b"],
            [
                _matrix((1.0, 0.0), (0.0, 1.0)),
                _matrix((0.5, 0.5), (1.0, 0.0)),
            ],
        ).pack()
        query = kayak.query(_matrix((1.0, 0.0), (0.5, 1.0)))

        model = _UpstreamLemurElm(
            input_dim=2,
            output_dim=1,
            final_hidden_dim=3,
            activation="gelu",
        )
        with torch.no_grad():
            random_features = model.feature_extractor[0]
            random_features.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]],
                    dtype=torch.float32,
                )
            )
            model.output_layer.weight.zero_()
        weights = torch.tensor(
            [[1.0, 0.0, -0.5], [0.25, 0.75, 0.5]],
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            mlp_path, w_path = _write_checkpoint_pair(
                tmp_root=tmp_root,
                model=model,
                weights=weights,
            )
            output_root = tmp_root / "latent_proxy"
            export_trained_reference_lemur_artifact(
                index,
                output_root,
                mlp_path=mlp_path,
                w_path=w_path,
                query_divisor=2.0,
                model_name="synthetic-elm",
            )

            reference = load_trained_reference_lemur(
                index,
                mlp_path=mlp_path,
                w_path=w_path,
                query_divisor=2.0,
            )
            artifact = load_latent_proxy_artifact(output_root)

        np.testing.assert_allclose(
            compute_query_latent_proxy_features(query, artifact),
            reference.compute_query_features(query),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            latent_proxy_similarity_scores(query, artifact).values,
            reference.similarity_scores(query).values,
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
