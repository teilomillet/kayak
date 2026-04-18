from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import kayak
from kayak_bridge.trained_reference_lemur import (
    _UpstreamLemurMlp,
    load_trained_reference_lemur,
)


def _matrix(*rows: tuple[float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


class TrainedReferenceLemurTests(unittest.TestCase):
    def test_loads_upstream_checkpoint_pair_and_scores_query(self) -> None:
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
            norm.weight.copy_(torch.ones(2, dtype=torch.float32))
            norm.bias.zero_()
            model.output_layer.weight.zero_()
        weights = torch.tensor(
            [[1.0, 0.0], [0.25, 0.75]],
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            mlp_path = Path(tmp_dir) / "mlp.pt"
            w_path = Path(tmp_dir) / "w.pt"
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

            loaded = load_trained_reference_lemur(
                index,
                mlp_path=mlp_path,
                w_path=w_path,
                query_divisor=1.0,
            )
            scores = loaded.similarity_scores(query)

        with torch.inference_mode():
            token_features = model.feature_extractor(
                torch.from_numpy(
                    np.array(query.as_vector_matrix(), dtype=np.float32, copy=True)
                )
            )
            expected_query = token_features.sum(dim=0)
            expected_scores = expected_query @ weights.T

        np.testing.assert_allclose(scores.values, expected_scores.numpy(), atol=1e-6)
        self.assertEqual(scores.doc_ids, ("doc-a", "doc-b"))

    def test_rejects_weight_rows_that_do_not_match_doc_count(self) -> None:
        index = kayak.documents(["doc-a"], [_matrix((1.0, 0.0))]).pack()
        model = _UpstreamLemurMlp(
            input_dim=2,
            output_dim=1,
            hidden_dim=2,
            final_hidden_dim=2,
            num_layers=1,
            activation="relu",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            mlp_path = Path(tmp_dir) / "mlp.pt"
            w_path = Path(tmp_dir) / "w.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": dict(model.config),
                    "output_mean": 0.0,
                    "output_std": 1.0,
                },
                mlp_path,
            )
            torch.save({"W": torch.ones((2, 2), dtype=torch.float32)}, w_path)

            with self.assertRaisesRegex(ValueError, "row count"):
                load_trained_reference_lemur(
                    index,
                    mlp_path=mlp_path,
                    w_path=w_path,
                )


if __name__ == "__main__":
    unittest.main()
