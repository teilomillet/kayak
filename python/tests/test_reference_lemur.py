from __future__ import annotations

import unittest

import numpy as np

import kayak
from kayak_bridge.reference_lemur import fit_reference_lemur


def _matrix(*rows: tuple[float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


class ReferenceLemurTests(unittest.TestCase):
    def test_fit_requires_positive_latent_dim_without_feature_weights(self) -> None:
        index = kayak.documents(["doc-a"], [_matrix((1.0, 0.0))]).pack()

        with self.assertRaisesRegex(ValueError, "latent_dim is required"):
            fit_reference_lemur(index)

    def test_similarity_scores_keep_doc_id_order_and_rank_exact_match_first(self) -> None:
        index = kayak.documents(
            ["doc-a", "doc-b"],
            [
                _matrix((1.0, 0.0), (1.0, 0.0)),
                _matrix((0.0, 1.0), (0.0, 1.0)),
            ],
        ).pack()
        model = fit_reference_lemur(
            index,
            feature_weights=np.eye(2, dtype=np.float32),
            landmark_vectors=_matrix((1.0, 0.0), (0.0, 1.0)),
            activation="relu",
            query_divisor=1.0,
            apply_layer_norm=False,
        )

        scores = model.similarity_scores(kayak.query(_matrix((1.0, 0.0))))

        self.assertEqual(scores.doc_ids, ("doc-a", "doc-b"))
        self.assertGreater(float(scores.values[0]), float(scores.values[1]))
        self.assertEqual(scores.topk(1)[0].doc_id, "doc-a")

    def test_multi_token_query_rewards_document_covering_both_concepts(self) -> None:
        index = kayak.documents(
            ["doc-a", "doc-b"],
            [
                _matrix((1.0, 0.0), (0.0, 1.0)),
                _matrix((1.0, 0.0), (1.0, 0.0)),
            ],
        ).pack()
        model = fit_reference_lemur(
            index,
            feature_weights=np.eye(2, dtype=np.float32),
            landmark_vectors=_matrix((1.0, 0.0), (0.0, 1.0)),
            activation="relu",
            query_divisor=1.0,
            apply_layer_norm=False,
        )

        scores = model.similarity_scores(
            kayak.query(_matrix((1.0, 0.0), (0.0, 1.0)))
        )

        self.assertGreater(float(scores.values[0]), float(scores.values[1]))


if __name__ == "__main__":
    unittest.main()
