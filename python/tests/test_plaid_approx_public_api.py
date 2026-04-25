from __future__ import annotations

import unittest

import numpy as np

import kayak


def _basis(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


class PlaidApproxPublicApiTests(unittest.TestCase):
    def test_search_accepts_explicit_plaid_approximation_parameter(self) -> None:
        query = kayak.query(np.stack([_basis(0), _basis(1)]))
        index = kayak.documents(
            ["doc-a", "doc-b", "doc-c"],
            [
                np.stack([_basis(0), _basis(1)]),
                np.stack([_basis(0), _basis(0)]),
                np.stack([_basis(1), _basis(1)]),
            ],
        ).pack()
        config = kayak.PlaidApproxConfig(
            centroid_count=3,
            centroids_per_query_vector=2,
            candidate_k=3,
        )

        exact_hits = kayak.search(
            query,
            index,
            k=2,
            backend=kayak.MOJO_EXACT_CPU_BACKEND,
        )
        approx_hits = kayak.search(query, index, k=2, approximation=config)

        self.assertEqual(approx_hits, exact_hits)

    def test_prepared_plaid_approx_index_supports_ragged_vector_counts(self) -> None:
        query_batch = kayak.query_batch(
            [
                np.stack([_basis(0), _basis(1)]),
                np.stack([_basis(1)]),
            ]
        )
        index = kayak.documents(
            ["short", "long", "other"],
            [
                np.stack([_basis(0)]),
                np.stack([_basis(0), _basis(1), _basis(1)]),
                np.stack([_basis(2), _basis(2)]),
            ],
        ).pack()
        config = kayak.PlaidApproxConfig(
            centroid_count=4,
            centroids_per_query_vector=2,
            candidate_k=3,
        )

        prepared = kayak.prepare_plaid_approx_index(index, config=config, final_k=2)
        hits_by_query = prepared.search_batch(query_batch, final_k=2)

        self.assertIsNone(prepared.document_vector_count)
        self.assertEqual(prepared.document_vector_counts, (1, 3, 2))
        self.assertEqual(
            hits_by_query,
            kayak.search_batch(
                query_batch,
                index,
                k=2,
                backend=kayak.MOJO_EXACT_CPU_BACKEND,
            ),
        )


if __name__ == "__main__":
    unittest.main()
