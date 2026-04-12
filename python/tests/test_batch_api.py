from __future__ import annotations

import shutil
import unittest

import numpy as np

import kayak
from kayak_bridge.cache_paths import REPO_ROOT


def _dim128_vector(*entries: tuple[int, float]) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    for index, value in entries:
        vector[index] = np.float32(value)
    return vector


class BatchApiTests(unittest.TestCase):
    @staticmethod
    def _mojo_backend_available() -> bool:
        if shutil.which("mojo") is not None:
            return True
        return (REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "mojo").exists()

    def _build_index(self) -> kayak.LateIndex:
        return kayak.documents(
            ["doc-a", "doc-b", "doc-c"],
            [
                np.stack(
                    [
                        _dim128_vector((0, 1.0)),
                        _dim128_vector((1, 1.0)),
                        _dim128_vector((2, 1.0)),
                    ]
                ),
                np.stack(
                    [
                        _dim128_vector((0, 1.0)),
                        _dim128_vector((1, 1.0)),
                    ]
                ),
                np.stack([_dim128_vector((2, 1.0))]),
            ],
        ).pack()

    def _build_query_batch(self) -> kayak.LateQueryBatch:
        return kayak.query_batch(
            [
                np.stack(
                    [
                        _dim128_vector((0, 1.0)),
                        _dim128_vector((1, 1.0)),
                    ]
                ),
                np.stack(
                    [
                        _dim128_vector((0, 1.0)),
                        _dim128_vector((1, 1.0)),
                        _dim128_vector((2, 1.0)),
                    ]
                ),
            ]
        )

    def test_backend_info_reports_available_reference_backend(self) -> None:
        info = kayak.backend_info(kayak.NUMPY_REFERENCE_BACKEND)

        self.assertEqual(info.name, kayak.NUMPY_REFERENCE_BACKEND)
        self.assertTrue(info.available)
        self.assertFalse(info.requires_mojo)
        self.assertIn(kayak.NUMPY_REFERENCE_BACKEND, kayak.available_backends())

    def test_query_batch_exposes_vector_counts_and_layout_conversion(self) -> None:
        query_batch = self._build_query_batch()

        self.assertEqual(query_batch.batch_size, 2)
        self.assertEqual(query_batch.vector_dim, 128)
        self.assertEqual(query_batch.vector_counts, (2, 3))
        self.assertEqual(query_batch.layouts, ("nested", "nested"))

        flat_batch = query_batch.to_layout("flat_dim128")
        self.assertEqual(flat_batch.layouts, ("flat_dim128", "flat_dim128"))

    def test_maxsim_batch_matches_individual_scores(self) -> None:
        query_batch = self._build_query_batch()
        index = self._build_index()

        expected = tuple(
            kayak.maxsim(query, index, backend=kayak.NUMPY_REFERENCE_BACKEND)
            for query in query_batch.queries
        )
        actual = kayak.maxsim_batch(
            query_batch,
            index,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(len(actual), len(expected))
        for actual_scores, expected_scores in zip(actual, expected, strict=True):
            np.testing.assert_allclose(actual_scores.numpy(), expected_scores.numpy())

    def test_search_batch_matches_individual_search(self) -> None:
        query_batch = self._build_query_batch()
        index = self._build_index()

        expected = tuple(
            kayak.search(query, index, k=2, backend=kayak.NUMPY_REFERENCE_BACKEND)
            for query in query_batch.queries
        )
        actual = kayak.search_batch(
            query_batch,
            index,
            k=2,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(actual, expected)

    def test_index_select_preserves_requested_document_order(self) -> None:
        index = self._build_index()
        selected = index.select(["doc-c", "doc-a"])
        query = self._build_query_batch().queries[1]

        self.assertEqual(selected.doc_ids, ("doc-c", "doc-a"))
        self.assertEqual(selected.vector_counts, (1, 3))
        np.testing.assert_allclose(
            kayak.maxsim(query, selected).numpy(),
            np.array([1.0, 3.0], dtype=np.float32),
        )

    @unittest.skipUnless(
        _mojo_backend_available.__func__(), "mojo_exact_cpu backend requires Mojo"
    )
    def test_mojo_batch_matches_numpy_batch_for_flat_dim128(self) -> None:
        query_batch = self._build_query_batch().to_layout("flat_dim128")
        index = self._build_index().to_layout("hybrid_flat_dim128")

        numpy_scores = kayak.maxsim_batch(
            query_batch,
            index,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        mojo_scores = kayak.maxsim_batch(
            query_batch,
            index,
            backend=kayak.MOJO_EXACT_CPU_BACKEND,
        )

        for actual_scores, expected_scores in zip(
            mojo_scores, numpy_scores, strict=True
        ):
            np.testing.assert_allclose(actual_scores.numpy(), expected_scores.numpy())


if __name__ == "__main__":
    unittest.main()
