from __future__ import annotations

import shutil
import unittest

import numpy as np
import torch

import kayak
from kayak_bridge.cache_paths import REPO_ROOT


def _dim128_vector(*entries: tuple[int, float]) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    for index, value in entries:
        vector[index] = np.float32(value)
    return vector


class LateInteractionApiTests(unittest.TestCase):
    @staticmethod
    def _mojo_backend_available() -> bool:
        if shutil.which("mojo") is not None:
            return True
        return (REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "mojo").exists()

    def _build_fixture(self) -> tuple[kayak.LateQuery, kayak.LateDocuments]:
        query_vectors = torch.tensor(
            np.stack(
                [
                    _dim128_vector((0, 1.0)),
                    _dim128_vector((1, 1.0)),
                ]
            ),
            dtype=torch.float32,
        )

        document_vectors = [
            np.stack(
                [
                    _dim128_vector((0, 1.0)),
                    _dim128_vector((1, 1.0)),
                ]
            ),
            np.stack(
                [
                    _dim128_vector((0, 1.0)),
                    _dim128_vector((0, 0.5), (1, 0.5)),
                ]
            ),
            np.stack(
                [
                    _dim128_vector((1, 1.0)),
                    _dim128_vector((0, 1.0)),
                ]
            ),
        ]

        return kayak.query(query_vectors), kayak.documents(
            ["doc-a", "doc-b", "doc-c"], document_vectors
        )

    def test_query_accepts_torch_and_exposes_shape(self) -> None:
        late_query, _ = self._build_fixture()

        self.assertEqual(late_query.layout, "nested")
        self.assertEqual(late_query.shape, (2, 128))
        self.assertEqual(late_query.vector_count, 2)
        self.assertEqual(late_query.vector_dim, 128)

    def test_documents_pack_preserves_offsets_and_vector_counts(self) -> None:
        _, late_documents = self._build_fixture()

        packed = late_documents.pack()

        self.assertEqual(packed.layout, "packed")
        self.assertEqual(tuple(int(offset) for offset in packed.doc_offsets), (0, 2, 4, 6))
        self.assertEqual(packed.vector_counts, (2, 2, 2))
        self.assertEqual(packed.total_vector_count, 6)

    def test_flat_query_requires_dim128(self) -> None:
        narrow_query = kayak.query(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "dim128"):
            narrow_query.to_layout("flat_dim128")

    def test_maxsim_matches_across_packed_and_hybrid_flat_layouts(self) -> None:
        late_query, late_documents = self._build_fixture()
        packed = late_documents.pack()
        hybrid = packed.to_layout("hybrid_flat_dim128")
        flat_query = late_query.to_layout("flat_dim128")

        packed_scores = kayak.maxsim(late_query, packed)
        hybrid_scores = kayak.maxsim(flat_query, hybrid)

        np.testing.assert_allclose(
            packed_scores.numpy(),
            np.array([2.0, 1.5, 2.0], dtype=np.float32),
        )
        np.testing.assert_allclose(packed_scores.numpy(), hybrid_scores.numpy())

    def test_search_preserves_stable_tie_order(self) -> None:
        late_query, late_documents = self._build_fixture()
        hits = kayak.search(late_query, late_documents.pack(), k=2)

        self.assertEqual([hit.doc_id for hit in hits], ["doc-a", "doc-c"])
        self.assertEqual([hit.score for hit in hits], [2.0, 2.0])

    def test_scores_convert_back_to_torch(self) -> None:
        late_query, late_documents = self._build_fixture()
        scores = kayak.maxsim(late_query, late_documents.pack())

        tensor = scores.torch()

        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), (3,))

    @unittest.skipUnless(
        _mojo_backend_available.__func__(), "mojo_exact_cpu backend requires Mojo"
    )
    def test_mojo_exact_cpu_matches_numpy_reference_for_packed_index(self) -> None:
        query = kayak.query(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )
        index = kayak.documents(
            ["doc-a", "doc-b"],
            [
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
            ],
        ).pack()

        numpy_scores = kayak.maxsim(
            query, index, backend=kayak.NUMPY_REFERENCE_BACKEND
        )
        mojo_scores = kayak.maxsim(
            query, index, backend=kayak.MOJO_EXACT_CPU_BACKEND
        )

        np.testing.assert_allclose(numpy_scores.numpy(), mojo_scores.numpy())

    @unittest.skipUnless(
        _mojo_backend_available.__func__(), "mojo_exact_cpu backend requires Mojo"
    )
    def test_mojo_exact_cpu_matches_numpy_reference_for_flat_dim128(self) -> None:
        late_query, late_documents = self._build_fixture()
        packed_index = late_documents.pack()
        hybrid_index = packed_index.to_layout("hybrid_flat_dim128")
        flat_query = late_query.to_layout("flat_dim128")

        numpy_scores = kayak.maxsim(
            flat_query, hybrid_index, backend=kayak.NUMPY_REFERENCE_BACKEND
        )
        mojo_scores = kayak.maxsim(
            flat_query, hybrid_index, backend=kayak.MOJO_EXACT_CPU_BACKEND
        )

        np.testing.assert_allclose(numpy_scores.numpy(), mojo_scores.numpy())


if __name__ == "__main__":
    unittest.main()
