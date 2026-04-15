from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

import kayak


_HAS_LANCEDB = importlib.util.find_spec("lancedb") is not None
_HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


def _matrix(*rows: tuple[float, ...]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


@unittest.skipUnless(_HAS_LANCEDB and _HAS_PYARROW, "lancedb and pyarrow are required")
class LanceDBStoreApiTests(unittest.TestCase):
    def _documents(self) -> kayak.LateDocuments:
        return kayak.documents(
            ["doc-a", "doc-b", "doc-c"],
            [
                _matrix((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                _matrix((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                _matrix((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ],
            texts=["alpha", "beta", "gamma"],
        )

    def _metadata(self) -> list[dict[str, object]]:
        return [
            {"topic": "hardware", "tenant": "a"},
            {"topic": "software", "tenant": "a"},
            {"topic": "hardware", "tenant": "b"},
        ]

    def test_lancedb_store_supports_upsert_filter_delete_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = kayak.LanceDBLateStore(Path(tmpdir) / "lancedb-store")
            store.upsert(self._documents(), metadata=self._metadata())

            reopened = kayak.open_store("lancedb", path=Path(tmpdir) / "lancedb-store")
            self.assertIsInstance(reopened, kayak.LanceDBLateStore)

            stats = reopened.stats()
            self.assertEqual(stats.kind, "lancedb")
            self.assertEqual(stats.document_count, 3)
            self.assertEqual(stats.total_vector_count, 6)
            self.assertEqual(stats.vector_dim, 3)
            self.assertTrue(stats.has_texts)
            self.assertTrue(stats.has_metadata)
            self.assertGreater(stats.storage_byte_size or 0, 0)

            full_index = reopened.load_index(include_text=True)
            self.assertEqual(full_index.doc_ids, ("doc-a", "doc-b", "doc-c"))
            self.assertEqual(full_index.doc_texts, ("alpha", "beta", "gamma"))

            subset = reopened.load_index(
                doc_ids=["doc-c", "doc-a"],
                include_text=True,
            )
            self.assertEqual(subset.doc_ids, ("doc-c", "doc-a"))
            self.assertEqual(subset.doc_texts, ("gamma", "alpha"))

            filtered = reopened.load_index(
                where={"tenant": "a"},
                include_text=True,
            )
            self.assertEqual(filtered.doc_ids, ("doc-a", "doc-b"))
            self.assertEqual(filtered.doc_texts, ("alpha", "beta"))

            filtered_subset = reopened.load_index(
                doc_ids=["doc-c", "doc-a", "doc-b"],
                where={"topic": "hardware"},
                include_text=True,
            )
            self.assertEqual(filtered_subset.doc_ids, ("doc-c", "doc-a"))
            self.assertEqual(filtered_subset.doc_texts, ("gamma", "alpha"))

            reopened.delete(["doc-b"])
            after_delete = reopened.load_index(include_text=True)
            self.assertEqual(after_delete.doc_ids, ("doc-a", "doc-c"))

    def test_lancedb_store_loads_exact_searchable_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = kayak.LanceDBLateStore(Path(tmpdir) / "lancedb-store")
            store.upsert(self._documents(), metadata=self._metadata())

            index = store.load_index()
            query = kayak.query([[0.0, 0.0, 1.0]])
            result = index.maxsim(query)

            self.assertEqual(result.topk(2)[0].doc_id, "doc-b")


if __name__ == "__main__":
    unittest.main()
