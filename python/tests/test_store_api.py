from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

import kayak


def _matrix(*rows: tuple[float, ...]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def _dim128(*indices: int) -> np.ndarray:
    rows = []
    for index in indices:
        row = np.zeros(128, dtype=np.float32)
        row[index] = np.float32(1.0)
        rows.append(row)
    return np.stack(rows)


class StoreApiTests(unittest.TestCase):
    def _documents(self) -> kayak.LateDocuments:
        return kayak.documents(
            ["doc-a", "doc-b", "doc-c"],
            [
                _matrix((1.0, 0.0), (0.0, 1.0)),
                _matrix((1.0, 0.0), (1.0, 0.0)),
                _matrix((0.0, 1.0), (0.0, 1.0)),
            ],
            texts=["alpha", "beta", "gamma"],
        )

    def _metadata(self) -> list[dict[str, object]]:
        return [
            {"topic": "hardware", "tenant": "a"},
            {"topic": "software", "tenant": "a"},
            {"topic": "hardware", "tenant": "b"},
        ]

    def test_memory_store_supports_upsert_filter_delete_and_layouts(self) -> None:
        store = kayak.MemoryLateStore()
        store.upsert(self._documents(), metadata=self._metadata())

        stats = store.stats()
        self.assertEqual(stats.kind, "memory")
        self.assertEqual(stats.document_count, 3)
        self.assertEqual(stats.total_vector_count, 6)
        self.assertTrue(stats.has_texts)
        self.assertTrue(stats.has_metadata)

        filtered = store.load_index(where={"topic": "hardware"}, include_text=True)
        self.assertEqual(filtered.doc_ids, ("doc-a", "doc-c"))
        self.assertEqual(filtered.doc_texts, ("alpha", "gamma"))

        reordered = store.load_index(
            doc_ids=["doc-c", "doc-a"],
            include_text=True,
        )
        self.assertEqual(reordered.doc_ids, ("doc-c", "doc-a"))
        self.assertEqual(reordered.doc_texts, ("gamma", "alpha"))

        store.delete(["doc-b"])
        after_delete = store.load_index(include_text=True)
        self.assertEqual(after_delete.doc_ids, ("doc-a", "doc-c"))

    def test_directory_store_persists_and_materializes_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kayak-store"
            store = kayak.DirectoryLateStore(path)
            store.upsert(self._documents(), metadata=self._metadata())

            reopened = kayak.DirectoryLateStore(path)
            stats = reopened.stats()
            self.assertEqual(stats.kind, "directory")
            self.assertEqual(stats.document_count, 3)
            self.assertGreater(stats.storage_byte_size or 0, 0)

            full_index = reopened.load_index(include_text=False)
            self.assertEqual(full_index.doc_ids, ("doc-a", "doc-b", "doc-c"))
            self.assertIsNone(full_index.doc_texts)
            self.assertIsInstance(full_index.token_vectors, np.memmap)

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

    def test_directory_store_can_load_hybrid_flat_dim128(self) -> None:
        documents = kayak.documents(
            ["doc-a", "doc-b"],
            [
                _dim128(0, 1),
                _dim128(1, 2),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = kayak.DirectoryLateStore(Path(tmpdir) / "kayak-store")
            store.upsert(documents)

            index = store.load_index(layout="hybrid_flat_dim128")

            self.assertEqual(index.layout, "hybrid_flat_dim128")
            self.assertEqual(index.vector_dim, 128)
            self.assertEqual(index.doc_ids, ("doc-a", "doc-b"))

    def test_store_registry_supports_builtin_and_custom_factories(self) -> None:
        memory = kayak.open_store("memory")
        self.assertIsInstance(memory, kayak.MemoryLateStore)

        class _CustomStore(kayak.MemoryLateStore):
            pass

        kayak.register_store("custom-memory", _CustomStore, replace=True)
        custom = kayak.open_store("custom-memory")
        self.assertIsInstance(custom, _CustomStore)

    def test_store_contract_supports_close_and_context_manager(self) -> None:
        memory = kayak.MemoryLateStore()
        memory.close()
        with memory as entered_memory:
            self.assertIs(entered_memory, memory)

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = kayak.DirectoryLateStore(Path(tmpdir) / "kayak-store")
            directory.close()
            with directory as entered_directory:
                self.assertIs(entered_directory, directory)


if __name__ == "__main__":
    unittest.main()
