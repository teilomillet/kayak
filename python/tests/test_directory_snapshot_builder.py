from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

import kayak
from kayak_bridge.directory_snapshot_builder import (
    StreamedPackedSnapshotDocument,
    write_streamed_packed_directory_snapshot,
)


def _matrix(*rows: tuple[float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


class DirectorySnapshotBuilderTests(unittest.TestCase):
    def test_streamed_snapshot_is_loadable_as_directory_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "store"
            summary = write_streamed_packed_directory_snapshot(
                root,
                [
                    StreamedPackedSnapshotDocument("doc-a", _matrix((1.0, 0.0))),
                    StreamedPackedSnapshotDocument(
                        "doc-b",
                        _matrix((0.0, 1.0), (1.0, 1.0)),
                    ),
                ],
                document_count=2,
                vector_dim=2,
                max_document_vector_count=3,
            )

            self.assertEqual(summary.document_count, 2)
            self.assertEqual(summary.total_vector_count, 3)
            self.assertEqual(summary.reserved_vector_capacity, 6)

            store = kayak.DirectoryLateStore(root)
            full_index = store.load_index()
            self.assertEqual(full_index.doc_ids, ("doc-a", "doc-b"))
            self.assertEqual(full_index.total_vector_count, 3)
            np.testing.assert_allclose(
                full_index.as_packed_token_matrix(),
                _matrix((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            )

    def test_streamed_snapshot_rejects_document_over_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "store"
            with self.assertRaises(ValueError):
                write_streamed_packed_directory_snapshot(
                    root,
                    [
                        StreamedPackedSnapshotDocument(
                            "doc-a",
                            _matrix((1.0, 0.0), (0.0, 1.0)),
                        )
                    ],
                    document_count=1,
                    vector_dim=2,
                    max_document_vector_count=1,
                )


if __name__ == "__main__":
    unittest.main()
