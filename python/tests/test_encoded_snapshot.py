from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from kayak_bridge.encoded_snapshot import (
    EncodedDocumentShardWriter,
    estimate_snapshot_payload_bytes,
    load_completed_shards,
)


class EncodedSnapshotTests(unittest.TestCase):
    def test_estimate_counts_vectors_token_ids_and_offsets(self) -> None:
        estimate = estimate_snapshot_payload_bytes(
            document_count=3,
            vector_count=10,
            vector_dim=4,
            vector_dtype="float16",
            token_id_dtype="uint32",
        )

        self.assertEqual(estimate.vector_bytes, 80)
        self.assertEqual(estimate.token_id_bytes, 40)
        self.assertEqual(estimate.doc_offset_bytes, 32)
        self.assertEqual(estimate.payload_bytes, 152)

    def test_writer_streams_shard_files_and_stats(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            writer = EncodedDocumentShardWriter(
                snapshot_root=root_path,
                shard_index=0,
                vector_dim=4,
                vector_dtype="float16",
                token_id_dtype="uint32",
            )
            writer.append(
                doc_id="d1",
                vectors=np.ones((2, 4), dtype=np.float32),
                token_ids=np.asarray([10, 11], dtype=np.int64),
            )
            summary = writer.close()

            shards = load_completed_shards(root_path)

        self.assertEqual(summary.document_count, 1)
        self.assertEqual(summary.vector_count, 2)
        self.assertEqual(len(shards), 1)
        self.assertEqual(shards[0].vector_bytes, 16)
        self.assertEqual(shards[0].token_id_bytes, 8)


if __name__ == "__main__":
    unittest.main()
