from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from kayak_bridge.encoded_snapshot import (
    EncodedDocumentShardWriter,
    write_snapshot_manifest,
)
from kayak_bridge.encoded_snapshot_loader import (
    load_packed_snapshot_documents,
    load_snapshot_queries,
)
from kayak_bridge.tachiom_snapshot_benchmark import benchmark_snapshot_with_tachiom
from kayak_bridge.tachiom_types import TachiomTacConfig


def _write_tiny_snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    snapshot.mkdir()
    (snapshot / "shards").mkdir()
    writer = EncodedDocumentShardWriter(
        snapshot_root=snapshot,
        shard_index=0,
        vector_dim=4,
        vector_dtype="float16",
        token_id_dtype="uint32",
    )
    writer.append(
        doc_id="d0",
        vectors=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        token_ids=np.asarray([10], dtype=np.int64),
    )
    writer.append(
        doc_id="d1",
        vectors=np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        token_ids=np.asarray([20], dtype=np.int64),
    )
    shard = writer.close()

    query_root = snapshot / "queries"
    query_root.mkdir()
    np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ).tofile(query_root / "query_vectors.f32")
    np.save(query_root / "query_offsets.u64.npy", np.asarray([0, 1, 2], dtype=np.uint64))
    (query_root / "queries.jsonl").write_text(
        '{"query_id": "q0", "relevant_doc_ids": ["d0"], "text": "alpha"}\n'
        '{"query_id": "q1", "relevant_doc_ids": ["d1"], "text": "beta"}\n',
        encoding="utf-8",
    )
    query_manifest = {
        "files": {
            "queries": "queries.jsonl",
            "query_offsets": "query_offsets.u64.npy",
            "query_vectors": "query_vectors.f32",
        },
        "k": 1,
        "primary_metric": "mrr",
        "query_count": 2,
        "query_vector_count": 2,
        "vector_dim": 4,
        "vector_dtype": "float32",
    }
    (query_root / "manifest.json").write_text(
        "{\n"
        '  "files": {\n'
        '    "queries": "queries.jsonl",\n'
        '    "query_offsets": "query_offsets.u64.npy",\n'
        '    "query_vectors": "query_vectors.f32"\n'
        "  },\n"
        '  "k": 1,\n'
        '  "primary_metric": "mrr",\n'
        '  "query_count": 2,\n'
        '  "query_vector_count": 2,\n'
        '  "vector_dim": 4,\n'
        '  "vector_dtype": "float32"\n'
        "}\n",
        encoding="utf-8",
    )
    write_snapshot_manifest(
        snapshot_root=snapshot,
        dataset_id="dataset://tiny",
        model_name="unit-model",
        vector_dim=4,
        vector_dtype="float16",
        token_id_dtype="uint32",
        shards=[shard],
        source={"unit": True},
        query_manifest=query_manifest,
    )
    return snapshot


class EncodedSnapshotLoaderTests(unittest.TestCase):
    def test_load_packed_documents_and_queries(self) -> None:
        with TemporaryDirectory() as root:
            snapshot = _write_tiny_snapshot(Path(root))

            documents = load_packed_snapshot_documents(snapshot)
            queries = load_snapshot_queries(snapshot)

        self.assertEqual(documents.doc_ids, ("d0", "d1"))
        self.assertEqual(documents.loaded_vector_count, 2)
        self.assertEqual(documents.doc_offsets.tolist(), [0, 1, 2])
        self.assertEqual(documents.token_ids.tolist(), [10, 20])
        self.assertEqual(queries.query_ids, ("q0", "q1"))
        self.assertEqual(queries.query_vector_count, 2)

    def test_benchmark_snapshot_with_tachiom_uses_binary_snapshot(self) -> None:
        with TemporaryDirectory() as root:
            snapshot = _write_tiny_snapshot(Path(root))

            summary = benchmark_snapshot_with_tachiom(
                snapshot,
                engine="tachiom_tac",
                tac_config=TachiomTacConfig(centroid_count=2, candidate_k=2),
                warmup_iterations=0,
                measurement_iterations=1,
                run_exact=True,
            )

        self.assertEqual(summary.document_count, 2)
        self.assertEqual(summary.document_vector_count_total, 2)
        self.assertEqual(summary.source_vector_dtype, "float16")
        self.assertAlmostEqual(summary.primary_value, 1.0)
        self.assertAlmostEqual(summary.exact_primary_value, 1.0)
        self.assertAlmostEqual(summary.final_recall_at_k_vs_exact, 1.0)


if __name__ == "__main__":
    unittest.main()
