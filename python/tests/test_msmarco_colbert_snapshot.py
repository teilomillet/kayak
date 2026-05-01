from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from kayak_bridge.encoded_snapshot_loader import load_packed_snapshot_documents
from kayak_bridge.msmarco_colbert_snapshot import build_msmarco_colbert_snapshot
from kayak_bridge.msmarco_passage_task import MsmarcoPassagePaths


def _fake_document_encoder(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    del model_name, batch_size
    rows = []
    for index, _text in enumerate(texts):
        vectors = np.full((2, 4), float(index + 1), dtype=np.float32)
        token_ids = np.asarray([100 + index, 200 + index], dtype=np.int64)
        rows.append((vectors, token_ids))
    return tuple(rows)


def _fake_query_encoder(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, ...]:
    del model_name, batch_size
    return tuple(np.ones((1, 4), dtype=np.float32) for _text in texts)


class MsmarcoColbertSnapshotTests(unittest.TestCase):
    def test_build_snapshot_writes_shards_queries_and_manifest(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            collection = root_path / "collection.tsv"
            queries = root_path / "queries.tsv"
            qrels = root_path / "qrels"
            output = root_path / "snapshot"
            collection.write_text(
                "d0\talpha\n"
                "d1\tbeta\n"
                "d2\tgamma\n",
                encoding="utf-8",
            )
            queries.write_text("q0\talpha?\nq1\tbeta?\n", encoding="utf-8")
            qrels.write_text("q0 0 d0 1\nq1 0 d1 1\n", encoding="utf-8")

            manifest = build_msmarco_colbert_snapshot(
                paths=MsmarcoPassagePaths(
                    collection=collection,
                    queries=queries,
                    qrels=qrels,
                ),
                output=output,
                document_limit=3,
                query_limit=2,
                model_name="unit-model",
                vector_dim=4,
                shard_max_vectors=4,
                document_batch_size=2,
                query_batch_size=2,
                encode_document_batch=_fake_document_encoder,
                encode_query_batch=_fake_query_encoder,
            )

            shard_dirs = sorted((output / "shards").glob("[0-9][0-9][0-9][0-9][0-9][0-9]"))

        self.assertEqual(manifest.document_count, 3)
        self.assertEqual(manifest.document_vector_count, 6)
        self.assertEqual(manifest.shard_count, 2)
        self.assertEqual(len(shard_dirs), 2)
        self.assertEqual(manifest.query_manifest["query_count"], 2)
        self.assertEqual(manifest.query_manifest["query_vector_count"], 2)
        self.assertEqual(manifest.vector_dtype, "float16")
        self.assertEqual(manifest.token_id_dtype, "uint32")

    def test_resume_skips_completed_document_shards(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            collection = root_path / "collection.tsv"
            queries = root_path / "queries.tsv"
            qrels = root_path / "qrels"
            output = root_path / "snapshot"
            collection.write_text(
                "d0\talpha\n"
                "d1\tbeta\n"
                "d2\tgamma\n",
                encoding="utf-8",
            )
            queries.write_text("q0\talpha?\n", encoding="utf-8")
            qrels.write_text("q0 0 d0 1\n", encoding="utf-8")

            build_msmarco_colbert_snapshot(
                paths=MsmarcoPassagePaths(
                    collection=collection,
                    queries=queries,
                    qrels=qrels,
                ),
                output=output,
                document_limit=2,
                query_limit=1,
                model_name="unit-model",
                vector_dim=4,
                shard_max_vectors=4,
                encode_document_batch=_fake_document_encoder,
                encode_query_batch=_fake_query_encoder,
            )
            manifest = build_msmarco_colbert_snapshot(
                paths=MsmarcoPassagePaths(
                    collection=collection,
                    queries=queries,
                    qrels=qrels,
                ),
                output=output,
                document_limit=3,
                query_limit=1,
                model_name="unit-model",
                vector_dim=4,
                shard_max_vectors=4,
                resume=True,
                encode_document_batch=_fake_document_encoder,
                encode_query_batch=_fake_query_encoder,
            )

        self.assertEqual(manifest.document_count, 3)
        self.assertEqual(manifest.document_vector_count, 6)
        self.assertEqual(manifest.shard_count, 2)

    def test_include_query_positives_extends_bounded_document_prefix(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            collection = root_path / "collection.tsv"
            queries = root_path / "queries.tsv"
            qrels = root_path / "qrels"
            output = root_path / "snapshot"
            collection.write_text(
                "d0\talpha\n"
                "d1\tbeta\n"
                "d2\tgamma\n"
                "d3\tdelta\n",
                encoding="utf-8",
            )
            queries.write_text("q0\tdelta?\n", encoding="utf-8")
            qrels.write_text("q0 0 d3 1\n", encoding="utf-8")

            manifest = build_msmarco_colbert_snapshot(
                paths=MsmarcoPassagePaths(
                    collection=collection,
                    queries=queries,
                    qrels=qrels,
                ),
                output=output,
                document_limit=2,
                query_limit=1,
                model_name="unit-model",
                vector_dim=4,
                shard_max_vectors=8,
                include_query_positives=True,
                encode_document_batch=_fake_document_encoder,
                encode_query_batch=_fake_query_encoder,
            )
            documents = load_packed_snapshot_documents(output)

        self.assertEqual(manifest.document_count, 3)
        self.assertEqual(documents.doc_ids, ("d0", "d1", "d3"))
        self.assertEqual(manifest.source["required_positive_doc_count"], 1)


if __name__ == "__main__":
    unittest.main()
