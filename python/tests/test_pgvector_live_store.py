from __future__ import annotations

import hashlib
import os
from pathlib import Path
import unittest
import uuid

import numpy as np

import kayak


DSN = os.environ.get("KAYAK_PGVECTOR_TEST_DSN")


def _token_vectors(text: str) -> np.ndarray:
    rows = []
    for token in text.lower().split():
        vector = np.zeros(128, dtype=np.float32)
        hot_index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % 128
        vector[hot_index] = np.float32(1.0)
        rows.append(vector)
    return np.stack(rows)


@unittest.skipUnless(
    DSN,
    "set KAYAK_PGVECTOR_TEST_DSN to run the live pgvector integration test",
)
class PgVectorLiveStoreTests(unittest.TestCase):
    def test_pgvector_store_round_trips_exact_index_against_real_postgres(self) -> None:
        table_name = f"kayak_live_{uuid.uuid4().hex[:12]}"
        encoder = kayak.open_encoder(
            "callable",
            query_encoder=_token_vectors,
            document_encoder=_token_vectors,
        )
        documents = encoder.encode_documents(
            ["doc-a", "doc-b", "doc-c"],
            [
                "pixi installs python mojo and kayak together",
                "pgvector keeps late interaction rows in postgres",
                "kayak loads the exact slice and searches it locally",
            ],
        )

        with kayak.open_store("pgvector", dsn=DSN, table_name=table_name) as store:
            store.upsert(
                documents,
                metadata=[
                    {"topic": "installation", "tenant": "acme"},
                    {"topic": "storage", "tenant": "acme"},
                    {"topic": "search", "tenant": "acme"},
                ],
            )

            stats = store.stats()
            self.assertEqual(stats.kind, "pgvector")
            self.assertEqual(stats.document_count, 3)
            self.assertEqual(stats.vector_dim, 128)

            index = store.load_index(where={"topic": "search"}, include_text=True)
            self.assertEqual(index.doc_ids, ("doc-c",))
            self.assertEqual(
                index.doc_texts,
                ("kayak loads the exact slice and searches it locally",),
            )

            query = encoder.encode_query("exact late interaction search")
            mojo_hits = kayak.search(
                query,
                index,
                k=1,
                backend=kayak.MOJO_EXACT_CPU_BACKEND,
            )
            numpy_hits = kayak.search(
                query,
                index,
                k=1,
                backend=kayak.NUMPY_REFERENCE_BACKEND,
            )
            self.assertEqual(mojo_hits[0].doc_id, "doc-c")
            self.assertEqual(numpy_hits[0].doc_id, "doc-c")
            self.assertAlmostEqual(mojo_hits[0].score, numpy_hits[0].score)


if __name__ == "__main__":
    unittest.main()
