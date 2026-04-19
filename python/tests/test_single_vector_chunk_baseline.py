from __future__ import annotations

import unittest

from kayak_bridge.single_vector_chunk_baseline import (
    build_single_vector_chunk_artifact,
)
from kayak_bridge.single_vector_chunk_baseline import (
    build_single_vector_chunk_sweep_bundle,
)
from kayak_bridge.source_text_chunking import RawTextChunkSpec


class _MockTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self.model_max_length = 4096
        self.model_name = "mock-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_ids: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            token_ids.append(self._token_to_id[token])
        return token_ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


class _FakeDenseEmbedder:
    DIM = 32

    def __init__(self) -> None:
        self._token_to_index: dict[str, int] = {}

    def _token_index(self, token: str) -> int:
        index = self._token_to_index.setdefault(token, len(self._token_to_index))
        if index >= self.DIM:
            raise ValueError("Increase DIM for fake dense embedder fixture.")
        return index

    def embed_texts(
        self,
        texts: list[str],
        model_name: str,
        batch_size: int,
    ):
        del model_name
        del batch_size
        rows = []
        for text in texts:
            vector = [0.0] * self.DIM
            for token in set(text.split()):
                vector[self._token_index(token)] = 1.0
            rows.append(vector)
        return tuple(rows)

    def encode_query_tokens(self, text: str) -> list[list[float]]:
        return [[value] for value in ()]


class SingleVectorChunkBaselineTests(unittest.TestCase):
    def test_build_artifact_reports_chunk_metadata(self) -> None:
        tokenizer = _MockTokenizer()
        embedder = _FakeDenseEmbedder()
        task = {
            "documents": [
                {"doc_id": "doc-a", "text": "alpha beta gamma delta epsilon"},
                {"doc_id": "doc-b", "text": "zeta eta theta iota"},
            ]
        }

        artifact = build_single_vector_chunk_artifact(
            task,
            spec=RawTextChunkSpec(3, 1),
            tokenizer=tokenizer,
            embed_texts_fn=embedder.embed_texts,
            embedding_model_name="mock-dense",
            embedding_batch_size=4,
        )

        self.assertEqual(artifact.embedding_model_name, "mock-dense")
        self.assertEqual(artifact.generated_chunk_count, 4)
        self.assertAlmostEqual(artifact.mean_chunks_per_document, 2.0)
        self.assertEqual(artifact.max_chunks_per_document, 2)
        self.assertAlmostEqual(artifact.mean_source_chunk_token_count, 2.75)
        self.assertEqual(artifact.max_source_chunk_token_count, 3)
        self.assertEqual(artifact.vector_dim, _FakeDenseEmbedder.DIM)
        self.assertTrue(artifact.normalize_embeddings)

    def test_build_bundle_records_dense_chunk_rows(self) -> None:
        tokenizer = _MockTokenizer()
        embedder = _FakeDenseEmbedder()
        task = {
            "dataset_id": "mock://dense",
            "family": "mock",
            "slice_name": "single_vector_chunk_fixture",
            "model_name": "mock-colbert",
            "primary_metric": "ndcg",
            "k": 2,
            "documents": [
                {
                    "doc_id": "doc-relevant",
                    "text": "alpha beta noise noise",
                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                    "vector_count": 2,
                },
                {
                    "doc_id": "doc-partial",
                    "text": "alpha alpha alpha alpha",
                    "vectors": [[1.0, 0.0], [1.0, 0.0]],
                    "vector_count": 2,
                },
                {
                    "doc_id": "doc-other",
                    "text": "gamma delta",
                    "vectors": [[0.0, 1.0], [1.0, 0.0]],
                    "vector_count": 2,
                },
            ],
            "queries": [
                {
                    "query_id": "q1",
                    "text": "alpha beta",
                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                    "vector_count": 2,
                    "relevant_doc_ids": ["doc-relevant"],
                }
            ],
        }

        bundle = build_single_vector_chunk_sweep_bundle(
            task,
            specs=(RawTextChunkSpec(2, 0),),
            tokenizer=tokenizer,
            embed_texts_fn=embedder.embed_texts,
            embedding_model_name="mock-dense",
            embedding_batch_size=8,
        )

        self.assertEqual(bundle.dataset_id, "mock://dense")
        self.assertEqual(bundle.exact_model_name, "mock-colbert")
        self.assertGreater(bundle.exact_primary_value, 0.0)
        self.assertEqual(len(bundle.rows), 1)
        row = bundle.rows[0]
        self.assertEqual(row["embedding_model_name"], "mock-dense")
        self.assertEqual(row["source_tokenizer_name"], "mock-tokenizer")
        self.assertEqual(row["source_chunk_tokens"], 2)
        self.assertEqual(row["source_overlap_tokens"], 0)
        self.assertEqual(row["generated_chunk_count"], 5)
        self.assertEqual(row["vector_dim"], _FakeDenseEmbedder.DIM)
        self.assertEqual(row["parent_aggregation"], "dedup_max_chunk_score")

    def test_inconsistent_embedding_dimensions_are_rejected(self) -> None:
        tokenizer = _MockTokenizer()

        def inconsistent_embedder(
            texts: list[str],
            model_name: str,
            batch_size: int,
        ):
            del texts
            del model_name
            del batch_size
            return ([1.0, 0.0], [1.0, 0.0, 0.0])

        task = {
            "documents": [
                {"doc_id": "doc-a", "text": "alpha beta"},
                {"doc_id": "doc-b", "text": "gamma delta"},
            ]
        }

        with self.assertRaisesRegex(ValueError, "share one vector dimension"):
            build_single_vector_chunk_artifact(
                task,
                spec=RawTextChunkSpec(2, 0),
                tokenizer=tokenizer,
                embed_texts_fn=inconsistent_embedder,
                embedding_model_name="mock-dense",
            )


if __name__ == "__main__":
    unittest.main()
