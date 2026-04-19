from __future__ import annotations

import unittest

import numpy as np

from kayak_bridge.raw_text_chunk_sweep import build_raw_text_chunk_sweep_bundle
from kayak_bridge.raw_text_chunk_sweep import build_raw_text_chunked_onevec_artifact
from kayak_bridge.raw_text_chunk_sweep import chunk_text_by_source_tokens
from kayak_bridge.raw_text_chunk_sweep import RawTextChunkSpec
from kayak_bridge.source_text_chunking import source_token_ids


class _MockTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self.model_max_length = 512

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

    def decode(
        self, token_ids, skip_special_tokens: bool = True
    ) -> str:
        del skip_special_tokens
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


class _VerboseAwareTokenizer:
    def __init__(self) -> None:
        self.model_max_length = 4
        self.last_call_kwargs: dict[str, object] | None = None

    def __call__(self, text: str, **kwargs):
        self.last_call_kwargs = kwargs
        return {"input_ids": list(range(1, len(text.split()) + 1))}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        raise AssertionError("source_token_ids should prefer the callable tokenizer path")

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"tok{token_id}" for token_id in token_ids)


class _FakeEncoder:
    DIM = 32

    def __init__(self) -> None:
        self._token_to_index: dict[str, int] = {}

    def _token_vector(self, token: str) -> list[float]:
        index = self._token_to_index.setdefault(token, len(self._token_to_index))
        if index >= self.DIM:
            raise ValueError("Increase DIM for fake encoder test fixture.")
        vector = np.zeros(self.DIM, dtype=np.float32)
        vector[index] = np.float32(1.0)
        return vector.tolist()

    def encode_document_texts(
        self,
        texts: list[str],
        model_name: str,
        batch_size: int,
    ):
        del model_name
        del batch_size
        return tuple(
            [self._token_vector(token) for token in text.split()]
            for text in texts
        )

    def encode_query(self, text: str) -> list[list[float]]:
        return [self._token_vector(token) for token in text.split()]


class RawTextChunkSweepTests(unittest.TestCase):
    def test_chunk_text_by_source_tokens_respects_overlap(self) -> None:
        tokenizer = _MockTokenizer()
        chunks = chunk_text_by_source_tokens(
            "a b c d e f g",
            tokenizer=tokenizer,
            spec=RawTextChunkSpec(4, 2),
        )

        self.assertEqual(chunks, ("a b c d", "c d e f", "e f g"))

    def test_source_token_ids_prefers_callable_tokenizer_without_verbose_warning(self) -> None:
        tokenizer = _VerboseAwareTokenizer()

        token_ids = source_token_ids(
            "a b c d e f",
            tokenizer=tokenizer,
        )

        self.assertEqual(token_ids, (1, 2, 3, 4, 5, 6))
        self.assertIsNotNone(tokenizer.last_call_kwargs)
        self.assertEqual(tokenizer.last_call_kwargs["truncation"], False)
        self.assertEqual(tokenizer.last_call_kwargs["verbose"], False)

    def test_build_artifact_reports_chunk_metadata(self) -> None:
        tokenizer = _MockTokenizer()
        encoder = _FakeEncoder()
        task = {
            "model_name": "mock-model",
            "documents": [
                {"doc_id": "doc-a", "text": "a b c d e f"},
                {"doc_id": "doc-b", "text": "g h i j k l"},
            ],
        }

        artifact = build_raw_text_chunked_onevec_artifact(
            task,
            spec=RawTextChunkSpec(4, 2),
            tokenizer=tokenizer,
            encode_document_texts_fn=encoder.encode_document_texts,
            document_vector_cap=4,
        )

        self.assertEqual(artifact.encoder_doc_maxlen, 4)
        self.assertEqual(artifact.generated_chunk_count, 4)
        self.assertAlmostEqual(artifact.mean_chunks_per_document, 2.0)
        self.assertEqual(artifact.max_chunks_per_document, 2)
        self.assertAlmostEqual(artifact.mean_source_chunk_token_count, 4.0)
        self.assertEqual(artifact.max_source_chunk_token_count, 4)
        self.assertAlmostEqual(artifact.mean_chunk_vector_count, 4.0)
        self.assertEqual(artifact.max_chunk_vector_count, 4)
        self.assertEqual(artifact.at_doc_maxlen_chunk_count, 4)
        self.assertAlmostEqual(artifact.at_doc_maxlen_chunk_fraction, 1.0)

    def test_build_bundle_compares_exact_onevec_and_chunked_baseline(self) -> None:
        tokenizer = _MockTokenizer()
        encoder = _FakeEncoder()
        task = {
            "dataset_id": "mock://raw-text",
            "family": "mock",
            "slice_name": "raw_text_chunk_sweep_fixture",
            "model_name": "mock-model",
            "primary_metric": "ndcg",
            "k": 2,
            "documents": [
                {
                    "doc_id": "doc-relevant",
                    "text": "alpha beta noise noise",
                    "vectors": encoder.encode_query("alpha beta noise noise"),
                    "vector_count": 4,
                },
                {
                    "doc_id": "doc-partial",
                    "text": "alpha alpha alpha alpha",
                    "vectors": encoder.encode_query("alpha alpha alpha alpha"),
                    "vector_count": 4,
                },
                {
                    "doc_id": "doc-other",
                    "text": "gamma delta",
                    "vectors": encoder.encode_query("gamma delta"),
                    "vector_count": 2,
                },
            ],
            "queries": [
                {
                    "query_id": "q1",
                    "text": "alpha beta",
                    "vectors": encoder.encode_query("alpha beta"),
                    "vector_count": 2,
                    "relevant_doc_ids": ["doc-relevant"],
                }
            ],
        }

        bundle = build_raw_text_chunk_sweep_bundle(
            task,
            specs=(RawTextChunkSpec(2, 0),),
            tokenizer=tokenizer,
            encode_document_texts_fn=encoder.encode_document_texts,
            document_vector_cap=None,
        )

        self.assertEqual(bundle.dataset_id, "mock://raw-text")
        self.assertEqual(bundle.primary_metric, "ndcg")
        self.assertEqual(len(bundle.rows), 1)
        self.assertGreater(bundle.exact_primary_value, bundle.onevec_primary_value)
        row = bundle.rows[0]
        self.assertEqual(row["source_chunk_tokens"], 2)
        self.assertEqual(row["source_overlap_tokens"], 0)
        self.assertIsNone(row["encoder_doc_maxlen"])
        self.assertEqual(row["generated_chunk_count"], 5)
        self.assertEqual(row["at_doc_maxlen_chunk_count"], 0)
        self.assertAlmostEqual(row["at_doc_maxlen_chunk_fraction"], 0.0)
        self.assertGreaterEqual(row["primary_value"], bundle.onevec_primary_value)

    def test_chunk_size_above_tokenizer_limit_is_rejected(self) -> None:
        tokenizer = _MockTokenizer()
        encoder = _FakeEncoder()
        task = {
            "model_name": "mock-model",
            "documents": [{"doc_id": "doc-a", "text": "a b c d"}],
        }

        with self.assertRaisesRegex(ValueError, "tokenizer model_max_length"):
            build_raw_text_chunked_onevec_artifact(
                task,
                spec=RawTextChunkSpec(600, 0),
                tokenizer=tokenizer,
                encode_document_texts_fn=encoder.encode_document_texts,
                document_vector_cap=None,
            )


if __name__ == "__main__":
    unittest.main()
