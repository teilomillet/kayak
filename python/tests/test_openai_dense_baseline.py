from __future__ import annotations

import tempfile
import unittest

from kayak_bridge.openai_dense_baseline import OpenAIEmbeddingTextBatcher


class _FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingItem(embedding) for embedding in embeddings]


class _FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, *, input, model, dimensions=None):
        self.calls.append(
            {
                "input": list(input),
                "model": model,
                "dimensions": dimensions,
            }
        )
        embeddings = []
        for text in input:
            vector = [
                float(len(text)),
                float(sum(ord(char) for char in text) % 97),
                float(len(text.split())),
            ]
            embeddings.append(vector)
        return _FakeEmbeddingResponse(embeddings)


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsEndpoint()


class OpenAIDenseBaselineTests(unittest.TestCase):
    def test_embed_texts_caches_by_model_dimensions_and_text(self) -> None:
        client = _FakeClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            batcher = OpenAIEmbeddingTextBatcher(
                client=client,
                dimensions=256,
                cache_dir=temp_dir,
            )

            first = batcher.embed_texts(
                ["alpha beta", "gamma"],
                "text-embedding-3-small",
                8,
            )
            second = batcher.embed_texts(
                ["alpha beta", "gamma"],
                "text-embedding-3-small",
                8,
            )

            self.assertEqual(len(client.embeddings.calls), 1)
            self.assertEqual(first, second)

            third = batcher.embed_texts(
                ["alpha beta", "delta"],
                "text-embedding-3-small",
                8,
            )

            self.assertEqual(len(client.embeddings.calls), 2)
            self.assertEqual(third[0], first[0])
            self.assertNotEqual(third[1], first[1])

    def test_embed_texts_requires_positive_batch_size(self) -> None:
        batcher = OpenAIEmbeddingTextBatcher(client=_FakeClient(), cache_dir=None)

        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            batcher.embed_texts(["alpha"], "text-embedding-3-small", 0)


if __name__ == "__main__":
    unittest.main()
