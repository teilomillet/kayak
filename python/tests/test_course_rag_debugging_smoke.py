from __future__ import annotations

import unittest

import numpy as np

import kayak


class CourseRagRetrievalDebuggingSmokeTests(unittest.TestCase):
    DIM = 64

    def setUp(self) -> None:
        self._token_to_index: dict[str, int] = {}

    def _token_vector(self, token: str) -> np.ndarray:
        index = self._token_to_index.setdefault(token, len(self._token_to_index))
        if index >= self.DIM:
            raise ValueError("Increase DIM for this smoke test.")
        vector = np.zeros(self.DIM, dtype=np.float32)
        vector[index] = np.float32(1.0)
        return vector

    def _encode_tokens(self, tokens: list[str]) -> np.ndarray:
        return np.stack([self._token_vector(token) for token in tokens])

    def _dense_mean(self, tokens: list[str]) -> np.ndarray:
        return self._encode_tokens(tokens).mean(axis=0)

    def _cosine_similarity(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))

    def _build_documents(self) -> dict[str, list[str]]:
        return {
            "doc-relevant": ["cancel", "subscription"]
            + [f"noise-{index}" for index in range(20)],
            "doc-partial": ["cancel", "cancel", "cancel", "cancel"],
            "doc-other": ["billing", "invoice"],
        }

    def test_kayak_beats_naive_mean_pooling_on_teaching_example(self) -> None:
        documents = self._build_documents()
        query_tokens = ["cancel", "subscription"]

        index = kayak.documents(
            list(documents.keys()),
            [self._encode_tokens(tokens) for tokens in documents.values()],
        ).pack()
        query = kayak.query(self._encode_tokens(query_tokens))

        hits = kayak.search(
            query,
            index,
            k=3,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        dense_scores = sorted(
            (
                (
                    doc_id,
                    self._cosine_similarity(
                        self._dense_mean(query_tokens),
                        self._dense_mean(tokens),
                    ),
                )
                for doc_id, tokens in documents.items()
            ),
            key=lambda row: row[1],
            reverse=True,
        )

        self.assertEqual(hits[0].doc_id, "doc-relevant")
        self.assertEqual(dense_scores[0][0], "doc-partial")
        self.assertEqual(hits[1].doc_id, "doc-partial")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_batch_search_matches_loop_on_same_teaching_index(self) -> None:
        documents = self._build_documents()
        index = kayak.documents(
            list(documents.keys()),
            [self._encode_tokens(tokens) for tokens in documents.values()],
        ).pack()

        queries = {
            "cancel subscription": ["cancel", "subscription"],
            "cancel only": ["cancel"],
            "invoice": ["invoice"],
        }
        batch = kayak.query_batch(
            [self._encode_tokens(tokens) for tokens in queries.values()]
        )

        expected = tuple(
            kayak.search(
                kayak.query(self._encode_tokens(tokens), text=name),
                index,
                k=1,
                backend=kayak.NUMPY_REFERENCE_BACKEND,
            )
            for name, tokens in queries.items()
        )
        actual = kayak.search_batch(
            batch,
            index,
            k=1,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(actual, expected)
        self.assertEqual(actual[0][0].doc_id, "doc-relevant")
        self.assertEqual(actual[1][0].doc_id, "doc-relevant")
        self.assertEqual(actual[2][0].doc_id, "doc-other")


if __name__ == "__main__":
    unittest.main()
