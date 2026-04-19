from __future__ import annotations

import unittest

import numpy as np

import kayak


class CourseChunkingBaselinesSmokeTests(unittest.TestCase):
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

    def _mean_vec(self, matrix: np.ndarray) -> np.ndarray:
        return np.mean(
            matrix,
            axis=0,
            dtype=np.float32,
            keepdims=True,
        ).astype(np.float32)

    def _build_chunked_index(
        self,
        documents: dict[str, list[str]],
        *,
        chunk_size: int,
    ) -> tuple[kayak.LateIndex, dict[str, str]]:
        chunk_ids: list[str] = []
        chunk_vectors: list[np.ndarray] = []
        parent_by_chunk: dict[str, str] = {}
        for doc_id, tokens in documents.items():
            matrix = self._encode_tokens(tokens)
            for chunk_index, start in enumerate(range(0, len(matrix), chunk_size)):
                chunk_id = f"{doc_id}::chunk{chunk_index}"
                chunk_ids.append(chunk_id)
                chunk_vectors.append(self._mean_vec(matrix[start : start + chunk_size]))
                parent_by_chunk[chunk_id] = doc_id
        return kayak.documents(chunk_ids, chunk_vectors).pack(), parent_by_chunk

    def _dedup_parent_docs(
        self,
        chunk_hits,
        parent_by_chunk: dict[str, str],
        *,
        k: int,
    ) -> tuple[str, ...]:
        ranked_doc_ids: list[str] = []
        seen: set[str] = set()
        for hit in chunk_hits:
            doc_id = parent_by_chunk[hit.doc_id]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            ranked_doc_ids.append(doc_id)
            if len(ranked_doc_ids) >= k:
                break
        return tuple(ranked_doc_ids)

    def test_document_proxy_matches_one_vector_per_document_baseline(self) -> None:
        query = kayak.query(
            self._encode_tokens(["cancel", "subscription"]),
        )
        index = kayak.documents(
            ["doc-a", "doc-b", "doc-c"],
            [
                self._encode_tokens(["cancel", "subscription"]),
                self._encode_tokens(["cancel", "cancel"]),
                self._encode_tokens(["invoice"]),
            ],
        ).pack()

        proxy_result = kayak.generate_candidates(
            query,
            index,
            kayak.document_proxy_candidate_generator(),
            k=3,
        )

        onevec_query = kayak.query(self._mean_vec(query.as_vector_matrix()))
        onevec_index = kayak.documents(
            list(index.doc_ids),
            [
                self._mean_vec(index.document_token_matrix(document_index))
                for document_index in range(index.document_count)
            ],
        ).pack()
        onevec_hits = kayak.search(
            onevec_query,
            onevec_index,
            k=3,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(
            proxy_result.candidate_doc_ids,
            tuple(hit.doc_id for hit in onevec_hits),
        )
        np.testing.assert_allclose(
            proxy_result.scores.numpy(),
            kayak.maxsim(
                onevec_query,
                onevec_index,
                backend=kayak.NUMPY_REFERENCE_BACKEND,
            ).numpy(),
        )

    def test_chunking_can_help_when_local_evidence_is_buried_in_noise(self) -> None:
        documents = {
            "relevant": [
                "noise1",
                "noise2",
                "noise3",
                "noise4",
                "cancel",
                "subscription",
                "noise5",
                "noise6",
            ],
            "partial": ["cancel", "noise7", "cancel", "noise8"],
            "other": ["invoice", "billing"],
        }
        query_onevec = kayak.query(
            self._mean_vec(self._encode_tokens(["cancel", "subscription"]))
        )
        onevec_index = kayak.documents(
            list(documents),
            [
                self._mean_vec(self._encode_tokens(tokens))
                for tokens in documents.values()
            ],
        ).pack()
        chunk_index, parent_by_chunk = self._build_chunked_index(
            documents,
            chunk_size=2,
        )

        onevec_hits = kayak.search(
            query_onevec,
            onevec_index,
            k=3,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        chunk_hits = kayak.search(
            query_onevec,
            chunk_index,
            k=chunk_index.document_count,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(onevec_hits[0].doc_id, "partial")
        self.assertEqual(
            self._dedup_parent_docs(chunk_hits, parent_by_chunk, k=3)[0],
            "relevant",
        )

    def test_chunking_can_hurt_when_evidence_spans_chunk_boundaries(self) -> None:
        documents = {
            "relevant": ["alpha", "noise1", "beta", "noise2", "gamma", "noise3"],
            "partial": ["alpha", "beta", "noise4", "noise5"],
            "other": ["invoice", "billing"],
        }
        exact_query = kayak.query(self._encode_tokens(["alpha", "beta", "gamma"]))
        exact_index = kayak.documents(
            list(documents),
            [self._encode_tokens(tokens) for tokens in documents.values()],
        ).pack()
        query_onevec = kayak.query(
            self._mean_vec(self._encode_tokens(["alpha", "beta", "gamma"]))
        )
        chunk_index, parent_by_chunk = self._build_chunked_index(
            documents,
            chunk_size=2,
        )

        exact_hits = kayak.search(
            exact_query,
            exact_index,
            k=3,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        chunk_hits = kayak.search(
            query_onevec,
            chunk_index,
            k=chunk_index.document_count,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(exact_hits[0].doc_id, "relevant")
        self.assertEqual(
            self._dedup_parent_docs(chunk_hits, parent_by_chunk, k=3)[0],
            "partial",
        )


if __name__ == "__main__":
    unittest.main()
