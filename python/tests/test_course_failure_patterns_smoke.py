from __future__ import annotations

import unittest

import numpy as np

import kayak


class CourseFailurePatternsSmokeTests(unittest.TestCase):
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

    def test_split_evidence_is_a_retrieval_unit_problem(self) -> None:
        query = kayak.query(self._encode_tokens(["cancel", "subscription"]))

        split_index = kayak.documents(
            ["chunk-a", "chunk-b", "other"],
            [
                self._encode_tokens(["cancel"]),
                self._encode_tokens(["subscription"]),
                self._encode_tokens(["invoice"]),
            ],
        ).pack()
        grouped_index = kayak.documents(
            ["doc-combined", "other"],
            [
                self._encode_tokens(["cancel", "subscription"]),
                self._encode_tokens(["invoice"]),
            ],
        ).pack()

        split_hits = kayak.search(
            query,
            split_index,
            k=3,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        grouped_hits = kayak.search(
            query,
            grouped_index,
            k=2,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(split_hits[0].score, 1.0)
        self.assertEqual(split_hits[1].score, 1.0)
        self.assertEqual({split_hits[0].doc_id, split_hits[1].doc_id}, {"chunk-a", "chunk-b"})
        self.assertEqual(grouped_hits[0].doc_id, "doc-combined")
        self.assertEqual(grouped_hits[0].score, 2.0)

    def test_narrow_candidate_stage_can_drop_oracle_before_rerank(self) -> None:
        query = kayak.query(self._encode_tokens(["cancel", "subscription"]))
        index = kayak.documents(
            ["doc-partial", "doc-relevant", "doc-other"],
            [
                self._encode_tokens(["cancel", "cancel", "cancel"]),
                self._encode_tokens(["cancel", "subscription"]),
                self._encode_tokens(["invoice"]),
            ],
        ).pack()

        exact_result = kayak.search_with_plan(
            query,
            index,
            kayak.exact_full_scan_search_plan(final_k=1, candidate_k=3),
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        narrow_result = kayak.search_with_plan(
            query,
            index,
            kayak.document_proxy_search_plan(
                final_k=1,
                candidate_k=1,
                query_vector_budget=1,
                document_vector_budget=1,
            ),
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        wider_result = kayak.search_with_plan(
            query,
            index,
            kayak.document_proxy_search_plan(
                final_k=1,
                candidate_k=2,
                query_vector_budget=1,
                document_vector_budget=1,
            ),
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual([hit.doc_id for hit in exact_result.hits], ["doc-relevant"])
        self.assertEqual(narrow_result.candidate_stage.candidate_doc_ids, ("doc-partial",))
        self.assertEqual([hit.doc_id for hit in narrow_result.hits], ["doc-partial"])
        self.assertEqual(
            wider_result.candidate_stage.candidate_doc_ids,
            ("doc-partial", "doc-relevant"),
        )
        self.assertEqual([hit.doc_id for hit in wider_result.hits], ["doc-relevant"])
        self.assertEqual(wider_result.stage2.stage_name, "exact_late_interaction")


if __name__ == "__main__":
    unittest.main()
