from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import kayak


def _token_vectors(text: str) -> np.ndarray:
    token_rows: list[np.ndarray] = []
    token_to_index = {
        "pixi": 0,
        "mojo": 1,
        "kayak": 2,
        "lancedb": 3,
        "qdrant": 4,
        "install": 5,
        "search": 6,
        "storage": 7,
    }
    for token in text.lower().split():
        vector = np.zeros(8, dtype=np.float32)
        hot_index = token_to_index.get(token, 7)
        vector[hot_index] = np.float32(1.0)
        token_rows.append(vector)
    if not token_rows:
        return np.zeros((1, 8), dtype=np.float32)
    return np.stack(token_rows)


class LateTextRetrieverApiTests(unittest.TestCase):
    def _open_retriever(self, **kwargs: object) -> kayak.LateTextRetriever:
        return kayak.open_text_retriever(
            encoder="callable",
            store="memory",
            encoder_kwargs={
                "query_encoder": _token_vectors,
                "document_encoder": _token_vectors,
            },
            **kwargs,
        )

    def test_open_text_retriever_can_ingest_and_search_text(self) -> None:
        retriever = self._open_retriever(backend=kayak.NUMPY_REFERENCE_BACKEND)
        retriever.upsert_texts(
            ["doc-install", "doc-storage", "doc-search"],
            [
                "pixi mojo install kayak",
                "lancedb storage kayak",
                "kayak search mojo",
            ],
            metadata=[
                {"topic": "install"},
                {"topic": "storage"},
                {"topic": "search"},
            ],
        )

        hits = retriever.search_text("pixi mojo install", k=2)

        self.assertEqual(hits[0].doc_id, "doc-install")
        self.assertEqual(retriever.stats().document_count, 3)
        self.assertTrue(retriever.capabilities().supports_metadata_filter)

    def test_search_text_supports_store_filters_and_text_materialization(self) -> None:
        retriever = self._open_retriever(
            backend=kayak.NUMPY_REFERENCE_BACKEND,
            default_include_text=True,
        )
        retriever.upsert_texts(
            ["doc-install", "doc-storage"],
            [
                "pixi mojo install kayak",
                "lancedb storage kayak",
            ],
            metadata=[
                {"topic": "install"},
                {"topic": "storage"},
            ],
        )

        index = retriever.load_index(where={"topic": "storage"}, include_text=True)
        hits = retriever.search_text(
            "storage kayak",
            k=1,
            where={"topic": "storage"},
            include_text=True,
        )

        self.assertEqual(index.doc_ids, ("doc-storage",))
        self.assertEqual(index.doc_texts, ("lancedb storage kayak",))
        self.assertEqual(hits[0].doc_id, "doc-storage")

    def test_search_text_with_exact_full_scan_plan_reports_noop_stage2(self) -> None:
        retriever = self._open_retriever(backend=kayak.NUMPY_REFERENCE_BACKEND)
        retriever.upsert_texts(
            ["doc-a", "doc-b"],
            [
                "kayak search mojo",
                "lancedb storage kayak",
            ],
        )

        result = retriever.search_text_with_plan(
            "kayak search",
            kayak.exact_full_scan_search_plan(final_k=1),
        )

        self.assertEqual(result.hits[0].doc_id, "doc-a")
        self.assertEqual(result.plan.stage2_reference_operator.kind, "noop_topk")
        self.assertEqual(result.stage2.stage_name, "noop_topk")

    def test_search_text_with_proxy_plan_runs_exact_stage2(self) -> None:
        retriever = self._open_retriever(backend=kayak.NUMPY_REFERENCE_BACKEND)
        retriever.upsert_texts(
            ["doc-a", "doc-b"],
            [
                "kayak search mojo",
                "lancedb storage kayak",
            ],
        )

        result = retriever.search_text_with_plan(
            "kayak search",
            kayak.document_proxy_search_plan(
                final_k=1,
                candidate_k=2,
                query_vector_budget=1,
                document_vector_budget=1,
            ),
        )

        self.assertEqual(result.hits[0].doc_id, "doc-a")
        self.assertEqual(result.stage2.stage_name, "exact_late_interaction")

    @patch("kayak.retrievers.backend_policy.backend_info")
    def test_open_text_retriever_prefers_mojo_when_available(
        self,
        backend_info_mock: object,
    ) -> None:
        backend_info_mock.return_value = kayak.BackendInfo(
            name=kayak.MOJO_EXACT_CPU_BACKEND,
            available=True,
            requires_mojo=True,
            query_layouts=("nested", "flat_dim128"),
            index_layouts=("packed", "hybrid_flat_dim128"),
            availability_reason="test fixture",
        )

        retriever = self._open_retriever()

        self.assertEqual(retriever.default_backend, kayak.MOJO_EXACT_CPU_BACKEND)

    @patch("kayak.retrievers.backend_policy.backend_info")
    def test_open_text_retriever_falls_back_to_numpy_without_mojo(
        self,
        backend_info_mock: object,
    ) -> None:
        backend_info_mock.return_value = kayak.BackendInfo(
            name=kayak.MOJO_EXACT_CPU_BACKEND,
            available=False,
            requires_mojo=True,
            query_layouts=("nested", "flat_dim128"),
            index_layouts=("packed", "hybrid_flat_dim128"),
            availability_reason="test fixture",
        )

        retriever = self._open_retriever()

        self.assertEqual(retriever.default_backend, kayak.NUMPY_REFERENCE_BACKEND)

    @patch("kayak.retrievers.backend_policy.backend_info")
    def test_open_text_retriever_keeps_explicit_backend_override(
        self,
        backend_info_mock: object,
    ) -> None:
        backend_info_mock.return_value = kayak.BackendInfo(
            name=kayak.MOJO_EXACT_CPU_BACKEND,
            available=True,
            requires_mojo=True,
            query_layouts=("nested", "flat_dim128"),
            index_layouts=("packed", "hybrid_flat_dim128"),
            availability_reason="test fixture",
        )

        retriever = self._open_retriever(backend=kayak.NUMPY_REFERENCE_BACKEND)

        self.assertEqual(retriever.default_backend, kayak.NUMPY_REFERENCE_BACKEND)


if __name__ == "__main__":
    unittest.main()
