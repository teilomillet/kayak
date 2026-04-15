from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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

    def test_open_text_retriever_can_bind_model_object_directly(self) -> None:
        class _Model:
            def encode_query_tokens(self, text: str) -> np.ndarray:
                return _token_vectors(text)

            def encode_document_tokens(self, text: str) -> np.ndarray:
                return _token_vectors(text)

        retriever = kayak.open_text_retriever(
            encoder=_Model(),
            store="memory",
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        retriever.upsert_texts(
            ["doc-install", "doc-storage"],
            [
                "pixi mojo install kayak",
                "lancedb storage kayak",
            ],
        )

        hits = retriever.search_text("pixi mojo install", k=1)

        self.assertEqual(hits[0].doc_id, "doc-install")

    def test_open_text_retriever_accepts_model_method_overrides(self) -> None:
        class _Model:
            def query_tokens(self, text: str) -> np.ndarray:
                return _token_vectors(text)

            def document_tokens(self, text: str) -> np.ndarray:
                return _token_vectors(text)

        retriever = kayak.open_text_retriever(
            encoder=_Model(),
            store="memory",
            backend=kayak.NUMPY_REFERENCE_BACKEND,
            encoder_kwargs={
                "query_method": "query_tokens",
                "document_method": "document_tokens",
            },
        )
        retriever.upsert_texts(
            ["doc-install", "doc-storage"],
            [
                "pixi mojo install kayak",
                "lancedb storage kayak",
            ],
        )

        hits = retriever.search_text("storage kayak", k=1)

        self.assertEqual(hits[0].doc_id, "doc-storage")

    def test_open_text_retriever_rejects_encoder_kwargs_for_ready_encoder(
        self,
    ) -> None:
        encoder = kayak.CallableLateTextEncoder(
            query_encoder=_token_vectors,
            document_encoder=_token_vectors,
        )

        with self.assertRaisesRegex(TypeError, "already-constructed encoder"):
            kayak.open_text_retriever(
                encoder=encoder,
                store="memory",
                encoder_kwargs={"query_method": "query_tokens"},
            )

    def test_open_text_retriever_rejects_store_kwargs_for_ready_store(self) -> None:
        store = kayak.MemoryLateStore()

        with self.assertRaisesRegex(TypeError, "already-constructed store"):
            kayak.open_text_retriever(
                encoder="callable",
                store=store,
                encoder_kwargs={
                    "query_encoder": _token_vectors,
                    "document_encoder": _token_vectors,
                },
                store_kwargs={"path": "/tmp/unused"},
            )

    def test_open_text_retriever_rejects_invalid_store_object(self) -> None:
        class _NotAStore:
            pass

        with self.assertRaisesRegex(
            TypeError,
            "LateStore protocol",
        ):
            kayak.open_text_retriever(
                encoder="callable",
                store=_NotAStore(),
                encoder_kwargs={
                    "query_encoder": _token_vectors,
                    "document_encoder": _token_vectors,
                },
            )

    def test_retriever_exposes_side_effect_free_encoding_helpers(self) -> None:
        retriever = self._open_retriever(backend=kayak.NUMPY_REFERENCE_BACKEND)

        encoded_query = retriever.encode_query("pixi mojo install")
        encoded_vectors = retriever.encode_document_vectors(
            "lancedb storage kayak"
        )
        encoded_documents = retriever.encode_documents(
            ["doc-a", "doc-b"],
            [
                "pixi mojo install kayak",
                "lancedb storage kayak",
            ],
        )

        self.assertIsInstance(encoded_query, kayak.LateQuery)
        self.assertEqual(encoded_query.vector_count, 3)
        self.assertEqual(tuple(encoded_vectors.shape), (3, 8))
        self.assertIsInstance(encoded_documents, kayak.LateDocuments)
        self.assertEqual(encoded_documents.doc_ids, ("doc-a", "doc-b"))
        self.assertEqual(retriever.stats().document_count, 0)

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

    def test_search_text_batch_matches_individual_search(self) -> None:
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

        expected = (
            retriever.search_text("pixi install", k=1),
            retriever.search_text("storage kayak", k=1),
            retriever.search_text("search mojo", k=1),
        )
        actual = retriever.search_text_batch(
            ["pixi install", "storage kayak", "search mojo"],
            k=1,
        )

        self.assertEqual(actual, expected)

    def test_session_reuses_one_loaded_index_across_repeated_searches(self) -> None:
        class _CountingStore(kayak.MemoryLateStore):
            def __init__(self) -> None:
                super().__init__()
                self.load_calls = 0

            def load_index(self, **kwargs: object) -> kayak.LateIndex:
                self.load_calls += 1
                return super().load_index(**kwargs)

        store = _CountingStore()
        retriever = kayak.open_text_retriever(
            encoder="callable",
            store=store,
            encoder_kwargs={
                "query_encoder": _token_vectors,
                "document_encoder": _token_vectors,
            },
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
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

        session = retriever.session(where={"topic": "search"}, include_text=True)
        first = session.search_text("search mojo", k=1)
        second = session.search_text("kayak search", k=1)

        self.assertEqual(store.load_calls, 1)
        self.assertEqual(session.index.doc_ids, ("doc-search",))
        self.assertEqual(session.index.doc_texts, ("kayak search mojo",))
        self.assertEqual(first[0].doc_id, "doc-search")
        self.assertEqual(second[0].doc_id, "doc-search")

    def test_session_batch_and_plan_calls_match_direct_search(self) -> None:
        retriever = self._open_retriever(backend=kayak.NUMPY_REFERENCE_BACKEND)
        retriever.upsert_texts(
            ["doc-a", "doc-b"],
            [
                "kayak search mojo",
                "lancedb storage kayak",
            ],
        )

        session = retriever.session()
        batch_hits = session.search_text_batch(
            ["kayak search", "storage kayak"],
            k=1,
        )
        planned = session.search_text_with_plan(
            "kayak search",
            kayak.exact_full_scan_search_plan(final_k=1),
        )

        self.assertEqual(batch_hits[0][0].doc_id, "doc-a")
        self.assertEqual(batch_hits[1][0].doc_id, "doc-b")
        self.assertEqual(planned.hits[0].doc_id, "doc-a")

    def test_session_can_serve_parallel_queries_from_one_loaded_slice(self) -> None:
        retriever = self._open_retriever(backend=kayak.NUMPY_REFERENCE_BACKEND)
        retriever.upsert_texts(
            ["doc-install", "doc-storage", "doc-search"],
            [
                "pixi mojo install kayak",
                "lancedb storage kayak",
                "kayak search mojo",
            ],
        )
        session = retriever.session()

        query_texts = (
            "pixi install",
            "storage kayak",
            "search mojo",
            "kayak mojo",
        )
        expected = [session.search_text(text, k=1) for text in query_texts]

        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = list(
                executor.map(
                    lambda text: session.search_text(text, k=1),
                    query_texts,
                )
            )

        self.assertEqual(actual, expected)

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

    def test_retriever_context_manager_closes_owned_store(self) -> None:
        class _ClosingStore(kayak.MemoryLateStore):
            def __init__(self) -> None:
                super().__init__()
                self.closed = False

            def close(self) -> None:
                self.closed = True

        store = _ClosingStore()
        retriever = kayak.open_text_retriever(
            encoder="callable",
            store=store,
            encoder_kwargs={
                "query_encoder": _token_vectors,
                "document_encoder": _token_vectors,
            },
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        with retriever as entered:
            self.assertIs(entered.store, store)

        self.assertTrue(store.closed)


if __name__ == "__main__":
    unittest.main()
