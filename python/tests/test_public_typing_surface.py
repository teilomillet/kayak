from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_overloads, get_type_hints
import unittest

import kayak


class PublicTypingSurfaceTests(unittest.TestCase):
    def test_encoder_factory_exposes_kind_specific_overloads(self) -> None:
        overloads = get_overloads(kayak.open_encoder)
        returns = {
            get_type_hints(overload)["return"] for overload in overloads
        }

        self.assertIn(kayak.CallableLateTextEncoder, returns)
        self.assertIn(kayak.ColBERTTextEncoder, returns)

    def test_store_factory_exposes_kind_specific_overloads(self) -> None:
        overloads = get_overloads(kayak.open_store)
        returns = {
            get_type_hints(overload)["return"] for overload in overloads
        }

        self.assertIn(kayak.MemoryLateStore, returns)
        self.assertIn(kayak.DirectoryLateStore, returns)
        self.assertIn(kayak.LanceDBLateStore, returns)
        self.assertIn(kayak.PgVectorLateStore, returns)
        self.assertIn(kayak.QdrantLateStore, returns)
        self.assertIn(kayak.WeaviateLateStore, returns)
        self.assertIn(kayak.ChromaLateStore, returns)

    def test_registry_listing_helpers_return_registered_strings(self) -> None:
        self.assertEqual(
            kayak.available_encoder_kinds(),
            ("callable", "colbert"),
        )
        self.assertIn("memory", kayak.available_store_kinds())
        self.assertIn("pgvector", kayak.available_store_kinds())
        self.assertIn("chromadb", kayak.available_store_kinds())

    def test_doctor_returns_typed_public_report(self) -> None:
        doctor_hints = get_type_hints(kayak.doctor)
        self.assertEqual(
            doctor_hints["return"],
            kayak.KayakDoctorReport,
        )

    def test_retriever_methods_return_concrete_public_types(self) -> None:
        encode_query_hints = get_type_hints(kayak.LateTextRetriever.encode_query)
        encode_documents_hints = get_type_hints(
            kayak.LateTextRetriever.encode_documents
        )
        search_query_hints = get_type_hints(kayak.LateTextRetriever.search_query)
        search_query_batch_hints = get_type_hints(
            kayak.LateTextRetriever.search_query_batch
        )
        session_hints = get_type_hints(kayak.LateTextRetriever.session)
        search_query_with_plan_hints = get_type_hints(
            kayak.LateTextRetriever.search_query_with_plan
        )
        session_search_hints = get_type_hints(
            kayak.LateTextSearchSession.search_query
        )

        self.assertEqual(
            encode_query_hints["return"],
            kayak.LateQuery,
        )
        self.assertEqual(
            encode_documents_hints["return"],
            kayak.LateDocuments,
        )
        self.assertEqual(
            search_query_hints["return"],
            tuple[kayak.SearchHit, ...],
        )
        self.assertEqual(
            search_query_batch_hints["return"],
            tuple[tuple[kayak.SearchHit, ...], ...],
        )
        self.assertEqual(
            search_query_with_plan_hints["return"],
            kayak.SearchPlanResult,
        )
        self.assertEqual(
            session_hints["return"],
            kayak.LateTextSearchSession,
        )
        self.assertEqual(
            session_search_hints["return"],
            tuple[kayak.SearchHit, ...],
        )

    def test_public_constructor_signatures_use_named_input_aliases(self) -> None:
        self.assertIn("TokenMatrixInput", str(inspect.signature(kayak.query)))
        self.assertIn(
            "DocumentMatricesInput",
            str(inspect.signature(kayak.documents)),
        )
        self.assertIn(
            "DocOffsetsInput",
            str(inspect.signature(kayak.packed_index)),
        )
        self.assertIn(
            "MetadataFilterInput",
            str(inspect.signature(kayak.LateTextRetriever.search_query)),
        )
        self.assertIn(
            "MetadataFilterInput",
            str(inspect.signature(kayak.LateTextRetriever.session)),
        )

    def test_public_signatures_limit_raw_object_usage(self) -> None:
        allowed = {
            "help",
            "open_encoder",
            "open_store",
            "open_text_retriever",
        }
        leaked: list[str] = []
        for name in sorted(kayak.PUBLIC_API):
            value = getattr(kayak, name)
            if not (inspect.isfunction(value) or inspect.isclass(value)):
                continue
            try:
                signature = str(inspect.signature(value))
            except Exception:
                continue
            if "object" in signature and name not in allowed:
                leaked.append(f"{name}: {signature}")

        self.assertEqual(
            leaked,
            [],
            f"unexpected raw object annotations leaked into public signatures: {leaked}",
        )

    def test_package_marks_itself_typed(self) -> None:
        typed_marker = Path(kayak.__file__).with_name("py.typed")
        self.assertTrue(typed_marker.exists(), typed_marker)

    def test_public_typing_module_exposes_stable_aliases(self) -> None:
        self.assertTrue(hasattr(kayak, "typing"))
        self.assertTrue(hasattr(kayak.typing, "TokenMatrixInput"))
        self.assertTrue(hasattr(kayak.typing, "DocumentMatricesInput"))
        self.assertTrue(hasattr(kayak.typing, "MetadataRowsInput"))
        self.assertIn(
            "stable public type aliases",
            (inspect.getdoc(kayak.typing) or "").lower(),
        )


if __name__ == "__main__":
    unittest.main()
