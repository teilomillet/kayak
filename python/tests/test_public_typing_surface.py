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

    def test_retriever_methods_return_concrete_public_types(self) -> None:
        search_query_hints = get_type_hints(kayak.LateTextRetriever.search_query)
        search_query_batch_hints = get_type_hints(
            kayak.LateTextRetriever.search_query_batch
        )
        search_query_with_plan_hints = get_type_hints(
            kayak.LateTextRetriever.search_query_with_plan
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
