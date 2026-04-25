from __future__ import annotations

import inspect
import unittest

import kayak


class HelpApiTests(unittest.TestCase):
    def test_help_overview_lists_public_categories_and_entries(self) -> None:
        text = kayak.help()

        self.assertIn("Public Python SDK", text)
        self.assertIn('kayak.help("mojo")', text)
        self.assertIn('kayak.help("doctor")', text)
        self.assertIn('kayak.help("search_text")', text)
        self.assertIn('kayak.help("typing")', text)
        self.assertIn("Encoders:", text)
        self.assertIn("Stores:", text)
        self.assertIn("Backends:", text)
        self.assertIn("Registered kinds: callable, colbert", text)
        self.assertIn("open_text_retriever", text)
        self.assertIn("mojo_bridge_info", text)
        self.assertIn("Stable aliases:", text)
        self.assertIn("TokenMatrixInput", text)

    def test_help_typing_topic_renders_public_typing_module(self) -> None:
        text = kayak.help("typing")

        self.assertIn("typing", text)
        self.assertIn("module: kayak.typing", text)
        self.assertIn("stable public type aliases", text.lower())
        self.assertIn("Stable aliases:", text)
        self.assertIn("MetadataRowsInput", text)

    def test_help_types_alias_resolves_to_typing_topic(self) -> None:
        text = kayak.help("types")

        self.assertIn("module: kayak.typing", text)
        self.assertIn("TokenMatrixInput", text)

    def test_help_exact_typing_alias_renders_user_facing_type_text(self) -> None:
        text = kayak.help("TokenMatrixInput")

        self.assertIn("TokenMatrixInput", text)
        self.assertIn("module: kayak.typing", text)
        self.assertIn("torch.Tensor", text)
        self.assertIn("2D float32 array", text)
        self.assertIn('kayak.help("typing")', text)

    def test_help_fuzzy_typing_alias_lists_alias_matches(self) -> None:
        text = kayak.help("Metadata")

        self.assertIn('Closest typing aliases for "Metadata"', text)
        self.assertIn("MetadataRowsInput", text)
        self.assertIn("MetadataFilterInput", text)

    def test_help_exact_topic_renders_signature_and_docstring(self) -> None:
        text = kayak.help("search")

        self.assertIn("search", text)
        self.assertIn("backend", text)
        self.assertIn("Return top-k hits for one query", text)
        self.assertIn("approximation", text)

    def test_help_category_lists_matching_public_entries(self) -> None:
        text = kayak.help("stores")

        self.assertIn("Stores", text)
        self.assertIn("Registered kinds:", text)
        self.assertIn("open_store", text)
        self.assertIn("DirectoryLateStore", text)

    def test_help_class_lists_public_methods(self) -> None:
        text = kayak.help(kayak.LateTextRetriever)

        self.assertIn("LateTextRetriever", text)
        self.assertIn("encode_query", text)
        self.assertIn("search_text", text)
        self.assertIn("search_text_batch", text)
        self.assertIn("load_index", text)
        self.assertIn("session", text)
        self.assertIn('kayak.help("Retrievers")', text)

    def test_help_session_topic_is_discoverable(self) -> None:
        text = kayak.help("session")

        self.assertIn("LateTextSearchSession", text)
        self.assertIn("search_text", text)
        self.assertIn("search_query_batch", text)

    def test_help_class_includes_classmethods(self) -> None:
        text = kayak.help(kayak.LateQuery)

        self.assertIn("from_vectors", text)
        self.assertIn("from_flat_values", text)

    def test_help_public_method_topic_renders_owner_and_related_topics(self) -> None:
        text = kayak.help("search_text")

        self.assertIn('Closest public methods for "search_text"', text)
        self.assertIn("LateTextRetriever.search_text", text)
        self.assertIn("LateTextSearchSession.search_text", text)
        self.assertIn('kayak.help("LateTextRetriever")', text)
        self.assertIn('kayak.help("LateTextSearchSession")', text)

    def test_help_callable_encoder_model_binding_method_is_discoverable(self) -> None:
        text = kayak.help("from_model")

        self.assertIn("CallableLateTextEncoder.from_model", text)
        self.assertIn("query_method", text)
        self.assertIn("document_method", text)

    def test_help_factory_topic_includes_coding_guidance(self) -> None:
        text = kayak.help("open_text_retriever")

        self.assertIn("Use this when you want one object", text)
        self.assertIn("Parameters", text)
        self.assertIn("encoder:", text)
        self.assertIn("Example", text)
        self.assertIn("retriever = kayak.open_text_retriever", text)
        self.assertIn("model object", text)

    def test_help_doctor_topic_is_discoverable(self) -> None:
        text = kayak.help("doctor")
        alias_text = kayak.help("diagnostics")

        self.assertIn("doctor", text)
        self.assertIn("KayakDoctorReport", text)
        self.assertIn("probe_mojo_load", text)
        self.assertIn("doctor", alias_text)

    def test_standard_python_docstrings_are_useful_for_editor_help(self) -> None:
        store_doc = inspect.getdoc(kayak.open_store)
        retriever_doc = inspect.getdoc(kayak.open_text_retriever)
        method_doc = inspect.getdoc(kayak.LateTextRetriever.search_query)
        doctor_doc = inspect.getdoc(kayak.doctor)

        assert store_doc is not None
        assert retriever_doc is not None
        assert method_doc is not None
        assert doctor_doc is not None

        self.assertIn("Built-in kinds:", store_doc)
        self.assertIn('"pgvector"', store_doc)
        self.assertIn("Use this when you want one object", retriever_doc)
        self.assertIn("Returns", method_doc)
        self.assertIn("tuple[SearchHit, ...]", method_doc)
        self.assertIn("factual environment report", doctor_doc)

    def test_help_unknown_topic_suggests_public_names(self) -> None:
        text = kayak.help("bridge")

        self.assertIn('Closest public matches for "bridge"', text)
        self.assertIn("mojo_bridge_info", text)

    def test_help_mojo_alias_resolves_to_backend_category(self) -> None:
        text = kayak.help("mojo")

        self.assertIn("Backends", text)
        self.assertIn("mojo_bridge_info", text)
        self.assertIn("MOJO_EXACT_CPU_BACKEND", text)
        self.assertIn('kayak.help("available_backends")', text)


if __name__ == "__main__":
    unittest.main()
