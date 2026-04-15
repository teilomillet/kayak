from __future__ import annotations

import unittest

from kayak_bridge.r2med_biology_full import _filter_relevant_doc_ids


class R2MEDBiologyFullTests(unittest.TestCase):
    def test_filter_relevant_doc_ids_keeps_only_available_docs(self) -> None:
        self.assertEqual(
            _filter_relevant_doc_ids(
                ("doc-a", "doc-b", "doc-c"),
                {"doc-b", "doc-c"},
            ),
            ("doc-b", "doc-c"),
        )

    def test_filter_relevant_doc_ids_keeps_full_set_without_filter(self) -> None:
        self.assertEqual(
            _filter_relevant_doc_ids(("doc-a", "doc-b"), None),
            ("doc-a", "doc-b"),
        )


if __name__ == "__main__":
    unittest.main()
