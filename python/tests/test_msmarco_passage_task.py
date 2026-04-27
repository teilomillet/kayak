from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kayak_bridge.msmarco_passage_task import (
    MsmarcoPassagePaths,
    parse_qrels_line,
    parse_tsv_id_text_line,
    select_msmarco_passage_documents_and_queries,
)


class MsmarcoPassageTaskTests(unittest.TestCase):
    def test_parse_qrels_line_accepts_trec_and_three_column_rows(self) -> None:
        self.assertEqual(parse_qrels_line("10 0 20 1"), ("10", "20", 1))
        self.assertEqual(parse_qrels_line("10 20 1"), ("10", "20", 1))

    def test_parse_tsv_id_text_line_keeps_tabs_inside_text(self) -> None:
        self.assertEqual(
            parse_tsv_id_text_line("p1\tleft\tright\n", row_name="collection"),
            ("p1", "left\tright"),
        )

    def test_selection_can_force_relevant_docs_beyond_limit(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            collection = root_path / "collection.tsv"
            queries = root_path / "queries.tsv"
            qrels = root_path / "qrels.dev.tsv"
            collection.write_text(
                "p0\tfirst passage\n"
                "p1\tsecond passage\n"
                "p2\trelevant passage\n",
                encoding="utf-8",
            )
            queries.write_text("q0\twhere is the relevant passage?\n", encoding="utf-8")
            qrels.write_text("q0 0 p2 1\n", encoding="utf-8")

            selection = select_msmarco_passage_documents_and_queries(
                MsmarcoPassagePaths(
                    collection=collection,
                    queries=queries,
                    qrels=qrels,
                ),
                document_limit=1,
                query_limit=1,
                include_relevant_documents=True,
            )

        self.assertEqual(selection.source_document_count, 3)
        self.assertEqual(selection.selected_document_count, 2)
        self.assertEqual(selection.required_relevant_doc_count, 1)
        self.assertEqual(selection.missing_required_relevant_doc_count, 0)
        self.assertEqual(
            [document["doc_id"] for document in selection.documents],
            ["p0", "p2"],
        )
        self.assertEqual(selection.queries[0]["relevant_doc_ids"], ["p2"])


if __name__ == "__main__":
    unittest.main()
