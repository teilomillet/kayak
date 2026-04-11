from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import SearchHit
from kayak.eval import recall_at_k, reciprocal_rank_at_k, success_at_k


def test_metrics_measure_first_relevant_rank_and_recall() raises:
    var hits = [
        SearchHit("doc-b", 3.0),
        SearchHit("doc-a", 2.0),
        SearchHit("doc-c", 1.0),
    ]
    var relevant_doc_ids = List[String]()
    relevant_doc_ids.append("doc-a")
    relevant_doc_ids.append("doc-c")

    assert_equal(reciprocal_rank_at_k(hits, relevant_doc_ids, 3), 0.5)
    assert_equal(success_at_k(hits, relevant_doc_ids, 1), 0.0)
    assert_equal(success_at_k(hits, relevant_doc_ids, 2), 1.0)
    assert_equal(recall_at_k(hits, relevant_doc_ids, 2), 0.5)
    assert_equal(recall_at_k(hits, relevant_doc_ids, 3), 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
