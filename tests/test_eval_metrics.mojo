from std.math import abs, log2
from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import SearchHit
from kayak.eval import ndcg_at_k, recall_at_k, reciprocal_rank_at_k, success_at_k


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


def test_ndcg_matches_reference_discounted_gain() raises:
    var hits = [
        SearchHit("doc-b", 3.0),
        SearchHit("doc-a", 2.0),
        SearchHit("doc-c", 1.0),
    ]
    var relevant_doc_ids = List[String]()
    relevant_doc_ids.append("doc-a")
    relevant_doc_ids.append("doc-c")

    var expected = (
        (1.0 / log2(Float64(3))) + (1.0 / log2(Float64(4)))
    ) / ((1.0 / log2(Float64(2))) + (1.0 / log2(Float64(3))))
    var actual = ndcg_at_k(hits, relevant_doc_ids, 3)

    assert_equal(abs(actual - expected) < 0.0000001, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
