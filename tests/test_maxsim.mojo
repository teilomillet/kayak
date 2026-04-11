from std.testing import TestSuite, assert_equal

from kayak import EncodedDocument, EncodedQuery, ExactCpuBackend, pack_documents
from kayak import search_exact


def test_exact_search_returns_hits_in_descending_score_order() raises:
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var documents = [
        EncodedDocument("doc-perfect", [[1.0, 0.0], [0.0, 1.0]]),
        EncodedDocument("doc-mixed", [[1.0, 0.0], [0.5, 0.5]]),
        EncodedDocument("doc-weak", [[1.0, 0.0], [0.0, 0.0]]),
    ]

    var index = pack_documents(documents)
    var hits = search_exact(ExactCpuBackend(), query, index, 2)

    assert_equal(len(hits), 2)
    assert_equal(hits[0].doc_id, "doc-perfect")
    assert_equal(hits[1].doc_id, "doc-mixed")
    assert_equal(hits[0].score, 2.0)
    assert_equal(hits[1].score, 1.5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
