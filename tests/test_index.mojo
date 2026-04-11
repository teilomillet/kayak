from std.testing import TestSuite, assert_equal

from kayak import EncodedDocument, pack_documents


def test_pack_documents_builds_monotonic_offsets() raises:
    var documents = [
        EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
        EncodedDocument("doc-b", [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]),
    ]

    var index = pack_documents(documents)

    assert_equal(index.document_count, 2)
    assert_equal(index.total_vector_count, 5)
    assert_equal(index.vector_dim, 2)
    assert_equal(index.doc_offsets[0], 0)
    assert_equal(index.doc_offsets[1], 2)
    assert_equal(index.doc_offsets[2], 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
