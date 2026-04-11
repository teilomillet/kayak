from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    ExactScoringConfig,
    pack_documents,
)
from kayak import search_exact


def basis_vector(vector_dim: Int, hot_index: Int) -> List[Float32]:
    var vector = List[Float32]()
    for dim_index in range(vector_dim):
        if dim_index == hot_index:
            vector.append(1.0)
        else:
            vector.append(0.0)
    return vector^


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


def test_exact_search_uses_dim128_fast_path_without_changing_scores() raises:
    var vector_dim = 128
    var query = EncodedQuery(
        [basis_vector(vector_dim, 3), basis_vector(vector_dim, 17)]
    )
    var documents = [
        EncodedDocument(
            "doc-perfect",
            [basis_vector(vector_dim, 3), basis_vector(vector_dim, 17)],
        ),
        EncodedDocument(
            "doc-partial",
            [basis_vector(vector_dim, 3), basis_vector(vector_dim, 33)],
        ),
        EncodedDocument(
            "doc-miss",
            [basis_vector(vector_dim, 81), basis_vector(vector_dim, 97)],
        ),
    ]

    var index = pack_documents(documents)
    var hits = search_exact(ExactCpuBackend(), query, index, 3)

    assert_equal(hits[0].doc_id, "doc-perfect")
    assert_equal(hits[1].doc_id, "doc-partial")
    assert_equal(hits[2].doc_id, "doc-miss")
    assert_equal(hits[0].score, 2.0)
    assert_equal(hits[1].score, 1.0)
    assert_equal(hits[2].score, 0.0)


def test_exact_search_toggle_keeps_dim128_scores_identical() raises:
    var vector_dim = 128
    var query = EncodedQuery(
        [basis_vector(vector_dim, 5), basis_vector(vector_dim, 29)]
    )
    var documents = [
        EncodedDocument(
            "doc-a",
            [basis_vector(vector_dim, 5), basis_vector(vector_dim, 29)],
        ),
        EncodedDocument(
            "doc-b",
            [basis_vector(vector_dim, 5), basis_vector(vector_dim, 61)],
        ),
        EncodedDocument(
            "doc-c",
            [basis_vector(vector_dim, 73), basis_vector(vector_dim, 97)],
        ),
    ]
    var index = pack_documents(documents)

    var default_hits = search_exact(ExactCpuBackend(), query, index, 3)

    var generic_config = ExactScoringConfig()
    generic_config.enable_dim128_fast_path = False
    var generic_hits = search_exact(
        ExactCpuBackend(generic_config^), query, index, 3
    )

    assert_equal(len(default_hits), len(generic_hits))
    for index in range(len(default_hits)):
        assert_equal(default_hits[index].doc_id, generic_hits[index].doc_id)
        assert_equal(default_hits[index].score, generic_hits[index].score)


def test_exact_search_toggle_keeps_parallel_scores_identical() raises:
    var vector_dim = 2
    var query_vectors = List[List[Float32]]()
    for hot_index in range(32):
        query_vectors.append(basis_vector(vector_dim, hot_index % vector_dim))

    var documents = List[EncodedDocument]()
    for document_index in range(8):
        var token_vectors = List[List[Float32]]()
        for token_index in range(256):
            token_vectors.append(
                basis_vector(vector_dim, (document_index + token_index) % vector_dim)
            )
        documents.append(
            EncodedDocument("doc-" + String(document_index), token_vectors^)
        )

    var query = EncodedQuery(query_vectors^)
    var index = pack_documents(documents^)

    var default_hits = search_exact(ExactCpuBackend(), query.copy(), index.copy(), 3)

    var serial_config = ExactScoringConfig()
    serial_config.enable_parallel_scoring = False
    var serial_hits = search_exact(
        ExactCpuBackend(serial_config^), query, index, 3
    )

    assert_equal(len(default_hits), len(serial_hits))
    for index in range(len(default_hits)):
        assert_equal(default_hits[index].doc_id, serial_hits[index].doc_id)
        assert_equal(default_hits[index].score, serial_hits[index].score)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
