from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import ExactScoringConfig, EncodedDocument, VectorScalar, pack_documents
from kayak.scoring.maxsim import (
    build_vector_balanced_boundaries,
    choose_parallel_work_item_count_for_shape,
)


def repeated_vector(vector_dim: Int) -> List[VectorScalar]:
    var vector = List[VectorScalar]()
    for _ in range(vector_dim):
        vector.append(1.0)
    return vector^


def make_document(doc_id: String, vector_count: Int) raises -> EncodedDocument:
    var token_vectors = List[List[VectorScalar]]()
    for _ in range(vector_count):
        token_vectors.append(repeated_vector(2))
    return EncodedDocument(doc_id, token_vectors^)


def test_vector_balanced_boundaries_follow_document_vector_mass() raises:
    var index = pack_documents(
        [
            make_document("doc-0", 1),
            make_document("doc-1", 10),
            make_document("doc-2", 2),
            make_document("doc-3", 9),
            make_document("doc-4", 3),
            make_document("doc-5", 8),
        ]
    )

    var boundaries = build_vector_balanced_boundaries(index, 3)

    assert_equal(boundaries, [0, 2, 4, 6])


def test_vector_balanced_boundaries_handle_single_partition() raises:
    var index = pack_documents(
        [make_document("doc-0", 4), make_document("doc-1", 7)]
    )

    assert_equal(build_vector_balanced_boundaries(index, 1), [0, 2])


def test_parallel_work_item_count_keeps_tiny_windows_single_partition() raises:
    var config = ExactScoringConfig()
    var work_item_count = choose_parallel_work_item_count_for_shape(
        32,
        10,
        10_000,
        config,
    )

    assert_equal(work_item_count, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
