from std.testing import TestSuite, assert_equal

from kayak import (
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_LAST,
    EncodedDocument,
    apply_document_representation_transform_to_document,
    apply_document_representation_transforms_to_packed_index,
    budgeted_token_pooling_document_representation_transform,
    hierarchical_token_pool_document,
    pack_documents,
    prefix_pruning_document_representation_transform,
    resolved_token_pooling_protected_token_count,
    sequential_token_pool_document,
    target_pooled_vector_count,
    token_pooling_document_representation_transform,
)


def test_target_pooled_vector_count_uses_ceiling_division() raises:
    assert_equal(target_pooled_vector_count(5, 2), 3)
    assert_equal(target_pooled_vector_count(6, 3), 2)
    assert_equal(target_pooled_vector_count(1, 8), 1)


def test_resolved_token_pooling_protected_token_count_clamps_to_leave_one_cluster() raises:
    assert_equal(resolved_token_pooling_protected_token_count(1, 4), 0)
    assert_equal(resolved_token_pooling_protected_token_count(3, 1), 1)
    assert_equal(resolved_token_pooling_protected_token_count(3, 5), 2)


def test_sequential_token_pool_document_mean_pools_contiguous_ranges() raises:
    var document = EncodedDocument(
        "doc-a",
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
    )

    var pooled = sequential_token_pool_document(document, 2)

    assert_equal(pooled.vector_count, 3)
    assert_equal(pooled.token_vectors[0][0], 0.5)
    assert_equal(pooled.token_vectors[0][1], 0.5)
    assert_equal(pooled.token_vectors[1][0], 1.0)
    assert_equal(pooled.token_vectors[1][1], 1.0)
    assert_equal(pooled.token_vectors[2][0], 1.0)
    assert_equal(pooled.token_vectors[2][1], 1.0)


def test_hierarchical_token_pool_document_merges_by_similarity_not_order() raises:
    var document = EncodedDocument(
        "doc-b",
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ],
    )

    var pooled = hierarchical_token_pool_document(document, 2)

    assert_equal(pooled.vector_count, 2)
    assert_equal(pooled.token_vectors[0][0], 0.95)
    assert_equal(pooled.token_vectors[0][1], 0.05)
    assert_equal(pooled.token_vectors[1][0], 0.05)
    assert_equal(pooled.token_vectors[1][1], 0.95)


def test_transform_dispatch_supports_sequential_and_hierarchical_pooling() raises:
    var document = EncodedDocument(
        "doc-c",
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ],
    )
    var sequential = apply_document_representation_transform_to_document(
        document,
        token_pooling_document_representation_transform(
            2,
            DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
        ),
    )
    var hierarchical = apply_document_representation_transform_to_document(
        document,
        token_pooling_document_representation_transform(
            2,
            DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
        ),
    )

    assert_equal(sequential.vector_count, 2)
    assert_equal(sequential.token_vectors[0][0], 0.5)
    assert_equal(hierarchical.vector_count, 2)
    assert_equal(hierarchical.token_vectors[0][0], 0.95)


def test_budgeted_sequential_pooling_preserves_protected_prefix_and_hits_target() raises:
    var document = EncodedDocument(
        "doc-d",
        [
            [10.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [5.0, 0.0],
            [7.0, 0.0],
        ],
    )

    var pooled = apply_document_representation_transform_to_document(
        document,
        budgeted_token_pooling_document_representation_transform(
            3,
            DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
            1,
            DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
        ),
    )

    assert_equal(pooled.vector_count, 3)
    assert_equal(pooled.token_vectors[0][0], 10.0)
    assert_equal(pooled.token_vectors[1][0], 2.0)
    assert_equal(pooled.token_vectors[2][0], 6.0)


def test_budgeted_hierarchical_pooling_preserves_protected_suffix_and_hits_target() raises:
    var document = EncodedDocument(
        "doc-e",
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [42.0, 42.0],
        ],
    )

    var pooled = apply_document_representation_transform_to_document(
        document,
        budgeted_token_pooling_document_representation_transform(
            3,
            DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
            1,
            DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_LAST,
        ),
    )

    assert_equal(pooled.vector_count, 3)
    assert_equal(pooled.token_vectors[0][0], 0.95)
    assert_equal(pooled.token_vectors[0][1], 0.05)
    assert_equal(pooled.token_vectors[1][0], 0.05)
    assert_equal(pooled.token_vectors[1][1], 0.95)
    assert_equal(pooled.token_vectors[2][0], 42.0)
    assert_equal(pooled.token_vectors[2][1], 42.0)


def test_budgeted_pooling_disables_protected_tokens_when_budget_is_one() raises:
    var document = EncodedDocument(
        "doc-f",
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [5.0, 0.0],
        ],
    )

    var pooled = apply_document_representation_transform_to_document(
        document,
        budgeted_token_pooling_document_representation_transform(
            1,
            DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
            2,
            DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
        ),
    )

    assert_equal(pooled.vector_count, 1)
    assert_equal(pooled.token_vectors[0][0], 3.0)


def test_transform_chain_applies_to_packed_index() raises:
    var index = pack_documents(
        [
            EncodedDocument(
                "doc-a",
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.1, 0.9],
                ],
            ),
            EncodedDocument(
                "doc-b",
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ],
            ),
        ]
    )
    var transformed = apply_document_representation_transforms_to_packed_index(
        index,
        [
            token_pooling_document_representation_transform(2),
            prefix_pruning_document_representation_transform(1),
        ],
    )

    assert_equal(transformed.document_count, 2)
    assert_equal(transformed.total_vector_count, 2)
    assert_equal(transformed.doc_ids[0], "doc-a")
    assert_equal(transformed.doc_offsets[0], 0)
    assert_equal(transformed.doc_offsets[1], 1)
    assert_equal(transformed.doc_offsets[2], 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
