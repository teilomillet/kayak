from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_COUNT,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_POSITION,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    EncodedDocument,
    TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT,
    TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING,
    TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING,
    apply_document_representation_transforms_to_documents,
    apply_training_free_sequence_compression_to_documents,
    document_representation_transform_config_value,
    pool_factor_for_target_document_vector_budget,
    training_free_sequence_compression_spec,
    training_free_sequence_compression_transforms,
)


def assert_documents_close(
    read expected: List[EncodedDocument], read observed: List[EncodedDocument]
) raises:
    assert_equal(len(expected), len(observed))
    for document_index in range(len(expected)):
        assert_equal(expected[document_index].doc_id, observed[document_index].doc_id)
        assert_equal(
            expected[document_index].vector_count,
            observed[document_index].vector_count,
        )
        for vector_index in range(expected[document_index].vector_count):
            for dim_index in range(len(expected[document_index].token_vectors[vector_index])):
                assert_equal(
                    abs(
                        Float64(
                            expected[document_index].token_vectors[vector_index][dim_index]
                        )
                        - Float64(
                            observed[document_index].token_vectors[vector_index][dim_index]
                        )
                    )
                    < 0.00001,
                    True,
                )


def test_full_exact_sequence_compression_spec_has_no_transforms() raises:
    var spec = training_free_sequence_compression_spec(
        8,
        TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT,
        3,
    )
    var transforms = training_free_sequence_compression_transforms(spec)

    assert_equal(spec.available_document_vector_count, 8)
    assert_equal(spec.requested_document_vector_budget, 8)
    assert_equal(spec.transform_policy, "")
    assert_equal(spec.derived_pool_factor, 0)
    assert_equal(spec.protected_token_count, 0)
    assert_equal(spec.protected_token_position, "")
    assert_equal(len(transforms), 0)


def test_token_pooling_sequence_compression_spec_derives_manifest_fields() raises:
    var spec = training_free_sequence_compression_spec(
        10,
        TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING,
        4,
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
        1,
        DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    )
    var transforms = training_free_sequence_compression_transforms(spec)

    assert_equal(pool_factor_for_target_document_vector_budget(10, 4), 3)
    assert_equal(spec.transform_policy, DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL)
    assert_equal(spec.derived_pool_factor, 3)
    assert_equal(spec.protected_token_count, 1)
    assert_equal(len(transforms), 1)
    assert_equal(
        document_representation_transform_config_value(
            transforms[0],
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "4",
    )
    assert_equal(
        document_representation_transform_config_value(
            transforms[0],
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
        ),
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    )
    assert_equal(
        document_representation_transform_config_value(
            transforms[0],
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_COUNT,
        ),
        "1",
    )
    assert_equal(
        document_representation_transform_config_value(
            transforms[0],
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_POSITION,
        ),
        DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    )


def test_sequence_compression_application_matches_direct_transform_runtime() raises:
    var documents = [
        EncodedDocument(
            "doc-a",
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ),
        EncodedDocument(
            "doc-b",
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
        ),
    ]
    var prefix_spec = training_free_sequence_compression_spec(
        4,
        TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING,
        2,
    )
    var pooling_spec = training_free_sequence_compression_spec(
        4,
        TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING,
        2,
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
        1,
        DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    )

    assert_documents_close(
        apply_document_representation_transforms_to_documents(
            documents,
            training_free_sequence_compression_transforms(prefix_spec),
        ),
        apply_training_free_sequence_compression_to_documents(
            documents,
            prefix_spec,
        ),
    )
    assert_documents_close(
        apply_document_representation_transforms_to_documents(
            documents,
            training_free_sequence_compression_transforms(pooling_spec),
        ),
        apply_training_free_sequence_compression_to_documents(
            documents,
            pooling_spec,
        ),
    )
    assert_equal(prefix_spec.transform_policy, DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
