from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
    DOCUMENT_ENCODER_COMPRESSION_KIND_ATTENTION_GUIDED_CLUSTERING,
    DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
    DOCUMENT_ENCODER_COMPRESSION_KIND_NONE,
    DOCUMENT_ENCODER_COMPRESSION_KIND_SEQUENCE_RESIZING,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_COUNT,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_POSITION,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    EncodedDocument,
    MultiVectorIndexCompressionLowering,
    MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_ENCODER,
    MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION,
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_ATTENTION_GUIDED_CLUSTERING,
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT,
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS,
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_SEQUENCE_RESIZING,
    apply_document_representation_transforms_to_documents,
    apply_multi_vector_index_compression_to_documents,
    budgeted_token_pooling_document_representation_transform,
    document_encoder_compression_config_value,
    document_representation_transform_config_value,
    has_document_encoder_compression,
    memory_tokens_document_encoder_compression,
    multi_vector_index_compression_document_encoder_compression,
    multi_vector_index_compression_lowering,
    multi_vector_index_compression_method_execution_boundary,
    multi_vector_index_compression_method_is_stored_representation_executable,
    multi_vector_index_compression_spec,
    multi_vector_index_compression_transforms,
    same_document_representation_transforms,
    pool_factor_for_multi_vector_index_compression_budget,
    supported_multi_vector_index_compression_methods,
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


def test_multi_vector_index_compression_methods_classify_all_paper_cases() raises:
    var methods = supported_multi_vector_index_compression_methods()

    assert_equal(len(methods), 5)
    assert_equal(
        multi_vector_index_compression_method_execution_boundary(
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT
        ),
        MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION,
    )
    assert_equal(
        multi_vector_index_compression_method_execution_boundary(
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING
        ),
        MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION,
    )
    assert_equal(
        multi_vector_index_compression_method_execution_boundary(
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_SEQUENCE_RESIZING
        ),
        MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_ENCODER,
    )
    assert_equal(
        multi_vector_index_compression_method_execution_boundary(
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS
        ),
        MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_ENCODER,
    )
    assert_equal(
        multi_vector_index_compression_method_execution_boundary(
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_ATTENTION_GUIDED_CLUSTERING
        ),
        MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_ENCODER,
    )
    assert_equal(
        multi_vector_index_compression_method_is_stored_representation_executable(
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS
        ),
        False,
    )


def test_hierarchical_pooling_multi_vector_compression_derives_manifest_fields() raises:
    var spec = multi_vector_index_compression_spec(
        10,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
        4,
        1,
        DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    )
    var transforms = multi_vector_index_compression_transforms(spec)

    assert_equal(pool_factor_for_multi_vector_index_compression_budget(10, 4), 3)
    assert_equal(
        spec.execution_boundary,
        MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION,
    )
    assert_equal(spec.transform_policy, DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL)
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
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
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


def test_hierarchical_pooling_application_matches_direct_transform_runtime() raises:
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
    var spec = multi_vector_index_compression_spec(
        4,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
        2,
        1,
        DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    )

    assert_documents_close(
        apply_document_representation_transforms_to_documents(
            documents,
            multi_vector_index_compression_transforms(spec),
        ),
        apply_multi_vector_index_compression_to_documents(
            documents,
            spec,
        ),
    )


def test_encoder_bound_multi_vector_methods_reject_transform_lowering() raises:
    for method_kind in [
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_SEQUENCE_RESIZING,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_ATTENTION_GUIDED_CLUSTERING,
    ]:
        var spec = multi_vector_index_compression_spec(8, method_kind, 4)
        var raised = False
        try:
            _ = multi_vector_index_compression_transforms(spec)
        except:
            raised = True

        assert_equal(
            spec.execution_boundary,
            MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_ENCODER,
        )
        assert_equal(raised, True)


def test_multi_vector_index_compression_methods_lower_encoder_compression_provenance() raises:
    var hierarchical_spec = multi_vector_index_compression_spec(
        8,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
        4,
    )
    var sequence_resizing_spec = multi_vector_index_compression_spec(
        8,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_SEQUENCE_RESIZING,
        4,
    )
    var memory_tokens_spec = multi_vector_index_compression_spec(
        8,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS,
        4,
    )
    var attention_guided_clustering_spec = multi_vector_index_compression_spec(
        8,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_ATTENTION_GUIDED_CLUSTERING,
        4,
    )

    var hierarchical_manifest = (
        multi_vector_index_compression_document_encoder_compression(
            hierarchical_spec
        )
    )
    var sequence_resizing_manifest = (
        multi_vector_index_compression_document_encoder_compression(
            sequence_resizing_spec
        )
    )
    var memory_tokens_manifest = (
        multi_vector_index_compression_document_encoder_compression(
            memory_tokens_spec
        )
    )
    var attention_guided_clustering_manifest = (
        multi_vector_index_compression_document_encoder_compression(
            attention_guided_clustering_spec
        )
    )

    assert_equal(hierarchical_manifest.kind, DOCUMENT_ENCODER_COMPRESSION_KIND_NONE)
    assert_equal(
        sequence_resizing_manifest.kind,
        DOCUMENT_ENCODER_COMPRESSION_KIND_SEQUENCE_RESIZING,
    )
    assert_equal(
        memory_tokens_manifest.kind,
        DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
    )
    assert_equal(
        attention_guided_clustering_manifest.kind,
        DOCUMENT_ENCODER_COMPRESSION_KIND_ATTENTION_GUIDED_CLUSTERING,
    )
    assert_equal(
        document_encoder_compression_config_value(
            sequence_resizing_manifest,
            DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "4",
    )
    assert_equal(
        document_encoder_compression_config_value(
            memory_tokens_manifest,
            DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "4",
    )
    assert_equal(
        document_encoder_compression_config_value(
            attention_guided_clustering_manifest,
            DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "4",
    )


def test_multi_vector_index_compression_lowering_uses_one_document_side_seam() raises:
    var full_exact = multi_vector_index_compression_lowering(
        multi_vector_index_compression_spec(
            8,
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT,
            4,
        )
    )
    var hierarchical = multi_vector_index_compression_lowering(
        multi_vector_index_compression_spec(
            8,
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
            4,
        )
    )
    var memory_tokens = multi_vector_index_compression_lowering(
        multi_vector_index_compression_spec(
            8,
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS,
            4,
        )
    )

    assert_equal(len(full_exact.document_representation_transforms), 0)
    assert_equal(
        has_document_encoder_compression(full_exact.document_encoder_compression),
        False,
    )
    assert_equal(len(hierarchical.document_representation_transforms), 1)
    assert_equal(
        same_document_representation_transforms(
            hierarchical.document_representation_transforms,
            multi_vector_index_compression_transforms(hierarchical.spec),
        ),
        True,
    )
    assert_equal(
        has_document_encoder_compression(
            hierarchical.document_encoder_compression
        ),
        False,
    )
    assert_equal(len(memory_tokens.document_representation_transforms), 0)
    assert_equal(
        memory_tokens.document_encoder_compression.kind,
        DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
    )
    assert_equal(
        document_encoder_compression_config_value(
            memory_tokens.document_encoder_compression,
            DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "4",
    )


def test_multi_vector_index_compression_lowering_rejects_cross_boundary_mixture() raises:
    var spec = multi_vector_index_compression_spec(
        8,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS,
        4,
    )
    var raised = False

    try:
        _ = MultiVectorIndexCompressionLowering(
            spec,
            [
                budgeted_token_pooling_document_representation_transform(
                    4,
                    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
                    1,
                    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
                )
            ],
            memory_tokens_document_encoder_compression(4),
        )
    except:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
