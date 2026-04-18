from std.testing import TestSuite, assert_equal

from kayak.collections import (
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_COUNT,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_POSITION,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_LAST,
    budgeted_token_pooling_document_representation_transform,
    DOCUMENT_REPRESENTATION_TRANSFORM_KIND_PREFIX_PRUNING,
    DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
    DocumentRepresentationTransformConfigEntry,
    DocumentRepresentationTransformManifest,
    document_representation_transform_config_value,
    document_representation_transforms_have_kind,
    prefix_pruning_document_representation_transform,
    same_document_representation_transforms,
    token_pooling_document_representation_transform,
)


def test_transform_helper_builders_preserve_kind_and_config() raises:
    var pooling = token_pooling_document_representation_transform(4)
    var budgeted_pooling = budgeted_token_pooling_document_representation_transform(
        3,
        "hierarchical",
        1,
        DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_LAST,
    )
    var pruning = prefix_pruning_document_representation_transform(16)

    assert_equal(pooling.kind, DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING)
    assert_equal(
        document_representation_transform_config_value(
            pooling,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR,
        ),
        "4",
    )
    assert_equal(
        document_representation_transform_config_value(
            budgeted_pooling,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "3",
    )
    assert_equal(
        document_representation_transform_config_value(
            budgeted_pooling,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_COUNT,
        ),
        "1",
    )
    assert_equal(
        document_representation_transform_config_value(
            budgeted_pooling,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_POSITION,
        ),
        DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_LAST,
    )
    assert_equal(
        document_representation_transform_config_value(
            pruning,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "16",
    )
    assert_equal(
        document_representation_transform_config_value(
            pruning,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
        ),
        "prefix",
    )


def test_transform_helpers_keep_ordered_chain_semantics() raises:
    var transforms = [
        token_pooling_document_representation_transform(2),
        prefix_pruning_document_representation_transform(8),
    ]

    assert_equal(
        document_representation_transforms_have_kind(
            transforms,
            DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
        ),
        True,
    )
    assert_equal(
        document_representation_transforms_have_kind(
            transforms,
            DOCUMENT_REPRESENTATION_TRANSFORM_KIND_PREFIX_PRUNING,
        ),
        True,
    )
    assert_equal(same_document_representation_transforms(transforms, transforms), True)


def test_transform_manifest_rejects_duplicate_config_keys() raises:
    var raised = False

    try:
        _ = DocumentRepresentationTransformManifest(
            DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
            [
                DocumentRepresentationTransformConfigEntry(
                    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR,
                    "2",
                ),
                DocumentRepresentationTransformConfigEntry(
                    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR,
                    "4",
                ),
            ],
        )
    except:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
