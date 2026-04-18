from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    JudgedQuery,
    JudgedTask,
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_SCALAR_NAME,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
)
from kayak.benchmarks import (
    TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT,
    TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING,
    TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING,
    build_training_free_sequence_compression_summary,
    pool_factor_for_target_document_vector_budget,
    standard_training_free_sequence_compression_budget_sizes,
    training_free_sequence_compression_summary_json,
)
from kayak.index import pack_documents
from kayak.runtime import ExactCpuBackend


def make_training_free_sequence_compression_fixture() raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "mock://training-free-sequence-compression",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "mock",
            "training_free_sequence_compression_fixture",
            "Small exact fixture for training-free sequence compression summaries.",
            "mrr",
            1,
            2,
            4,
            4,
            [
                EncodedDocument(
                    "doc-a",
                    [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                ),
                EncodedDocument(
                    "doc-b",
                    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                ),
            ],
            [
                JudgedQuery(
                    "q-1",
                    "sequence compression query",
                    EncodedQuery([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                    ["doc-a"],
                )
            ],
        ),
    )


def test_standard_training_free_sequence_compression_budget_sizes_reuse_pruning_grid() raises:
    var sizes = standard_training_free_sequence_compression_budget_sizes(10)
    assert_equal(len(sizes) >= 3, True)
    assert_equal(sizes[0], 4)
    assert_equal(sizes[len(sizes) - 1], 10)


def test_pool_factor_for_target_document_vector_budget_uses_ceiling_division() raises:
    assert_equal(pool_factor_for_target_document_vector_budget(10, 4), 3)
    assert_equal(pool_factor_for_target_document_vector_budget(10, 10), 1)


def test_training_free_sequence_compression_summary_json_contains_method_fields() raises:
    var stored_task = make_training_free_sequence_compression_fixture()
    var stored_index = StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )
    var baseline_summary = build_training_free_sequence_compression_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        Path("/tmp/kayak-training-free-sequence-compression-full"),
        TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT,
        stored_task.task.nominal_document_vector_count,
    )
    var pruning_summary = build_training_free_sequence_compression_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        Path("/tmp/kayak-training-free-sequence-compression-pruning"),
        TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING,
        2,
    )
    var pooling_summary = build_training_free_sequence_compression_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        Path("/tmp/kayak-training-free-sequence-compression-pooling"),
        TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING,
        2,
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    )
    var json = training_free_sequence_compression_summary_json(pooling_summary)

    assert_equal(baseline_summary.method_kind, "full_exact")
    assert_equal(baseline_summary.mean_reference_recall_at_k, 1.0)
    assert_equal(pruning_summary.transformed_vector_count < pruning_summary.full_vector_count, True)
    assert_equal(pooling_summary.derived_pool_factor, 2)
    assert_equal(pooling_summary.protected_token_count, 1)
    assert_equal(pooling_summary.protected_token_position, "first")
    assert_equal(json.find("\"method_kind\":\"token_pooling\"") != -1, True)
    assert_equal(json.find("\"transform_policy\":\"hierarchical\"") != -1, True)
    assert_equal(json.find("\"requested_document_vector_budget\":2") != -1, True)
    assert_equal(json.find("\"derived_pool_factor\":2") != -1, True)
    assert_equal(json.find("\"protected_token_count\":1") != -1, True)
    assert_equal(json.find("\"protected_token_position\":\"first\"") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
