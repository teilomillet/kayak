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
)
from kayak.benchmarks import (
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT,
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS,
    build_multi_vector_index_compression_summary,
    multi_vector_index_compression_summary_json,
    standard_multi_vector_index_compression_budget_sizes,
)
from kayak.index import pack_documents
from kayak.runtime import ExactCpuBackend


def make_multi_vector_index_compression_fixture() raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "mock://multi-vector-index-compression",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "mock",
            "multi_vector_index_compression_fixture",
            "Small exact fixture for multi-vector index compression summaries.",
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
                    "multi-vector compression query",
                    EncodedQuery([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                    ["doc-a"],
                )
            ],
        ),
    )


def test_standard_multi_vector_index_compression_budget_sizes_reuse_pruning_grid() raises:
    var sizes = standard_multi_vector_index_compression_budget_sizes(10)
    assert_equal(len(sizes) >= 3, True)
    assert_equal(sizes[0], 4)
    assert_equal(sizes[len(sizes) - 1], 10)


def test_multi_vector_index_compression_summary_json_contains_boundary_fields() raises:
    var stored_task = make_multi_vector_index_compression_fixture()
    var stored_index = StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )
    var baseline_summary = build_multi_vector_index_compression_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        Path("/tmp/kayak-multi-vector-index-compression-full"),
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT,
        stored_task.task.nominal_document_vector_count,
    )
    var pooling_summary = build_multi_vector_index_compression_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        Path("/tmp/kayak-multi-vector-index-compression-hierarchical"),
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
        2,
    )
    var json = multi_vector_index_compression_summary_json(pooling_summary)

    assert_equal(baseline_summary.method_kind, "full_exact")
    assert_equal(baseline_summary.execution_boundary, "stored_representation")
    assert_equal(baseline_summary.mean_reference_recall_at_k, 1.0)
    assert_equal(pooling_summary.method_kind, "hierarchical_pooling")
    assert_equal(pooling_summary.execution_boundary, "stored_representation")
    assert_equal(pooling_summary.transformed_vector_count < pooling_summary.full_vector_count, True)
    assert_equal(pooling_summary.derived_pool_factor, 2)
    assert_equal(pooling_summary.protected_token_count, 1)
    assert_equal(pooling_summary.protected_token_position, "first")
    assert_equal(json.find("\"method_kind\":\"hierarchical_pooling\"") != -1, True)
    assert_equal(json.find("\"execution_boundary\":\"stored_representation\"") != -1, True)
    assert_equal(json.find("\"requested_document_vector_budget\":2") != -1, True)
    assert_equal(json.find("\"derived_pool_factor\":2") != -1, True)


def test_multi_vector_index_compression_summary_rejects_encoder_bound_method() raises:
    var stored_task = make_multi_vector_index_compression_fixture()
    var stored_index = StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )

    var raised = False
    try:
        _ = build_multi_vector_index_compression_summary(
            ExactCpuBackend(),
            stored_task,
            stored_index,
            Path("/tmp/kayak-multi-vector-index-compression-memory"),
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS,
            2,
        )
    except:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
