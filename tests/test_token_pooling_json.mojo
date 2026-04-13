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
    build_token_pooling_summary,
    standard_token_pooling_factors,
    token_pooling_summary_json,
)
from kayak.index import pack_documents
from kayak.runtime import ExactCpuBackend


def make_token_pooling_fixture() raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "mock://token-pooling",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "mock",
            "token_pooling_fixture",
            "Small exact fixture for token-pooling summaries.",
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
                    "token pooling query",
                    EncodedQuery([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                    ["doc-a"],
                )
            ],
        ),
    )


def test_standard_token_pooling_factors_cap_and_deduplicate() raises:
    var factors = standard_token_pooling_factors(6)
    assert_equal(len(factors) >= 3, True)
    assert_equal(factors[0], 2)
    assert_equal(factors[len(factors) - 1], 6)


def test_token_pooling_summary_json_contains_policy_and_pool_factor() raises:
    var stored_task = make_token_pooling_fixture()
    var stored_index = StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )
    var summary = build_token_pooling_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        Path("/tmp/kayak-token-pooling-summary"),
        2,
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    )
    var json = token_pooling_summary_json(summary)

    assert_equal(json.find("\"pool_factor\":2") != -1, True)
    assert_equal(json.find("\"pooling_policy\":\"hierarchical\"") != -1, True)
    assert_equal(json.find("\"mean_reference_recall_at_k\":") != -1, True)
    assert_equal(summary.pooled_vector_count < summary.full_vector_count, True)
    assert_equal(summary.artifact_bytes_per_vector > 0.0, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
