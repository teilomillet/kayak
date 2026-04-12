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
    build_vector_pruning_summary,
    standard_vector_pruning_budget_sizes,
    vector_pruning_summary_json,
)
from kayak.index import pack_documents
from kayak.runtime import ExactCpuBackend


def make_vector_pruning_fixture() raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "mock://vector-pruning",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "mock",
            "vector_pruning_fixture",
            "Small exact fixture for vector-pruning summaries.",
            "mrr",
            1,
            2,
            4,
            4,
            [
                EncodedDocument(
                    "doc-a",
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                ),
                EncodedDocument(
                    "doc-b",
                    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                ),
            ],
            [
                JudgedQuery(
                    "q-1",
                    "vector pruning query",
                    EncodedQuery([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                    ["doc-a"],
                )
            ],
        ),
    )


def test_standard_vector_pruning_budget_sizes_cap_and_deduplicate() raises:
    var sizes = standard_vector_pruning_budget_sizes(10)
    assert_equal(len(sizes) >= 3, True)
    assert_equal(sizes[0], 4)
    assert_equal(sizes[len(sizes) - 1], 10)


def test_vector_pruning_summary_json_contains_budget_fields() raises:
    var stored_task = make_vector_pruning_fixture()
    var stored_index = StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )
    var summary = build_vector_pruning_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        Path("/tmp/kayak-vector-pruning-summary"),
        2,
    )
    var json = vector_pruning_summary_json(summary)

    assert_equal(json.find("\"document_vector_budget\":2") != -1, True)
    assert_equal(json.find("\"mean_reference_recall_at_k\":") != -1, True)
    assert_equal(summary.pruned_vector_count < summary.full_vector_count, True)
    assert_equal(summary.artifact_bytes_per_vector > 0.0, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
