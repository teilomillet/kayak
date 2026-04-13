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
    build_token_pooling_stage_aware_summary,
    token_pooling_stage_aware_summary_json,
)
from kayak.index import pack_documents
from kayak.runtime import ExactCpuBackend


def make_stage_aware_fixture() raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "mock://token-pooling-stage-aware",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "mock",
            "token_pooling_stage_aware_fixture",
            "Small exact fixture for stage-aware token-pooling summaries.",
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


def test_stage_aware_summary_json_contains_candidate_and_restore_fields() raises:
    var stored_task = make_stage_aware_fixture()
    var stored_index = StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )
    var summary = build_token_pooling_stage_aware_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        Path("/tmp/kayak-token-pooling-stage-aware-summary"),
        2,
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
        2,
    )
    var json = token_pooling_stage_aware_summary_json(summary)

    assert_equal(json.find("\"candidate_k\":2") != -1, True)
    assert_equal(
        json.find("\"stage2_reference_operator\":\"exact_late_interaction\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"mean_restored_reference_recall_at_final_k\":") != -1,
        True,
    )
    assert_equal(summary.mean_restored_reference_recall_at_final_k, 1.0)
    assert_equal(summary.mean_restored_recall_at_k, 1.0)
    assert_equal(summary.mean_stage2_window_vector_count > 0.0, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
