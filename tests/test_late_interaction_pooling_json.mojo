from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    JudgedQuery,
    JudgedTask,
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_SCALAR_NAME,
)
from kayak.benchmarks import (
    LateInteractionPoolingSummary,
    build_late_interaction_pooling_summary,
    late_interaction_pooling_summary_json,
)
from kayak.index import pack_documents


def make_late_interaction_pooling_fixture() raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "mock://late-interaction-pooling",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "mock",
            "late_interaction_pooling_fixture",
            "Small exact fixture for late-interaction pooling summaries.",
            "mrr",
            1,
            1,
            4,
            2,
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [1.0, 0.0]]),
                EncodedDocument("doc-b", [[1.0, 0.0], [0.0, 1.0]]),
            ],
            [
                JudgedQuery(
                    "q-1",
                    "late interaction pooling query",
                    EncodedQuery([[1.0, 0.0]]),
                    ["doc-a"],
                )
            ],
        ),
    )


def test_late_interaction_pooling_summary_json_contains_operator_fields() raises:
    var stored_task = make_late_interaction_pooling_fixture()
    var stored_index = StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )
    var maxsim_summary = build_late_interaction_pooling_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        "maxsim",
        4,
    )
    var topk_summary = build_late_interaction_pooling_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        "topk_mean",
        2,
    )
    var json = late_interaction_pooling_summary_json(topk_summary)

    assert_equal(maxsim_summary.match_k, 1)
    assert_equal(maxsim_summary.mean_reference_recall_at_k, 1.0)
    assert_equal(topk_summary.pooling_kind, "topk_mean")
    assert_equal(topk_summary.match_k, 2)
    assert_equal(topk_summary.primary_value >= maxsim_summary.primary_value, True)
    assert_equal(json.find("\"pooling_kind\":\"topk_mean\"") != -1, True)
    assert_equal(json.find("\"match_k\":2") != -1, True)
    assert_equal(json.find("\"mean_reference_recall_at_k\":") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
