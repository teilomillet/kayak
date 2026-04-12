from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    JudgedQuery,
    JudgedTask,
    NamespaceId,
    SnapshotId,
    StoredJudgedTask,
    StoredPackedIndex,
    TenantId,
    VECTOR_SCALAR_NAME,
    best_effort_faithfulness_policy,
    document_proxy_search_plan,
    ensure_one_segment_collection_mirror,
    evaluate_query_hits,
    explain_collection_search,
    final_hits_to_search_hits,
    load_resolved_collection_snapshot,
    pack_documents,
    search_collection_for_plan,
)
from kayak.benchmarks import (
    StageAwareSearchSummary,
    build_stage_aware_search_summary_from_measurement,
    stage_aware_search_summary_json,
)


def unique_collection_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def test_stage_aware_search_summary_json_contains_stage_one_fields() raises:
    var json = stage_aware_search_summary_json(
        StageAwareSearchSummary(
            "mock://dataset",
            "mock-model",
            "browsecomp",
            "gold",
            "mock_collection",
            "snapshot-0001",
            "document_proxy",
            "best_effort",
            "ndcg",
            0.4,
            0.4,
            0.5,
            0.5,
            1.0,
            0.25,
            0.012,
            10,
            100,
            4,
            128,
            512,
            512,
            4096,
            32.0,
            8.0,
            128,
        )
    )

    assert_equal(
        json.find("\"candidate_generator_kind\":\"document_proxy\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"faithfulness_policy_kind\":\"best_effort\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"mean_candidate_recall_at_final_k\":0.25") != -1,
        True,
    )
    assert_equal(json.find("\"bytes_per_vector\":8.0") != -1, True)


def test_build_stage_aware_search_summary_reports_proxy_recall_gap() raises:
    var documents = [
        EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
        EncodedDocument("doc-b", [[0.6, 0.6], [0.6, 0.6]]),
        EncodedDocument("doc-c", [[0.0, 1.0], [0.0, 1.0]]),
    ]
    var task = StoredJudgedTask(
        "mock://hard-recall",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "browsecomp",
            "proxy-hard-recall",
            "proxy first stage misses the exact winner at candidate_k=1",
            "ndcg",
            1,
            2,
            2,
            2,
            documents.copy(),
            [
                JudgedQuery(
                    "q-1",
                    "find the exact two-token match",
                    EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
                    ["doc-a"],
                )
            ],
        ),
    )
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-stage-aware-summary"),
        CollectionId("stage-aware-summary"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        StoredPackedIndex(
            "mock://hard-recall",
            "mock-model",
            VECTOR_SCALAR_NAME,
            pack_documents(documents),
        ),
        0,
        0,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var plan = document_proxy_search_plan(
        1,
        1,
        best_effort_faithfulness_policy(),
    )
    var final_hits = search_collection_for_plan(
        ExactCpuBackend(),
        task.task.queries[0].query,
        snapshot,
        plan,
    )
    var evaluation = evaluate_query_hits(
        task.task.queries[0],
        final_hits_to_search_hits(final_hits),
        task.task.k,
        task.task.primary_metric,
    )
    var explain = explain_collection_search(
        ExactCpuBackend(),
        task.task.queries[0].query,
        snapshot,
        plan,
    )
    var summary = build_stage_aware_search_summary_from_measurement(
        task,
        snapshot,
        plan,
        Float64(evaluation.primary_value),
        Float64(evaluation.ndcg_at_k),
        Float64(evaluation.reciprocal_rank_at_k),
        Float64(evaluation.recall_at_k),
        Float64(evaluation.success_at_k),
        Float64(explain.candidate_recall_at_final_k),
        0.0,
    )

    assert_equal(summary.candidate_generator_kind, "document_proxy")
    assert_equal(summary.faithfulness_policy_kind, "best_effort")
    assert_equal(summary.mean_candidate_recall_at_final_k, 0.0)
    assert_equal(summary.mean_recall_at_k, 0.0)
    assert_equal(summary.document_count, 3)
    assert_equal(summary.vector_count, 6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
