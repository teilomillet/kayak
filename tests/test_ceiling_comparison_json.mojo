from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    EncodedDocument,
    EncodedQuery,
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
    load_resolved_collection_snapshot,
)
from kayak.benchmarks import (
    CeilingComparisonSummary,
    build_exact_clause_text_ceiling_summary,
    build_stage_aware_ceiling_summary_for_plan,
    ceiling_comparison_summary_json,
)
from kayak.index import pack_documents
from kayak.runtime import ExactCpuBackend
from kayak.text import DocumentTextCorpus


def unique_collection_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def make_ceiling_fixture() raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "mock://ceiling",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "mock",
            "ceiling_fixture",
            "Small exact fixture for ceiling summaries.",
            "mrr",
            1,
            1,
            2,
            4,
            [
                EncodedDocument(
                    "doc-a",
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                ),
                EncodedDocument(
                    "doc-b",
                    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                ),
            ],
            [
                JudgedQuery(
                    "q-1",
                    "alpha. beta",
                    EncodedQuery([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                    ["doc-a"],
                )
            ],
        ),
    )


def test_ceiling_comparison_summary_json_contains_method_fields() raises:
    var summary = CeilingComparisonSummary(
        "mock://ceiling",
        "mock-model",
        "mock",
        "ceiling_fixture",
        "exact_clause_text_ceiling",
        "exact_full_scan",
        "clause_text",
        "text",
        True,
        ["document_text"],
        20,
        10,
        "mrr",
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.01,
    )
    var json = ceiling_comparison_summary_json(summary)

    assert_equal(json.find("\"method_kind\":\"exact_clause_text_ceiling\"") != -1, True)
    assert_equal(json.find("\"stage2_kind\":\"clause_text\"") != -1, True)
    assert_equal(
        json.find("\"stage2_materialized_artifact_families\":[\"document_text\"]")
            != -1,
        True,
    )
    assert_equal(json.find("\"reranker_kind\":\"clause_text\"") != -1, True)
    assert_equal(json.find("\"candidate_k\":20") != -1, True)


def test_build_exact_clause_text_ceiling_summary_reports_candidate_window() raises:
    var stored_task = make_ceiling_fixture()
    var stored_index = StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )
    var summary = build_exact_clause_text_ceiling_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        DocumentTextCorpus(
            ["doc-a", "doc-b"],
            ["alpha beta answer", "gamma delta distractor"],
        ),
        2,
    )

    assert_equal(summary.candidate_k, 2)
    assert_equal(summary.stage2_kind, "clause_text")
    assert_equal(summary.stage2_family, "text")
    assert_equal(summary.stage2_requires_query_text, True)
    assert_equal(len(summary.stage2_materialized_artifact_families), 1)
    assert_equal(summary.stage2_materialized_artifact_families[0], "document_text")
    assert_equal(summary.mean_candidate_recall_at_final_k, 1.0)
    assert_equal(summary.mean_search_seconds >= 0.0, True)


def test_build_stage_aware_ceiling_summary_propagates_stage2_materialized_families() raises:
    var stored_task = StoredJudgedTask(
        "mock://ceiling-stage-aware",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "mock",
            "ceiling_stage_aware_fixture",
            "Small stage-aware fixture for ceiling summaries.",
            "mrr",
            1,
            2,
            2,
            2,
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                EncodedDocument("doc-b", [[0.6, 0.6], [0.6, 0.6]]),
            ],
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
        unique_collection_root("kayak-ceiling-stage-aware"),
        CollectionId("ceiling-stage-aware"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        StoredPackedIndex(
            stored_task.dataset_id.copy(),
            stored_task.model_name.copy(),
            stored_task.vector_scalar_name.copy(),
            pack_documents(stored_task.task.documents),
        ),
        0,
        0,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )

    var summary = build_stage_aware_ceiling_summary_for_plan(
        ExactCpuBackend(),
        stored_task,
        snapshot,
        document_proxy_search_plan(
            1,
            2,
            best_effort_faithfulness_policy(),
        ),
    )

    assert_equal(len(summary.stage2_materialized_artifact_families), 1)
    assert_equal(summary.stage2_materialized_artifact_families[0], "late_interaction")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
