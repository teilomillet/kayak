from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    evaluate_query_hits,
    final_hits_to_search_hits,
    load_resolved_collection_snapshot,
    search_collection_for_plan,
)
from kayak.benchmarks import (
    ContradictionHardRecallFixture,
    ContradictionHardRecallProfile,
    build_stage_aware_search_summary,
    make_contradiction_hard_recall_fixture,
)
from kayak.collections import (
    ResolvedCollectionSnapshot,
    ensure_one_segment_collection_mirror,
)
from kayak.planning import (
    SearchPlan,
    best_effort_faithfulness_policy,
    document_proxy_search_plan,
    exact_full_scan_search_plan,
)
from kayak.runtime import ExactCpuBackend


def unique_collection_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def assert_stage_aware_plan_runs(
    read fixture: ContradictionHardRecallFixture,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read profile: ContradictionHardRecallProfile,
) raises:
    var summary = build_stage_aware_search_summary(
        ExactCpuBackend(),
        fixture.stored_task,
        snapshot,
        plan,
    )
    assert_equal(summary.query_count, profile.query_count)
    assert_equal(summary.candidate_stage.vector_count > 0, True)
    assert_equal(summary.mean_candidate_recall_at_final_k >= 0.0, True)


def contradiction_profile(slice_name: String) raises -> ContradictionHardRecallProfile:
    return ContradictionHardRecallProfile(
        "contradiction_hard_recall",
        slice_name,
        "Small polarity-sensitive fixture for contradiction hard-recall checks.",
        4,
        3,
        6,
        12,
        2,
        4,
        2,
        16,
    )


def test_contradiction_hard_recall_exact_full_scan_is_perfect() raises:
    var profile = contradiction_profile("test_exact")
    var fixture = make_contradiction_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-contradiction-hard-recall-exact"),
        CollectionId("contradiction-hard-recall-exact"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        fixture.stored_index,
        0,
        0,
        profile.centroid_head_posting_cap,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var query = fixture.stored_task.task.queries[0].copy()
    var evaluation = evaluate_query_hits(
        query,
        final_hits_to_search_hits(
            search_collection_for_plan(
                ExactCpuBackend(),
                query.query,
                snapshot,
                exact_full_scan_search_plan(profile.final_k, profile.final_k),
            )
        ),
        profile.final_k,
        fixture.stored_task.task.primary_metric,
    )

    assert_equal(evaluation.recall_at_k, 1.0)
    assert_equal(evaluation.primary_value, 1.0)


def test_contradiction_hard_recall_summary_reports_expected_vector_counts() raises:
    var profile = contradiction_profile("test_summary")
    var fixture = make_contradiction_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-contradiction-hard-recall-summary"),
        CollectionId("contradiction-hard-recall-summary"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        fixture.stored_index,
        0,
        0,
        profile.centroid_head_posting_cap,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var summary = build_stage_aware_search_summary(
        ExactCpuBackend(),
        fixture.stored_task,
        snapshot,
        document_proxy_search_plan(
            profile.final_k,
            4,
            best_effort_faithfulness_policy(),
        ),
    )

    assert_equal(summary.family, "contradiction_hard_recall")
    assert_equal(summary.query_count, profile.query_count)
    assert_equal(summary.nominal_query_vector_count, profile.slot_count + 1)
    assert_equal(
        summary.nominal_document_vector_count,
        profile.slot_count + 1 + profile.filler_vector_count,
    )
    assert_equal(summary.document_count, len(fixture.stored_task.task.documents))
    assert_equal(summary.vector_count > 0, True)
    assert_equal(summary.candidate_stage.vector_count > 0, True)
    assert_equal(summary.stage2_reference.vector_count >= profile.final_k, True)


def test_contradiction_hard_recall_corpus_contains_opposite_polarity_docs() raises:
    var profile = contradiction_profile("test_polarity")
    var fixture = make_contradiction_hard_recall_fixture(profile)
    var document_ids = fixture.stored_task.task.documents.copy()
    var saw_support = False
    var saw_refute = False

    for document in document_ids:
        if String(document.doc_id).find("-support-") != -1:
            saw_support = True
        if String(document.doc_id).find("-refute-") != -1:
            saw_refute = True

    assert_equal(saw_support, True)
    assert_equal(saw_refute, True)


def test_contradiction_hard_recall_stage_aware_plan_runs() raises:
    var profile = contradiction_profile("test_stage_aware")
    var fixture = make_contradiction_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-contradiction-hard-recall-stage-aware"),
        CollectionId("contradiction-hard-recall-stage-aware"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        fixture.stored_index,
        0,
        0,
        profile.centroid_head_posting_cap,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )

    assert_stage_aware_plan_runs(
        fixture,
        snapshot,
        document_proxy_search_plan(
            profile.final_k,
            4,
            best_effort_faithfulness_policy(),
        ),
        profile,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
