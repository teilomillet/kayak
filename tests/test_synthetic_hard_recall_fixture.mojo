from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    evaluate_query_hits,
    explain_collection_search,
    final_hits_to_search_hits,
    load_resolved_collection_snapshot,
    search_collection_for_plan,
)
from kayak.benchmarks import (
    SyntheticHardRecallProfile,
    build_stage_aware_search_summary,
    make_synthetic_hard_recall_fixture,
)
from kayak.collections import ensure_one_segment_collection_mirror
from kayak.planning import (
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


def test_synthetic_hard_recall_exact_full_scan_is_perfect() raises:
    var profile = SyntheticHardRecallProfile(
        "synthetic_hard_recall",
        "test_exact",
        "Small shared-slot conjunction fixture for correctness checks.",
        4,
        3,
        6,
        12,
        2,
        4,
        2,
        16,
    )
    var fixture = make_synthetic_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-synthetic-hard-recall-exact"),
        CollectionId("synthetic-hard-recall-exact"),
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


def test_synthetic_hard_recall_summary_reports_expected_vector_counts() raises:
    var profile = SyntheticHardRecallProfile(
        "synthetic_hard_recall",
        "test_summary",
        "Small shared-slot conjunction fixture for summary checks.",
        4,
        3,
        6,
        12,
        2,
        4,
        2,
        16,
    )
    var fixture = make_synthetic_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-synthetic-hard-recall-summary"),
        CollectionId("synthetic-hard-recall-summary"),
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

    assert_equal(summary.family, "synthetic_hard_recall")
    assert_equal(summary.query_count, profile.query_count)
    assert_equal(summary.nominal_query_vector_count, profile.slot_count)
    assert_equal(
        summary.nominal_document_vector_count,
        profile.slot_count + profile.filler_vector_count,
    )
    assert_equal(summary.document_count, len(fixture.stored_task.task.documents))
    assert_equal(summary.vector_count > 0, True)
    assert_equal(summary.candidate_stage_vector_count > 0, True)
    assert_equal(summary.exact_stage_vector_count >= profile.final_k, True)


def test_synthetic_hard_recall_exact_rerank_recovers_when_candidates_cover_oracle() raises:
    var profile = SyntheticHardRecallProfile(
        "synthetic_hard_recall",
        "test_full_candidate_cover",
        "Small shared-slot conjunction fixture for rerank faithfulness checks.",
        4,
        3,
        6,
        12,
        2,
        4,
        2,
        16,
    )
    var fixture = make_synthetic_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-synthetic-hard-recall-rerank"),
        CollectionId("synthetic-hard-recall-rerank"),
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
    var plan = document_proxy_search_plan(
        profile.final_k,
        len(fixture.stored_task.task.documents),
        best_effort_faithfulness_policy(),
    )
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query.query,
        snapshot,
        plan,
    )
    var evaluation = evaluate_query_hits(
        query,
        final_hits_to_search_hits(explain.final_hits),
        profile.final_k,
        fixture.stored_task.task.primary_metric,
    )

    assert_equal(explain.candidate_recall_at_final_k, 1.0)
    assert_equal(evaluation.recall_at_k, 1.0)
    assert_equal(evaluation.primary_value, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
