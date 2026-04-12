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
    build_stage_aware_search_summary,
    default_single_core_scale_profiles,
    make_single_core_scale_fixture,
)
from kayak.collections import ensure_one_segment_collection_mirror
from kayak.planning import (
    centroid_heads_search_plan,
    exact_full_scan_search_plan,
    best_effort_faithfulness_policy,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig


def unique_collection_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def single_core_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def test_single_core_scale_fixture_exact_full_scan_is_perfect() raises:
    var profile = default_single_core_scale_profiles()[0].copy()
    var fixture = make_single_core_scale_fixture(profile)
    var evaluation = evaluate_query_hits(
        fixture.stored_task.task.queries[0],
        final_hits_to_search_hits(
            search_collection_for_plan(
                ExactCpuBackend(),
                fixture.stored_task.task.queries[0].query,
                load_resolved_collection_snapshot(
                    ensure_one_segment_collection_mirror(
                        unique_collection_root("kayak-single-core-scale-exact"),
                        CollectionId("single-core-scale-exact"),
                        TenantId("public"),
                        NamespaceId("benchmark"),
                        SnapshotId("snapshot-0001"),
                        1,
                        fixture.stored_index,
                        0,
                        0,
                        profile.centroid_head_posting_cap,
                    ),
                    SnapshotId("snapshot-0001"),
                ),
                exact_full_scan_search_plan(profile.final_k, profile.final_k),
            )
        ),
        profile.final_k,
        fixture.stored_task.task.primary_metric,
    )

    assert_equal(evaluation.success_at_k, 1.0)
    assert_equal(evaluation.primary_value, 1.0)


def test_single_core_scale_stage_aware_summary_reports_stage_density() raises:
    var profile = default_single_core_scale_profiles()[0].copy()
    var fixture = make_single_core_scale_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-single-core-scale-summary"),
        CollectionId("single-core-scale-summary"),
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
        single_core_backend(),
        fixture.stored_task,
        snapshot,
        centroid_heads_search_plan(
            profile.final_k,
            profile.candidate_k,
            best_effort_faithfulness_policy(),
        ),
    )

    assert_equal(summary.family, "synthetic_scale")
    assert_equal(summary.document_count, profile.document_count)
    assert_equal(summary.nominal_query_vector_count, profile.query_vector_count)
    assert_equal(
        summary.nominal_document_vector_count,
        profile.document_vector_count,
    )
    assert_equal(summary.candidate_stage_document_count, profile.document_count)
    assert_equal(summary.candidate_stage_vector_count > 0, True)
    assert_equal(summary.candidate_stage_byte_size > 0, True)
    assert_equal(summary.exact_stage_document_count, profile.candidate_k)
    assert_equal(
        summary.exact_stage_vector_count,
        profile.candidate_k * profile.document_vector_count,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
