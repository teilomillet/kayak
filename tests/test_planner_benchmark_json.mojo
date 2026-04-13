from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
)
from kayak.benchmarks import (
    FaithfulnessFrontierSummary,
    PlannerBenchmarkSummary,
    build_planner_benchmark_summary,
    default_single_core_scale_profiles,
    faithfulness_frontier_summary_json,
    make_single_core_scale_fixture,
    planner_benchmark_summary_json,
)
from kayak.collections import (
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    load_snapshot_search_artifact_availability,
)
from kayak.planning import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SearchPlanSelectionRequest,
    best_effort_faithfulness_policy,
    clause_text_stage2_operator,
    document_proxy_search_plan,
    exact_late_interaction_stage2_operator,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig
from kayak.text import DocumentTextCorpus


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


def test_planner_benchmark_summary_json_contains_selection_fields() raises:
    var measured_json = faithfulness_frontier_summary_json(
        FaithfulnessFrontierSummary(
            "mock://dataset",
            "mock-model",
            "synthetic_scale",
            "docs_64",
            "mock_collection",
            "snapshot-0001",
            document_proxy_search_plan(
                2,
                16,
                best_effort_faithfulness_policy(),
            ).candidate_generator,
            2,
            16,
            4,
            64,
            0,
            8,
            64,
            512,
            512,
            64,
            0.001,
            0.002,
            16.0,
            0.75,
            0.5,
            0.5,
            0.5,
            1.0,
            64,
            128,
            4096,
            64.0,
            64.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )
    var json = planner_benchmark_summary_json(
        PlannerBenchmarkSummary(
            "balanced",
            document_proxy_search_plan(
                2,
                16,
                best_effort_faithfulness_policy(),
                exact_late_interaction_stage2_operator(),
            ),
            "promoted",
            "planner used the default candidate-generator order for goal balanced",
            ["exact_full_scan", "document_proxy"],
            ["document_proxy", "exact_full_scan"],
            FaithfulnessFrontierSummary(
                "mock://dataset",
                "mock-model",
                "synthetic_scale",
                "docs_64",
                "mock_collection",
                "snapshot-0001",
                document_proxy_search_plan(
                    2,
                    16,
                    best_effort_faithfulness_policy(),
                ).candidate_generator,
                2,
                16,
                4,
                64,
                0,
                8,
                64,
                512,
                512,
                64,
                0.001,
                0.002,
                16.0,
                0.75,
                0.5,
                0.5,
                0.5,
                1.0,
                64,
                128,
                4096,
                64.0,
                64.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        )
    )

    assert_equal(json.find("\"planning_goal\":\"balanced\"") != -1, True)
    assert_equal(
        json.find("\"reference_scoring_semantics_kind\":\"exact_late_interaction\"")
            != -1,
        True,
    )
    assert_equal(
        json.find("\"stage2_reference_kind\":\"exact_late_interaction\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"selected_candidate_generator_status\":\"promoted\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"effective_candidate_generator_order\":[\"document_proxy\",\"exact_full_scan\"]")
            != -1,
        True,
    )
    assert_equal(json.find("\"measured\":") != -1, True)
    assert_equal(json.find(measured_json) != -1, True)


def test_build_planner_benchmark_summary_reports_selected_generator() raises:
    var profile = default_single_core_scale_profiles()[0].copy()
    var fixture = make_single_core_scale_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-planner-benchmark-summary"),
        CollectionId("planner-benchmark-summary"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        fixture.stored_index,
        profile.document_vector_count,
        profile.vector_dim,
        8,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var availability = load_snapshot_search_artifact_availability(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var summary = build_planner_benchmark_summary(
        single_core_backend(),
        fixture.stored_task,
        snapshot,
        availability,
        SearchPlanSelectionRequest(
            profile.final_k,
            profile.candidate_k,
            best_effort_faithfulness_policy(),
            goal=SEARCH_PLANNING_GOAL_BALANCED,
        ),
        exact_late_interaction_stage2_operator(),
        profile.query_vector_count,
        profile.vector_dim,
        8,
    )

    assert_equal(summary.planning_goal, "balanced")
    assert_equal(summary.plan.reference_scoring_semantics.kind, "exact_late_interaction")
    assert_equal(summary.plan.stage2_reference_operator.kind, "exact_late_interaction")
    assert_equal(summary.plan.candidate_generator.kind, "centroid_postings_imputed_flat")
    assert_equal(summary.selected_candidate_generator_status, "promoted")
    assert_equal(summary.measured.mean_candidate_recall_at_final_k >= 0.0, True)
    assert_equal(summary.measured.stage1_byte_size > 0, True)


def test_build_planner_benchmark_summary_supports_clause_text_on_mirrored_text_corpus() raises:
    var profile = default_single_core_scale_profiles()[0].copy()
    var fixture = make_single_core_scale_fixture(profile)
    var doc_ids = fixture.stored_index.index.doc_ids.copy()
    var texts = List[String]()
    for doc_id in doc_ids:
        texts.append("benchmark text for " + doc_id)

    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-planner-benchmark-clause-text"),
        CollectionId("planner-benchmark-clause-text"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        fixture.stored_index,
        DocumentTextCorpus(doc_ids^, texts^),
        profile.document_vector_count,
        profile.vector_dim,
        8,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var availability = load_snapshot_search_artifact_availability(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var summary = build_planner_benchmark_summary(
        single_core_backend(),
        fixture.stored_task,
        snapshot,
        availability,
        SearchPlanSelectionRequest(
            profile.final_k,
            profile.candidate_k,
            best_effort_faithfulness_policy(),
            goal=SEARCH_PLANNING_GOAL_BALANCED,
        ),
        clause_text_stage2_operator(),
        profile.query_vector_count,
        profile.vector_dim,
        8,
    )

    assert_equal(summary.plan.stage3_verifier.kind, "clause_text")
    assert_equal(summary.plan.candidate_generator.kind, "centroid_postings_imputed_flat")
    assert_equal(summary.measured.stage1_byte_size > 0, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
