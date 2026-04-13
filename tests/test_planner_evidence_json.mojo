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
    PlannerEvidenceCandidateSummary,
    PlannerEvidenceSummary,
    build_planner_evidence_summary,
    default_single_core_scale_profiles,
    make_single_core_scale_fixture,
    planner_evidence_summary_json,
)
from kayak.collections import (
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    load_snapshot_search_artifact_availability,
)
from kayak.planning import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SEARCH_PLANNING_GOAL_EXACT_ONLY,
    SearchPlanSelectionRequest,
    best_effort_faithfulness_policy,
    clause_text_stage2_operator,
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


def manual_measured_summary(
    candidate_generator_kind: String,
) -> FaithfulnessFrontierSummary:
    return FaithfulnessFrontierSummary(
        "mock://dataset",
        "mock-model",
        "synthetic_scale",
        "docs_64",
        "mock_collection",
        "snapshot-0001",
        candidate_generator_kind.copy(),
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


def test_planner_evidence_summary_json_contains_candidates_and_advisory() raises:
    var json = planner_evidence_summary_json(
        PlannerEvidenceSummary(
            "balanced",
            "exact_late_interaction",
            "late_interaction",
            False,
            ["late_interaction"],
            "document_proxy",
            "promoted",
            "planner used the default candidate-generator order for goal balanced",
            ["exact_full_scan", "document_proxy"],
            ["document_proxy", "exact_full_scan"],
            True,
            "",
            "selected candidate is undominated on current evidence",
            [
                PlannerEvidenceCandidateSummary(
                    "document_proxy",
                    "promoted",
                    True,
                    manual_measured_summary("document_proxy"),
                ),
                PlannerEvidenceCandidateSummary(
                    "exact_full_scan",
                    "exact_fallback",
                    False,
                    manual_measured_summary("exact_full_scan"),
                ),
            ],
        )
    )

    assert_equal(json.find("\"stage2_kind\":\"exact_late_interaction\"") != -1, True)
    assert_equal(
        json.find("\"stage2_family\":\"late_interaction\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"stage2_requires_query_text\":false") != -1,
        True,
    )
    assert_equal(
        json.find("\"stage2_materialized_artifact_families\":[\"late_interaction\"]")
            != -1,
        True,
    )
    assert_equal(
        json.find("\"selected_candidate_generator_kind\":\"document_proxy\"") != -1,
        True,
    )
    assert_equal(json.find("\"selected_is_undominated\":true") != -1, True)
    assert_equal(json.find("\"candidates\":[") != -1, True)
    assert_equal(json.find("\"planner_status\":\"promoted\"") != -1, True)


def test_build_planner_evidence_summary_reports_selected_candidate_and_stage2() raises:
    var profile = default_single_core_scale_profiles()[0].copy()
    var fixture = make_single_core_scale_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-planner-evidence-summary"),
        CollectionId("planner-evidence-summary"),
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
    var summary = build_planner_evidence_summary(
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
    assert_equal(summary.stage2_kind, "exact_late_interaction")
    assert_equal(summary.stage2_family, "late_interaction")
    assert_equal(summary.stage2_requires_query_text, False)
    assert_equal(len(summary.stage2_materialized_artifact_families), 1)
    assert_equal(summary.stage2_materialized_artifact_families[0], "late_interaction")
    assert_equal(
        summary.selected_candidate_generator_kind,
        "centroid_postings_imputed_flat",
    )
    assert_equal(summary.selected_candidate_generator_status, "promoted")
    assert_equal(len(summary.candidates) > 0, True)
    assert_equal(summary.candidates[0].measured.stage1_byte_size > 0, True)

    var selected_count = 0
    for candidate in summary.candidates:
        if candidate.is_selected:
            selected_count += 1
    assert_equal(selected_count, 1)


def test_build_planner_evidence_summary_supports_clause_text_on_mirrored_text_corpus() raises:
    var profile = default_single_core_scale_profiles()[0].copy()
    var fixture = make_single_core_scale_fixture(profile)
    var doc_ids = fixture.stored_index.index.doc_ids.copy()
    var texts = List[String]()
    for doc_id in doc_ids:
        texts.append("evidence for " + doc_id)

    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-planner-evidence-clause-text"),
        CollectionId("planner-evidence-clause-text"),
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
    var summary = build_planner_evidence_summary(
        single_core_backend(),
        fixture.stored_task,
        snapshot,
        availability,
        SearchPlanSelectionRequest(
            profile.final_k,
            profile.candidate_k,
            best_effort_faithfulness_policy(),
            goal=SEARCH_PLANNING_GOAL_EXACT_ONLY,
        ),
        clause_text_stage2_operator(),
        profile.query_vector_count,
        profile.vector_dim,
        8,
    )

    assert_equal(summary.planning_goal, "exact_only")
    assert_equal(summary.stage2_kind, "clause_text")
    assert_equal(summary.selected_candidate_generator_kind, "exact_full_scan")
    assert_equal(len(summary.candidates) > 0, True)
    assert_equal(summary.candidates[0].measured.stage1_byte_size > 0, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
