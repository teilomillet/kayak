from std.testing import TestSuite, assert_equal

from kayak import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
    SnapshotSearchArtifactAvailability,
    SearchPlanSelectionRequest,
    best_effort_faithfulness_policy,
    oracle_full_recall_required_faithfulness_policy,
    search_plan_for_candidate_generator_kind,
    select_search_plan_for_availability,
)
from kayak.filters import match_all_filter, one_of_filter


def test_balanced_planner_prefers_native_frontier_when_available() raises:
    var selection = select_search_plan_for_availability(
        SnapshotSearchArtifactAvailability(
            1,
            ["document_proxy", "centroid_postings", "gem_graph"],
            ["document_proxy", "centroid_postings", "gem_graph"],
        ),
        SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            match_all_filter(),
            SEARCH_PLANNING_GOAL_BALANCED,
        ),
    )

    assert_equal(
        selection.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )


def test_native_multivector_planner_prefers_current_warp_shaped_path() raises:
    var selection = select_search_plan_for_availability(
        SnapshotSearchArtifactAvailability(
            1,
            ["document_proxy", "centroid_postings", "gem_graph"],
            ["document_proxy", "centroid_postings", "gem_graph"],
        ),
        SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            match_all_filter(),
            SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
        ),
    )

    assert_equal(
        selection.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )


def test_planner_uses_explicit_preferred_order_when_supported() raises:
    var selection = select_search_plan_for_availability(
        SnapshotSearchArtifactAvailability(
            1,
            ["centroid_postings"],
            ["centroid_postings"],
        ),
        SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            match_all_filter(),
            SEARCH_PLANNING_GOAL_BALANCED,
            ["centroid_postings_head_auto", "centroid_postings_imputed_flat"],
        ),
    )

    assert_equal(selection.plan.candidate_generator.kind, "centroid_postings_head_auto")
    assert_equal(
        selection.reason.find("preferred_candidate_generator_kinds") != -1,
        True,
    )


def test_planner_uses_exact_for_unavailable_structured_filters_and_oracle_guardrails() raises:
    var structured_filter_selection = select_search_plan_for_availability(
        SnapshotSearchArtifactAvailability(
            1,
            ["document_proxy"],
            ["document_proxy"],
        ),
        SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            one_of_filter("source", ["wire"]),
            SEARCH_PLANNING_GOAL_BALANCED,
        ),
    )
    var oracle_selection = select_search_plan_for_availability(
        SnapshotSearchArtifactAvailability(
            1,
            ["document_proxy"],
            ["document_proxy"],
        ),
        SearchPlanSelectionRequest(
            5,
            20,
            oracle_full_recall_required_faithfulness_policy(),
            match_all_filter(),
            SEARCH_PLANNING_GOAL_BALANCED,
            False,
        ),
    )

    assert_equal(
        structured_filter_selection.plan.candidate_generator.kind,
        "exact_full_scan",
    )
    assert_equal(oracle_selection.plan.candidate_generator.kind, "exact_full_scan")
    assert_equal(structured_filter_selection.plan.candidate_budget.candidate_k, 20)
    assert_equal(oracle_selection.plan.candidate_budget.candidate_k, 20)
    assert_equal(
        structured_filter_selection.reason.find(
            "default candidate-generator order"
        ) != -1,
        True,
    )
    assert_equal(
        oracle_selection.reason.find("oracle_full_recall_required") != -1,
        True,
    )


def test_planner_keeps_native_stage1_for_exact_doc_id_filters() raises:
    var selection = select_search_plan_for_availability(
        SnapshotSearchArtifactAvailability(
            1,
            ["document_proxy", "centroid_postings"],
            ["document_proxy", "centroid_postings"],
        ),
        SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            one_of_filter("doc_id", ["doc-a"]),
            SEARCH_PLANNING_GOAL_BALANCED,
        ),
    )

    assert_equal(
        selection.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )
    assert_equal(
        selection.reason.find("default candidate-generator order") != -1,
        True,
    )


def test_planner_exact_fallback_preserves_requested_candidate_window() raises:
    var selection = select_search_plan_for_availability(
        SnapshotSearchArtifactAvailability(
            1,
            [],
            [],
        ),
        SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            match_all_filter(),
            SEARCH_PLANNING_GOAL_BALANCED,
        ),
    )

    assert_equal(selection.plan.candidate_generator.kind, "exact_full_scan")
    assert_equal(selection.plan.candidate_budget.final_k, 5)
    assert_equal(selection.plan.candidate_budget.candidate_k, 20)
    assert_equal(selection.selected_candidate_generator_status, "exact_fallback")


def test_search_plan_for_candidate_generator_kind_uses_default_contracts() raises:
    var exact_plan = search_plan_for_candidate_generator_kind(
        "exact_full_scan",
        SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            match_all_filter(),
            SEARCH_PLANNING_GOAL_BALANCED,
        ),
    )
    var gem_plan = search_plan_for_candidate_generator_kind(
        "gem_graph",
        SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            match_all_filter(),
            SEARCH_PLANNING_GOAL_BALANCED,
            False,
            7,
            9,
        ),
    )

    assert_equal(exact_plan.candidate_generator.is_exact, True)
    assert_equal(exact_plan.stage2_operator.kind, "noop_topk")
    assert_equal(exact_plan.faithfulness_policy.kind, "exact_stage1_required")
    assert_equal(gem_plan.candidate_generator.kind, "gem_graph")
    assert_equal(
        gem_plan.candidate_generator.cluster_top_k_per_query_token,
        7,
    )
    assert_equal(gem_plan.candidate_generator.beam_width, 9)
    assert_equal(gem_plan.stage2_operator.kind, "exact_late_interaction")
    assert_equal(gem_plan.faithfulness_policy.kind, "best_effort")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
