from std.testing import TestSuite, assert_equal

from kayak import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
    SnapshotSearchArtifactAvailability,
    SearchPlanSelectionRequest,
    best_effort_faithfulness_policy,
    oracle_full_recall_required_faithfulness_policy,
    select_search_plan_for_availability,
)
from kayak.filters import match_all_filter, one_of_filter


def test_balanced_planner_prefers_proxy_frontier_when_available() raises:
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

    assert_equal(selection.plan.candidate_generator.kind, "document_proxy")


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


def test_planner_falls_back_to_exact_for_filters_and_oracle_guardrails() raises:
    var filtered_selection = select_search_plan_for_availability(
        SnapshotSearchArtifactAvailability(
            1,
            ["document_proxy"],
            ["document_proxy"],
        ),
        SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            one_of_filter("doc_id", ["doc-a"]),
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

    assert_equal(filtered_selection.plan.candidate_generator.kind, "exact_full_scan")
    assert_equal(oracle_selection.plan.candidate_generator.kind, "exact_full_scan")
    assert_equal(filtered_selection.plan.candidate_budget.candidate_k, 20)
    assert_equal(oracle_selection.plan.candidate_budget.candidate_k, 20)
    assert_equal(
        filtered_selection.reason.find("require exact stage-1") != -1,
        True,
    )
    assert_equal(
        oracle_selection.reason.find("oracle_full_recall_required") != -1,
        True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
