from std.testing import TestSuite, assert_equal

from kayak import (
    FILTER_FIELD_INTERNAL_TENANT_ID,
    SEARCH_PLANNING_GOAL_BALANCED,
    SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
    SEARCH_PLAN_ORDER_POLICY_CONSTRAINT_OVERRIDE,
    SEARCH_PLAN_ORDER_POLICY_GOAL_DEFAULT,
    SEARCH_PLAN_ORDER_POLICY_PREFERRED_OVERRIDE,
    SEARCH_PLAN_SELECTION_CONSTRAINT_NONE,
    SEARCH_PLAN_SELECTION_CONSTRAINT_ORACLE_REQUIRES_DEBUG,
    SEARCH_PLAN_SELECTION_OUTCOME_SELECTED_AVAILABLE,
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
        selection.decision.order_policy_kind,
        SEARCH_PLAN_ORDER_POLICY_PREFERRED_OVERRIDE,
    )
    assert_equal(
        selection.decision.constraint_kind,
        SEARCH_PLAN_SELECTION_CONSTRAINT_NONE,
    )
    assert_equal(
        selection.decision.outcome_kind,
        SEARCH_PLAN_SELECTION_OUTCOME_SELECTED_AVAILABLE,
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
        structured_filter_selection.decision.order_policy_kind,
        SEARCH_PLAN_ORDER_POLICY_GOAL_DEFAULT,
    )
    assert_equal(
        structured_filter_selection.decision.constraint_kind,
        SEARCH_PLAN_SELECTION_CONSTRAINT_NONE,
    )
    assert_equal(
        structured_filter_selection.decision.outcome_kind,
        SEARCH_PLAN_SELECTION_OUTCOME_SELECTED_AVAILABLE,
    )
    assert_equal(
        oracle_selection.decision.order_policy_kind,
        SEARCH_PLAN_ORDER_POLICY_CONSTRAINT_OVERRIDE,
    )
    assert_equal(
        oracle_selection.decision.constraint_kind,
        SEARCH_PLAN_SELECTION_CONSTRAINT_ORACLE_REQUIRES_DEBUG,
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
        selection.decision.order_policy_kind,
        SEARCH_PLAN_ORDER_POLICY_GOAL_DEFAULT,
    )
    assert_equal(
        selection.decision.constraint_kind,
        SEARCH_PLAN_SELECTION_CONSTRAINT_NONE,
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
    assert_equal(exact_plan.stage2_reference_operator.kind, "noop_topk")
    assert_equal(exact_plan.stage3_verifier.kind, "none")
    assert_equal(exact_plan.faithfulness_policy.kind, "exact_stage1_required")
    assert_equal(gem_plan.candidate_generator.kind, "gem_graph")
    assert_equal(
        gem_plan.candidate_generator.cluster_top_k_per_query_token,
        7,
    )
    assert_equal(gem_plan.candidate_generator.beam_width, 9)
    assert_equal(gem_plan.stage2_reference_operator.kind, "exact_late_interaction")
    assert_equal(gem_plan.stage3_verifier.kind, "none")
    assert_equal(gem_plan.faithfulness_policy.kind, "best_effort")


def test_search_plan_selection_request_rejects_reserved_internal_scope_filters() raises:
    var raised = False

    try:
        _ = SearchPlanSelectionRequest(
            5,
            20,
            best_effort_faithfulness_policy(),
            one_of_filter(FILTER_FIELD_INTERNAL_TENANT_ID, ["tenant-a"]),
            SEARCH_PLANNING_GOAL_BALANCED,
        )
    except:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
