from std.collections import List

from kayak.collections import SnapshotSearchArtifactAvailability
from kayak.collections.validation import require_non_empty_string
from kayak.filters import FilterExpression, match_all_filter
from kayak.index import (
    DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH,
    DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
)

from .planning_goal import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SEARCH_PLANNING_GOAL_EXACT_ONLY,
    SEARCH_PLANNING_GOAL_LATENCY_FIRST,
    SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
    require_search_planning_goal,
)
from .planner_registry import (
    default_candidate_generator_order_for_goal,
    registered_search_planner_candidate_generator_kinds,
    search_planner_registry_entry,
)
from .candidate_budget import CandidateBudget
from .faithfulness import FaithfulnessPolicy
from .search_plan import (
    SearchPlan,
    centroid_heads_search_plan,
    centroid_postings_flat_search_plan,
    centroid_postings_head_auto_search_plan,
    centroid_postings_blockmax_search_plan,
    centroid_postings_head_search_plan,
    centroid_postings_imputed_search_plan,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_search_plan,
    document_proxy_search_plan,
    exact_full_scan_search_plan,
    gem_graph_search_plan,
)
from .stage1_capabilities import (
    stage1_capabilities_for_candidate_generator_kind,
    stage1_required_search_artifact_families,
)


def append_unique_generator_kind(
    mut kinds: List[String], candidate_generator_kind: String
) raises:
    _ = stage1_capabilities_for_candidate_generator_kind(candidate_generator_kind)
    for existing in kinds:
        if existing == candidate_generator_kind:
            return

    kinds.append(candidate_generator_kind.copy())


def search_planning_goal_kinds(goal: String) raises -> List[String]:
    return default_candidate_generator_order_for_goal(goal)


def candidate_generator_kind_is_available(
    read availability: SnapshotSearchArtifactAvailability,
    candidate_generator_kind: String,
) raises -> Bool:
    if candidate_generator_kind == "exact_full_scan":
        return True

    var required_families = stage1_required_search_artifact_families(
        candidate_generator_kind
    )
    for family in required_families:
        if not availability.has_search_artifact_family_on_all_segments(family):
            return False

    return True


def available_candidate_generator_kinds(
    read availability: SnapshotSearchArtifactAvailability
) raises -> List[String]:
    var kinds = List[String]()
    for candidate_generator_kind in registered_search_planner_candidate_generator_kinds():
        if candidate_generator_kind_is_available(
            availability, candidate_generator_kind
        ):
            append_unique_generator_kind(kinds, candidate_generator_kind)

    return kinds^


struct SearchPlanSelectionRequest(Copyable):
    var goal: String
    var candidate_budget: CandidateBudget
    var faithfulness_policy: FaithfulnessPolicy
    var filter_expression: FilterExpression
    var preferred_candidate_generator_kinds: List[String]
    var debug_mode: Bool
    var gem_graph_cluster_top_k_per_query_token: Int
    var gem_graph_beam_width: Int

    def __init__(
        out self,
        final_k: Int,
        candidate_k: Int,
        faithfulness_policy: FaithfulnessPolicy,
        filter_expression: FilterExpression = match_all_filter(),
        goal: String = SEARCH_PLANNING_GOAL_BALANCED,
        debug_mode: Bool = False,
        gem_graph_cluster_top_k_per_query_token: Int = DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
        gem_graph_beam_width: Int = DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH,
    ) raises:
        self = SearchPlanSelectionRequest(
            final_k,
            candidate_k,
            faithfulness_policy,
            filter_expression,
            goal,
            [],
            debug_mode,
            gem_graph_cluster_top_k_per_query_token,
            gem_graph_beam_width,
        )

    def __init__(
        out self,
        final_k: Int,
        candidate_k: Int,
        faithfulness_policy: FaithfulnessPolicy,
        filter_expression: FilterExpression,
        goal: String,
        read preferred_candidate_generator_kinds: List[String],
        debug_mode: Bool = False,
        gem_graph_cluster_top_k_per_query_token: Int = DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
        gem_graph_beam_width: Int = DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH,
    ) raises:
        if gem_graph_cluster_top_k_per_query_token <= 0:
            raise Error(
                "search planning gem_graph_cluster_top_k_per_query_token must be positive"
            )

        if gem_graph_beam_width <= 0:
            raise Error("search planning gem_graph_beam_width must be positive")

        self.goal = require_search_planning_goal(goal)
        self.candidate_budget = CandidateBudget(final_k, candidate_k)
        self.faithfulness_policy = faithfulness_policy.copy()
        self.filter_expression = filter_expression.copy()
        self.preferred_candidate_generator_kinds = List[String]()
        for candidate_generator_kind in preferred_candidate_generator_kinds:
            append_unique_generator_kind(
                self.preferred_candidate_generator_kinds,
                candidate_generator_kind,
            )
        self.debug_mode = debug_mode
        self.gem_graph_cluster_top_k_per_query_token = (
            gem_graph_cluster_top_k_per_query_token
        )
        self.gem_graph_beam_width = gem_graph_beam_width


struct SearchPlanSelection(Copyable):
    var goal: String
    var selected_candidate_generator_status: String
    var available_candidate_generator_kinds: List[String]
    var effective_candidate_generator_order: List[String]
    var plan: SearchPlan
    var reason: String

    def __init__(
        out self,
        goal: String,
        var selected_candidate_generator_status: String,
        read available_candidate_generator_kinds: List[String],
        read effective_candidate_generator_order: List[String],
        plan: SearchPlan,
        var reason: String,
    ) raises:
        self.goal = require_search_planning_goal(goal)
        self.selected_candidate_generator_status = require_non_empty_string(
            selected_candidate_generator_status,
            "selected_candidate_generator_status",
        )
        self.available_candidate_generator_kinds = List[String]()
        self.effective_candidate_generator_order = List[String]()
        for candidate_generator_kind in available_candidate_generator_kinds:
            append_unique_generator_kind(
                self.available_candidate_generator_kinds,
                candidate_generator_kind,
            )
        for candidate_generator_kind in effective_candidate_generator_order:
            append_unique_generator_kind(
                self.effective_candidate_generator_order,
                candidate_generator_kind,
            )
        self.plan = plan.copy()
        self.reason = reason^


struct CandidateGeneratorOrderDecision(Copyable):
    var order: List[String]
    var reason: String

    def __init__(out self, read order: List[String], var reason: String) raises:
        self.order = List[String]()
        for candidate_generator_kind in order:
            append_unique_generator_kind(self.order, candidate_generator_kind)
        self.reason = reason^


def selected_plan_for_kind(
    candidate_generator_kind: String,
    read request: SearchPlanSelectionRequest,
) raises -> SearchPlan:
    if candidate_generator_kind == "exact_full_scan":
        return exact_full_scan_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
        )

    if candidate_generator_kind == "document_proxy":
        return document_proxy_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
        )

    if candidate_generator_kind == "centroid_heads":
        return centroid_heads_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
        )

    if candidate_generator_kind == "centroid_postings":
        return centroid_postings_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
        )

    if candidate_generator_kind == "centroid_postings_flat":
        return centroid_postings_flat_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
        )

    if candidate_generator_kind == "centroid_postings_head":
        return centroid_postings_head_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
        )

    if candidate_generator_kind == "centroid_postings_head_auto":
        return centroid_postings_head_auto_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
        )

    if candidate_generator_kind == "centroid_postings_blockmax":
        return centroid_postings_blockmax_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
        )

    if candidate_generator_kind == "centroid_postings_imputed":
        return centroid_postings_imputed_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
        )

    if candidate_generator_kind == "centroid_postings_imputed_flat":
        return centroid_postings_imputed_flat_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
        )

    if candidate_generator_kind == "gem_graph":
        return gem_graph_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.candidate_k,
            request.faithfulness_policy,
            request.gem_graph_cluster_top_k_per_query_token,
            request.gem_graph_beam_width,
        )

    raise Error("unknown search planning candidate generator kind: " + candidate_generator_kind)


def search_plan_for_candidate_generator_kind(
    candidate_generator_kind: String,
    read request: SearchPlanSelectionRequest,
) raises -> SearchPlan:
    return selected_plan_for_kind(candidate_generator_kind, request)


def effective_candidate_generator_order(
    read request: SearchPlanSelectionRequest
) raises -> CandidateGeneratorOrderDecision:
    if not request.filter_expression.is_match_all():
        return CandidateGeneratorOrderDecision(
            ["exact_full_scan"],
            "non-match_all filters currently require exact stage-1 candidate generation",
        )

    if request.faithfulness_policy.kind == "exact_stage1_required":
        return CandidateGeneratorOrderDecision(
            ["exact_full_scan"],
            "exact_stage1_required faithfulness policy requires exact stage-1 candidate generation",
        )

    if (
        request.faithfulness_policy.kind == "oracle_full_recall_required"
        and not request.debug_mode
    ):
        return CandidateGeneratorOrderDecision(
            ["exact_full_scan"],
            "oracle_full_recall_required without debug_mode falls back to exact stage-1 candidate generation",
        )

    if len(request.preferred_candidate_generator_kinds) != 0:
        var preferred = List[String]()
        for candidate_generator_kind in request.preferred_candidate_generator_kinds:
            append_unique_generator_kind(preferred, candidate_generator_kind)

        append_unique_generator_kind(preferred, "exact_full_scan")
        return CandidateGeneratorOrderDecision(
            preferred,
            "planner used explicit preferred_candidate_generator_kinds order",
        )

    return CandidateGeneratorOrderDecision(
        search_planning_goal_kinds(request.goal),
        "planner used the default candidate-generator order for goal "
        + request.goal,
    )


def select_search_plan_for_availability(
    read availability: SnapshotSearchArtifactAvailability,
    read request: SearchPlanSelectionRequest,
) raises -> SearchPlanSelection:
    var available_kinds = available_candidate_generator_kinds(availability)
    var decision = effective_candidate_generator_order(request)

    for candidate_generator_kind in decision.order:
        if candidate_generator_kind_is_available(
            availability, candidate_generator_kind
        ):
            return SearchPlanSelection(
                request.goal,
                search_planner_registry_entry(candidate_generator_kind).planner_status,
                available_kinds,
                decision.order,
                selected_plan_for_kind(candidate_generator_kind, request),
                decision.reason,
            )

    return SearchPlanSelection(
        request.goal,
        search_planner_registry_entry("exact_full_scan").planner_status,
        available_kinds,
        decision.order,
        exact_full_scan_search_plan(
            request.candidate_budget.final_k,
            request.candidate_budget.final_k,
        ),
        decision.reason
        + "; no requested non-exact stage-1 candidate generator was available, so the planner fell back to exact_full_scan",
    )
