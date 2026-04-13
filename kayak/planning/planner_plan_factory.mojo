from .candidate_budget import CandidateBudget
from .candidate_generator import CandidateGenerator
from .faithfulness import (
    FaithfulnessPolicy,
    exact_stage1_required_faithfulness_policy,
)
from .reference_scoring_semantics import (
    exact_late_interaction_reference_scoring_semantics,
)
from .search_plan import SearchPlan
from .stage2_reference_operator import (
    exact_late_interaction_stage2_reference_operator,
    noop_topk_stage2_reference_operator,
)
from .stage3_verifier_operator import (
    none_stage3_verifier_operator,
)


def planner_candidate_generator_for_kind(
    candidate_generator_kind: String,
    gem_graph_cluster_top_k_per_query_token: Int,
    gem_graph_beam_width: Int,
) raises -> CandidateGenerator:
    if candidate_generator_kind == "gem_graph":
        return CandidateGenerator(
            candidate_generator_kind.copy(),
            gem_graph_cluster_top_k_per_query_token,
            gem_graph_beam_width,
        )

    return CandidateGenerator(candidate_generator_kind.copy())


def planner_default_search_plan_for_candidate_generator(
    read candidate_generator: CandidateGenerator,
    candidate_budget: CandidateBudget,
    read requested_faithfulness_policy: FaithfulnessPolicy,
) raises -> SearchPlan:
    if candidate_generator.is_exact:
        return SearchPlan(
            candidate_generator,
            candidate_budget,
            exact_stage1_required_faithfulness_policy(),
            exact_late_interaction_reference_scoring_semantics(),
            noop_topk_stage2_reference_operator(),
            none_stage3_verifier_operator(),
        )

    return SearchPlan(
        candidate_generator,
        candidate_budget,
        requested_faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def planner_default_search_plan_for_kind(
    candidate_generator_kind: String,
    candidate_budget: CandidateBudget,
    read requested_faithfulness_policy: FaithfulnessPolicy,
    gem_graph_cluster_top_k_per_query_token: Int,
    gem_graph_beam_width: Int,
) raises -> SearchPlan:
    return planner_default_search_plan_for_candidate_generator(
        planner_candidate_generator_for_kind(
            candidate_generator_kind,
            gem_graph_cluster_top_k_per_query_token,
            gem_graph_beam_width,
        ),
        candidate_budget,
        requested_faithfulness_policy,
    )
