from kayak.planning import (
    CandidateGenerator,
    SearchPlan,
)

from .json_common import append_json_string_list, json_escape


def append_candidate_generator_semantics_json_fields(
    mut buffer: String,
    read generator: CandidateGenerator,
):
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(generator.kind) + "\","
    buffer += "\"candidate_generator_family\":\""
    buffer += json_escape(generator.family) + "\","
    buffer += "\"stage1_interaction_semantics\":\""
    buffer += json_escape(generator.interaction_semantics) + "\","
    buffer += "\"stage1_alignment_granularity\":\""
    buffer += json_escape(generator.alignment_granularity) + "\","
    buffer += "\"stage1_score_kind\":\""
    buffer += json_escape(generator.score_kind) + "\","
    buffer += "\"stage1_required_artifact_families\":"
    append_json_string_list(
        buffer,
        generator.required_search_artifact_families,
    )
    buffer += ","
    buffer += "\"graph_cluster_top_k_per_query_token\":"
    buffer += String(generator.cluster_top_k_per_query_token) + ","
    buffer += "\"graph_beam_width\":"
    buffer += String(generator.beam_width) + ","
    buffer += "\"graph_frontier_policy_kind\":\""
    buffer += json_escape(generator.graph_frontier_policy_kind) + "\""


def append_search_plan_semantics_json_fields(
    mut buffer: String,
    read plan: SearchPlan,
):
    append_candidate_generator_semantics_json_fields(
        buffer,
        plan.candidate_generator,
    )
    buffer += ","
    buffer += "\"faithfulness_policy_kind\":\""
    buffer += json_escape(plan.faithfulness_policy.kind) + "\","
    buffer += "\"final_k\":" + String(plan.candidate_budget.final_k) + ","
    buffer += "\"candidate_k\":" + String(plan.candidate_budget.candidate_k) + ","
    buffer += "\"reference_scoring_semantics_kind\":\""
    buffer += json_escape(plan.reference_scoring_semantics.kind) + "\","
    buffer += "\"reference_scoring_semantics_family\":\""
    buffer += json_escape(plan.reference_scoring_semantics.family) + "\","
    buffer += "\"reference_scoring_score_kind\":\""
    buffer += json_escape(plan.reference_scoring_semantics.score_kind) + "\","
    buffer += "\"reference_scoring_required_artifact_families\":"
    append_json_string_list(
        buffer,
        plan.reference_scoring_semantics.required_artifact_families,
    )
    buffer += ","
    buffer += "\"stage2_reference_kind\":\""
    buffer += json_escape(plan.stage2_reference_operator.kind) + "\","
    buffer += "\"stage2_reference_family\":\""
    buffer += json_escape(plan.stage2_reference_operator.family) + "\","
    buffer += "\"stage2_reference_executes_reference_scoring\":"
    if plan.stage2_reference_operator.executes_reference_scoring:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"stage2_reference_required_artifact_families\":"
    append_json_string_list(
        buffer,
        plan.stage2_reference_operator.required_artifact_families,
    )
    buffer += ","
    buffer += "\"stage3_verifier_kind\":\""
    buffer += json_escape(plan.stage3_verifier.kind) + "\","
    buffer += "\"stage3_verifier_family\":\""
    buffer += json_escape(plan.stage3_verifier.family) + "\","
    buffer += "\"stage3_verifier_requires_query_text\":"
    if plan.stage3_verifier.requires_query_text:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"stage3_verifier_required_artifact_families\":"
    append_json_string_list(
        buffer,
        plan.stage3_verifier.required_artifact_families,
    )
