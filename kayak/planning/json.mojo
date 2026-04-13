from std.collections import List

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .explain import CollectionSearchExplain
from .faithfulness import FaithfulnessAssessment
from .filter_application_profile import FilterApplicationProfile
from .graph_search_counters import GraphSearchCounters
from .score_histogram import ScoreHistogram
from .selection_decision import SearchPlanSelectionDecision
from .stage_artifact_materialization import StageArtifactMaterialization
from .stage_profile import SearchStageProfile


def json_escape(text: String) -> String:
    var escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\"", "\\\"")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


def append_json_hit(mut buffer: String, read hit: CollectionHit):
    buffer += "{"
    buffer += "\"segment_id\":\"" + json_escape(hit.segment_id) + "\","
    buffer += "\"doc_id\":\"" + json_escape(hit.doc_id) + "\","
    buffer += "\"score\":" + String(hit.score)
    buffer += "}"


def append_json_hit_list(mut buffer: String, read hits: List[CollectionHit]):
    buffer += "["
    for index in range(len(hits)):
        if index > 0:
            buffer += ","
        append_json_hit(buffer, hits[index])
    buffer += "]"


def append_json_string_list(mut buffer: String, read values: List[String]):
    buffer += "["
    for index in range(len(values)):
        if index > 0:
            buffer += ","
        buffer += "\""
        buffer += json_escape(values[index])
        buffer += "\""
    buffer += "]"


def append_json_stage_profile(
    mut buffer: String, read profile: SearchStageProfile
):
    buffer += "{"
    buffer += "\"stage_name\":\"" + json_escape(profile.stage_name) + "\","
    buffer += "\"input_hit_count\":" + String(profile.input_hit_count) + ","
    buffer += "\"output_hit_count\":" + String(profile.output_hit_count) + ","
    buffer += "\"segment_count\":" + String(profile.segment_count) + ","
    buffer += "\"document_count\":" + String(profile.document_count) + ","
    buffer += "\"token_count\":" + String(profile.token_count) + ","
    buffer += "\"vector_count\":" + String(profile.vector_count) + ","
    buffer += "\"byte_size\":" + String(profile.byte_size) + ","
    buffer += "\"tracks_graph_search\":"
    if profile.tracks_graph_search:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"graph_search_counters\":"
    append_json_graph_search_counters(buffer, profile.graph_search_counters)
    buffer += ","
    buffer += "\"score_histogram\":"
    append_json_score_histogram(buffer, profile.score_histogram)
    buffer += ","
    buffer += "\"materialized_artifacts\":"
    append_json_stage_artifact_materialization_list(
        buffer,
        profile.materialized_artifacts,
    )
    buffer += "}"


def append_json_graph_search_counters(
    mut buffer: String, read counters: GraphSearchCounters
):
    buffer += "{"
    buffer += "\"visited_vertex_count\":"
    buffer += String(counters.visited_vertex_count) + ","
    buffer += "\"expanded_edge_count\":"
    buffer += String(counters.expanded_edge_count) + ","
    buffer += "\"visited_cluster_count\":"
    buffer += String(counters.visited_cluster_count) + ","
    buffer += "\"entry_point_count\":"
    buffer += String(counters.entry_point_count) + ","
    buffer += "\"max_frontier_size\":"
    buffer += String(counters.max_frontier_size)
    buffer += "}"


def append_json_score_histogram(
    mut buffer: String, read histogram: ScoreHistogram
):
    buffer += "{"
    buffer += "\"bin_count\":" + String(histogram.bin_count) + ","
    buffer += "\"min_score\":" + String(histogram.min_score) + ","
    buffer += "\"max_score\":" + String(histogram.max_score) + ","
    buffer += "\"counts\":["
    for index in range(len(histogram.counts)):
        if index > 0:
            buffer += ","
        buffer += String(histogram.counts[index])
    buffer += "]"
    buffer += "}"


def append_json_stage_artifact_materialization(
    mut buffer: String, read materialization: StageArtifactMaterialization
):
    buffer += "{"
    buffer += "\"family\":\"" + json_escape(materialization.family) + "\","
    buffer += "\"segment_count\":" + String(materialization.segment_count) + ","
    buffer += "\"document_count\":" + String(materialization.document_count) + ","
    buffer += "\"token_count\":" + String(materialization.token_count) + ","
    buffer += "\"vector_count\":" + String(materialization.vector_count) + ","
    buffer += "\"byte_size\":" + String(materialization.byte_size)
    buffer += "}"


def append_json_stage_artifact_materialization_list(
    mut buffer: String, read materializations: List[StageArtifactMaterialization]
):
    buffer += "["
    for index in range(len(materializations)):
        if index > 0:
            buffer += ","
        append_json_stage_artifact_materialization(
            buffer,
            materializations[index],
        )
    buffer += "]"


def append_json_filter_application_profile(
    mut buffer: String,
    read profile: FilterApplicationProfile,
):
    buffer += "{"
    buffer += "\"public_filter_applied\":"
    if profile.public_filter_applied:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"logical_scope_applied\":"
    if profile.logical_scope_applied:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"uses_document_filter_index\":"
    if profile.uses_document_filter_index:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"input_document_count\":"
    buffer += String(profile.input_document_count) + ","
    buffer += "\"matching_document_count\":"
    buffer += String(profile.matching_document_count) + ","
    buffer += "\"artifact_byte_size\":"
    buffer += String(profile.artifact_byte_size) + ","
    buffer += "\"selectivity\":"
    buffer += String(profile.selectivity())
    buffer += "}"


def append_json_candidate_set(mut buffer: String, read candidate_set: CandidateSet):
    buffer += "{"
    buffer += "\"generator_kind\":\"" + json_escape(candidate_set.generator_kind) + "\","
    buffer += "\"generator_family\":\"" + json_escape(candidate_set.generator_family) + "\","
    buffer += "\"interaction_semantics\":\""
    buffer += json_escape(candidate_set.interaction_semantics) + "\","
    buffer += "\"alignment_granularity\":\""
    buffer += json_escape(candidate_set.alignment_granularity) + "\","
    buffer += "\"score_kind\":\"" + json_escape(candidate_set.score_kind) + "\","
    buffer += "\"segment_count\":" + String(candidate_set.segment_count) + ","
    buffer += "\"document_count\":" + String(candidate_set.document_count) + ","
    buffer += "\"token_count\":" + String(candidate_set.token_count) + ","
    buffer += "\"vector_count\":" + String(candidate_set.vector_count) + ","
    buffer += "\"byte_size\":" + String(candidate_set.byte_size) + ","
    buffer += "\"tracks_graph_search\":"
    if candidate_set.tracks_graph_search:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"graph_search_counters\":"
    append_json_graph_search_counters(
        buffer,
        candidate_set.graph_search_counters,
    )
    buffer += ","
    buffer += "\"filter_application\":"
    append_json_filter_application_profile(
        buffer,
        candidate_set.filter_application_profile,
    )
    buffer += ","
    buffer += "\"hit_count\":" + String(len(candidate_set.hits)) + ","
    buffer += "\"hits\":"
    append_json_hit_list(buffer, candidate_set.hits)
    buffer += "}"


def append_json_faithfulness_assessment(
    mut buffer: String, read assessment: FaithfulnessAssessment
):
    buffer += "{"
    buffer += "\"policy_kind\":\"" + json_escape(assessment.policy_kind) + "\","
    buffer += "\"evidence_kind\":\"" + json_escape(assessment.evidence_kind) + "\","
    buffer += "\"stage1_is_exact\":"
    if assessment.stage1_is_exact:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"has_oracle_recall_measurement\":"
    if assessment.has_oracle_recall_measurement:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"observed_candidate_recall_at_final_k\":"
    buffer += String(assessment.observed_candidate_recall_at_final_k) + ","
    buffer += "\"passes\":"
    if assessment.passes:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"message\":\"" + json_escape(assessment.message) + "\""
    buffer += "}"


def append_json_search_plan_selection_decision(
    mut buffer: String,
    read decision: SearchPlanSelectionDecision,
):
    buffer += "{"
    buffer += "\"order_policy_kind\":\""
    buffer += json_escape(decision.order_policy_kind) + "\","
    buffer += "\"constraint_kind\":\""
    buffer += json_escape(decision.constraint_kind) + "\","
    buffer += "\"outcome_kind\":\""
    buffer += json_escape(decision.outcome_kind) + "\","
    buffer += "\"explanation\":\""
    buffer += json_escape(decision.explanation) + "\""
    buffer += "}"


def collection_search_explain_json(
    read explain: CollectionSearchExplain
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(explain.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(explain.snapshot_id) + "\","
    buffer += "\"serving_scope_kind\":\""
    buffer += json_escape(explain.serving_scope.kind) + "\","
    buffer += "\"serving_scope_requires_logical_pushdown\":"
    if explain.serving_scope.requires_logical_scope_pushdown:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"plan\":{"
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(explain.plan.candidate_generator.kind)
    buffer += "\","
    buffer += "\"candidate_generator_family\":\""
    buffer += json_escape(explain.plan.candidate_generator.family)
    buffer += "\","
    buffer += "\"stage1_interaction_semantics\":\""
    buffer += json_escape(explain.plan.candidate_generator.interaction_semantics)
    buffer += "\","
    buffer += "\"stage1_alignment_granularity\":\""
    buffer += json_escape(explain.plan.candidate_generator.alignment_granularity)
    buffer += "\","
    buffer += "\"stage1_score_kind\":\""
    buffer += json_escape(explain.plan.candidate_generator.score_kind)
    buffer += "\","
    buffer += "\"stage1_required_artifact_families\":"
    append_json_string_list(
        buffer,
        explain.plan.candidate_generator.required_search_artifact_families,
    )
    buffer += ","
    buffer += "\"faithfulness_policy_kind\":\""
    buffer += json_escape(explain.plan.faithfulness_policy.kind)
    buffer += "\","
    buffer += "\"graph_cluster_top_k_per_query_token\":"
    buffer += String(
        explain.plan.candidate_generator.cluster_top_k_per_query_token
    ) + ","
    buffer += "\"graph_beam_width\":"
    buffer += String(explain.plan.candidate_generator.beam_width) + ","
    buffer += "\"candidate_k\":" + String(explain.plan.candidate_budget.candidate_k) + ","
    buffer += "\"final_k\":" + String(explain.plan.candidate_budget.final_k) + ","
    buffer += "\"reference_scoring_semantics_kind\":\""
    buffer += json_escape(explain.plan.reference_scoring_semantics.kind) + "\","
    buffer += "\"reference_scoring_semantics_family\":\""
    buffer += json_escape(explain.plan.reference_scoring_semantics.family) + "\","
    buffer += "\"reference_scoring_score_kind\":\""
    buffer += json_escape(explain.plan.reference_scoring_semantics.score_kind) + "\","
    buffer += "\"reference_scoring_required_artifact_families\":"
    append_json_string_list(
        buffer,
        explain.plan.reference_scoring_semantics.required_artifact_families,
    )
    buffer += ","
    buffer += "\"stage2_reference_kind\":\""
    buffer += json_escape(explain.plan.stage2_reference_operator.kind) + "\","
    buffer += "\"stage2_reference_family\":\""
    buffer += json_escape(explain.plan.stage2_reference_operator.family) + "\","
    buffer += "\"stage2_reference_executes_reference_scoring\":"
    if explain.plan.stage2_reference_operator.executes_reference_scoring:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"stage2_reference_required_artifact_families\":"
    append_json_string_list(
        buffer,
        explain.plan.stage2_reference_operator.required_artifact_families,
    )
    buffer += ","
    buffer += "\"stage3_verifier_kind\":\""
    buffer += json_escape(explain.plan.stage3_verifier.kind) + "\","
    buffer += "\"stage3_verifier_family\":\""
    buffer += json_escape(explain.plan.stage3_verifier.family) + "\","
    buffer += "\"stage3_verifier_requires_query_text\":"
    if explain.plan.stage3_verifier.requires_query_text:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"stage3_verifier_required_artifact_families\":"
    append_json_string_list(
        buffer,
        explain.plan.stage3_verifier.required_artifact_families,
    )
    buffer += "},"
    buffer += "\"candidate_set\":"
    append_json_candidate_set(buffer, explain.candidate_set)
    buffer += ","
    buffer += "\"candidate_stage\":"
    append_json_stage_profile(buffer, explain.candidate_stage)
    buffer += ","
    buffer += "\"stage2\":"
    append_json_stage_profile(buffer, explain.stage2)
    buffer += ","
    buffer += "\"stage3_verifier\":"
    append_json_stage_profile(buffer, explain.stage3_verifier)
    buffer += ","
    buffer += "\"exact_stage\":"
    append_json_stage_profile(buffer, explain.exact_stage)
    buffer += ","
    buffer += "\"candidate_recall_at_final_k\":"
    buffer += String(explain.candidate_recall_at_final_k)
    buffer += ","
    buffer += "\"faithfulness\":"
    append_json_faithfulness_assessment(buffer, explain.faithfulness)
    buffer += ","
    buffer += "\"final_hits\":"
    append_json_hit_list(buffer, explain.final_hits)
    buffer += "}"
    return buffer^
