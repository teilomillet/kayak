from std.collections import List

from kayak.collections.reclaim_json import (
    append_snapshot_ids_json,
    collection_reclaim_execution_result_json,
    collection_reclaim_plan_json,
)
from kayak.collections import SearchArtifactBuildPolicy
from kayak.collections.document_metadata import DocumentMetadataUpdate
from kayak.filters import FilterClause, FilterExpression, FilterTerm
from kayak.numeric import VectorScalar
from kayak.planning import (
    CollectionHit,
    SearchPlan,
    SearchPlanSelection,
    collection_search_explain_json,
)

from .collection_requests import (
    CreateCollectionRequest,
    UpdateCollectionRetentionPolicyRequest,
)
from .document_requests import DeleteDocumentsRequest, UpsertDocument, UpsertDocumentsRequest
from .lifecycle_contracts import (
    BuildReclaimPlanRequest,
    BuildReclaimPlanResponse,
    CollectionLifecycleRequest,
    CollectionLifecycleResponse,
    ExecuteReclaimRequest,
    ExecuteReclaimResponse,
    UpdateCollectionRetentionPolicyResponse,
)
from .search_contracts import (
    DebugSearchResponse,
    ExplainRequest,
    ExplainResponse,
    PlannedDebugSearchResponse,
    PlannedExplainResponse,
    PlannedSearchRequest,
    PlannedSearchResponse,
    SearchRequest,
    SearchResponse,
)
from .service_status import ServiceHealthStatus, ServiceMetricsSnapshot
from .snapshot_requests import (
    CreateSnapshotRequest,
    ExportSnapshotRequest,
    ImportSnapshotRequest,
)


def json_escape(text: String) -> String:
    var escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\"", "\\\"")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


def append_json_string_list(mut buffer: String, read values: List[String]):
    buffer += "["

    for index in range(len(values)):
        if index > 0:
            buffer += ","

        buffer += "\""
        buffer += json_escape(values[index])
        buffer += "\""

    buffer += "]"


def append_json_vector(mut buffer: String, read values: List[VectorScalar]):
    buffer += "["

    for index in range(len(values)):
        if index > 0:
            buffer += ","

        buffer += String(values[index])

    buffer += "]"


def append_json_vector_list(
    mut buffer: String, read vectors: List[List[VectorScalar]]
):
    buffer += "["

    for index in range(len(vectors)):
        if index > 0:
            buffer += ","

        append_json_vector(buffer, vectors[index])

    buffer += "]"


def append_json_filter_term(mut buffer: String, read term: FilterTerm):
    buffer += "{"
    buffer += "\"field\":\"" + json_escape(term.field.name) + "\","
    buffer += "\"operator\":\"" + json_escape(term.operator) + "\","
    buffer += "\"values\":"
    append_json_string_list(buffer, term.values)
    buffer += "}"


def append_json_filter_clause(mut buffer: String, read clause: FilterClause):
    buffer += "{"
    buffer += "\"terms\":["

    for index in range(len(clause.terms)):
        if index > 0:
            buffer += ","

        append_json_filter_term(buffer, clause.terms[index])

    buffer += "]}"


def append_json_filter_expression(
    mut buffer: String, read expression: FilterExpression
):
    buffer += "{"
    buffer += "\"clauses\":["

    for index in range(len(expression.clauses)):
        if index > 0:
            buffer += ","

        append_json_filter_clause(buffer, expression.clauses[index])

    buffer += "]}"


def append_json_metadata_updates(
    mut buffer: String, read metadata_updates: List[DocumentMetadataUpdate]
):
    buffer += "["

    for index in range(len(metadata_updates)):
        if index > 0:
            buffer += ","

        buffer += "{"
        buffer += "\"key\":\"" + json_escape(metadata_updates[index].key) + "\","
        buffer += "\"value\":\"" + json_escape(metadata_updates[index].value) + "\""
        buffer += "}"

    buffer += "]"


def append_json_search_artifact_build_policy(
    mut buffer: String, read policy: SearchArtifactBuildPolicy
):
    buffer += "["

    for index in range(len(policy.stage1_artifacts)):
        if index > 0:
            buffer += ","

        buffer += "{"
        buffer += "\"family\":\""
        buffer += json_escape(policy.stage1_artifacts[index].family) + "\","
        buffer += "\"root\":\""
        buffer += json_escape(policy.stage1_artifacts[index].root) + "\","
        buffer += "\"config\":["
        for config_index in range(len(policy.stage1_artifacts[index].config)):
            if config_index > 0:
                buffer += ","

            buffer += "{"
            buffer += "\"key\":\""
            buffer += json_escape(
                policy.stage1_artifacts[index].config[config_index].key
            ) + "\","
            buffer += "\"value\":\""
            buffer += json_escape(
                policy.stage1_artifacts[index].config[config_index].value
            ) + "\""
            buffer += "}"
        buffer += "]"
        buffer += "}"

    buffer += "]"


def append_json_search_plan_only(mut buffer: String, read plan: SearchPlan):
    buffer += "{"
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(plan.candidate_generator.kind)
    buffer += "\","
    buffer += "\"candidate_generator_family\":\""
    buffer += json_escape(plan.candidate_generator.family)
    buffer += "\","
    buffer += "\"stage1_interaction_semantics\":\""
    buffer += json_escape(plan.candidate_generator.interaction_semantics)
    buffer += "\","
    buffer += "\"stage1_alignment_granularity\":\""
    buffer += json_escape(plan.candidate_generator.alignment_granularity)
    buffer += "\","
    buffer += "\"stage1_score_kind\":\""
    buffer += json_escape(plan.candidate_generator.score_kind)
    buffer += "\","
    buffer += "\"faithfulness_policy_kind\":\""
    buffer += json_escape(plan.faithfulness_policy.kind)
    buffer += "\","
    buffer += "\"graph_cluster_top_k_per_query_token\":"
    buffer += String(plan.candidate_generator.cluster_top_k_per_query_token) + ","
    buffer += "\"graph_beam_width\":"
    buffer += String(plan.candidate_generator.beam_width) + ","
    buffer += "\"final_k\":"
    buffer += String(plan.candidate_budget.final_k) + ","
    buffer += "\"candidate_k\":"
    buffer += String(plan.candidate_budget.candidate_k) + ","
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
    buffer += ","
    buffer += "\"stage2_kind\":\""
    buffer += json_escape(plan.stage2_operator.kind) + "\","
    buffer += "\"stage2_family\":\""
    buffer += json_escape(plan.stage2_operator.family) + "\","
    buffer += "\"stage2_requires_query_text\":"
    if plan.stage2_operator.requires_query_text:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"stage2_required_artifact_families\":"
    append_json_string_list(
        buffer,
        plan.stage2_operator.required_artifact_families,
    )
    buffer += ","
    buffer += "\"exact_stage_kind\":\""
    buffer += json_escape(plan.exact_stage_kind) + "\","
    buffer += "\"reranker_kind\":\""
    buffer += json_escape(plan.reranker_kind) + "\""
    buffer += "}"


def append_json_search_plan(mut buffer: String, read request: SearchRequest):
    append_json_search_plan_only(buffer, request.plan)


def append_json_search_plan_selection(
    mut buffer: String, read selection: SearchPlanSelection
):
    buffer += "{"
    buffer += "\"goal\":\"" + json_escape(selection.goal) + "\","
    buffer += "\"selected_candidate_generator_status\":\""
    buffer += json_escape(selection.selected_candidate_generator_status) + "\","
    buffer += "\"available_candidate_generator_kinds\":"
    append_json_string_list(
        buffer, selection.available_candidate_generator_kinds
    )
    buffer += ",\"effective_candidate_generator_order\":"
    append_json_string_list(
        buffer, selection.effective_candidate_generator_order
    )
    buffer += ",\"reason\":\"" + json_escape(selection.reason) + "\","
    buffer += "\"plan\":"
    append_json_search_plan_only(buffer, selection.plan)
    buffer += "}"


def append_json_search_planning_request(
    mut buffer: String, read request: PlannedSearchRequest
):
    buffer += "{"
    buffer += "\"goal\":\"" + json_escape(request.planning.goal) + "\","
    buffer += "\"preferred_candidate_generator_kinds\":"
    append_json_string_list(
        buffer, request.planning.preferred_candidate_generator_kinds
    )
    buffer += ",\"faithfulness_policy_kind\":\""
    buffer += json_escape(request.planning.faithfulness_policy.kind) + "\","
    buffer += "\"final_k\":"
    buffer += String(request.planning.candidate_budget.final_k) + ","
    buffer += "\"candidate_k\":"
    buffer += String(request.planning.candidate_budget.candidate_k) + ","
    buffer += "\"debug_mode\":"
    if request.planning.debug_mode:
        buffer += "true"
    else:
        buffer += "false"
    buffer += ",\"graph_cluster_top_k_per_query_token\":"
    buffer += String(
        request.planning.gem_graph_cluster_top_k_per_query_token
    ) + ","
    buffer += "\"graph_beam_width\":"
    buffer += String(request.planning.gem_graph_beam_width)
    buffer += "}"


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


def append_json_upsert_document(mut buffer: String, read document: UpsertDocument):
    buffer += "{"
    buffer += "\"doc_id\":\"" + json_escape(document.document.doc_id) + "\","
    buffer += "\"token_vectors\":"
    append_json_vector_list(buffer, document.document.token_vectors)
    buffer += ",\"has_text\":"
    if document.has_text:
        buffer += "true"
    else:
        buffer += "false"
    buffer += ",\"has_metadata_updates\":"
    if document.has_metadata_updates:
        buffer += "true"
    else:
        buffer += "false"

    if document.has_text:
        buffer += ",\"text\":\"" + json_escape(document.text) + "\""
    if document.has_metadata_updates:
        buffer += ",\"metadata_updates\":"
        append_json_metadata_updates(buffer, document.metadata_updates)

    buffer += "}"


def create_collection_request_json(read request: CreateCollectionRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"model_name\":\"" + json_escape(request.model_name) + "\","
    buffer += "\"vector_scalar_name\":\""
    buffer += json_escape(request.vector_scalar_name) + "\","
    buffer += "\"vector_dim\":" + String(request.vector_dim) + ","
    buffer += "\"default_keep_latest_inactive_count\":"
    buffer += String(request.default_keep_latest_inactive_count) + ","
    buffer += "\"search_artifact_build_policy\":"
    append_json_search_artifact_build_policy(
        buffer, request.search_artifact_build_policy
    )
    buffer += "}"
    return buffer^


def append_json_policy_override(
    mut buffer: String, read request: CollectionLifecycleRequest
):
    buffer += "\"has_policy_override\":"
    if request.has_policy_override:
        buffer += "true"
        buffer += ",\"policy_override_keep_latest_inactive_count\":"
        buffer += String(request.policy_override.keep_latest_inactive_count)
        buffer += ",\"policy_override_pinned_snapshot_ids\":"
        append_snapshot_ids_json(
            buffer, request.policy_override.pinned_snapshot_ids
        )
        return

    buffer += "false"


def append_json_build_reclaim_policy_override(
    mut buffer: String, read request: BuildReclaimPlanRequest
):
    buffer += "\"has_policy_override\":"
    if request.has_policy_override:
        buffer += "true"
        buffer += ",\"policy_override_keep_latest_inactive_count\":"
        buffer += String(request.policy_override.keep_latest_inactive_count)
        buffer += ",\"policy_override_pinned_snapshot_ids\":"
        append_snapshot_ids_json(
            buffer, request.policy_override.pinned_snapshot_ids
        )
        return

    buffer += "false"


def collection_lifecycle_request_json(
    read request: CollectionLifecycleRequest
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    append_json_policy_override(buffer, request)
    buffer += "}"
    return buffer^


def collection_lifecycle_response_json(
    read response: CollectionLifecycleResponse
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(response.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(response.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(response.namespace_id.value) + "\","
    buffer += "\"model_name\":\"" + json_escape(response.model_name) + "\","
    buffer += "\"vector_scalar_name\":\""
    buffer += json_escape(response.vector_scalar_name) + "\","
    buffer += "\"vector_dim\":" + String(response.vector_dim) + ","
    buffer += "\"latest_generation\":" + String(response.latest_generation) + ","
    buffer += "\"active_snapshot_id\":\""
    buffer += json_escape(response.active_snapshot_id) + "\","
    buffer += "\"default_keep_latest_inactive_count\":"
    buffer += String(response.default_keep_latest_inactive_count) + ","
    buffer += "\"search_artifact_build_policy\":"
    append_json_search_artifact_build_policy(
        buffer, response.search_artifact_build_policy
    )
    buffer += ","
    buffer += "\"effective_keep_latest_inactive_count\":"
    buffer += String(response.effective_keep_latest_inactive_count) + ","
    buffer += "\"effective_pinned_snapshot_ids\":"
    append_snapshot_ids_json(buffer, response.effective_pinned_snapshot_ids)
    buffer += ",\"draft_document_count\":"
    buffer += String(response.draft_document_count) + ","
    buffer += "\"pending_draft_mutation_count\":"
    buffer += String(response.pending_draft_mutation_count) + ","
    buffer += "\"reclaim_plan\":"
    buffer += collection_reclaim_plan_json(response.reclaim_plan)
    buffer += "}"
    return buffer^


def build_reclaim_plan_request_json(read request: BuildReclaimPlanRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    append_json_build_reclaim_policy_override(buffer, request)
    buffer += "}"
    return buffer^


def build_reclaim_plan_response_json(
    read response: BuildReclaimPlanResponse
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"effective_keep_latest_inactive_count\":"
    buffer += String(response.effective_keep_latest_inactive_count) + ","
    buffer += "\"effective_pinned_snapshot_ids\":"
    append_snapshot_ids_json(buffer, response.effective_pinned_snapshot_ids)
    buffer += ",\"plan\":"
    buffer += collection_reclaim_plan_json(response.plan)
    buffer += "}"
    return buffer^


def execute_reclaim_request_json(read request: ExecuteReclaimRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"dry_run\":"
    if request.dry_run:
        buffer += "true"
    else:
        buffer += "false"
    buffer += ",\"plan\":"
    buffer += collection_reclaim_plan_json(request.plan)
    buffer += "}"
    return buffer^


def execute_reclaim_response_json(read response: ExecuteReclaimResponse) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"result\":"
    buffer += collection_reclaim_execution_result_json(response.result)
    buffer += "}"
    return buffer^


def update_collection_retention_policy_request_json(
    read request: UpdateCollectionRetentionPolicyRequest
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"default_keep_latest_inactive_count\":"
    buffer += String(request.default_keep_latest_inactive_count)
    buffer += "}"
    return buffer^


def update_collection_retention_policy_response_json(
    read response: UpdateCollectionRetentionPolicyResponse
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(response.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(response.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(response.namespace_id.value) + "\","
    buffer += "\"latest_generation\":" + String(response.latest_generation) + ","
    buffer += "\"active_snapshot_id\":\""
    buffer += json_escape(response.active_snapshot_id) + "\","
    buffer += "\"default_keep_latest_inactive_count\":"
    buffer += String(response.default_keep_latest_inactive_count)
    buffer += "}"
    return buffer^


def upsert_documents_request_json(read request: UpsertDocumentsRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"documents\":["

    for index in range(len(request.documents)):
        if index > 0:
            buffer += ","

        append_json_upsert_document(buffer, request.documents[index])

    buffer += "]}"
    return buffer^


def delete_documents_request_json(read request: DeleteDocumentsRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"doc_ids\":"
    append_json_string_list(buffer, request.doc_ids)
    buffer += "}"
    return buffer^


def create_snapshot_request_json(read request: CreateSnapshotRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(request.snapshot_id.value) + "\","
    buffer += "\"reason\":\"" + json_escape(request.reason) + "\""
    buffer += "}"
    return buffer^


def export_snapshot_request_json(read request: ExportSnapshotRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(request.snapshot_id.value) + "\""
    buffer += "}"
    return buffer^


def import_snapshot_request_json(read request: ImportSnapshotRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(request.snapshot_id.value) + "\","
    buffer += "\"source_uri\":\"" + json_escape(request.source_uri) + "\""
    buffer += "}"
    return buffer^


def search_request_json(read request: SearchRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(request.snapshot_id.value) + "\","
    buffer += "\"query\":"
    append_json_vector_list(buffer, request.query.token_vectors)
    buffer += ",\"query_text\":\""
    buffer += json_escape(request.query_text) + "\""
    buffer += ",\"filter_expression\":"
    append_json_filter_expression(buffer, request.filter_expression)
    buffer += ",\"plan\":"
    append_json_search_plan(buffer, request)
    buffer += ",\"debug_mode\":"
    if request.debug_mode:
        buffer += "true"
    else:
        buffer += "false"

    buffer += "}"
    return buffer^


def planned_search_request_json(read request: PlannedSearchRequest) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(request.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(request.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(request.namespace_id.value) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(request.snapshot_id.value) + "\","
    buffer += "\"query\":"
    append_json_vector_list(buffer, request.query.token_vectors)
    buffer += ",\"query_text\":\""
    buffer += json_escape(request.query_text) + "\""
    buffer += ",\"stage2_operator_kind\":\""
    buffer += json_escape(request.stage2_operator_kind) + "\""
    buffer += ",\"filter_expression\":"
    append_json_filter_expression(buffer, request.filter_expression)
    buffer += ",\"planning\":"
    append_json_search_planning_request(buffer, request)
    buffer += "}"
    return buffer^


def search_response_json(read response: SearchResponse) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(response.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(response.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(response.namespace_id.value) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(response.snapshot_id.value) + "\","
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(response.plan.candidate_generator.kind) + "\","
    buffer += "\"faithfulness_policy_kind\":\""
    buffer += json_escape(response.plan.faithfulness_policy.kind) + "\","
    buffer += "\"final_k\":" + String(response.plan.candidate_budget.final_k) + ","
    buffer += "\"candidate_k\":" + String(response.plan.candidate_budget.candidate_k) + ","
    buffer += "\"hits\":"
    append_json_hit_list(buffer, response.hits)
    buffer += "}"
    return buffer^


def debug_search_response_json(read response: DebugSearchResponse) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"search\":"
    buffer += search_response_json(response.search)
    buffer += ",\"debug\":"
    buffer += collection_search_explain_json(response.explain)
    buffer += "}"
    return buffer^


def planned_search_response_json(read response: PlannedSearchResponse) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"selection\":"
    append_json_search_plan_selection(buffer, response.selection)
    buffer += ",\"search\":"
    buffer += search_response_json(response.search)
    buffer += "}"
    return buffer^


def planned_debug_search_response_json(
    read response: PlannedDebugSearchResponse
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"selection\":"
    append_json_search_plan_selection(buffer, response.selection)
    buffer += ",\"debug\":"
    buffer += debug_search_response_json(response.debug)
    buffer += "}"
    return buffer^


def explain_request_json(read request: ExplainRequest) -> String:
    return search_request_json(request.search)


def explain_response_json(read response: ExplainResponse) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"explain\":"
    buffer += collection_search_explain_json(response.explain)
    buffer += "}"
    return buffer^


def planned_explain_response_json(
    read response: PlannedExplainResponse
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"selection\":"
    append_json_search_plan_selection(buffer, response.selection)
    buffer += ",\"explain\":"
    buffer += collection_search_explain_json(response.explain.explain)
    buffer += "}"
    return buffer^


def service_health_status_json(read status: ServiceHealthStatus) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"status\":\"" + json_escape(status.status) + "\","
    buffer += "\"collection_count\":" + String(status.collection_count) + ","
    buffer += "\"live_snapshot_count\":"
    buffer += String(status.live_snapshot_count) + ","
    buffer += "\"pending_compaction_count\":"
    buffer += String(status.pending_compaction_count)
    buffer += "}"
    return buffer^


def service_metrics_snapshot_json(read metrics: ServiceMetricsSnapshot) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_count\":" + String(metrics.collection_count) + ","
    buffer += "\"segment_count\":" + String(metrics.segment_count) + ","
    buffer += "\"document_count\":" + String(metrics.document_count) + ","
    buffer += "\"vector_count\":" + String(metrics.vector_count) + ","
    buffer += "\"byte_size\":" + String(metrics.byte_size) + ","
    buffer += "\"published_snapshot_count\":"
    buffer += String(metrics.published_snapshot_count) + ","
    buffer += "\"inactive_snapshot_count\":"
    buffer += String(metrics.inactive_snapshot_count) + ","
    buffer += "\"inactive_unique_segment_count\":"
    buffer += String(metrics.inactive_unique_segment_count) + ","
    buffer += "\"inactive_unique_byte_size\":"
    buffer += String(metrics.inactive_unique_byte_size) + ","
    buffer += "\"pending_draft_collection_count\":"
    buffer += String(metrics.pending_draft_collection_count) + ","
    buffer += "\"pending_draft_mutation_count\":"
    buffer += String(metrics.pending_draft_mutation_count)
    buffer += "}"
    return buffer^
