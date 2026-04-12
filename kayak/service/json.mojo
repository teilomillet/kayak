from std.collections import List

from kayak.filters import FilterClause, FilterExpression, FilterTerm
from kayak.numeric import VectorScalar
from kayak.planning import CollectionHit, collection_search_explain_json

from .collection_requests import CreateCollectionRequest
from .document_requests import DeleteDocumentsRequest, UpsertDocument, UpsertDocumentsRequest
from .search_contracts import (
    DebugSearchResponse,
    ExplainRequest,
    ExplainResponse,
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


def append_json_search_plan(mut buffer: String, read request: SearchRequest):
    buffer += "{"
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(request.plan.candidate_generator.kind)
    buffer += "\","
    buffer += "\"faithfulness_policy_kind\":\""
    buffer += json_escape(request.plan.faithfulness_policy.kind)
    buffer += "\","
    buffer += "\"final_k\":"
    buffer += String(request.plan.candidate_budget.final_k) + ","
    buffer += "\"candidate_k\":"
    buffer += String(request.plan.candidate_budget.candidate_k) + ","
    buffer += "\"exact_stage_kind\":\""
    buffer += json_escape(request.plan.exact_stage_kind) + "\","
    buffer += "\"reranker_kind\":\""
    buffer += json_escape(request.plan.reranker_kind) + "\""
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

    if document.has_text:
        buffer += ",\"text\":\"" + json_escape(document.text) + "\""

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
    buffer += "\"vector_dim\":" + String(request.vector_dim)
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


def explain_request_json(read request: ExplainRequest) -> String:
    return search_request_json(request.search)


def explain_response_json(read response: ExplainResponse) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"explain\":"
    buffer += collection_search_explain_json(response.explain)
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
    buffer += "\"byte_size\":" + String(metrics.byte_size)
    buffer += "}"
    return buffer^
