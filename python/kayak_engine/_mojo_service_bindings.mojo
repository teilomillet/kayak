from std.collections import List
from std.os import abort
from std.pathlib import Path
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from kayak.collections import (
    COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED,
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
)
from kayak.collections.document_metadata import DocumentMetadataUpdate
from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.filters import FilterClause, FilterExpression, FilterField, FilterTerm
from kayak.numeric import VECTOR_SCALAR_NAME, VectorScalar
from kayak.planning import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SearchPlanSelectionRequest,
    best_effort_faithfulness_policy,
)
from kayak.runtime import ExactCpuBackend
from kayak.service import (
    CreateCollectionRequest,
    CreateSnapshotRequest,
    ExplainResponse,
    PlannedDebugSearchResponse,
    PlannedExplainResponse,
    PlannedSearchRequest,
    PlannedSearchResponse,
    SearchResponse,
    ServiceHealthStatus,
    ServiceMetricsSnapshot,
    UpsertDocument,
    UpsertDocumentsRequest,
    create_collection,
    create_snapshot,
    debug_search_response_json,
    execute_debug_search,
    execute_explain,
    execute_planned_debug_search,
    execute_planned_explain,
    execute_planned_search,
    execute_search,
    explain_response_json,
    planned_debug_search_response_json,
    planned_explain_response_json,
    planned_search_response_json,
    search_response_json,
    service_health_status_json,
    service_metrics_snapshot_json,
    upsert_documents,
)
from kayak.service.metrics_runtime import (
    build_service_health_status,
    build_service_metrics_snapshot,
)
from kayak.service.search_contracts import (
    SearchRequest,
    default_exact_search_request,
)


def decode_string_list(py_values: PythonObject) raises -> List[String]:
    var values = List[String]()
    for index in range(len(py_values)):
        values.append(String(py=py_values[index]))
    return values^


def decode_nested_string_lists(py_values: PythonObject) raises -> List[List[String]]:
    var values = List[List[String]]()
    for index in range(len(py_values)):
        values.append(decode_string_list(py_values[index]))
    return values^


def decode_float_vector(py_values: PythonObject) raises -> List[VectorScalar]:
    var values = List[VectorScalar]()
    for index in range(len(py_values)):
        values.append(VectorScalar(py=py_values[index]))
    return values^


def decode_float_vectors(py_values: PythonObject) raises -> List[List[VectorScalar]]:
    var vectors = List[List[VectorScalar]]()
    for index in range(len(py_values)):
        vectors.append(decode_float_vector(py_values[index]))
    return vectors^


def decode_document_vectors(
    py_values: PythonObject
) raises -> List[List[List[VectorScalar]]]:
    var documents = List[List[List[VectorScalar]]]()
    for index in range(len(py_values)):
        documents.append(decode_float_vectors(py_values[index]))
    return documents^


def decode_filter_expression(
    py_clause_fields: PythonObject,
    py_clause_operators: PythonObject,
    py_clause_values: PythonObject,
) raises -> FilterExpression:
    if len(py_clause_fields) == 0:
        return FilterExpression([])

    var clauses = List[FilterClause]()
    for clause_index in range(len(py_clause_fields)):
        var fields = decode_string_list(py_clause_fields[clause_index])
        var operators = decode_string_list(py_clause_operators[clause_index])
        var values = decode_nested_string_lists(py_clause_values[clause_index])
        if len(fields) != len(operators) or len(fields) != len(values):
            raise Error("filter clause field/operator/value lengths must match")

        var terms = List[FilterTerm]()
        for term_index in range(len(fields)):
            terms.append(
                FilterTerm(
                    FilterField(fields[term_index]),
                    operators[term_index],
                    values[term_index],
                )
            )
        clauses.append(FilterClause(terms))
    return FilterExpression(clauses)


def decode_metadata_updates(
    py_keys: PythonObject, py_values: PythonObject
) raises -> List[List[DocumentMetadataUpdate]]:
    if len(py_keys) != len(py_values):
        raise Error("metadata key/value row counts must match")

    var rows = List[List[DocumentMetadataUpdate]]()
    for row_index in range(len(py_keys)):
        var keys = decode_string_list(py_keys[row_index])
        var values = decode_string_list(py_values[row_index])
        if len(keys) != len(values):
            raise Error("metadata update keys and values must align per document")

        var updates = List[DocumentMetadataUpdate]()
        for update_index in range(len(keys)):
            updates.append(DocumentMetadataUpdate(keys[update_index], values[update_index]))
        rows.append(updates^)
    return rows^


def upsert_documents_request_from_python(
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
    py_doc_ids: PythonObject,
    py_document_vectors: PythonObject,
    py_texts: PythonObject,
    py_metadata_keys: PythonObject,
    py_metadata_values: PythonObject,
) raises -> UpsertDocumentsRequest:
    var doc_ids = decode_string_list(py_doc_ids)
    var document_vectors = decode_document_vectors(py_document_vectors)
    var texts = decode_string_list(py_texts)
    var metadata_rows = decode_metadata_updates(py_metadata_keys, py_metadata_values)
    if len(doc_ids) != len(document_vectors):
        raise Error("doc_ids and document_vectors lengths must match")
    if len(doc_ids) != len(texts):
        raise Error("doc_ids and texts lengths must match")
    if len(doc_ids) != len(metadata_rows):
        raise Error("doc_ids and metadata rows lengths must match")

    var documents = List[UpsertDocument]()
    for index in range(len(doc_ids)):
        var document = EncodedDocument(
            doc_ids[index].copy(),
            document_vectors[index].copy(),
        )
        documents.append(
            UpsertDocument(document, texts[index].copy(), metadata_rows[index])
        )
    return UpsertDocumentsRequest(
        CollectionId(collection_id),
        TenantId(tenant_id),
        NamespaceId(namespace_id),
        documents^,
    )


def append_json_string_field(mut buffer: String, key: String, value: String):
    buffer += "\""
    buffer += key
    buffer += "\":\""
    buffer += value.replace("\\", "\\\\").replace("\"", "\\\"")
    buffer += "\""


def create_collection_response_json(
    collection_root: Path,
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
    collection_layout_family: String,
    model_name: String,
    vector_dim: Int,
) -> String:
    var buffer = String()
    buffer += "{"
    append_json_string_field(buffer, "collection_root", collection_root.__fspath__())
    buffer += ","
    append_json_string_field(buffer, "collection_id", collection_id)
    buffer += ","
    append_json_string_field(buffer, "tenant_id", tenant_id)
    buffer += ","
    append_json_string_field(buffer, "namespace_id", namespace_id)
    buffer += ","
    append_json_string_field(buffer, "collection_layout_family", collection_layout_family)
    buffer += ","
    append_json_string_field(buffer, "model_name", model_name)
    buffer += ",\"vector_dim\":"
    buffer += String(vector_dim)
    buffer += "}"
    return buffer^


def upsert_documents_response_json(
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
    upserted_count: Int,
    draft_document_count: Int,
) -> String:
    var buffer = String()
    buffer += "{"
    append_json_string_field(buffer, "collection_id", collection_id)
    buffer += ","
    append_json_string_field(buffer, "tenant_id", tenant_id)
    buffer += ","
    append_json_string_field(buffer, "namespace_id", namespace_id)
    buffer += ",\"upserted_count\":"
    buffer += String(upserted_count)
    buffer += ",\"draft_document_count\":"
    buffer += String(draft_document_count)
    buffer += "}"
    return buffer^


def create_snapshot_response_json(
    snapshot_id: String,
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
    generation: Int,
    segment_count: Int,
    document_count: Int,
    vector_count: Int,
    byte_size: Int,
) -> String:
    var buffer = String()
    buffer += "{"
    append_json_string_field(buffer, "snapshot_id", snapshot_id)
    buffer += ","
    append_json_string_field(buffer, "collection_id", collection_id)
    buffer += ","
    append_json_string_field(buffer, "tenant_id", tenant_id)
    buffer += ","
    append_json_string_field(buffer, "namespace_id", namespace_id)
    buffer += ",\"generation\":"
    buffer += String(generation)
    buffer += ",\"segment_count\":"
    buffer += String(segment_count)
    buffer += ",\"document_count\":"
    buffer += String(document_count)
    buffer += ",\"vector_count\":"
    buffer += String(vector_count)
    buffer += ",\"byte_size\":"
    buffer += String(byte_size)
    buffer += "}"
    return buffer^


def python_string(read value: String) raises -> PythonObject:
    return Python.str(value)


def create_collection_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_collection_layout_family: PythonObject,
    py_model_name: PythonObject,
    py_vector_scalar_name: PythonObject,
    py_vector_dim: PythonObject,
    py_default_keep_latest_inactive_count: PythonObject,
) raises -> PythonObject:
    var collection_id = String(py=py_collection_id)
    var tenant_id = String(py=py_tenant_id)
    var namespace_id = String(py=py_namespace_id)
    var model_name = String(py=py_model_name)
    var collection_layout_family = String(py=py_collection_layout_family)
    if collection_layout_family.byte_length() == 0:
        collection_layout_family = COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED
    var vector_scalar_name = String(py=py_vector_scalar_name)
    if vector_scalar_name.byte_length() == 0:
        vector_scalar_name = VECTOR_SCALAR_NAME
    var request = CreateCollectionRequest(
        CollectionId(collection_id),
        TenantId(tenant_id),
        NamespaceId(namespace_id),
        collection_layout_family,
        model_name,
        vector_scalar_name,
        Int(py=py_vector_dim),
        Int(py=py_default_keep_latest_inactive_count),
    )
    var collection_root = create_collection(
        Path(String(py=py_service_root)),
        request,
    )
    return python_string(
        create_collection_response_json(
            collection_root,
            collection_id,
            tenant_id,
            namespace_id,
            collection_layout_family,
            model_name,
            request.vector_dim,
        )
    )


def upsert_documents_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_doc_ids: PythonObject,
    py_document_vectors: PythonObject,
    py_texts: PythonObject,
    py_metadata_keys: PythonObject,
    py_metadata_values: PythonObject,
) raises -> PythonObject:
    var collection_id = String(py=py_collection_id)
    var tenant_id = String(py=py_tenant_id)
    var namespace_id = String(py=py_namespace_id)
    var request = upsert_documents_request_from_python(
        collection_id,
        tenant_id,
        namespace_id,
        py_doc_ids,
        py_document_vectors,
        py_texts,
        py_metadata_keys,
        py_metadata_values,
    )
    var draft_document_count = upsert_documents(
        Path(String(py=py_service_root)),
        request,
    )
    return python_string(
        upsert_documents_response_json(
            collection_id,
            tenant_id,
            namespace_id,
            len(request.documents),
            draft_document_count,
        )
    )


def create_snapshot_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_snapshot_id: PythonObject,
    py_reason: PythonObject,
) raises -> PythonObject:
    var snapshot = create_snapshot(
        Path(String(py=py_service_root)),
        CreateSnapshotRequest(
            CollectionId(String(py=py_collection_id)),
            TenantId(String(py=py_tenant_id)),
            NamespaceId(String(py=py_namespace_id)),
            SnapshotId(String(py=py_snapshot_id)),
            String(py=py_reason),
        ),
    )
    return python_string(
        create_snapshot_response_json(
            snapshot.snapshot_id.value,
            snapshot.collection_id.value,
            snapshot.tenant_id.value,
            snapshot.namespace_id.value,
            snapshot.generation,
            snapshot.stats.segment_count,
            snapshot.stats.document_count,
            snapshot.stats.total_vector_count,
            snapshot.stats.byte_size,
        )
    )


def exact_search_request_from_python(
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
    snapshot_id: String,
    py_query_vectors: PythonObject,
    query_model_name: String,
    query_text: String,
    final_k: Int,
    debug_mode: Bool,
    py_clause_fields: PythonObject,
    py_clause_operators: PythonObject,
    py_clause_values: PythonObject,
) raises -> SearchRequest:
    var request = default_exact_search_request(
        CollectionId(collection_id),
        TenantId(tenant_id),
        NamespaceId(namespace_id),
        SnapshotId(snapshot_id),
        EncodedQuery(decode_float_vectors(py_query_vectors)),
        final_k,
        query_model_name,
        debug_mode,
        query_text,
    )
    return SearchRequest(
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
        request.snapshot_id,
        request.query,
        request.query_model_name,
        request.query_text,
        decode_filter_expression(py_clause_fields, py_clause_operators, py_clause_values),
        request.plan,
        request.debug_mode,
    )


def planned_search_request_from_python(
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
    snapshot_id: String,
    py_query_vectors: PythonObject,
    query_model_name: String,
    query_text: String,
    final_k: Int,
    candidate_k: Int,
    goal: String,
    py_preferred_candidate_generator_kinds: PythonObject,
    debug_mode: Bool,
    faithfulness_policy_kind: String,
    stage2_reference_kind: String,
    stage3_verifier_kind: String,
    py_clause_fields: PythonObject,
    py_clause_operators: PythonObject,
    py_clause_values: PythonObject,
    gem_graph_cluster_top_k_per_query_token: Int,
    gem_graph_beam_width: Int,
) raises -> PlannedSearchRequest:
    var faithfulness_policy = best_effort_faithfulness_policy()
    if faithfulness_policy_kind != "" and faithfulness_policy_kind != faithfulness_policy.kind:
        raise Error(
            "planned search transport currently supports only best_effort faithfulness_policy_kind"
        )

    var resolved_goal = goal
    if resolved_goal.byte_length() == 0:
        resolved_goal = SEARCH_PLANNING_GOAL_BALANCED
    return PlannedSearchRequest(
        CollectionId(collection_id),
        TenantId(tenant_id),
        NamespaceId(namespace_id),
        SnapshotId(snapshot_id),
        EncodedQuery(decode_float_vectors(py_query_vectors)),
        query_model_name,
        query_text,
        stage2_reference_kind,
        stage3_verifier_kind,
        decode_filter_expression(py_clause_fields, py_clause_operators, py_clause_values),
        SearchPlanSelectionRequest(
            final_k,
            candidate_k,
            faithfulness_policy,
            decode_filter_expression(py_clause_fields, py_clause_operators, py_clause_values),
            resolved_goal,
            decode_string_list(py_preferred_candidate_generator_kinds),
            debug_mode,
            gem_graph_cluster_top_k_per_query_token,
            gem_graph_beam_width,
        ),
    )


def exact_search_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_snapshot_id: PythonObject,
    py_query_vectors: PythonObject,
    py_query_model_name: PythonObject,
    py_query_text: PythonObject,
    py_final_k: PythonObject,
    py_debug_mode: PythonObject,
    py_clause_fields: PythonObject,
    py_clause_operators: PythonObject,
    py_clause_values: PythonObject,
) raises -> PythonObject:
    var backend = ExactCpuBackend()
    var response = execute_search(
        backend,
        Path(String(py=py_service_root)),
        exact_search_request_from_python(
            String(py=py_collection_id),
            String(py=py_tenant_id),
            String(py=py_namespace_id),
            String(py=py_snapshot_id),
            py_query_vectors,
            String(py=py_query_model_name),
            String(py=py_query_text),
            Int(py=py_final_k),
            Bool(py=py_debug_mode),
            py_clause_fields,
            py_clause_operators,
            py_clause_values,
        ),
    )
    return python_string(search_response_json(response))


def exact_explain_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_snapshot_id: PythonObject,
    py_query_vectors: PythonObject,
    py_query_model_name: PythonObject,
    py_query_text: PythonObject,
    py_final_k: PythonObject,
    py_debug_mode: PythonObject,
    py_clause_fields: PythonObject,
    py_clause_operators: PythonObject,
    py_clause_values: PythonObject,
) raises -> PythonObject:
    var backend = ExactCpuBackend()
    var response = execute_explain(
        backend,
        Path(String(py=py_service_root)),
        exact_search_request_from_python(
            String(py=py_collection_id),
            String(py=py_tenant_id),
            String(py=py_namespace_id),
            String(py=py_snapshot_id),
            py_query_vectors,
            String(py=py_query_model_name),
            String(py=py_query_text),
            Int(py=py_final_k),
            Bool(py=py_debug_mode),
            py_clause_fields,
            py_clause_operators,
            py_clause_values,
        ),
    )
    return python_string(explain_response_json(response))


def planned_search_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_snapshot_id: PythonObject,
    py_query_vectors: PythonObject,
    py_query_model_name: PythonObject,
    py_query_text: PythonObject,
    py_final_k: PythonObject,
    py_candidate_k: PythonObject,
    py_goal: PythonObject,
    py_preferred_candidate_generator_kinds: PythonObject,
    py_debug_mode: PythonObject,
    py_faithfulness_policy_kind: PythonObject,
    py_stage2_reference_kind: PythonObject,
    py_stage3_verifier_kind: PythonObject,
    py_clause_fields: PythonObject,
    py_clause_operators: PythonObject,
    py_clause_values: PythonObject,
    py_gem_graph_cluster_top_k_per_query_token: PythonObject,
    py_gem_graph_beam_width: PythonObject,
) raises -> PythonObject:
    var backend = ExactCpuBackend()
    var response = execute_planned_search(
        backend,
        Path(String(py=py_service_root)),
        planned_search_request_from_python(
            String(py=py_collection_id),
            String(py=py_tenant_id),
            String(py=py_namespace_id),
            String(py=py_snapshot_id),
            py_query_vectors,
            String(py=py_query_model_name),
            String(py=py_query_text),
            Int(py=py_final_k),
            Int(py=py_candidate_k),
            String(py=py_goal),
            py_preferred_candidate_generator_kinds,
            Bool(py=py_debug_mode),
            String(py=py_faithfulness_policy_kind),
            String(py=py_stage2_reference_kind),
            String(py=py_stage3_verifier_kind),
            py_clause_fields,
            py_clause_operators,
            py_clause_values,
            Int(py=py_gem_graph_cluster_top_k_per_query_token),
            Int(py=py_gem_graph_beam_width),
        ),
    )
    return python_string(planned_search_response_json(response))


def planned_explain_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_snapshot_id: PythonObject,
    py_query_vectors: PythonObject,
    py_query_model_name: PythonObject,
    py_query_text: PythonObject,
    py_final_k: PythonObject,
    py_candidate_k: PythonObject,
    py_goal: PythonObject,
    py_preferred_candidate_generator_kinds: PythonObject,
    py_debug_mode: PythonObject,
    py_faithfulness_policy_kind: PythonObject,
    py_stage2_reference_kind: PythonObject,
    py_stage3_verifier_kind: PythonObject,
    py_clause_fields: PythonObject,
    py_clause_operators: PythonObject,
    py_clause_values: PythonObject,
    py_gem_graph_cluster_top_k_per_query_token: PythonObject,
    py_gem_graph_beam_width: PythonObject,
) raises -> PythonObject:
    var backend = ExactCpuBackend()
    var response = execute_planned_explain(
        backend,
        Path(String(py=py_service_root)),
        planned_search_request_from_python(
            String(py=py_collection_id),
            String(py=py_tenant_id),
            String(py=py_namespace_id),
            String(py=py_snapshot_id),
            py_query_vectors,
            String(py=py_query_model_name),
            String(py=py_query_text),
            Int(py=py_final_k),
            Int(py=py_candidate_k),
            String(py=py_goal),
            py_preferred_candidate_generator_kinds,
            Bool(py=py_debug_mode),
            String(py=py_faithfulness_policy_kind),
            String(py=py_stage2_reference_kind),
            String(py=py_stage3_verifier_kind),
            py_clause_fields,
            py_clause_operators,
            py_clause_values,
            Int(py=py_gem_graph_cluster_top_k_per_query_token),
            Int(py=py_gem_graph_beam_width),
        ),
    )
    return python_string(planned_explain_response_json(response))


def service_health_json(py_service_root: PythonObject) raises -> PythonObject:
    return python_string(
        service_health_status_json(
            build_service_health_status(Path(String(py=py_service_root)))
        )
    )


def service_metrics_json(py_service_root: PythonObject) raises -> PythonObject:
    return python_string(
        service_metrics_snapshot_json(
            build_service_metrics_snapshot(Path(String(py=py_service_root)))
        )
    )


def create_collection_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return create_collection_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["collection_layout_family"],
        py_request["model_name"],
        py_request["vector_scalar_name"],
        py_request["vector_dim"],
        py_request["default_keep_latest_inactive_count"],
    )


def upsert_documents_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return upsert_documents_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["doc_ids"],
        py_request["document_vectors"],
        py_request["texts"],
        py_request["metadata_keys"],
        py_request["metadata_values"],
    )


def create_snapshot_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return create_snapshot_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["snapshot_id"],
        py_request["reason"],
    )


def exact_search_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return exact_search_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["snapshot_id"],
        py_request["query"],
        py_request["query_model_name"],
        py_request["query_text"],
        py_request["final_k"],
        py_request["debug_mode"],
        py_request["clause_fields"],
        py_request["clause_operators"],
        py_request["clause_values"],
    )


def exact_explain_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return exact_explain_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["snapshot_id"],
        py_request["query"],
        py_request["query_model_name"],
        py_request["query_text"],
        py_request["final_k"],
        py_request["debug_mode"],
        py_request["clause_fields"],
        py_request["clause_operators"],
        py_request["clause_values"],
    )


def planned_search_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return planned_search_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["snapshot_id"],
        py_request["query"],
        py_request["query_model_name"],
        py_request["query_text"],
        py_request["final_k"],
        py_request["candidate_k"],
        py_request["goal"],
        py_request["preferred_candidate_generator_kinds"],
        py_request["debug_mode"],
        py_request["faithfulness_policy_kind"],
        py_request["stage2_reference_kind"],
        py_request["stage3_verifier_kind"],
        py_request["clause_fields"],
        py_request["clause_operators"],
        py_request["clause_values"],
        py_request["gem_graph_cluster_top_k_per_query_token"],
        py_request["gem_graph_beam_width"],
    )


def planned_explain_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return planned_explain_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["snapshot_id"],
        py_request["query"],
        py_request["query_model_name"],
        py_request["query_text"],
        py_request["final_k"],
        py_request["candidate_k"],
        py_request["goal"],
        py_request["preferred_candidate_generator_kinds"],
        py_request["debug_mode"],
        py_request["faithfulness_policy_kind"],
        py_request["stage2_reference_kind"],
        py_request["stage3_verifier_kind"],
        py_request["clause_fields"],
        py_request["clause_operators"],
        py_request["clause_values"],
        py_request["gem_graph_cluster_top_k_per_query_token"],
        py_request["gem_graph_beam_width"],
    )


@export
def PyInit__mojo_service_bindings() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo_service_bindings")
        module.def_function[create_collection_json_bridge](
            "create_collection_json",
            docstring="Create a hosted collection and return a JSON response payload.",
        )
        module.def_function[upsert_documents_json_bridge](
            "upsert_documents_json",
            docstring="Upsert hosted collection documents and return a JSON response payload.",
        )
        module.def_function[create_snapshot_json_bridge](
            "create_snapshot_json",
            docstring="Create a hosted snapshot and return a JSON response payload.",
        )
        module.def_function[exact_search_json_bridge](
            "exact_search_json",
            docstring="Execute exact hosted search and return a JSON response payload.",
        )
        module.def_function[exact_explain_json_bridge](
            "exact_explain_json",
            docstring="Execute exact hosted explain and return a JSON response payload.",
        )
        module.def_function[planned_search_json_bridge](
            "planned_search_json",
            docstring="Execute planner-driven hosted search and return a JSON response payload.",
        )
        module.def_function[planned_explain_json_bridge](
            "planned_explain_json",
            docstring="Execute planner-driven hosted explain and return a JSON response payload.",
        )
        module.def_function[service_health_json](
            "service_health_json",
            docstring="Build the hosted service health payload as JSON.",
        )
        module.def_function[service_metrics_json](
            "service_metrics_json",
            docstring="Build the hosted service metrics payload as JSON.",
        )
        return module.finalize()
    except e:
        abort(String("error creating hosted engine Mojo service bindings:", e))
