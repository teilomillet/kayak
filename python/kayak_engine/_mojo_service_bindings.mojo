from std.collections import List
from std.format import Writable, Writer
from std.os import abort
from std.pathlib import Path
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from kayak.benchmarks import (
    build_materialized_collection_search_summary,
    build_latent_proxy_projection_profile_summary,
    faithfulness_frontier_summary_json,
    latent_proxy_projection_profile_summary_json,
    materialize_native_latent_proxy_task_collection,
    native_latent_proxy_collection_summary_json,
)
from kayak.collections import (
    COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED,
    CollectionReclaimPlan,
    CollectionId,
    NamespaceId,
    SegmentId,
    SnapshotId,
    SnapshotRetentionDecision,
    SnapshotRetentionPolicy,
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
from kayak.scoring import ExactScoringConfig
from kayak.service import (
    BuildReclaimPlanRequest,
    CollectionLifecycleRequest,
    CreateCollectionRequest,
    CreateSnapshotRequest,
    DeleteDocumentsRequest,
    ExplainResponse,
    ExecuteReclaimRequest,
    ExportSnapshotRequest,
    ImportSnapshotRequest,
    PreparedExactSearchBatchConfig,
    PreparedSearchSnapshot,
    PlannedDebugSearchResponse,
    PlannedExplainResponse,
    PlannedSearchRequest,
    PlannedSearchResponse,
    SearchResponse,
    ServiceHealthStatus,
    ServiceMetricsSnapshot,
    UpdateCollectionRetentionPolicyRequest,
    UpsertDocument,
    UpsertDocumentsRequest,
    build_collection_lifecycle_report,
    build_reclaim_plan,
    build_reclaim_plan_response_json,
    collection_lifecycle_response_json,
    create_collection,
    create_snapshot,
    debug_search_response_json,
    delete_documents,
    delete_documents_response_json,
    execute_debug_search,
    execute_explain,
    execute_planned_debug_search,
    execute_planned_explain,
    execute_planned_search,
    execute_reclaim,
    execute_reclaim_response_json,
    execute_search,
    execute_search_batch_with_prepared_snapshot,
    execute_search_with_prepared_snapshot,
    explain_response_json,
    exact_cpu_backend_for_scoring_config,
    export_snapshot,
    planned_debug_search_response_json,
    planned_explain_response_json,
    planned_search_response_json,
    prepare_service_exact_search_snapshot,
    search_response_json,
    service_health_status_json,
    service_metrics_snapshot_json,
    snapshot_export_bundle_manifest_json,
    import_snapshot,
    update_collection_retention_policy,
    update_collection_retention_policy_response_json,
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


struct PreparedExactSearchSession(Movable, Writable):
    var prepared: PreparedSearchSnapshot

    def __init__(out self, var prepared: PreparedSearchSnapshot):
        self.prepared = prepared^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedExactSearchSession(collection_id=",
            self.prepared.snapshot.collection.collection_id.value,
            ", snapshot_id=",
            self.prepared.snapshot.snapshot.snapshot_id.value,
            ", text_corpus_loaded=",
            self.prepared.text_corpus_loaded,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


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


def decode_snapshot_ids(py_values: PythonObject) raises -> List[SnapshotId]:
    var snapshot_ids = List[SnapshotId]()
    for index in range(len(py_values)):
        snapshot_ids.append(SnapshotId(String(py=py_values[index])))
    return snapshot_ids^


def decode_segment_ids(py_values: PythonObject) raises -> List[SegmentId]:
    var segment_ids = List[SegmentId]()
    for index in range(len(py_values)):
        segment_ids.append(SegmentId(String(py=py_values[index])))
    return segment_ids^


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


def exact_scoring_config_from_python(
    enable_parallel_scoring: Bool,
    enable_dim128_fast_path: Bool,
    enable_parallel_work_item_oversubscription: Bool,
    parallel_work_item_count_override: Int,
) -> ExactScoringConfig:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = enable_parallel_scoring
    config.enable_dim128_fast_path = enable_dim128_fast_path
    config.enable_parallel_work_item_oversubscription = (
        enable_parallel_work_item_oversubscription
    )
    config.parallel_work_item_count_override = parallel_work_item_count_override
    return config^


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


def snapshot_retention_policy_from_python(
    has_policy_override: Bool,
    keep_latest_inactive_count: Int,
    py_pinned_snapshot_ids: PythonObject,
) raises -> SnapshotRetentionPolicy:
    if not has_policy_override:
        return SnapshotRetentionPolicy(0)

    return SnapshotRetentionPolicy(
        keep_latest_inactive_count,
        decode_snapshot_ids(py_pinned_snapshot_ids),
    )


def decode_snapshot_retention_decision(
    py_decision: PythonObject
) raises -> SnapshotRetentionDecision:
    return SnapshotRetentionDecision(
        SnapshotId(String(py=py_decision["snapshot_id"])),
        Int(py=py_decision["generation"]),
        Int(py=py_decision["segment_count"]),
        Int(py=py_decision["byte_size"]),
        Bool(py=py_decision["retain"]),
        String(py=py_decision["reason"]),
    )


def decode_snapshot_retention_decisions(
    py_decisions: PythonObject
) raises -> List[SnapshotRetentionDecision]:
    var decisions = List[SnapshotRetentionDecision]()
    for index in range(len(py_decisions)):
        decisions.append(decode_snapshot_retention_decision(py_decisions[index]))
    return decisions^


def decode_collection_reclaim_plan(
    py_plan: PythonObject
) raises -> CollectionReclaimPlan:
    return CollectionReclaimPlan(
        CollectionId(String(py=py_plan["collection_id"])),
        TenantId(String(py=py_plan["tenant_id"])),
        NamespaceId(String(py=py_plan["namespace_id"])),
        String(py=py_plan["active_snapshot_id"]),
        Int(py=py_plan["total_snapshot_count"]),
        Int(py=py_plan["inactive_snapshot_count"]),
        Int(py=py_plan["retained_inactive_snapshot_count"]),
        Int(py=py_plan["reclaimable_snapshot_count"]),
        Int(py=py_plan["reclaimable_unique_segment_count"]),
        Int(py=py_plan["reclaimable_unique_byte_size"]),
        decode_snapshot_retention_decisions(py_plan["decisions"]),
        decode_segment_ids(py_plan["reclaimable_unique_segment_ids"]),
    )


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


def delete_documents_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_doc_ids: PythonObject,
) raises -> PythonObject:
    return python_string(
        delete_documents_response_json(
            delete_documents(
                Path(String(py=py_service_root)),
                DeleteDocumentsRequest(
                    CollectionId(String(py=py_collection_id)),
                    TenantId(String(py=py_tenant_id)),
                    NamespaceId(String(py=py_namespace_id)),
                    decode_string_list(py_doc_ids),
                ),
            )
        )
    )


def update_collection_retention_policy_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_default_keep_latest_inactive_count: PythonObject,
) raises -> PythonObject:
    return python_string(
        update_collection_retention_policy_response_json(
            update_collection_retention_policy(
                Path(String(py=py_service_root)),
                UpdateCollectionRetentionPolicyRequest(
                    CollectionId(String(py=py_collection_id)),
                    TenantId(String(py=py_tenant_id)),
                    NamespaceId(String(py=py_namespace_id)),
                    Int(py=py_default_keep_latest_inactive_count),
                ),
            )
        )
    )


def export_snapshot_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_snapshot_id: PythonObject,
    py_bundle_root: PythonObject,
) raises -> PythonObject:
    return python_string(
        snapshot_export_bundle_manifest_json(
            export_snapshot(
                Path(String(py=py_service_root)),
                ExportSnapshotRequest(
                    CollectionId(String(py=py_collection_id)),
                    TenantId(String(py=py_tenant_id)),
                    NamespaceId(String(py=py_namespace_id)),
                    SnapshotId(String(py=py_snapshot_id)),
                ),
                Path(String(py=py_bundle_root)),
            )
        )
    )


def import_snapshot_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_snapshot_id: PythonObject,
    py_source_uri: PythonObject,
) raises -> PythonObject:
    return python_string(
        snapshot_export_bundle_manifest_json(
            import_snapshot(
                Path(String(py=py_service_root)),
                ImportSnapshotRequest(
                    CollectionId(String(py=py_collection_id)),
                    TenantId(String(py=py_tenant_id)),
                    NamespaceId(String(py=py_namespace_id)),
                    SnapshotId(String(py=py_snapshot_id)),
                    String(py=py_source_uri),
                ),
            )
        )
    )


def collection_lifecycle_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_has_policy_override: PythonObject,
    py_policy_override_keep_latest_inactive_count: PythonObject,
    py_policy_override_pinned_snapshot_ids: PythonObject,
) raises -> PythonObject:
    var has_policy_override = Bool(py=py_has_policy_override)
    var request = CollectionLifecycleRequest(
        CollectionId(String(py=py_collection_id)),
        TenantId(String(py=py_tenant_id)),
        NamespaceId(String(py=py_namespace_id)),
    )
    if has_policy_override:
        request = CollectionLifecycleRequest(
            CollectionId(String(py=py_collection_id)),
            TenantId(String(py=py_tenant_id)),
            NamespaceId(String(py=py_namespace_id)),
            snapshot_retention_policy_from_python(
                has_policy_override,
                Int(py=py_policy_override_keep_latest_inactive_count),
                py_policy_override_pinned_snapshot_ids,
            ),
        )

    return python_string(
        collection_lifecycle_response_json(
            build_collection_lifecycle_report(
                Path(String(py=py_service_root)),
                request,
            )
        )
    )


def build_reclaim_plan_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_has_policy_override: PythonObject,
    py_policy_override_keep_latest_inactive_count: PythonObject,
    py_policy_override_pinned_snapshot_ids: PythonObject,
) raises -> PythonObject:
    var has_policy_override = Bool(py=py_has_policy_override)
    var request = BuildReclaimPlanRequest(
        CollectionId(String(py=py_collection_id)),
        TenantId(String(py=py_tenant_id)),
        NamespaceId(String(py=py_namespace_id)),
    )
    if has_policy_override:
        request = BuildReclaimPlanRequest(
            CollectionId(String(py=py_collection_id)),
            TenantId(String(py=py_tenant_id)),
            NamespaceId(String(py=py_namespace_id)),
            snapshot_retention_policy_from_python(
                has_policy_override,
                Int(py=py_policy_override_keep_latest_inactive_count),
                py_policy_override_pinned_snapshot_ids,
            ),
        )

    return python_string(
        build_reclaim_plan_response_json(
            build_reclaim_plan(
                Path(String(py=py_service_root)),
                request,
            )
        )
    )


def execute_reclaim_json(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_dry_run: PythonObject,
    py_plan: PythonObject,
) raises -> PythonObject:
    return python_string(
        execute_reclaim_response_json(
            execute_reclaim(
                Path(String(py=py_service_root)),
                ExecuteReclaimRequest(
                    CollectionId(String(py=py_collection_id)),
                    TenantId(String(py=py_tenant_id)),
                    NamespaceId(String(py=py_namespace_id)),
                    decode_collection_reclaim_plan(py_plan),
                    Bool(py=py_dry_run),
                ),
            )
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


def decode_exact_search_requests(py_requests: PythonObject) raises -> List[SearchRequest]:
    var requests = List[SearchRequest]()
    for index in range(len(py_requests)):
        var py_request = py_requests[index]
        requests.append(
            exact_search_request_from_python(
                String(py=py_request["collection_id"]),
                String(py=py_request["tenant_id"]),
                String(py=py_request["namespace_id"]),
                String(py=py_request["snapshot_id"]),
                py_request["query"],
                String(py=py_request["query_model_name"]),
                String(py=py_request["query_text"]),
                Int(py=py_request["final_k"]),
                Bool(py=py_request["debug_mode"]),
                py_request["clause_fields"],
                py_request["clause_operators"],
                py_request["clause_values"],
            )
        )
    return requests^


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


def search_responses_json_to_python(
    read responses: List[SearchResponse]
) raises -> PythonObject:
    var py_responses = Python.list()
    for response in responses:
        py_responses.append(Python.str(search_response_json(response)))
    return py_responses


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


def debug_search_json(
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
    var response = execute_debug_search(
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
    return python_string(debug_search_response_json(response))


def prepare_exact_search_session(
    py_service_root: PythonObject,
    py_collection_id: PythonObject,
    py_tenant_id: PythonObject,
    py_namespace_id: PythonObject,
    py_snapshot_id: PythonObject,
    py_load_text_corpus: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=PreparedExactSearchSession(
            prepare_service_exact_search_snapshot(
                Path(String(py=py_service_root)),
                CollectionId(String(py=py_collection_id)),
                TenantId(String(py=py_tenant_id)),
                NamespaceId(String(py=py_namespace_id)),
                SnapshotId(String(py=py_snapshot_id)),
                Bool(py=py_load_text_corpus),
            )
        )
    )


def prepared_exact_search_json(
    py_prepared_session: PythonObject,
    py_request: PythonObject,
    py_enable_parallel_scoring: PythonObject,
    py_enable_dim128_fast_path: PythonObject,
    py_enable_parallel_work_item_oversubscription: PythonObject,
    py_parallel_work_item_count_override: PythonObject,
) raises -> PythonObject:
    var prepared_session = py_prepared_session.downcast_value_ptr[
        PreparedExactSearchSession
    ]()
    var backend = exact_cpu_backend_for_scoring_config(
        exact_scoring_config_from_python(
            Bool(py=py_enable_parallel_scoring),
            Bool(py=py_enable_dim128_fast_path),
            Bool(py=py_enable_parallel_work_item_oversubscription),
            Int(py=py_parallel_work_item_count_override),
        )
    )
    var response = execute_search_with_prepared_snapshot(
        backend,
        prepared_session[].prepared,
        exact_search_request_from_python(
            String(py=py_request["collection_id"]),
            String(py=py_request["tenant_id"]),
            String(py=py_request["namespace_id"]),
            String(py=py_request["snapshot_id"]),
            py_request["query"],
            String(py=py_request["query_model_name"]),
            String(py=py_request["query_text"]),
            Int(py=py_request["final_k"]),
            Bool(py=py_request["debug_mode"]),
            py_request["clause_fields"],
            py_request["clause_operators"],
            py_request["clause_values"],
        ),
    )
    return python_string(search_response_json(response))


def prepared_exact_search_batch_json(
    py_prepared_session: PythonObject,
    py_requests: PythonObject,
    py_worker_count: PythonObject,
    py_enable_parallel_scoring: PythonObject,
    py_enable_dim128_fast_path: PythonObject,
    py_enable_parallel_work_item_oversubscription: PythonObject,
    py_parallel_work_item_count_override: PythonObject,
) raises -> PythonObject:
    var prepared_session = py_prepared_session.downcast_value_ptr[
        PreparedExactSearchSession
    ]()
    var responses = execute_search_batch_with_prepared_snapshot(
        prepared_session[].prepared,
        decode_exact_search_requests(py_requests),
        PreparedExactSearchBatchConfig(
            Int(py=py_worker_count),
            exact_scoring_config_from_python(
                Bool(py=py_enable_parallel_scoring),
                Bool(py=py_enable_dim128_fast_path),
                Bool(py=py_enable_parallel_work_item_oversubscription),
                Int(py=py_parallel_work_item_count_override),
            ),
        ),
    )
    return search_responses_json_to_python(responses)


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


def planned_debug_search_json(
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
    var response = execute_planned_debug_search(
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
    return python_string(planned_debug_search_response_json(response))


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


def materialize_latent_proxy_collection_json_bridge(
    py_request: PythonObject
) raises -> PythonObject:
    return python_string(
        native_latent_proxy_collection_summary_json(
            materialize_native_latent_proxy_task_collection(
                String(py=py_request["dataset_id"]),
                String(py=py_request["model_name"]),
                String(py=py_request["task_path"]),
                Path(String(py=py_request["artifact_root"])),
                Path(String(py=py_request["collection_root"])),
                CollectionId(String(py=py_request["collection_id"])),
                TenantId(String(py=py_request["tenant_id"])),
                NamespaceId(String(py=py_request["namespace_id"])),
                SnapshotId(String(py=py_request["snapshot_id"])),
                Bool(py=py_request["load_text_corpus"]),
            )
        )
    )


def benchmark_materialized_collection_search_json_bridge(
    py_request: PythonObject
) raises -> PythonObject:
    return python_string(
        faithfulness_frontier_summary_json(
            build_materialized_collection_search_summary(
                String(py=py_request["dataset_id"]),
                String(py=py_request["model_name"]),
                String(py=py_request["task_path"]),
                Path(String(py=py_request["collection_root"])),
                SnapshotId(String(py=py_request["snapshot_id"])),
                String(py=py_request["candidate_generator_kind"]),
                Int(py=py_request["candidate_k"]),
            )
        )
    )


def benchmark_latent_proxy_projection_profile_json_bridge(
    py_request: PythonObject
) raises -> PythonObject:
    return python_string(
        latent_proxy_projection_profile_summary_json(
            build_latent_proxy_projection_profile_summary(
                String(py=py_request["dataset_id"]),
                String(py=py_request["model_name"]),
                String(py=py_request["task_path"]),
                String(py=py_request["artifact_root"]),
            )
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


def delete_documents_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return delete_documents_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["doc_ids"],
    )


def update_collection_retention_policy_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return update_collection_retention_policy_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["default_keep_latest_inactive_count"],
    )


def export_snapshot_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return export_snapshot_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["snapshot_id"],
        py_request["bundle_root"],
    )


def import_snapshot_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return import_snapshot_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["snapshot_id"],
        py_request["source_uri"],
    )


def collection_lifecycle_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return collection_lifecycle_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["has_policy_override"],
        py_request["policy_override_keep_latest_inactive_count"],
        py_request["policy_override_pinned_snapshot_ids"],
    )


def build_reclaim_plan_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return build_reclaim_plan_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["has_policy_override"],
        py_request["policy_override_keep_latest_inactive_count"],
        py_request["policy_override_pinned_snapshot_ids"],
    )


def execute_reclaim_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return execute_reclaim_json(
        py_service_root,
        py_request["collection_id"],
        py_request["tenant_id"],
        py_request["namespace_id"],
        py_request["dry_run"],
        py_request["plan"],
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


def prepared_exact_search_json_bridge(
    py_prepared_session: PythonObject,
    py_request: PythonObject,
    py_scoring_config: PythonObject,
) raises -> PythonObject:
    return prepared_exact_search_json(
        py_prepared_session,
        py_request,
        py_scoring_config["enable_parallel_scoring"],
        py_scoring_config["enable_dim128_fast_path"],
        py_scoring_config["enable_parallel_work_item_oversubscription"],
        py_scoring_config["parallel_work_item_count_override"],
    )


def prepared_exact_search_batch_json_bridge(
    py_prepared_session: PythonObject,
    py_requests: PythonObject,
    py_batch_config: PythonObject,
) raises -> PythonObject:
    return prepared_exact_search_batch_json(
        py_prepared_session,
        py_requests,
        py_batch_config["worker_count"],
        py_batch_config["enable_parallel_scoring"],
        py_batch_config["enable_dim128_fast_path"],
        py_batch_config["enable_parallel_work_item_oversubscription"],
        py_batch_config["parallel_work_item_count_override"],
    )


def debug_search_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return debug_search_json(
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


def planned_debug_search_json_bridge(
    py_service_root: PythonObject, py_request: PythonObject
) raises -> PythonObject:
    return planned_debug_search_json(
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
        _ = module.add_type[PreparedExactSearchSession]("PreparedExactSearchSession")
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
        module.def_function[delete_documents_json_bridge](
            "delete_documents_json",
            docstring="Delete hosted collection documents and return a JSON response payload.",
        )
        module.def_function[update_collection_retention_policy_json_bridge](
            "update_collection_retention_policy_json",
            docstring="Update collection retention policy and return a JSON response payload.",
        )
        module.def_function[export_snapshot_json_bridge](
            "export_snapshot_json",
            docstring="Export a hosted snapshot bundle and return a JSON response payload.",
        )
        module.def_function[import_snapshot_json_bridge](
            "import_snapshot_json",
            docstring="Import a hosted snapshot bundle and return a JSON response payload.",
        )
        module.def_function[collection_lifecycle_json_bridge](
            "collection_lifecycle_json",
            docstring="Build the collection lifecycle report and return a JSON response payload.",
        )
        module.def_function[build_reclaim_plan_json_bridge](
            "build_reclaim_plan_json",
            docstring="Build a reclaim plan and return a JSON response payload.",
        )
        module.def_function[execute_reclaim_json_bridge](
            "execute_reclaim_json",
            docstring="Execute a reclaim plan and return a JSON response payload.",
        )
        module.def_function[prepare_exact_search_session](
            "prepare_exact_search_session",
            docstring="Prepare a hosted exact-search session pinned to one published snapshot.",
        )
        module.def_function[exact_search_json_bridge](
            "exact_search_json",
            docstring="Execute exact hosted search and return a JSON response payload.",
        )
        module.def_function[prepared_exact_search_json_bridge](
            "prepared_exact_search_json",
            docstring="Execute exact hosted search on a prepared pinned snapshot and return a JSON response payload.",
        )
        module.def_function[prepared_exact_search_batch_json_bridge](
            "prepared_exact_search_batch_json",
            docstring="Execute a batch of exact hosted searches on a prepared pinned snapshot and return JSON response payloads.",
        )
        module.def_function[debug_search_json_bridge](
            "debug_search_json",
            docstring="Execute exact hosted debug search and return a JSON response payload.",
        )
        module.def_function[exact_explain_json_bridge](
            "exact_explain_json",
            docstring="Execute exact hosted explain and return a JSON response payload.",
        )
        module.def_function[planned_search_json_bridge](
            "planned_search_json",
            docstring="Execute planner-driven hosted search and return a JSON response payload.",
        )
        module.def_function[planned_debug_search_json_bridge](
            "planned_debug_search_json",
            docstring="Execute planner-driven hosted debug search and return a JSON response payload.",
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
        module.def_function[materialize_latent_proxy_collection_json_bridge](
            "materialize_latent_proxy_collection_json",
            docstring="Materialize one packed-index plus latent-proxy collection mirror and return a JSON summary.",
        )
        module.def_function[benchmark_materialized_collection_search_json_bridge](
            "benchmark_materialized_collection_search_json",
            docstring="Benchmark one explicit candidate-generator plan on a materialized collection and return a JSON summary.",
        )
        module.def_function[benchmark_latent_proxy_projection_profile_json_bridge](
            "benchmark_latent_proxy_projection_profile_json",
            docstring="Benchmark latent-proxy projection and scan microkernels and return a JSON summary.",
        )
        return module.finalize()
    except e:
        abort(String("error creating hosted engine Mojo service bindings:", e))
