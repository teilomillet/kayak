# Executable hosted-collection loop for create, mutate, snapshot, search, and explain.

from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    CollectionManifest,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA,
    CollectionStats,
    SegmentId,
    SealedSegmentManifest,
    SnapshotExportBundleManifest,
    SnapshotSearchArtifactAvailability,
    SnapshotManifest,
    SnapshotLoadRequirements,
    collection_layout_family_is_shared_pool,
    collection_manifest_exists,
    exact_only_snapshot_requirements,
    export_snapshot_bundle,
    import_snapshot_bundle,
    load_snapshot_search_artifact_availability,
    load_collection_manifest,
    load_resolved_collection_snapshot,
    publish_collection_snapshot,
    save_collection_manifest,
    seal_single_segment,
    snapshot_manifest_exists,
)
from kayak.contracts import EncodedDocument
from kayak.collections.document_metadata import (
    DocumentMetadataMap,
    empty_document_metadata_map,
    merge_document_metadata,
)
from kayak.filters import (
    FilterExpression,
    filter_expression_is_exact_doc_id_filter,
    filter_expression_requires_document_metadata,
)
from kayak.planning import (
    explain_collection_search,
    SearchPlan,
    SearchPlanSelection,
    search_plan_selection_request_with_serving_scope,
    search_serving_scope_for_collection,
    select_search_plan_for_availability,
    search_collection_for_plan,
)
from kayak.planning.filter_scope import effective_filter_expression_for_collection
from kayak.runtime import ExactScoringBackend

from .collection_requests import CreateCollectionRequest
from .document_requests import DeleteDocumentsRequest, UpsertDocumentsRequest
from .draft_state import (
    append_draft_delete_batch,
    append_draft_upsert_batch,
    compact_draft_collection_state,
    empty_draft_collection_state,
    load_draft_collection_state,
    save_draft_collection_state,
)
from .paths import (
    draft_state_root,
    service_collection_root,
)
from .planned_search_stage_override import (
    planned_search_has_stage_override,
    search_plan_with_planned_search_stage_override,
)
from .search_contracts import (
    DebugSearchResponse,
    ExplainResponse,
    PlannedDebugSearchResponse,
    PlannedExplainResponse,
    PlannedSearchRequest,
    PlannedSearchResponse,
    SearchRequest,
    SearchResponse,
)
from .snapshot_requests import (
    CreateSnapshotRequest,
    ExportSnapshotRequest,
    ImportSnapshotRequest,
)


def require_filter_supported_for_request(
    read collection: CollectionManifest,
    read request: SearchRequest,
) raises:
    var effective_filter_expression = effective_filter_expression_for_collection(
        collection,
        request.filter_expression,
    )
    if effective_filter_expression.is_match_all():
        if not request.plan.candidate_generator.supports_match_all_filter:
            raise Error(
                "candidate generator does not support match_all filters: "
                + request.plan.candidate_generator.kind
            )
        return

    if filter_expression_is_exact_doc_id_filter(effective_filter_expression):
        if request.plan.candidate_generator.supports_exact_doc_id_filter:
            return
        raise Error(
            "exact doc_id filters currently require a stage-1 generator with exact-doc_id-filter support: "
            + request.plan.candidate_generator.kind
        )

    if filter_expression_requires_document_metadata(effective_filter_expression):
        if request.plan.candidate_generator.supports_structured_filter:
            return
        raise Error(
            "metadata/logical-scope filters currently require a stage-1 generator with structured-filter support: "
            + request.plan.candidate_generator.kind
        )

    raise Error("unsupported filter expression for search request")


def require_request_matches_collection(
    read collection: CollectionManifest,
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
) raises:
    if collection.collection_id.value != collection_id:
        raise Error("request collection_id does not match collection manifest")

    if collection.tenant_id.value != tenant_id:
        raise Error("request tenant_id does not match collection manifest")

    if collection.namespace_id.value != namespace_id:
        raise Error("request namespace_id does not match collection manifest")


def require_query_matches_collection(
    read collection: CollectionManifest,
    query_model_name: String,
    query_vector_dim: Int,
) raises:
    if query_model_name != collection.model_name:
        raise Error("query_model_name does not match collection manifest")

    if query_vector_dim != collection.vector_dim:
        raise Error("query vector_dim does not match collection manifest")


def load_collection_for_request(
    service_root: Path,
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
    collection_root: Path,
) raises -> CollectionManifest:
    _ = service_root
    if not collection_manifest_exists(collection_root):
        raise Error("collection does not exist")

    var collection = load_collection_manifest(collection_root)
    require_request_matches_collection(
        collection, collection_id, tenant_id, namespace_id
    )
    return collection^


def create_collection(service_root: Path, request: CreateCollectionRequest) raises -> Path:
    var collection_root = service_collection_root(
        service_root,
        request.tenant_id,
        request.namespace_id,
        request.collection_id,
    )
    if collection_manifest_exists(collection_root):
        raise Error("collection already exists")

    var manifest = request.to_manifest()
    save_collection_manifest(collection_root, manifest)
    save_draft_collection_state(
        draft_state_root(collection_root),
        manifest,
        empty_draft_collection_state(),
    )
    return collection_root


def find_document_index(
    read documents: List[EncodedDocument], doc_id: String
) -> Int:
    for index in range(len(documents)):
        if documents[index].doc_id == doc_id:
            return index

    return -1


def upsert_documents(service_root: Path, request: UpsertDocumentsRequest) raises -> Int:
    var collection_root = service_collection_root(
        service_root,
        request.tenant_id,
        request.namespace_id,
        request.collection_id,
    )
    var collection = load_collection_for_request(
        service_root,
        request.collection_id.value,
        request.tenant_id.value,
        request.namespace_id.value,
        collection_root,
    )
    var state = load_draft_collection_state(draft_state_root(collection_root), collection)
    var documents = state.documents.copy()
    var texts = state.texts.copy()
    var metadata_maps = state.metadata_maps.copy()
    var normalized_documents = List[EncodedDocument]()
    var normalized_texts = List[String]()
    var normalized_metadata_maps = List[DocumentMetadataMap]()

    for upsert in request.documents:
        if upsert.document.vector_dim != collection.vector_dim:
            raise Error("upsert document vector_dim does not match collection manifest")

        var existing_index = find_document_index(documents, upsert.document.doc_id)
        var next_text = String()
        var next_metadata = empty_document_metadata_map()
        if existing_index != -1:
            next_text = texts[existing_index].copy()
            next_metadata = metadata_maps[existing_index].copy()
        if upsert.has_text:
            next_text = upsert.text.copy()
        if upsert.has_metadata_updates:
            next_metadata = merge_document_metadata(
                next_metadata,
                upsert.metadata_updates,
            )

        if existing_index == -1:
            documents.append(upsert.document.copy())
            texts.append(next_text.copy())
            metadata_maps.append(next_metadata.copy())
        else:
            documents[existing_index] = upsert.document.copy()
            texts[existing_index] = next_text.copy()
            metadata_maps[existing_index] = next_metadata.copy()

        normalized_documents.append(upsert.document.copy())
        normalized_texts.append(next_text^)
        normalized_metadata_maps.append(next_metadata.copy())

    append_draft_upsert_batch(
        draft_state_root(collection_root),
        collection,
        normalized_documents,
        normalized_texts,
        normalized_metadata_maps,
        len(documents),
    )
    return len(documents)


def delete_documents(service_root: Path, request: DeleteDocumentsRequest) raises -> Int:
    var collection_root = service_collection_root(
        service_root,
        request.tenant_id,
        request.namespace_id,
        request.collection_id,
    )
    var collection = load_collection_for_request(
        service_root,
        request.collection_id.value,
        request.tenant_id.value,
        request.namespace_id.value,
        collection_root,
    )
    var state = load_draft_collection_state(draft_state_root(collection_root), collection)
    var kept_documents = List[EncodedDocument]()
    var kept_texts = List[String]()

    for index in range(len(state.documents)):
        var keep_document = True
        for doc_id in request.doc_ids:
            if state.documents[index].doc_id == doc_id:
                keep_document = False
                break

        if keep_document:
            kept_documents.append(state.documents[index].copy())
            kept_texts.append(state.texts[index].copy())

    var kept_document_count = len(kept_documents)
    append_draft_delete_batch(
        draft_state_root(collection_root),
        collection,
        request.doc_ids,
        kept_document_count,
    )
    return kept_document_count


def parse_file_uri(source_uri: String) raises -> Path:
    if source_uri.find("file://") != 0:
        raise Error("snapshot import currently requires a file:// source_uri")

    return Path(source_uri.replace("file://", ""))


def snapshot_load_requirements_for_request(
    read collection: CollectionManifest,
    read request: SearchRequest,
) raises -> SnapshotLoadRequirements:
    var effective_filter_expression = effective_filter_expression_for_collection(
        collection,
        request.filter_expression,
    )
    var needs_document_metadata = filter_expression_requires_document_metadata(
        effective_filter_expression
    )
    var needs_document_text = request.plan.stage3_verifier.requires_artifact_family(
        "document_text"
    )
    var required_artifacts = (
        request.plan.candidate_generator.required_search_artifact_families.copy()
    )
    if collection_layout_family_is_shared_pool(collection.collection_layout_family):
        required_artifacts.append(
            SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX
        )
    elif needs_document_metadata:
        if request.plan.candidate_generator.is_exact:
            required_artifacts.append(SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA)
        elif request.plan.candidate_generator.supports_structured_filter:
            required_artifacts.append(
                SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX
            )
        else:
            required_artifacts.append(SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA)

    if len(required_artifacts) == 0:
        return exact_only_snapshot_requirements(needs_document_text)

    return SnapshotLoadRequirements(
        False,
        required_artifacts,
        needs_document_text,
    )


def require_shared_pool_snapshot_filter_index_availability(
    read collection: CollectionManifest,
    read availability: SnapshotSearchArtifactAvailability,
) raises:
    if not collection_layout_family_is_shared_pool(
        collection.collection_layout_family
    ):
        return

    if availability.has_search_artifact_family_on_all_segments(
        SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX
    ):
        return

    raise Error(
        "shared_pool collections require a document_filter_index sidecar on every segment"
    )


def snapshot_manifest_for_segment(
    read request: CreateSnapshotRequest,
    read collection: CollectionManifest,
    generation: Int,
    read segment: SealedSegmentManifest,
) raises -> SnapshotManifest:
    return SnapshotManifest(
        request.snapshot_id,
        collection.collection_id,
        collection.tenant_id,
        collection.namespace_id,
        generation,
        [segment.segment_id.copy()],
        CollectionStats(
            1,
            segment.stats.document_count,
            segment.stats.token_count,
            segment.stats.total_vector_count,
            segment.stats.byte_size,
        ),
    )


def create_snapshot(
    service_root: Path, request: CreateSnapshotRequest
) raises -> SnapshotManifest:
    var collection_root = service_collection_root(
        service_root,
        request.tenant_id,
        request.namespace_id,
        request.collection_id,
    )
    var collection = load_collection_for_request(
        service_root,
        request.collection_id.value,
        request.tenant_id.value,
        request.namespace_id.value,
        collection_root,
    )
    if snapshot_manifest_exists(collection_root / "snapshots" / request.snapshot_id.value):
        raise Error("snapshot already exists")

    var draft_state = load_draft_collection_state(
        draft_state_root(collection_root), collection
    )
    if draft_state.is_empty():
        raise Error("cannot create a snapshot from an empty draft collection")

    var generation = collection.latest_generation + 1
    var segment_id = SegmentId("segment-" + String(generation))
    var sealed_segment = seal_single_segment(
        collection_root,
        collection,
        segment_id,
        generation,
        draft_state.documents,
        draft_state.texts,
        draft_state.metadata_maps,
    )
    var snapshot = snapshot_manifest_for_segment(
        request,
        collection,
        generation,
        sealed_segment,
    )
    _ = publish_collection_snapshot(collection_root, collection, snapshot)
    _ = compact_draft_collection_state(
        draft_state_root(collection_root),
        collection,
    )
    return snapshot^


def export_snapshot(
    service_root: Path,
    request: ExportSnapshotRequest,
    bundle_root: Path,
) raises -> SnapshotExportBundleManifest:
    _ = load_collection_for_request(
        service_root,
        request.collection_id.value,
        request.tenant_id.value,
        request.namespace_id.value,
        service_collection_root(
            service_root,
            request.tenant_id,
            request.namespace_id,
            request.collection_id,
        ),
    )
    return export_snapshot_bundle(
        service_collection_root(
            service_root,
            request.tenant_id,
            request.namespace_id,
            request.collection_id,
        ),
        request.snapshot_id,
        bundle_root,
    )


def import_snapshot(
    service_root: Path, request: ImportSnapshotRequest
) raises -> SnapshotExportBundleManifest:
    return import_snapshot_bundle(
        parse_file_uri(request.source_uri),
        service_collection_root(
            service_root,
            request.tenant_id,
            request.namespace_id,
            request.collection_id,
        ),
    )


def execute_search[Backend: ExactScoringBackend](
    read backend: Backend,
    service_root: Path,
    read request: SearchRequest,
) raises -> SearchResponse:
    var collection_root = service_collection_root(
        service_root,
        request.tenant_id,
        request.namespace_id,
        request.collection_id,
    )
    var collection = load_collection_for_request(
        service_root,
        request.collection_id.value,
        request.tenant_id.value,
        request.namespace_id.value,
        collection_root,
    )
    require_query_matches_collection(
        collection,
        request.query_model_name,
        request.query.vector_dim,
    )
    require_filter_supported_for_request(collection, request)
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        request.snapshot_id,
        snapshot_load_requirements_for_request(collection, request),
    )
    return SearchResponse(
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
        request.snapshot_id,
        request.plan,
        search_collection_for_plan(
            backend,
            request.query,
            snapshot,
            request.plan,
            request.filter_expression,
            request.query_text,
        ),
    )


def search_request_for_planned_request(
    read request: PlannedSearchRequest, plan: SearchPlan
) raises -> SearchRequest:
    var effective_plan = search_plan_with_planned_search_stage_override(
        plan,
        request.stage_override,
    )

    return SearchRequest(
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
        request.snapshot_id,
        request.query,
        request.query_model_name,
        request.query_text,
        request.filter_expression,
        effective_plan,
        request.planning.debug_mode,
    )


def selection_for_planned_request(
    read request: PlannedSearchRequest,
    read selection: SearchPlanSelection,
) raises -> SearchPlanSelection:
    if not planned_search_has_stage_override(request.stage_override):
        return selection.copy()

    return SearchPlanSelection(
        selection.goal,
        selection.selected_candidate_generator_status.copy(),
        selection.available_candidate_generator_kinds,
        selection.effective_candidate_generator_order,
        selection.serving_scope,
        search_plan_with_planned_search_stage_override(
            selection.plan,
            request.stage_override,
        ),
        selection.decision,
    )


def select_search_plan_for_request(
    service_root: Path,
    read request: PlannedSearchRequest,
) raises -> SearchPlanSelection:
    var collection_root = service_collection_root(
        service_root,
        request.tenant_id,
        request.namespace_id,
        request.collection_id,
    )
    var collection = load_collection_for_request(
        service_root,
        request.collection_id.value,
        request.tenant_id.value,
        request.namespace_id.value,
        collection_root,
    )
    require_query_matches_collection(
        collection,
        request.query_model_name,
        request.query.vector_dim,
    )
    var availability = load_snapshot_search_artifact_availability(
        collection_root,
        request.snapshot_id,
    )
    require_shared_pool_snapshot_filter_index_availability(
        collection,
        availability,
    )
    return select_search_plan_for_availability(
        availability,
        search_plan_selection_request_with_serving_scope(
            request.planning,
            search_serving_scope_for_collection(collection),
        ),
    )


def execute_planned_search[Backend: ExactScoringBackend](
    read backend: Backend,
    service_root: Path,
    read request: PlannedSearchRequest,
) raises -> PlannedSearchResponse:
    var selection = selection_for_planned_request(
        request,
        select_search_plan_for_request(service_root, request),
    )
    var search = execute_search(
        backend,
        service_root,
        search_request_for_planned_request(request, selection.plan),
    )
    return PlannedSearchResponse(selection, search)


def execute_debug_search[Backend: ExactScoringBackend](
    read backend: Backend,
    service_root: Path,
    read request: SearchRequest,
) raises -> DebugSearchResponse:
    var search = execute_search(backend, service_root, request)
    var explain = execute_explain(backend, service_root, request)
    return DebugSearchResponse(search, explain.explain)


def execute_planned_debug_search[Backend: ExactScoringBackend](
    read backend: Backend,
    service_root: Path,
    read request: PlannedSearchRequest,
) raises -> PlannedDebugSearchResponse:
    var selection = selection_for_planned_request(
        request,
        select_search_plan_for_request(service_root, request),
    )
    var debug = execute_debug_search(
        backend,
        service_root,
        search_request_for_planned_request(request, selection.plan),
    )
    return PlannedDebugSearchResponse(selection, debug)


def execute_explain[Backend: ExactScoringBackend](
    read backend: Backend,
    service_root: Path,
    read request: SearchRequest,
) raises -> ExplainResponse:
    var collection_root = service_collection_root(
        service_root,
        request.tenant_id,
        request.namespace_id,
        request.collection_id,
    )
    var collection = load_collection_for_request(
        service_root,
        request.collection_id.value,
        request.tenant_id.value,
        request.namespace_id.value,
        collection_root,
    )
    require_query_matches_collection(
        collection,
        request.query_model_name,
        request.query.vector_dim,
    )
    require_filter_supported_for_request(collection, request)
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        request.snapshot_id,
        snapshot_load_requirements_for_request(collection, request),
    )
    return ExplainResponse(
        explain_collection_search(
            backend,
            request.query,
            snapshot,
            request.plan,
            request.filter_expression,
            request.query_text,
        )
    )


def execute_planned_explain[Backend: ExactScoringBackend](
    read backend: Backend,
    service_root: Path,
    read request: PlannedSearchRequest,
) raises -> PlannedExplainResponse:
    var selection = selection_for_planned_request(
        request,
        select_search_plan_for_request(service_root, request),
    )
    var explain = execute_explain(
        backend,
        service_root,
        search_request_for_planned_request(request, selection.plan),
    )
    return PlannedExplainResponse(selection, explain)
