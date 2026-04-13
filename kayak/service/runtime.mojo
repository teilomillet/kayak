# Executable hosted-collection loop for create, mutate, snapshot, search, and explain.

from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    CollectionManifest,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA,
    CollectionStats,
    SegmentId,
    SealedSegmentManifest,
    SnapshotExportBundleManifest,
    SnapshotManifest,
    SnapshotLoadRequirements,
    collection_manifest_exists,
    exact_only_snapshot_requirements,
    export_snapshot_bundle,
    import_snapshot_bundle,
    load_collection_manifest,
    load_resolved_collection_snapshot,
    promote_collection_generation,
    publish_snapshot_manifest,
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
    filter_expression_requires_document_metadata,
)
from kayak.planning import (
    explain_collection_search,
    search_collection_for_plan,
)
from kayak.runtime import ExactScoringBackend

from .collection_requests import CreateCollectionRequest
from .document_requests import DeleteDocumentsRequest, UpsertDocumentsRequest
from .draft_state import (
    append_draft_delete_batch,
    append_draft_upsert_batch,
    empty_draft_collection_state,
    load_draft_collection_state,
    save_draft_collection_state,
)
from .paths import (
    draft_state_root,
    service_collection_root,
)
from .search_contracts import (
    DebugSearchResponse,
    ExplainResponse,
    SearchRequest,
    SearchResponse,
)
from .snapshot_requests import (
    CreateSnapshotRequest,
    ExportSnapshotRequest,
    ImportSnapshotRequest,
)


def require_filter_supported_for_request(read request: SearchRequest) raises:
    if request.filter_expression.is_match_all():
        return

    if request.plan.candidate_generator.kind != "exact_full_scan":
        raise Error(
            "non-match_all filters currently require exact_full_scan stage-1"
        )


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
    read request: SearchRequest
) raises -> SnapshotLoadRequirements:
    var needs_document_metadata = filter_expression_requires_document_metadata(
        request.filter_expression
    )

    if request.plan.candidate_generator.artifact_family.byte_length() == 0:
        if needs_document_metadata:
            return SnapshotLoadRequirements(
                False,
                [SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA],
                False,
            )
        return exact_only_snapshot_requirements()

    var required_artifacts = List[String]()
    required_artifacts.append(request.plan.candidate_generator.artifact_family.copy())
    if needs_document_metadata:
        required_artifacts.append(SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA)

    return SnapshotLoadRequirements(
        False,
        required_artifacts,
        False,
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
    publish_snapshot_manifest(collection_root, snapshot)
    _ = promote_collection_generation(collection_root, collection, generation)
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
    require_filter_supported_for_request(request)
    var collection_root = service_collection_root(
        service_root,
        request.tenant_id,
        request.namespace_id,
        request.collection_id,
    )
    _ = load_collection_for_request(
        service_root,
        request.collection_id.value,
        request.tenant_id.value,
        request.namespace_id.value,
        collection_root,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        request.snapshot_id,
        snapshot_load_requirements_for_request(request),
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
        ),
    )


def execute_debug_search[Backend: ExactScoringBackend](
    read backend: Backend,
    service_root: Path,
    read request: SearchRequest,
) raises -> DebugSearchResponse:
    var search = execute_search(backend, service_root, request)
    var explain = execute_explain(backend, service_root, request)
    return DebugSearchResponse(search, explain.explain)


def execute_explain[Backend: ExactScoringBackend](
    read backend: Backend,
    service_root: Path,
    read request: SearchRequest,
) raises -> ExplainResponse:
    require_filter_supported_for_request(request)
    var collection_root = service_collection_root(
        service_root,
        request.tenant_id,
        request.namespace_id,
        request.collection_id,
    )
    _ = load_collection_for_request(
        service_root,
        request.collection_id.value,
        request.tenant_id.value,
        request.namespace_id.value,
        collection_root,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        request.snapshot_id,
        snapshot_load_requirements_for_request(request),
    )
    return ExplainResponse(
        explain_collection_search(
            backend,
            request.query,
            snapshot,
            request.plan,
            request.filter_expression,
        )
    )
