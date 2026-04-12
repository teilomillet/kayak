# Executable hosted-collection loop for create, mutate, snapshot, search, and explain.

from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    CollectionManifest,
    CollectionStats,
    SegmentId,
    SegmentStats,
    SealedSegmentManifest,
    SnapshotExportBundleManifest,
    SnapshotManifest,
    StoredDocumentTextCorpus,
    collection_manifest_exists,
    export_snapshot_bundle,
    import_snapshot_bundle,
    load_collection_manifest,
    load_resolved_collection_snapshot,
    save_collection_manifest,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
    save_stored_document_text_corpus,
    snapshot_manifest_exists,
)
from kayak.contracts import EncodedDocument
from kayak.filters import FilterExpression
from kayak.index import pack_documents
from kayak.planning import (
    explain_collection_search,
    final_hits_for_plan,
)
from kayak.runtime import ExactScoringBackend
from kayak.storage import StoredPackedIndex, save_stored_packed_index
from kayak.text import DocumentTextCorpus

from .collection_requests import CreateCollectionRequest
from .document_requests import DeleteDocumentsRequest, UpsertDocumentsRequest
from .draft_state import (
    DraftCollectionState,
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


def require_filter_is_match_all(read filter_expression: FilterExpression) raises:
    if not filter_expression.is_match_all():
        raise Error("hosted collection runtime currently supports match_all filters only")


def packed_index_storage_byte_size(root: Path) raises -> Int:
    var total = (root / "manifest.tsv").read_text().byte_length()
    total += (root / "doc_ids.tsv").read_text().byte_length()
    total += (root / "doc_offsets.tsv").read_text().byte_length()
    total += len((root / "token_vectors.bin").read_bytes())
    return total


def text_corpus_storage_byte_size(
    root: Path, document_count: Int
) raises -> Int:
    var total = (root / "manifest.tsv").read_text().byte_length()
    total += (root / "entries.tsv").read_text().byte_length()

    for index in range(document_count):
        total += (root / "texts" / (String(index) + ".txt")).read_text().byte_length()

    return total


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

    for upsert in request.documents:
        if upsert.document.vector_dim != collection.vector_dim:
            raise Error("upsert document vector_dim does not match collection manifest")

        var existing_index = find_document_index(documents, upsert.document.doc_id)
        var next_text = String()
        if existing_index != -1:
            next_text = texts[existing_index].copy()
        if upsert.has_text:
            next_text = upsert.text.copy()

        if existing_index == -1:
            documents.append(upsert.document.copy())
            texts.append(next_text^)
        else:
            documents[existing_index] = upsert.document.copy()
            texts[existing_index] = next_text^

    var final_document_count = len(documents)
    save_draft_collection_state(
        draft_state_root(collection_root),
        collection,
        DraftCollectionState(documents^, texts^),
    )
    return final_document_count


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
    save_draft_collection_state(
        draft_state_root(collection_root),
        collection,
        DraftCollectionState(kept_documents^, kept_texts^),
    )
    return kept_document_count


def parse_file_uri(source_uri: String) raises -> Path:
    if source_uri.find("file://") != 0:
        raise Error("snapshot import currently requires a file:// source_uri")

    return Path(source_uri.replace("file://", ""))


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
    var packed_index = pack_documents(draft_state.documents)
    var segment_id = SegmentId("segment-" + String(generation))
    var segment_root = collection_root / "segments" / segment_id.value
    var packed_index_root = segment_root / "packed_index"

    save_stored_packed_index(
        packed_index_root,
        StoredPackedIndex(
            "collection://" + collection.collection_id.value,
            collection.model_name.copy(),
            collection.vector_scalar_name.copy(),
            packed_index.copy(),
        ),
    )

    var byte_size = packed_index_storage_byte_size(packed_index_root)
    var text_corpus_root_name = ""
    if draft_state.has_any_text():
        text_corpus_root_name = "text_corpus"
        var doc_ids = List[String]()
        for document in draft_state.documents:
            doc_ids.append(document.doc_id.copy())

        save_stored_document_text_corpus(
            segment_root / text_corpus_root_name,
            StoredDocumentTextCorpus(
                collection.collection_id,
                segment_id.copy(),
                DocumentTextCorpus(doc_ids^, draft_state.texts.copy()),
            ),
        )
        byte_size += text_corpus_storage_byte_size(
            segment_root / text_corpus_root_name,
            len(draft_state.documents),
        )

    var segment_stats = SegmentStats(
        packed_index.document_count,
        packed_index.total_vector_count,
        packed_index.total_vector_count,
        byte_size,
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            segment_id.copy(),
            collection.collection_id,
            collection.tenant_id,
            collection.namespace_id,
            generation,
            collection.model_name.copy(),
            collection.vector_scalar_name.copy(),
            collection.vector_dim,
            "packed_index",
            text_corpus_root_name.copy(),
            segment_stats.copy(),
        ),
    )

    var snapshot = SnapshotManifest(
        request.snapshot_id,
        collection.collection_id,
        collection.tenant_id,
        collection.namespace_id,
        generation,
        [segment_id.copy()],
        CollectionStats(
            1,
            segment_stats.document_count,
            segment_stats.token_count,
            segment_stats.total_vector_count,
            segment_stats.byte_size,
        ),
    )
    save_snapshot_manifest(collection_root / "snapshots" / request.snapshot_id.value, snapshot)
    save_collection_manifest(
        collection_root,
        CollectionManifest(
            collection.collection_id,
            collection.tenant_id,
            collection.namespace_id,
            collection.model_name.copy(),
            collection.vector_scalar_name.copy(),
            collection.vector_dim,
            generation,
        ),
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
    require_filter_is_match_all(request.filter_expression)
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
    var snapshot = load_resolved_collection_snapshot(collection_root, request.snapshot_id)
    var explain = explain_collection_search(backend, request.query, snapshot, request.plan)
    return SearchResponse(
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
        request.snapshot_id,
        request.plan,
        final_hits_for_plan(explain.candidate_set, request.plan),
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
    require_filter_is_match_all(request.filter_expression)
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
    var snapshot = load_resolved_collection_snapshot(collection_root, request.snapshot_id)
    return ExplainResponse(
        explain_collection_search(backend, request.query, snapshot, request.plan)
    )
