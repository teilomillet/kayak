# Explicit prepared search snapshots for repeated search on one published snapshot.
#
# This module owns the reusable loaded-snapshot seam. It does not own mutation,
# snapshot publication, or hidden cache policy.

from std.format import Writable, Writer
from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    NamespaceId,
    ResolvedCollectionSnapshot,
    SnapshotId,
    SnapshotSearchArtifactAvailability,
    TenantId,
    load_all_snapshot_requirements,
    load_resolved_collection_snapshot,
    load_snapshot_search_artifact_availability,
)
from kayak.planning import (
    SearchPlanSelection,
    search_plan_selection_request_with_serving_scope,
    search_serving_scope_for_collection,
    select_search_plan_for_availability,
)
from kayak.runtime import ExactScoringBackend

from .paths import service_collection_root
from .runtime import (
    load_collection_for_request,
    require_filter_supported_for_request,
    require_query_matches_collection,
    require_shared_pool_snapshot_filter_index_availability,
    search_request_for_planned_request,
    selection_for_planned_request,
    snapshot_load_requirements_for_request,
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
from kayak.planning import explain_collection_search, search_collection_for_plan


struct PreparedSearchSnapshot(Movable, Writable):
    var snapshot: ResolvedCollectionSnapshot
    var availability: SnapshotSearchArtifactAvailability
    var text_corpus_loaded: Bool

    def __init__(
        out self,
        snapshot: ResolvedCollectionSnapshot,
        availability: SnapshotSearchArtifactAvailability,
        text_corpus_loaded: Bool,
    ):
        self.snapshot = snapshot.copy()
        self.availability = availability.copy()
        self.text_corpus_loaded = text_corpus_loaded

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedSearchSnapshot(collection_id=",
            self.snapshot.collection.collection_id.value,
            ", snapshot_id=",
            self.snapshot.snapshot.snapshot_id.value,
            ", segment_count=",
            len(self.snapshot.segments),
            ", text_corpus_loaded=",
            self.text_corpus_loaded,
            ")",
        )


def prepare_collection_search_snapshot(
    collection_root: Path,
    snapshot_id: SnapshotId,
    load_text_corpus: Bool = True,
) raises -> PreparedSearchSnapshot:
    return PreparedSearchSnapshot(
        load_resolved_collection_snapshot(
            collection_root,
            snapshot_id,
            load_all_snapshot_requirements(load_text_corpus),
        ),
        load_snapshot_search_artifact_availability(collection_root, snapshot_id),
        load_text_corpus,
    )


def prepare_service_search_snapshot(
    service_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    load_text_corpus: Bool = True,
) raises -> PreparedSearchSnapshot:
    var collection_root = service_collection_root(
        service_root,
        tenant_id,
        namespace_id,
        collection_id,
    )
    _ = load_collection_for_request(
        service_root,
        collection_id.value,
        tenant_id.value,
        namespace_id.value,
        collection_root,
    )
    return prepare_collection_search_snapshot(
        collection_root,
        snapshot_id,
        load_text_corpus,
    )


def require_prepared_snapshot_matches_request(
    read prepared: PreparedSearchSnapshot,
    read request: SearchRequest,
) raises:
    if prepared.snapshot.collection.collection_id.value != request.collection_id.value:
        raise Error("prepared snapshot collection_id does not match search request")
    if prepared.snapshot.collection.tenant_id.value != request.tenant_id.value:
        raise Error("prepared snapshot tenant_id does not match search request")
    if prepared.snapshot.collection.namespace_id.value != request.namespace_id.value:
        raise Error("prepared snapshot namespace_id does not match search request")
    if prepared.snapshot.snapshot.snapshot_id.value != request.snapshot_id.value:
        raise Error("prepared snapshot snapshot_id does not match search request")

    require_query_matches_collection(
        prepared.snapshot.collection,
        request.query_model_name,
        request.query.vector_dim,
    )
    require_filter_supported_for_request(prepared.snapshot.collection, request)

    var requirements = snapshot_load_requirements_for_request(
        prepared.snapshot.collection,
        request,
    )
    if requirements.load_text_corpus and not prepared.text_corpus_loaded:
        raise Error(
            "prepared snapshot does not include text_corpus required by the search request"
        )


def execute_search_with_prepared_snapshot[Backend: ExactScoringBackend](
    read backend: Backend,
    read prepared: PreparedSearchSnapshot,
    read request: SearchRequest,
) raises -> SearchResponse:
    require_prepared_snapshot_matches_request(prepared, request)
    return SearchResponse(
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
        request.snapshot_id,
        request.plan,
        search_collection_for_plan(
            backend,
            request.query,
            prepared.snapshot,
            request.plan,
            request.filter_expression,
            request.query_text,
        ),
    )


def execute_explain_with_prepared_snapshot[Backend: ExactScoringBackend](
    read backend: Backend,
    read prepared: PreparedSearchSnapshot,
    read request: SearchRequest,
) raises -> ExplainResponse:
    require_prepared_snapshot_matches_request(prepared, request)
    return ExplainResponse(
        explain_collection_search(
            backend,
            request.query,
            prepared.snapshot,
            request.plan,
            request.filter_expression,
            request.query_text,
        )
    )


def execute_debug_search_with_prepared_snapshot[Backend: ExactScoringBackend](
    read backend: Backend,
    read prepared: PreparedSearchSnapshot,
    read request: SearchRequest,
) raises -> DebugSearchResponse:
    return DebugSearchResponse(
        execute_search_with_prepared_snapshot(backend, prepared, request),
        execute_explain_with_prepared_snapshot(backend, prepared, request).explain,
    )


def select_search_plan_for_prepared_snapshot(
    read prepared: PreparedSearchSnapshot,
    read request: PlannedSearchRequest,
) raises -> SearchPlanSelection:
    if prepared.snapshot.collection.collection_id.value != request.collection_id.value:
        raise Error(
            "prepared snapshot collection_id does not match planned search request"
        )
    if prepared.snapshot.collection.tenant_id.value != request.tenant_id.value:
        raise Error(
            "prepared snapshot tenant_id does not match planned search request"
        )
    if prepared.snapshot.collection.namespace_id.value != request.namespace_id.value:
        raise Error(
            "prepared snapshot namespace_id does not match planned search request"
        )
    if prepared.snapshot.snapshot.snapshot_id.value != request.snapshot_id.value:
        raise Error(
            "prepared snapshot snapshot_id does not match planned search request"
        )

    require_query_matches_collection(
        prepared.snapshot.collection,
        request.query_model_name,
        request.query.vector_dim,
    )
    require_shared_pool_snapshot_filter_index_availability(
        prepared.snapshot.collection,
        prepared.availability,
    )

    return selection_for_planned_request(
        request,
        select_search_plan_for_availability(
            prepared.availability,
            search_plan_selection_request_with_serving_scope(
                request.planning,
                search_serving_scope_for_collection(prepared.snapshot.collection),
            ),
        ),
    )


def execute_planned_search_with_prepared_snapshot[Backend: ExactScoringBackend](
    read backend: Backend,
    read prepared: PreparedSearchSnapshot,
    read request: PlannedSearchRequest,
) raises -> PlannedSearchResponse:
    var selection = select_search_plan_for_prepared_snapshot(prepared, request)
    var search = execute_search_with_prepared_snapshot(
        backend,
        prepared,
        search_request_for_planned_request(request, selection.plan),
    )
    return PlannedSearchResponse(selection, search)


def execute_planned_explain_with_prepared_snapshot[Backend: ExactScoringBackend](
    read backend: Backend,
    read prepared: PreparedSearchSnapshot,
    read request: PlannedSearchRequest,
) raises -> PlannedExplainResponse:
    var selection = select_search_plan_for_prepared_snapshot(prepared, request)
    var explain = execute_explain_with_prepared_snapshot(
        backend,
        prepared,
        search_request_for_planned_request(request, selection.plan),
    )
    return PlannedExplainResponse(selection, explain)


def execute_planned_debug_search_with_prepared_snapshot[Backend: ExactScoringBackend](
    read backend: Backend,
    read prepared: PreparedSearchSnapshot,
    read request: PlannedSearchRequest,
) raises -> PlannedDebugSearchResponse:
    var selection = select_search_plan_for_prepared_snapshot(prepared, request)
    var debug = execute_debug_search_with_prepared_snapshot(
        backend,
        prepared,
        search_request_for_planned_request(request, selection.plan),
    )
    return PlannedDebugSearchResponse(selection, debug)
