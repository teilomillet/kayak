# Explicit worker-local exact-search executor for repeated search on one
# published snapshot.
#
# This module owns the reusable exact execution seam specialized for exact
# full-scan dim128 late interaction when a flat persisted layout is available.
# It does not own transport concurrency, snapshot publication, or generic
# prepared-snapshot cache policy.

from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    CollectionManifest,
    NamespaceId,
    SealedSegmentManifest,
    SnapshotId,
    SnapshotManifest,
    SnapshotSearchArtifactAvailability,
    TenantId,
    load_collection_manifest,
    load_sealed_segment_manifest,
    load_snapshot_manifest,
    load_snapshot_search_artifact_availability,
)
from kayak.collections.paths import (
    collection_segment_root,
    collection_snapshot_root,
    resolve_segment_artifact_root,
)
from kayak.contracts import FlatQueryDim128, build_flat_query_dim128
from kayak.index import HybridFlatDim128Index
from kayak.planning import (
    CollectionHit,
    SearchPlanSelection,
    search_plan_selection_request_with_serving_scope,
    search_serving_scope_for_collection,
    select_search_plan_for_availability,
)
from kayak.planning.topk import insert_descending_collection_hit
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig
from kayak.search import search_exact_hybrid_flat_only_dim128_with_flat_query
from kayak.storage import (
    hybrid_flat_dim128_default_root,
    hybrid_flat_dim128_index_exists,
    load_stored_hybrid_flat_dim128_index,
    materialize_stored_hybrid_flat_dim128_index_from_packed_storage,
    packed_storage_supports_direct_hybrid_flat_dim128_materialization,
)

from .paths import service_collection_root
from .prepared_snapshot_runtime import (
    PreparedSearchSnapshot,
    execute_debug_search_with_prepared_snapshot,
    execute_explain_with_prepared_snapshot,
    execute_planned_debug_search_with_prepared_snapshot,
    execute_planned_explain_with_prepared_snapshot,
    execute_search_with_prepared_snapshot,
    prepare_collection_search_snapshot,
)
from .runtime import (
    load_collection_for_request,
    require_filter_supported_for_request,
    require_query_matches_collection,
    require_shared_pool_snapshot_filter_index_availability,
    search_request_for_planned_request,
    selection_for_planned_request,
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


def exact_cpu_backend_for_scoring_config(
    read scoring_config: ExactScoringConfig
) -> ExactCpuBackend:
    var copied = ExactScoringConfig()
    copied.enable_parallel_scoring = scoring_config.enable_parallel_scoring
    copied.enable_dim128_fast_path = scoring_config.enable_dim128_fast_path
    copied.enable_parallel_work_item_oversubscription = (
        scoring_config.enable_parallel_work_item_oversubscription
    )
    copied.parallel_work_item_count_override = (
        scoring_config.parallel_work_item_count_override
    )
    return ExactCpuBackend(copied^)


def require_collection_and_snapshot_match(
    read collection: CollectionManifest, read snapshot: SnapshotManifest
) raises:
    if snapshot.collection_id.value != collection.collection_id.value:
        raise Error("snapshot collection_id does not match collection manifest")

    if snapshot.tenant_id.value != collection.tenant_id.value:
        raise Error("snapshot tenant_id does not match collection manifest")

    if snapshot.namespace_id.value != collection.namespace_id.value:
        raise Error("snapshot namespace_id does not match collection manifest")

    if snapshot.generation > collection.latest_generation:
        raise Error("snapshot generation exceeds collection latest_generation")


def require_segment_matches_collection(
    read collection: CollectionManifest,
    read snapshot: SnapshotManifest,
    read segment: SealedSegmentManifest,
) raises:
    if segment.collection_id.value != collection.collection_id.value:
        raise Error("segment collection_id does not match collection manifest")

    if segment.tenant_id.value != collection.tenant_id.value:
        raise Error("segment tenant_id does not match collection manifest")

    if segment.namespace_id.value != collection.namespace_id.value:
        raise Error("segment namespace_id does not match collection manifest")

    if segment.model_name != collection.model_name:
        raise Error("segment model_name does not match collection manifest")

    if segment.vector_scalar_name != collection.vector_scalar_name:
        raise Error("segment vector_scalar_name does not match collection manifest")

    if segment.vector_dim != collection.vector_dim:
        raise Error("segment vector_dim does not match collection manifest")

    if segment.generation > snapshot.generation:
        raise Error("segment generation exceeds snapshot generation")


struct PreparedExactSegment(Copyable):
    var segment_id: String
    var index: HybridFlatDim128Index

    def __init__(
        out self,
        var segment_id: String,
        read index: HybridFlatDim128Index,
    ):
        self.segment_id = segment_id^
        self.index = index.copy()


struct PreparedExactSnapshot(Copyable):
    var collection_root: Path
    var collection: CollectionManifest
    var snapshot: SnapshotManifest
    var availability: SnapshotSearchArtifactAvailability
    var load_text_corpus: Bool
    var exact_full_scan_ready: Bool
    var segments: List[PreparedExactSegment]

    def __init__(
        out self,
        collection_root: Path,
        collection: CollectionManifest,
        snapshot: SnapshotManifest,
        availability: SnapshotSearchArtifactAvailability,
        load_text_corpus: Bool,
        exact_full_scan_ready: Bool,
        read segments: List[PreparedExactSegment],
    ):
        self.collection_root = collection_root.copy()
        self.collection = collection.copy()
        self.snapshot = snapshot.copy()
        self.availability = availability.copy()
        self.load_text_corpus = load_text_corpus
        self.exact_full_scan_ready = exact_full_scan_ready
        self.segments = segments.copy()


def prepare_collection_exact_snapshot(
    collection_root: Path,
    snapshot_id: SnapshotId,
    load_text_corpus: Bool = True,
) raises -> PreparedExactSnapshot:
    var collection = load_collection_manifest(collection_root)
    var snapshot = load_snapshot_manifest(
        collection_snapshot_root(collection_root, snapshot_id)
    )
    require_collection_and_snapshot_match(collection, snapshot)

    var availability = load_snapshot_search_artifact_availability(
        collection_root, snapshot_id
    )
    var segments = List[PreparedExactSegment]()
    var exact_full_scan_ready = collection.vector_dim == 128

    if exact_full_scan_ready:
        for segment_id in snapshot.segment_ids:
            var segment_root = collection_segment_root(collection_root, segment_id)
            var segment = load_sealed_segment_manifest(segment_root)
            require_segment_matches_collection(collection, snapshot, segment)

            var hybrid_root = hybrid_flat_dim128_default_root(segment_root)
            if not hybrid_flat_dim128_index_exists(hybrid_root):
                var packed_root = resolve_segment_artifact_root(
                    segment_root,
                    segment.packed_index_root,
                    "packed_index_root",
                )
                if not packed_storage_supports_direct_hybrid_flat_dim128_materialization(
                    packed_root
                ):
                    exact_full_scan_ready = False
                    break

                materialize_stored_hybrid_flat_dim128_index_from_packed_storage(
                    hybrid_root,
                    packed_root,
                )

            var loaded = load_stored_hybrid_flat_dim128_index(hybrid_root)
            segments.append(
                PreparedExactSegment(segment.segment_id.value.copy(), loaded.index)
            )

    return PreparedExactSnapshot(
        collection_root,
        collection,
        snapshot,
        availability,
        load_text_corpus,
        exact_full_scan_ready,
        segments,
    )


def require_prepared_exact_snapshot_matches_request(
    read prepared: PreparedExactSnapshot,
    read request: SearchRequest,
) raises:
    if prepared.collection.collection_id.value != request.collection_id.value:
        raise Error("prepared snapshot collection_id does not match search request")
    if prepared.collection.tenant_id.value != request.tenant_id.value:
        raise Error("prepared snapshot tenant_id does not match search request")
    if prepared.collection.namespace_id.value != request.namespace_id.value:
        raise Error("prepared snapshot namespace_id does not match search request")
    if prepared.snapshot.snapshot_id.value != request.snapshot_id.value:
        raise Error("prepared snapshot snapshot_id does not match search request")

    require_query_matches_collection(
        prepared.collection,
        request.query_model_name,
        request.query.vector_dim,
    )
    require_filter_supported_for_request(prepared.collection, request)


def prepared_exact_snapshot_supports_fast_search(
    read prepared: PreparedExactSnapshot,
    read request: SearchRequest,
) -> Bool:
    return (
        prepared.exact_full_scan_ready
        and request.query.vector_dim == 128
        and request.filter_expression.is_match_all()
        and request.plan.candidate_generator.kind == "exact_full_scan"
        and request.plan.stage2_reference_operator.kind == "noop_topk"
        and request.plan.stage3_verifier.kind == "none"
    )


def search_prepared_exact_snapshot_fast(
    read backend: ExactCpuBackend,
    read prepared: PreparedExactSnapshot,
    read query: FlatQueryDim128,
    final_k: Int,
) raises -> List[CollectionHit]:
    var final_hits = List[CollectionHit]()

    for segment in prepared.segments:
        var hits = search_exact_hybrid_flat_only_dim128_with_flat_query(
            query,
            segment.index,
            final_k,
            backend.scoring_config,
        )
        for hit in hits:
            insert_descending_collection_hit(
                final_hits,
                CollectionHit(
                    segment.segment_id.copy(),
                    hit.doc_id.copy(),
                    hit.score,
                ),
                final_k,
            )

    return final_hits^


def load_fallback_prepared_snapshot(
    read prepared: PreparedExactSnapshot
) raises -> PreparedSearchSnapshot:
    return prepare_collection_search_snapshot(
        prepared.collection_root,
        prepared.snapshot.snapshot_id,
        prepared.load_text_corpus,
    )


struct PreparedExactSearchExecutor(Movable):
    var prepared: PreparedExactSnapshot
    var backend: ExactCpuBackend

    def __init__(out self, read prepared: PreparedExactSnapshot):
        self.prepared = prepared.copy()
        self.backend = ExactCpuBackend()

    def __init__(
        out self,
        read prepared: PreparedExactSnapshot,
        read scoring_config: ExactScoringConfig,
    ):
        self.prepared = prepared.copy()
        self.backend = exact_cpu_backend_for_scoring_config(scoring_config)

    def execute_search(
        self, read request: SearchRequest
    ) raises -> SearchResponse:
        require_prepared_exact_snapshot_matches_request(self.prepared, request)
        if prepared_exact_snapshot_supports_fast_search(self.prepared, request):
            return SearchResponse(
                request.collection_id,
                request.tenant_id,
                request.namespace_id,
                request.snapshot_id,
                request.plan,
                search_prepared_exact_snapshot_fast(
                    self.backend,
                    self.prepared,
                    build_flat_query_dim128(request.query),
                    request.plan.candidate_budget.final_k,
                ),
            )

        return execute_search_with_prepared_snapshot(
            self.backend,
            load_fallback_prepared_snapshot(self.prepared),
            request,
        )

    def execute_explain(
        self, read request: SearchRequest
    ) raises -> ExplainResponse:
        require_prepared_exact_snapshot_matches_request(self.prepared, request)
        return execute_explain_with_prepared_snapshot(
            self.backend,
            load_fallback_prepared_snapshot(self.prepared),
            request,
        )

    def execute_debug_search(
        self, read request: SearchRequest
    ) raises -> DebugSearchResponse:
        return DebugSearchResponse(
            self.execute_search(request),
            self.execute_explain(request).explain,
        )

    def select_search_plan(
        self, read request: PlannedSearchRequest
    ) raises -> SearchPlanSelection:
        if self.prepared.collection.collection_id.value != request.collection_id.value:
            raise Error(
                "prepared snapshot collection_id does not match planned search request"
            )
        if self.prepared.collection.tenant_id.value != request.tenant_id.value:
            raise Error(
                "prepared snapshot tenant_id does not match planned search request"
            )
        if self.prepared.collection.namespace_id.value != request.namespace_id.value:
            raise Error(
                "prepared snapshot namespace_id does not match planned search request"
            )
        if self.prepared.snapshot.snapshot_id.value != request.snapshot_id.value:
            raise Error(
                "prepared snapshot snapshot_id does not match planned search request"
            )

        require_query_matches_collection(
            self.prepared.collection,
            request.query_model_name,
            request.query.vector_dim,
        )
        require_shared_pool_snapshot_filter_index_availability(
            self.prepared.collection,
            self.prepared.availability,
        )

        return selection_for_planned_request(
            request,
            select_search_plan_for_availability(
                self.prepared.availability,
                search_plan_selection_request_with_serving_scope(
                    request.planning,
                    search_serving_scope_for_collection(self.prepared.collection),
                ),
            ),
        )

    def execute_planned_search(
        self, read request: PlannedSearchRequest
    ) raises -> PlannedSearchResponse:
        var selection = self.select_search_plan(request)
        return PlannedSearchResponse(
            selection,
            self.execute_search(
                search_request_for_planned_request(request, selection.plan)
            ),
        )

    def execute_planned_explain(
        self, read request: PlannedSearchRequest
    ) raises -> PlannedExplainResponse:
        return execute_planned_explain_with_prepared_snapshot(
            self.backend,
            load_fallback_prepared_snapshot(self.prepared),
            request,
        )

    def execute_planned_debug_search(
        self, read request: PlannedSearchRequest
    ) raises -> PlannedDebugSearchResponse:
        return execute_planned_debug_search_with_prepared_snapshot(
            self.backend,
            load_fallback_prepared_snapshot(self.prepared),
            request,
        )


def prepare_collection_exact_search_executor(
    collection_root: Path,
    snapshot_id: SnapshotId,
    load_text_corpus: Bool = True,
) raises -> PreparedExactSearchExecutor:
    return PreparedExactSearchExecutor(
        prepare_collection_exact_snapshot(
            collection_root,
            snapshot_id,
            load_text_corpus,
        )
    )


def prepare_collection_exact_search_executor_with_config(
    collection_root: Path,
    snapshot_id: SnapshotId,
    read scoring_config: ExactScoringConfig,
    load_text_corpus: Bool = True,
) raises -> PreparedExactSearchExecutor:
    return PreparedExactSearchExecutor(
        prepare_collection_exact_snapshot(
            collection_root,
            snapshot_id,
            load_text_corpus,
        ),
        scoring_config,
    )


def prepare_service_exact_search_executor(
    service_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    load_text_corpus: Bool = True,
) raises -> PreparedExactSearchExecutor:
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
    return prepare_collection_exact_search_executor(
        collection_root,
        snapshot_id,
        load_text_corpus,
    )


def prepare_service_exact_search_executor_with_config(
    service_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    read scoring_config: ExactScoringConfig,
    load_text_corpus: Bool = True,
) raises -> PreparedExactSearchExecutor:
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
    return prepare_collection_exact_search_executor_with_config(
        collection_root,
        snapshot_id,
        scoring_config,
        load_text_corpus,
    )
