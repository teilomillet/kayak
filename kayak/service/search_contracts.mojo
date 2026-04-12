# Canonical search and explain contracts for the hosted service boundary.

from std.collections import List

from kayak.collections import CollectionId, NamespaceId, SnapshotId, TenantId
from kayak.collections.validation import require_positive_int
from kayak.contracts import EncodedQuery
from kayak.planning import (
    CollectionHit,
    CollectionSearchExplain,
    SearchPlan,
    exact_full_scan_search_plan,
)


struct SearchRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var snapshot_id: SnapshotId
    var query: EncodedQuery
    var plan: SearchPlan
    var debug_mode: Bool

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
        query: EncodedQuery,
        plan: SearchPlan,
        debug_mode: Bool,
    ) raises:
        _ = require_positive_int(plan.candidate_budget.final_k, "search final_k")
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.snapshot_id = snapshot_id.copy()
        self.query = query.copy()
        self.plan = plan.copy()
        self.debug_mode = debug_mode


def default_exact_search_request(
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    query: EncodedQuery,
    final_k: Int,
    debug_mode: Bool = False,
) raises -> SearchRequest:
    return SearchRequest(
        collection_id,
        tenant_id,
        namespace_id,
        snapshot_id,
        query,
        exact_full_scan_search_plan(final_k, final_k),
        debug_mode,
    )


struct SearchResponse(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var snapshot_id: SnapshotId
    var plan: SearchPlan
    var hits: List[CollectionHit]

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
        plan: SearchPlan,
        var hits: List[CollectionHit],
    ) raises:
        if len(hits) > plan.candidate_budget.final_k:
            raise Error("search hits exceed search plan final_k")

        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.snapshot_id = snapshot_id.copy()
        self.plan = plan.copy()
        self.hits = hits^


struct DebugSearchResponse(Copyable):
    var search: SearchResponse
    var explain: CollectionSearchExplain

    def __init__(
        out self, search: SearchResponse, explain: CollectionSearchExplain
    ) raises:
        if search.collection_id.value != explain.collection_id:
            raise Error("search collection_id does not match explain payload")

        if search.snapshot_id.value != explain.snapshot_id:
            raise Error("search snapshot_id does not match explain payload")

        self.search = search.copy()
        self.explain = explain.copy()


struct ExplainRequest(Copyable):
    var search: SearchRequest

    def __init__(out self, search: SearchRequest):
        self.search = search.copy()


struct ExplainResponse(Copyable):
    var explain: CollectionSearchExplain

    def __init__(out self, explain: CollectionSearchExplain):
        self.explain = explain.copy()
