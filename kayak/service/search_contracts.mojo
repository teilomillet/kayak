# Canonical search and explain contracts for the hosted service boundary.

from std.collections import List

from kayak.collections import CollectionId, NamespaceId, SnapshotId, TenantId
from kayak.collections.validation import require_non_empty_string, require_positive_int
from kayak.contracts import EncodedQuery
from kayak.filters import (
    FilterExpression,
    match_all_filter,
    require_user_visible_filter_expression,
)
from kayak.planning import (
    CollectionHit,
    CollectionSearchExplain,
    SearchPlan,
    SearchPlanSelection,
    SearchPlanSelectionRequest,
    exact_full_scan_search_plan,
)
from .planned_search_stage_override import (
    PlannedSearchStageOverride,
)


struct SearchRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var snapshot_id: SnapshotId
    var query: EncodedQuery
    var query_model_name: String
    var query_text: String
    var filter_expression: FilterExpression
    var plan: SearchPlan
    var debug_mode: Bool

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
        query: EncodedQuery,
        query_model_name: String,
        var query_text: String,
        filter_expression: FilterExpression,
        plan: SearchPlan,
        debug_mode: Bool,
    ) raises:
        _ = require_positive_int(plan.candidate_budget.final_k, "search final_k")
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.snapshot_id = snapshot_id.copy()
        self.query = query.copy()
        self.query_model_name = require_non_empty_string(
            query_model_name, "query_model_name"
        )
        self.query_text = query_text^
        require_user_visible_filter_expression(
            filter_expression,
            "search request filter_expression",
        )
        self.filter_expression = filter_expression.copy()
        self.plan = plan.copy()
        self.debug_mode = debug_mode

        if (
            self.plan.faithfulness_policy.kind == "oracle_full_recall_required"
            and not self.plan.candidate_generator.is_exact
            and not self.debug_mode
        ):
            raise Error(
                "oracle_full_recall_required faithfulness policy requires debug_mode for non-exact stage-1 search"
            )

        if self.plan.stage3_verifier.requires_query_text and self.query_text.byte_length() == 0:
            raise Error(
                "stage3 verifier "
                + self.plan.stage3_verifier.kind
                + " requires non-empty query_text"
            )


    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
        query: EncodedQuery,
        query_model_name: String,
        filter_expression: FilterExpression,
        plan: SearchPlan,
        debug_mode: Bool,
    ) raises:
        self = SearchRequest(
            collection_id,
            tenant_id,
            namespace_id,
            snapshot_id,
            query,
            query_model_name,
            "",
            filter_expression,
            plan,
            debug_mode,
        )


def default_exact_search_request(
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    query: EncodedQuery,
    final_k: Int,
    query_model_name: String,
    debug_mode: Bool = False,
    query_text: String = "",
) raises -> SearchRequest:
    return SearchRequest(
        collection_id,
        tenant_id,
        namespace_id,
        snapshot_id,
        query,
        query_model_name,
        query_text,
        match_all_filter(),
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


def same_search_plan(read left: SearchPlan, read right: SearchPlan) -> Bool:
    return (
        left.candidate_generator.kind == right.candidate_generator.kind
        and left.candidate_generator.cluster_top_k_per_query_token
        == right.candidate_generator.cluster_top_k_per_query_token
        and left.candidate_generator.beam_width
        == right.candidate_generator.beam_width
        and left.candidate_budget.final_k == right.candidate_budget.final_k
        and left.candidate_budget.candidate_k == right.candidate_budget.candidate_k
        and left.reference_scoring_semantics.kind
        == right.reference_scoring_semantics.kind
        and left.stage2_reference_operator.kind
        == right.stage2_reference_operator.kind
        and left.stage3_verifier.kind == right.stage3_verifier.kind
        and left.faithfulness_policy.kind == right.faithfulness_policy.kind
    )


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


struct PlannedSearchRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var snapshot_id: SnapshotId
    var query: EncodedQuery
    var query_model_name: String
    var query_text: String
    var stage_override: PlannedSearchStageOverride
    var filter_expression: FilterExpression
    var planning: SearchPlanSelectionRequest

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
        query: EncodedQuery,
        query_model_name: String,
        var query_text: String,
        var stage2_reference_kind: String,
        var stage3_verifier_kind: String,
        filter_expression: FilterExpression,
        planning: SearchPlanSelectionRequest,
    ) raises:
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.snapshot_id = snapshot_id.copy()
        self.query = query.copy()
        self.query_model_name = require_non_empty_string(
            query_model_name, "query_model_name"
        )
        self.query_text = query_text^
        self.stage_override = PlannedSearchStageOverride(
            self.query_text,
            stage2_reference_kind^,
            stage3_verifier_kind^,
        )
        require_user_visible_filter_expression(
            filter_expression,
            "planned search request filter_expression",
        )
        self.filter_expression = filter_expression.copy()
        self.planning = planning.copy()

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
        query: EncodedQuery,
        query_model_name: String,
        var query_text: String,
        filter_expression: FilterExpression,
        planning: SearchPlanSelectionRequest,
    ) raises:
        self = PlannedSearchRequest(
            collection_id,
            tenant_id,
            namespace_id,
            snapshot_id,
            query,
            query_model_name,
            query_text^,
            "",
            "",
            filter_expression,
            planning,
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
        query: EncodedQuery,
        query_model_name: String,
        filter_expression: FilterExpression,
        planning: SearchPlanSelectionRequest,
    ) raises:
        self = PlannedSearchRequest(
            collection_id,
            tenant_id,
            namespace_id,
            snapshot_id,
            query,
            query_model_name,
            "",
            "",
            "",
            filter_expression,
            planning,
        )


struct PlannedSearchResponse(Copyable):
    var selection: SearchPlanSelection
    var search: SearchResponse

    def __init__(
        out self, selection: SearchPlanSelection, search: SearchResponse
    ) raises:
        if not same_search_plan(selection.plan, search.plan):
            raise Error("planned search selection does not match search response plan")

        self.selection = selection.copy()
        self.search = search.copy()


struct PlannedDebugSearchResponse(Copyable):
    var selection: SearchPlanSelection
    var debug: DebugSearchResponse

    def __init__(
        out self, selection: SearchPlanSelection, debug: DebugSearchResponse
    ) raises:
        if not same_search_plan(selection.plan, debug.search.plan):
            raise Error(
                "planned search selection does not match debug search response plan"
            )

        self.selection = selection.copy()
        self.debug = debug.copy()


struct PlannedExplainResponse(Copyable):
    var selection: SearchPlanSelection
    var explain: ExplainResponse

    def __init__(
        out self, selection: SearchPlanSelection, explain: ExplainResponse
    ) raises:
        if not same_search_plan(selection.plan, explain.explain.plan):
            raise Error(
                "planned search selection does not match explain response plan"
            )

        self.selection = selection.copy()
        self.explain = explain.copy()
