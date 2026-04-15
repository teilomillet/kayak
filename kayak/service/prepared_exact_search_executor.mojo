# Explicit worker-local exact-search executor for repeated search on one
# published snapshot.
#
# This module owns the reusable exact execution seam. It does not own transport
# concurrency, snapshot cache policy, or query planning policy.

from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig

from .prepared_snapshot_runtime import (
    PreparedSearchSnapshot,
    execute_debug_search_with_prepared_snapshot,
    execute_explain_with_prepared_snapshot,
    execute_planned_debug_search_with_prepared_snapshot,
    execute_planned_explain_with_prepared_snapshot,
    execute_planned_search_with_prepared_snapshot,
    execute_search_with_prepared_snapshot,
    prepare_collection_search_snapshot,
    prepare_service_search_snapshot,
    select_search_plan_for_prepared_snapshot,
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
from kayak.planning import SearchPlanSelection


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


struct PreparedExactSearchExecutor(Movable):
    var prepared: PreparedSearchSnapshot
    var backend: ExactCpuBackend

    def __init__(out self, var prepared: PreparedSearchSnapshot):
        self.prepared = prepared^
        self.backend = ExactCpuBackend()

    def __init__(
        out self,
        var prepared: PreparedSearchSnapshot,
        read scoring_config: ExactScoringConfig,
    ):
        self.prepared = prepared^
        self.backend = exact_cpu_backend_for_scoring_config(scoring_config)

    def execute_search(
        self, read request: SearchRequest
    ) raises -> SearchResponse:
        return execute_search_with_prepared_snapshot(
            self.backend,
            self.prepared,
            request,
        )

    def execute_explain(
        self, read request: SearchRequest
    ) raises -> ExplainResponse:
        return execute_explain_with_prepared_snapshot(
            self.backend,
            self.prepared,
            request,
        )

    def execute_debug_search(
        self, read request: SearchRequest
    ) raises -> DebugSearchResponse:
        return execute_debug_search_with_prepared_snapshot(
            self.backend,
            self.prepared,
            request,
        )

    def select_search_plan(
        self, read request: PlannedSearchRequest
    ) raises -> SearchPlanSelection:
        return select_search_plan_for_prepared_snapshot(self.prepared, request)

    def execute_planned_search(
        self, read request: PlannedSearchRequest
    ) raises -> PlannedSearchResponse:
        return execute_planned_search_with_prepared_snapshot(
            self.backend,
            self.prepared,
            request,
        )

    def execute_planned_explain(
        self, read request: PlannedSearchRequest
    ) raises -> PlannedExplainResponse:
        return execute_planned_explain_with_prepared_snapshot(
            self.backend,
            self.prepared,
            request,
        )

    def execute_planned_debug_search(
        self, read request: PlannedSearchRequest
    ) raises -> PlannedDebugSearchResponse:
        return execute_planned_debug_search_with_prepared_snapshot(
            self.backend,
            self.prepared,
            request,
        )


def prepare_collection_exact_search_executor(
    collection_root: Path,
    snapshot_id: SnapshotId,
    load_text_corpus: Bool = True,
) raises -> PreparedExactSearchExecutor:
    return PreparedExactSearchExecutor(
        prepare_collection_search_snapshot(
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
        prepare_collection_search_snapshot(
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
    return PreparedExactSearchExecutor(
        prepare_service_search_snapshot(
            service_root,
            collection_id,
            tenant_id,
            namespace_id,
            snapshot_id,
            load_text_corpus,
        )
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
    return PreparedExactSearchExecutor(
        prepare_service_search_snapshot(
            service_root,
            collection_id,
            tenant_id,
            namespace_id,
            snapshot_id,
            load_text_corpus,
        ),
        scoring_config,
    )
