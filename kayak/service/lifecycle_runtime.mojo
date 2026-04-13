from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    CollectionManifest,
    NamespaceId,
    CollectionReclaimPlan,
    SnapshotRetentionPolicy,
    TenantId,
    build_collection_reclaim_plan,
    execute_collection_reclaim_plan,
    save_collection_manifest,
)

from .collection_requests import UpdateCollectionRetentionPolicyRequest
from .draft_state import load_draft_state_metadata
from .lifecycle_contracts import (
    BuildReclaimPlanRequest,
    BuildReclaimPlanResponse,
    CollectionLifecycleRequest,
    CollectionLifecycleResponse,
    ExecuteReclaimRequest,
    ExecuteReclaimResponse,
    UpdateCollectionRetentionPolicyResponse,
)
from .paths import draft_state_root, service_collection_root
from .runtime import load_collection_for_request


def effective_retention_policy(
    read collection: CollectionManifest,
    has_policy_override: Bool,
    read policy_override: SnapshotRetentionPolicy,
) raises -> SnapshotRetentionPolicy:
    if has_policy_override:
        return policy_override.copy()

    return SnapshotRetentionPolicy(collection.default_keep_latest_inactive_count)


def collection_root_for_request(
    service_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
) -> Path:
    return service_collection_root(
        service_root,
        tenant_id,
        namespace_id,
        collection_id,
    )


def load_collection_for_lifecycle_request(
    service_root: Path,
    collection_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
) raises -> CollectionManifest:
    return load_collection_for_request(
        service_root,
        collection_id.value,
        tenant_id.value,
        namespace_id.value,
        collection_root,
    )


def build_collection_lifecycle_report(
    service_root: Path,
    read request: CollectionLifecycleRequest,
) raises -> CollectionLifecycleResponse:
    var collection_root = collection_root_for_request(
        service_root,
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
    )
    var collection = load_collection_for_lifecycle_request(
        service_root,
        collection_root,
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
    )
    var draft_metadata = load_draft_state_metadata(
        draft_state_root(collection_root),
        collection,
    )
    var policy = effective_retention_policy(
        collection, request.has_policy_override, request.policy_override
    )
    var reclaim_plan = build_collection_reclaim_plan(collection_root, policy)

    return CollectionLifecycleResponse(
        collection.collection_id,
        collection.tenant_id,
        collection.namespace_id,
        collection.model_name.copy(),
        collection.vector_scalar_name.copy(),
        collection.vector_dim,
        collection.latest_generation,
        collection.active_snapshot_id.copy(),
        collection.default_keep_latest_inactive_count,
        policy.keep_latest_inactive_count,
        policy.pinned_snapshot_ids,
        draft_metadata.document_count,
        draft_metadata.mutation_count,
        reclaim_plan,
    )


def build_reclaim_plan(
    service_root: Path,
    read request: BuildReclaimPlanRequest,
) raises -> BuildReclaimPlanResponse:
    var collection_root = collection_root_for_request(
        service_root,
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
    )
    var collection = load_collection_for_lifecycle_request(
        service_root,
        collection_root,
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
    )
    var policy = effective_retention_policy(
        collection, request.has_policy_override, request.policy_override
    )
    var plan = build_collection_reclaim_plan(collection_root, policy)
    return BuildReclaimPlanResponse(
        plan,
        policy.keep_latest_inactive_count,
        policy.pinned_snapshot_ids,
    )


def execute_reclaim(
    service_root: Path,
    read request: ExecuteReclaimRequest,
) raises -> ExecuteReclaimResponse:
    var collection_root = collection_root_for_request(
        service_root,
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
    )
    var collection = load_collection_for_lifecycle_request(
        service_root,
        collection_root,
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
    )
    if request.plan.collection_id.value != collection.collection_id.value:
        raise Error("execute reclaim plan collection_id does not match request")
    if request.plan.tenant_id.value != collection.tenant_id.value:
        raise Error("execute reclaim plan tenant_id does not match request")
    if request.plan.namespace_id.value != collection.namespace_id.value:
        raise Error("execute reclaim plan namespace_id does not match request")

    return ExecuteReclaimResponse(
        execute_collection_reclaim_plan(collection_root, request.plan, request.dry_run)
    )


def update_collection_retention_policy(
    service_root: Path,
    read request: UpdateCollectionRetentionPolicyRequest,
) raises -> UpdateCollectionRetentionPolicyResponse:
    var collection_root = collection_root_for_request(
        service_root,
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
    )
    var collection = load_collection_for_lifecycle_request(
        service_root,
        collection_root,
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
    )
    var updated = CollectionManifest(
        collection.collection_id,
        collection.tenant_id,
        collection.namespace_id,
        collection.model_name.copy(),
        collection.vector_scalar_name.copy(),
        collection.vector_dim,
        collection.latest_generation,
        collection.active_snapshot_id.copy(),
        request.default_keep_latest_inactive_count,
    )
    save_collection_manifest(collection_root, updated)
    return UpdateCollectionRetentionPolicyResponse(
        updated.collection_id,
        updated.tenant_id,
        updated.namespace_id,
        updated.latest_generation,
        updated.active_snapshot_id.copy(),
        updated.default_keep_latest_inactive_count,
    )
