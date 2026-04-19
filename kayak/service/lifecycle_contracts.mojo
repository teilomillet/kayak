from std.collections import List

from kayak.collections import (
    CollectionId,
    DocumentEncoderCompressionManifest,
    CollectionReclaimExecutionResult,
    CollectionReclaimPlan,
    NamespaceId,
    SearchArtifactBuildPolicy,
    SnapshotId,
    SnapshotRetentionPolicy,
    TenantId,
    default_document_encoder_compression_manifest,
    require_collection_layout_family_supported,
)
from kayak.collections.validation import (
    require_non_empty_string,
    require_non_negative_int,
)


struct CollectionLifecycleRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var has_policy_override: Bool
    var policy_override: SnapshotRetentionPolicy

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
    ) raises:
        self = CollectionLifecycleRequest(
            collection_id,
            tenant_id,
            namespace_id,
            SnapshotRetentionPolicy(0),
            False,
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        read policy_override: SnapshotRetentionPolicy,
    ) raises:
        self = CollectionLifecycleRequest(
            collection_id,
            tenant_id,
            namespace_id,
            policy_override,
            True,
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        read policy_override: SnapshotRetentionPolicy,
        has_policy_override: Bool,
    ):
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.has_policy_override = has_policy_override
        self.policy_override = policy_override.copy()


struct CollectionLifecycleResponse(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var collection_layout_family: String
    var model_name: String
    var vector_scalar_name: String
    var vector_dim: Int
    var latest_generation: Int
    var active_snapshot_id: String
    var default_keep_latest_inactive_count: Int
    var search_artifact_build_policy: SearchArtifactBuildPolicy
    var document_encoder_compression: DocumentEncoderCompressionManifest
    var effective_keep_latest_inactive_count: Int
    var effective_pinned_snapshot_ids: List[SnapshotId]
    var draft_document_count: Int
    var pending_draft_mutation_count: Int
    var reclaim_plan: CollectionReclaimPlan

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        collection_layout_family: String,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
        active_snapshot_id: String,
        default_keep_latest_inactive_count: Int,
        read search_artifact_build_policy: SearchArtifactBuildPolicy,
        read document_encoder_compression: DocumentEncoderCompressionManifest,
        effective_keep_latest_inactive_count: Int,
        read effective_pinned_snapshot_ids: List[SnapshotId],
        draft_document_count: Int,
        pending_draft_mutation_count: Int,
        read reclaim_plan: CollectionReclaimPlan,
    ) raises:
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.collection_layout_family = require_collection_layout_family_supported(
            collection_layout_family
        )
        self.model_name = require_non_empty_string(model_name, "model_name")
        self.vector_scalar_name = require_non_empty_string(
            vector_scalar_name, "vector_scalar_name"
        )
        self.vector_dim = require_non_negative_int(vector_dim, "vector_dim")
        self.latest_generation = require_non_negative_int(
            latest_generation, "latest_generation"
        )
        self.active_snapshot_id = active_snapshot_id.copy()
        self.default_keep_latest_inactive_count = require_non_negative_int(
            default_keep_latest_inactive_count,
            "default_keep_latest_inactive_count",
        )
        self.search_artifact_build_policy = search_artifact_build_policy.copy()
        self.document_encoder_compression = document_encoder_compression.copy()
        self.effective_keep_latest_inactive_count = require_non_negative_int(
            effective_keep_latest_inactive_count,
            "effective_keep_latest_inactive_count",
        )
        self.effective_pinned_snapshot_ids = effective_pinned_snapshot_ids.copy()
        self.draft_document_count = require_non_negative_int(
            draft_document_count, "draft_document_count"
        )
        self.pending_draft_mutation_count = require_non_negative_int(
            pending_draft_mutation_count, "pending_draft_mutation_count"
        )
        self.reclaim_plan = reclaim_plan.copy()

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        collection_layout_family: String,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
        active_snapshot_id: String,
        default_keep_latest_inactive_count: Int,
        read search_artifact_build_policy: SearchArtifactBuildPolicy,
        effective_keep_latest_inactive_count: Int,
        read effective_pinned_snapshot_ids: List[SnapshotId],
        draft_document_count: Int,
        pending_draft_mutation_count: Int,
        read reclaim_plan: CollectionReclaimPlan,
    ) raises:
        self = CollectionLifecycleResponse(
            collection_id,
            tenant_id,
            namespace_id,
            collection_layout_family,
            model_name,
            vector_scalar_name,
            vector_dim,
            latest_generation,
            active_snapshot_id,
            default_keep_latest_inactive_count,
            search_artifact_build_policy,
            default_document_encoder_compression_manifest(),
            effective_keep_latest_inactive_count,
            effective_pinned_snapshot_ids,
            draft_document_count,
            pending_draft_mutation_count,
            reclaim_plan,
        )


struct BuildReclaimPlanRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var has_policy_override: Bool
    var policy_override: SnapshotRetentionPolicy

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
    ) raises:
        self = BuildReclaimPlanRequest(
            collection_id,
            tenant_id,
            namespace_id,
            SnapshotRetentionPolicy(0),
            False,
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        read policy_override: SnapshotRetentionPolicy,
    ) raises:
        self = BuildReclaimPlanRequest(
            collection_id,
            tenant_id,
            namespace_id,
            policy_override,
            True,
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        read policy_override: SnapshotRetentionPolicy,
        has_policy_override: Bool,
    ):
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.has_policy_override = has_policy_override
        self.policy_override = policy_override.copy()


struct BuildReclaimPlanResponse(Copyable):
    var plan: CollectionReclaimPlan
    var effective_keep_latest_inactive_count: Int
    var effective_pinned_snapshot_ids: List[SnapshotId]

    def __init__(
        out self,
        read plan: CollectionReclaimPlan,
        effective_keep_latest_inactive_count: Int,
        read effective_pinned_snapshot_ids: List[SnapshotId],
    ) raises:
        self.plan = plan.copy()
        self.effective_keep_latest_inactive_count = require_non_negative_int(
            effective_keep_latest_inactive_count,
            "effective_keep_latest_inactive_count",
        )
        self.effective_pinned_snapshot_ids = effective_pinned_snapshot_ids.copy()


struct ExecuteReclaimRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var plan: CollectionReclaimPlan
    var dry_run: Bool

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        read plan: CollectionReclaimPlan,
        dry_run: Bool = True,
    ):
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.plan = plan.copy()
        self.dry_run = dry_run


struct ExecuteReclaimResponse(Copyable):
    var result: CollectionReclaimExecutionResult

    def __init__(out self, read result: CollectionReclaimExecutionResult):
        self.result = result.copy()


struct UpdateCollectionRetentionPolicyResponse(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var latest_generation: Int
    var active_snapshot_id: String
    var default_keep_latest_inactive_count: Int

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        latest_generation: Int,
        active_snapshot_id: String,
        default_keep_latest_inactive_count: Int,
    ) raises:
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.latest_generation = require_non_negative_int(
            latest_generation, "latest_generation"
        )
        self.active_snapshot_id = active_snapshot_id.copy()
        self.default_keep_latest_inactive_count = require_non_negative_int(
            default_keep_latest_inactive_count,
            "default_keep_latest_inactive_count",
        )
