# Collection-level manifest for hosted late-interaction data.

from .ids import CollectionId, NamespaceId, TenantId
from .search_artifact_builders import (
    require_search_artifact_build_policy_supported_for_segment_sealing,
)
from .search_artifact_policy import (
    SearchArtifactBuildPolicy,
    default_search_artifact_build_policy,
    require_search_artifact_build_policy_layout_safe,
)
from .validation import require_non_empty_string, require_non_negative_int, require_positive_int


struct CollectionManifest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var model_name: String
    var vector_scalar_name: String
    var vector_dim: Int
    var latest_generation: Int
    var active_snapshot_id: String
    var default_keep_latest_inactive_count: Int
    var search_artifact_build_policy: SearchArtifactBuildPolicy

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
    ) raises:
        self = CollectionManifest(
            collection_id,
            tenant_id,
            namespace_id,
            model_name,
            vector_scalar_name,
            vector_dim,
            latest_generation,
            "",
            1,
            default_search_artifact_build_policy(),
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
        default_keep_latest_inactive_count: Int,
    ) raises:
        self = CollectionManifest(
            collection_id,
            tenant_id,
            namespace_id,
            model_name,
            vector_scalar_name,
            vector_dim,
            latest_generation,
            "",
            default_keep_latest_inactive_count,
            default_search_artifact_build_policy(),
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
        active_snapshot_id: String,
    ) raises:
        self = CollectionManifest(
            collection_id,
            tenant_id,
            namespace_id,
            model_name,
            vector_scalar_name,
            vector_dim,
            latest_generation,
            active_snapshot_id,
            1,
            default_search_artifact_build_policy(),
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
        read search_artifact_build_policy: SearchArtifactBuildPolicy,
    ) raises:
        self = CollectionManifest(
            collection_id,
            tenant_id,
            namespace_id,
            model_name,
            vector_scalar_name,
            vector_dim,
            latest_generation,
            "",
            1,
            search_artifact_build_policy,
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
        default_keep_latest_inactive_count: Int,
        read search_artifact_build_policy: SearchArtifactBuildPolicy,
    ) raises:
        self = CollectionManifest(
            collection_id,
            tenant_id,
            namespace_id,
            model_name,
            vector_scalar_name,
            vector_dim,
            latest_generation,
            "",
            default_keep_latest_inactive_count,
            search_artifact_build_policy,
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
        active_snapshot_id: String,
        default_keep_latest_inactive_count: Int,
    ) raises:
        self = CollectionManifest(
            collection_id,
            tenant_id,
            namespace_id,
            model_name,
            vector_scalar_name,
            vector_dim,
            latest_generation,
            active_snapshot_id,
            default_keep_latest_inactive_count,
            default_search_artifact_build_policy(),
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
        active_snapshot_id: String,
        read search_artifact_build_policy: SearchArtifactBuildPolicy,
    ) raises:
        self = CollectionManifest(
            collection_id,
            tenant_id,
            namespace_id,
            model_name,
            vector_scalar_name,
            vector_dim,
            latest_generation,
            active_snapshot_id,
            1,
            search_artifact_build_policy,
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        latest_generation: Int,
        active_snapshot_id: String,
        default_keep_latest_inactive_count: Int,
        read search_artifact_build_policy: SearchArtifactBuildPolicy,
    ) raises:
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.model_name = require_non_empty_string(model_name, "model_name")
        self.vector_scalar_name = require_non_empty_string(
            vector_scalar_name, "vector_scalar_name"
        )
        self.vector_dim = require_positive_int(vector_dim, "vector_dim")
        self.latest_generation = require_non_negative_int(
            latest_generation, "latest_generation"
        )
        self.active_snapshot_id = active_snapshot_id.copy()
        self.default_keep_latest_inactive_count = require_non_negative_int(
            default_keep_latest_inactive_count,
            "default_keep_latest_inactive_count",
        )
        self.search_artifact_build_policy = search_artifact_build_policy.copy()
        require_search_artifact_build_policy_layout_safe(
            self.search_artifact_build_policy
        )
        require_search_artifact_build_policy_supported_for_segment_sealing(
            self.search_artifact_build_policy
        )
        if self.active_snapshot_id.byte_length() != 0 and self.latest_generation == 0:
            raise Error(
                "active_snapshot_id requires a positive latest_generation"
            )

    def has_active_snapshot(self) -> Bool:
        return self.active_snapshot_id.byte_length() != 0
