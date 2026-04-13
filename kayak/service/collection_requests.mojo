# Canonical collection-management requests for the hosted service boundary.

from kayak.collections import (
    CollectionId,
    CollectionManifest,
    NamespaceId,
    TenantId,
)
from kayak.collections.validation import (
    require_non_empty_string,
    require_non_negative_int,
    require_positive_int,
)


struct CreateCollectionRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var model_name: String
    var vector_scalar_name: String
    var vector_dim: Int
    var default_keep_latest_inactive_count: Int

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
    ) raises:
        self = CreateCollectionRequest(
            collection_id,
            tenant_id,
            namespace_id,
            model_name,
            vector_scalar_name,
            vector_dim,
            1,
        )

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        default_keep_latest_inactive_count: Int,
    ) raises:
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.model_name = require_non_empty_string(model_name, "model_name")
        self.vector_scalar_name = require_non_empty_string(
            vector_scalar_name, "vector_scalar_name"
        )
        self.vector_dim = require_positive_int(vector_dim, "vector_dim")
        self.default_keep_latest_inactive_count = require_non_negative_int(
            default_keep_latest_inactive_count,
            "default_keep_latest_inactive_count",
        )

    def to_manifest(self) raises -> CollectionManifest:
        return CollectionManifest(
            self.collection_id,
            self.tenant_id,
            self.namespace_id,
            self.model_name,
            self.vector_scalar_name,
            self.vector_dim,
            0,
            self.default_keep_latest_inactive_count,
        )


struct UpdateCollectionRetentionPolicyRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var default_keep_latest_inactive_count: Int

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        default_keep_latest_inactive_count: Int,
    ) raises:
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.default_keep_latest_inactive_count = require_non_negative_int(
            default_keep_latest_inactive_count,
            "default_keep_latest_inactive_count",
        )
