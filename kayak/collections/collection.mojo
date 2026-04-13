# Collection-level manifest for hosted late-interaction data.

from .ids import CollectionId, NamespaceId, TenantId
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
        if self.active_snapshot_id.byte_length() != 0 and self.latest_generation == 0:
            raise Error(
                "active_snapshot_id requires a positive latest_generation"
            )

    def has_active_snapshot(self) -> Bool:
        return self.active_snapshot_id.byte_length() != 0
