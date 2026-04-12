# Export manifest for a self-contained snapshot bundle.

from .ids import CollectionId, NamespaceId, SnapshotId, TenantId
from .validation import require_non_negative_int


struct SnapshotExportBundleManifest(Copyable):
    var snapshot_id: SnapshotId
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var generation: Int
    var segment_count: Int

    def __init__(
        out self,
        snapshot_id: SnapshotId,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        generation: Int,
        segment_count: Int,
    ) raises:
        self.snapshot_id = snapshot_id.copy()
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.generation = require_non_negative_int(generation, "generation")
        self.segment_count = require_non_negative_int(
            segment_count, "segment_count"
        )
