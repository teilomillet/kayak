# Canonical snapshot-management requests for the hosted service boundary.

from kayak.collections import CollectionId, NamespaceId, SnapshotId, TenantId
from kayak.collections.validation import require_non_empty_string


struct CreateSnapshotRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var snapshot_id: SnapshotId
    var reason: String

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
        reason: String,
    ) raises:
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.snapshot_id = snapshot_id.copy()
        self.reason = require_non_empty_string(reason, "reason")


struct ExportSnapshotRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var snapshot_id: SnapshotId

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
    ):
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.snapshot_id = snapshot_id.copy()


struct ImportSnapshotRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var snapshot_id: SnapshotId
    var source_uri: String

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        snapshot_id: SnapshotId,
        source_uri: String,
    ) raises:
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.snapshot_id = snapshot_id.copy()
        self.source_uri = require_non_empty_string(source_uri, "source_uri")
