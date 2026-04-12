# Strongly named identifiers for hosted collection storage.

from .validation import require_non_empty_string


struct TenantId(Copyable):
    var value: String

    def __init__(out self, value: String) raises:
        self.value = require_non_empty_string(value, "tenant_id")


struct NamespaceId(Copyable):
    var value: String

    def __init__(out self, value: String) raises:
        self.value = require_non_empty_string(value, "namespace_id")


struct CollectionId(Copyable):
    var value: String

    def __init__(out self, value: String) raises:
        self.value = require_non_empty_string(value, "collection_id")


struct SegmentId(Copyable):
    var value: String

    def __init__(out self, value: String) raises:
        self.value = require_non_empty_string(value, "segment_id")


struct SnapshotId(Copyable):
    var value: String

    def __init__(out self, value: String) raises:
        self.value = require_non_empty_string(value, "snapshot_id")
