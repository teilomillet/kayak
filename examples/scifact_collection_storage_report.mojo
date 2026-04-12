from std.pathlib import Path

from kayak import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    collection_storage_report_json,
    ensure_one_segment_collection_mirror,
    load_collection_storage_report,
)
from kayak.storage import ensure_scifact_real_subset_cache


def ensure_scifact_collection_root() raises -> Path:
    var cache = ensure_scifact_real_subset_cache()
    return ensure_one_segment_collection_mirror(
        Path(".cache/kayak/scifact_real_subset_collection"),
        CollectionId("scifact_real_subset"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        cache.stored_index,
        0,
    )


def main() raises:
    var collection_root = ensure_scifact_collection_root()
    var report = load_collection_storage_report(
        collection_root, SnapshotId("snapshot-0001")
    )
    print(collection_storage_report_json(report))
