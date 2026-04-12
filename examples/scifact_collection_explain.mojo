from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactCpuBackend,
    NamespaceId,
    SnapshotId,
    TenantId,
    collection_search_explain_json,
    ensure_one_segment_collection_mirror,
    exact_full_scan_search_plan,
    explain_collection_search,
    load_resolved_collection_snapshot,
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
    var cache = ensure_scifact_real_subset_cache()
    var resolved = load_resolved_collection_snapshot(
        collection_root, SnapshotId("snapshot-0001")
    )
    var plan = exact_full_scan_search_plan(
        cache.stored_task.task.k, cache.stored_task.task.k * 2
    )
    var explain = explain_collection_search(
        ExactCpuBackend(),
        cache.stored_task.task.queries[0].query,
        resolved,
        plan,
    )

    print(collection_search_explain_json(explain))
