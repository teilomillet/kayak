from std.pathlib import Path

from kayak import (
    CollectionId,
    CollectionManifest,
    CollectionStats,
    ExactCpuBackend,
    NamespaceId,
    SegmentId,
    SegmentStats,
    SealedSegmentManifest,
    SnapshotId,
    SnapshotManifest,
    TenantId,
    collection_manifest_exists,
    collection_search_explain_json,
    exact_full_scan_search_plan,
    explain_collection_search,
    load_resolved_collection_snapshot,
    save_collection_manifest,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
    save_stored_packed_index,
)
from kayak.storage import ensure_scifact_real_subset_cache


def ensure_scifact_collection_root() raises -> Path:
    var collection_root = Path(".cache/kayak/scifact_real_subset_collection")
    if collection_manifest_exists(collection_root):
        return collection_root

    print("building one-segment SciFact collection mirror...")
    var cache = ensure_scifact_real_subset_cache()
    var stored_index = cache.stored_index.copy()
    var segment_root = collection_root / "segments" / "segment-0001"

    save_collection_manifest(
        collection_root,
        CollectionManifest(
            CollectionId("scifact_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            stored_index.model_name.copy(),
            stored_index.vector_scalar_name.copy(),
            stored_index.index.vector_dim,
            1,
        ),
    )
    save_stored_packed_index(segment_root / "packed_index", stored_index.copy())

    var segment_stats = SegmentStats(
        stored_index.index.document_count,
        stored_index.index.total_vector_count,
        stored_index.index.total_vector_count,
        (segment_root / "packed_index" / "manifest.tsv").read_text().byte_length()
            + (segment_root / "packed_index" / "doc_ids.tsv").read_text().byte_length()
            + (segment_root / "packed_index" / "doc_offsets.tsv").read_text().byte_length()
            + len((segment_root / "packed_index" / "token_vectors.bin").read_bytes()),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("scifact_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            1,
            stored_index.model_name.copy(),
            stored_index.vector_scalar_name.copy(),
            stored_index.index.vector_dim,
            "packed_index",
            "",
            segment_stats.copy(),
        ),
    )
    save_snapshot_manifest(
        collection_root / "snapshots" / "snapshot-0001",
        SnapshotManifest(
            SnapshotId("snapshot-0001"),
            CollectionId("scifact_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            1,
            [SegmentId("segment-0001")],
            CollectionStats(
                1,
                segment_stats.document_count,
                segment_stats.token_count,
                segment_stats.total_vector_count,
                segment_stats.byte_size,
            ),
        ),
    )
    return collection_root^


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
