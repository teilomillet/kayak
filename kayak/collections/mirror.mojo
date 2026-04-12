from std.pathlib import Path

from kayak.storage import (
    StoredPackedIndex,
    centroid_postings_storage_byte_size,
    ensure_stored_centroid_posting_index,
    document_proxy_storage_byte_size,
    ensure_stored_document_proxy_index,
    save_stored_packed_index,
)

from .collection import CollectionManifest
from .collection_store import save_collection_manifest
from .ids import CollectionId, NamespaceId, SegmentId, SnapshotId, TenantId
from .segment import SealedSegmentManifest
from .segment_store import save_sealed_segment_manifest
from .snapshot import SnapshotManifest
from .snapshot_store import save_snapshot_manifest
from .stats import CollectionStats, SegmentStats


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())

    return path.read_text().byte_length()


def packed_index_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "doc_ids.tsv")
    total += file_size_bytes(root / "doc_offsets.tsv")

    if (root / "token_vectors.bin").exists():
        total += file_size_bytes(root / "token_vectors.bin")
    else:
        total += file_size_bytes(root / "token_vectors.tsv")

    return total


def ensure_one_segment_collection_mirror(
    collection_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    generation: Int,
    read stored_index: StoredPackedIndex,
    document_proxy_vector_budget: Int = 0,
    centroid_postings_vector_budget: Int = 0,
) raises -> Path:
    var segment_id = SegmentId("segment-0001")
    var segment_root = collection_root / "segments" / segment_id.value
    save_collection_manifest(
        collection_root,
        CollectionManifest(
            collection_id,
            tenant_id,
            namespace_id,
            stored_index.model_name.copy(),
            stored_index.vector_scalar_name.copy(),
            stored_index.index.vector_dim,
            generation,
        ),
    )
    save_stored_packed_index(segment_root / "packed_index", stored_index.copy())
    _ = ensure_stored_document_proxy_index(
        segment_root / "document_proxy",
        stored_index,
        document_proxy_vector_budget,
    )
    _ = ensure_stored_centroid_posting_index(
        segment_root / "centroid_postings",
        stored_index,
        centroid_postings_vector_budget,
    )

    var segment_stats = SegmentStats(
        stored_index.index.document_count,
        stored_index.index.total_vector_count,
        stored_index.index.total_vector_count,
        packed_index_storage_byte_size(segment_root / "packed_index")
            + centroid_postings_storage_byte_size(segment_root / "centroid_postings")
            + document_proxy_storage_byte_size(segment_root / "document_proxy"),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            segment_id.copy(),
            collection_id,
            tenant_id,
            namespace_id,
            generation,
            stored_index.model_name.copy(),
            stored_index.vector_scalar_name.copy(),
            stored_index.index.vector_dim,
            "packed_index",
            "centroid_postings",
            "document_proxy",
            "",
            segment_stats.copy(),
        ),
    )
    save_snapshot_manifest(
        collection_root / "snapshots" / snapshot_id.value,
        SnapshotManifest(
            snapshot_id,
            collection_id,
            tenant_id,
            namespace_id,
            generation,
            [segment_id.copy()],
            CollectionStats(
                1,
                segment_stats.document_count,
                segment_stats.token_count,
                segment_stats.total_vector_count,
                segment_stats.byte_size,
            ),
        ),
    )
    return collection_root
