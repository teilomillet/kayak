from std.pathlib import Path

from .ids import SnapshotId
from .resolved_snapshot import ResolvedCollectionSnapshot
from .resolver import load_resolved_collection_snapshot
from .stats import CollectionStats


struct CollectionStorageReport(Copyable):
    var stats: CollectionStats
    var segment_count_with_text: Int
    var segment_count_without_text: Int

    def __init__(
        out self,
        stats: CollectionStats,
        segment_count_with_text: Int,
        segment_count_without_text: Int,
    ) raises:
        if segment_count_with_text < 0:
            raise Error("segment_count_with_text must be non-negative")

        if segment_count_without_text < 0:
            raise Error("segment_count_without_text must be non-negative")

        if segment_count_with_text + segment_count_without_text != stats.segment_count:
            raise Error("text-bearing segment counts do not match stats.segment_count")

        self.stats = stats.copy()
        self.segment_count_with_text = segment_count_with_text
        self.segment_count_without_text = segment_count_without_text


def build_collection_storage_report(
    read resolved_snapshot: ResolvedCollectionSnapshot
) raises -> CollectionStorageReport:
    var segment_count_with_text = 0

    for segment in resolved_snapshot.segments:
        if segment.has_text_corpus:
            segment_count_with_text += 1

    return CollectionStorageReport(
        resolved_snapshot.snapshot.stats,
        segment_count_with_text,
        len(resolved_snapshot.segments) - segment_count_with_text,
    )


def load_collection_storage_report(
    collection_root: Path, snapshot_id: SnapshotId
) raises -> CollectionStorageReport:
    return build_collection_storage_report(
        load_resolved_collection_snapshot(collection_root, snapshot_id)
    )
