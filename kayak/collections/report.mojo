from std.collections import List
from std.pathlib import Path

from .density import StorageDensity
from .ids import SnapshotId
from .resolved_snapshot import ResolvedCollectionSnapshot
from .resolver import load_resolved_collection_snapshot
from .segment_report import SegmentStorageReport, build_segment_storage_report
from .stats import CollectionStats


struct CollectionStorageReport(Copyable):
    var stats: CollectionStats
    var density: StorageDensity
    var segment_count_with_text: Int
    var segment_count_without_text: Int
    var segment_reports: List[SegmentStorageReport]

    def __init__(
        out self,
        stats: CollectionStats,
        segment_count_with_text: Int,
        segment_count_without_text: Int,
        read segment_reports: List[SegmentStorageReport],
    ) raises:
        if segment_count_with_text < 0:
            raise Error("segment_count_with_text must be non-negative")

        if segment_count_without_text < 0:
            raise Error("segment_count_without_text must be non-negative")

        if segment_count_with_text + segment_count_without_text != stats.segment_count:
            raise Error("text-bearing segment counts do not match stats.segment_count")

        if len(segment_reports) != stats.segment_count:
            raise Error("segment_reports length does not match stats.segment_count")

        self.stats = stats.copy()
        self.density = StorageDensity(
            stats.byte_size,
            stats.document_count,
            stats.token_count,
            stats.total_vector_count,
        )
        self.segment_count_with_text = segment_count_with_text
        self.segment_count_without_text = segment_count_without_text
        self.segment_reports = segment_reports.copy()


def build_collection_storage_report(
    read resolved_snapshot: ResolvedCollectionSnapshot
) raises -> CollectionStorageReport:
    var segment_count_with_text = 0
    var segment_reports = List[SegmentStorageReport]()

    for segment in resolved_snapshot.segments:
        if segment.has_text_corpus:
            segment_count_with_text += 1
        segment_reports.append(build_segment_storage_report(segment))

    return CollectionStorageReport(
        resolved_snapshot.snapshot.stats,
        segment_count_with_text,
        len(resolved_snapshot.segments) - segment_count_with_text,
        segment_reports,
    )


def load_collection_storage_report(
    collection_root: Path, snapshot_id: SnapshotId
) raises -> CollectionStorageReport:
    return build_collection_storage_report(
        load_resolved_collection_snapshot(collection_root, snapshot_id)
    )
