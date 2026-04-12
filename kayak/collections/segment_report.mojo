from .density import StorageDensity
from .resolved_snapshot import LoadedSealedSegment
from .stats import SegmentStats
from .validation import require_non_empty_string


struct SegmentStorageReport(Copyable):
    var segment_id: String
    var has_text_corpus: Bool
    var stats: SegmentStats
    var density: StorageDensity

    def __init__(
        out self,
        segment_id: String,
        has_text_corpus: Bool,
        stats: SegmentStats,
    ) raises:
        self.segment_id = require_non_empty_string(segment_id, "segment_id")
        self.has_text_corpus = has_text_corpus
        self.stats = stats.copy()
        self.density = StorageDensity(
            stats.byte_size,
            stats.document_count,
            stats.token_count,
            stats.total_vector_count,
        )


def build_segment_storage_report(
    read segment: LoadedSealedSegment
) raises -> SegmentStorageReport:
    return SegmentStorageReport(
        segment.manifest.segment_id.value,
        segment.has_text_corpus,
        segment.manifest.stats,
    )
