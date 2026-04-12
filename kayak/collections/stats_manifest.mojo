from std.collections import List

from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import parse_int

from .stats import CollectionStats, SegmentStats


def segment_stats_manifest_entries(
    read stats: SegmentStats
) -> List[ManifestEntry]:
    var entries = List[ManifestEntry]()
    entries.append(ManifestEntry("document_count", String(stats.document_count)))
    entries.append(ManifestEntry("token_count", String(stats.token_count)))
    entries.append(
        ManifestEntry("total_vector_count", String(stats.total_vector_count))
    )
    entries.append(ManifestEntry("byte_size", String(stats.byte_size)))
    return entries^


def collection_stats_manifest_entries(
    read stats: CollectionStats
) -> List[ManifestEntry]:
    var entries = List[ManifestEntry]()
    entries.append(ManifestEntry("segment_count", String(stats.segment_count)))
    entries.append(ManifestEntry("document_count", String(stats.document_count)))
    entries.append(ManifestEntry("token_count", String(stats.token_count)))
    entries.append(
        ManifestEntry("total_vector_count", String(stats.total_vector_count))
    )
    entries.append(ManifestEntry("byte_size", String(stats.byte_size)))
    return entries^


def load_segment_stats_from_manifest(
    entries: List[ManifestEntry]
) raises -> SegmentStats:
    return SegmentStats(
        parse_int(
            require_manifest_value(entries, "document_count"),
            "segment document_count",
        ),
        parse_int(
            require_manifest_value(entries, "token_count"), "segment token_count"
        ),
        parse_int(
            require_manifest_value(entries, "total_vector_count"),
            "segment total_vector_count",
        ),
        parse_int(require_manifest_value(entries, "byte_size"), "segment byte_size"),
    )


def load_collection_stats_from_manifest(
    entries: List[ManifestEntry]
) raises -> CollectionStats:
    return CollectionStats(
        parse_int(
            require_manifest_value(entries, "segment_count"),
            "collection segment_count",
        ),
        parse_int(
            require_manifest_value(entries, "document_count"),
            "collection document_count",
        ),
        parse_int(
            require_manifest_value(entries, "token_count"), "collection token_count"
        ),
        parse_int(
            require_manifest_value(entries, "total_vector_count"),
            "collection total_vector_count",
        ),
        parse_int(
            require_manifest_value(entries, "byte_size"), "collection byte_size"
        ),
    )
