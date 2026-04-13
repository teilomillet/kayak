from std.pathlib import Path

from .ids import SegmentId, SnapshotId
from .validation import require_non_empty_string


def collection_manifest_path(root: Path) -> Path:
    return root / "collection.manifest.tsv"


def collection_segments_root(root: Path) -> Path:
    return root / "segments"


def collection_snapshots_root(root: Path) -> Path:
    return root / "snapshots"


def collection_segment_root(root: Path, read segment_id: SegmentId) -> Path:
    return collection_segments_root(root) / segment_id.value


def collection_snapshot_root(root: Path, read snapshot_id: SnapshotId) -> Path:
    return collection_snapshots_root(root) / snapshot_id.value


def segment_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def snapshot_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def snapshot_segment_ids_path(root: Path) -> Path:
    return root / "segment_ids.tsv"


def text_corpus_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def text_corpus_entries_path(root: Path) -> Path:
    return root / "entries.tsv"


def text_corpus_payload_root(root: Path) -> Path:
    return root / "texts"


def document_metadata_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def document_metadata_entries_path(root: Path) -> Path:
    return root / "entries.tsv"


def document_metadata_payload_root(root: Path) -> Path:
    return root / "metadata"


def document_filter_index_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def document_filter_index_entries_path(root: Path) -> Path:
    return root / "entries.tsv"


def require_relative_artifact_root(
    artifact_root: String, field_name: String
) raises -> String:
    var normalized = require_non_empty_string(artifact_root, field_name)

    if normalized.find("/") == 0:
        raise Error(field_name + " must be relative, not absolute")

    if normalized.find("..") != -1:
        raise Error(field_name + " must not contain parent-directory traversal")

    if normalized.find("\\") != -1:
        raise Error(field_name + " must use forward slashes only")

    if normalized.find(":") != -1:
        raise Error(field_name + " must not contain drive or URI separators")

    return normalized^


def resolve_segment_artifact_root(
    segment_root: Path, artifact_root: String, field_name: String
) raises -> Path:
    return segment_root / require_relative_artifact_root(artifact_root, field_name)
