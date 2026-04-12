from std.collections import List
from std.pathlib import Path

from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.storage.manifest import (
    ManifestEntry,
    read_manifest,
    require_manifest_value,
    write_manifest,
)
from kayak.storage.text_codec import parse_int

comptime COLLECTION_ARTIFACT_FORMAT_VERSION = 1


def write_collection_artifact_manifest(
    path: Path, artifact_kind: String, read artifact_entries: List[ManifestEntry]
) raises:
    var entries = List[ManifestEntry]()
    entries.append(
        ManifestEntry("format_version", String(COLLECTION_ARTIFACT_FORMAT_VERSION))
    )
    entries.append(ManifestEntry("artifact_kind", artifact_kind.copy()))

    for entry in artifact_entries:
        entries.append(ManifestEntry(entry.key.copy(), entry.value.copy()))

    write_manifest(path, entries)


def read_collection_artifact_manifest(
    path: Path, expected_artifact_kind: String
) raises -> List[ManifestEntry]:
    var entries = read_manifest(path)
    var format_version = parse_int(
        require_manifest_value(entries, "format_version"),
        "collection artifact format_version",
    )
    if format_version != COLLECTION_ARTIFACT_FORMAT_VERSION:
        raise Error("unsupported collection artifact format version")

    if require_manifest_value(entries, "artifact_kind") != expected_artifact_kind:
        raise Error("collection artifact is not " + expected_artifact_kind)

    return entries^


def require_current_vector_scalar_name(
    entries: List[ManifestEntry], owner: String
) raises -> String:
    var stored_vector_scalar_name = require_manifest_value(
        entries, "vector_scalar_name"
    )
    if stored_vector_scalar_name != VECTOR_SCALAR_NAME:
        raise Error(
            owner
            + " vector scalar type does not match current build: "
            + stored_vector_scalar_name
            + " vs "
            + VECTOR_SCALAR_NAME
        )

    return stored_vector_scalar_name
