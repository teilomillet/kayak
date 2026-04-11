from std.collections import List
from std.pathlib import Path

from kayak.numeric import STORAGE_FORMAT_VERSION, VECTOR_SCALAR_NAME

from .text_codec import append_line, parse_int, read_non_empty_lines, split_tab_fields


struct ManifestEntry(Copyable):
    var key: String
    var value: String

    def __init__(out self, var key: String, var value: String):
        self.key = key^
        self.value = value^


def write_manifest(path: Path, entries: List[ManifestEntry]) raises:
    var buffer = String()

    for entry in entries:
        append_line(buffer, entry.key + "\t" + entry.value)

    path.write_text(buffer)


def read_manifest(path: Path) raises -> List[ManifestEntry]:
    var entries = List[ManifestEntry]()

    for line in read_non_empty_lines(path):
        var fields = split_tab_fields(line, 2, "manifest line")
        entries.append(ManifestEntry(fields[0], fields[1]))

    return entries^


def require_manifest_value(
    entries: List[ManifestEntry], key: String
) raises -> String:
    for entry in entries:
        if entry.key == key:
            return entry.value.copy()

    raise Error("missing manifest key: " + key)


def require_supported_storage_format(entries: List[ManifestEntry]) raises -> Int:
    var format_version = parse_int(
        require_manifest_value(entries, "format_version"),
        "storage format_version",
    )

    if format_version != 1 and format_version != STORAGE_FORMAT_VERSION:
        raise Error("unsupported storage format version")

    var stored_vector_scalar_name = require_manifest_value(
        entries, "vector_scalar_name"
    )
    if stored_vector_scalar_name != VECTOR_SCALAR_NAME:
        raise Error(
            "stored vector scalar type does not match current build: "
            + stored_vector_scalar_name
            + " vs "
            + VECTOR_SCALAR_NAME
        )

    return format_version
