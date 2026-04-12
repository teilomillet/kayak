from std.collections import List

from kayak.storage.manifest import ManifestEntry


def load_optional_manifest_value(
    read entries: List[ManifestEntry], key: String
) -> String:
    for entry in entries:
        if entry.key == key:
            return entry.value.copy()

    return ""
