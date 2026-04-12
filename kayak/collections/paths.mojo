from std.pathlib import Path


def collection_manifest_path(root: Path) -> Path:
    return root / "collection.manifest.tsv"


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
