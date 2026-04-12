from std.os import makedirs
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import EncodedDocument, StoredPackedIndex, VECTOR_SCALAR_NAME, pack_documents
from kayak.storage import (
    CENTROID_POSTINGS_ORDER_UNSPECIFIED,
    CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC,
    build_stored_centroid_posting_index,
    centroid_postings_storage_byte_size,
    ensure_stored_centroid_posting_index,
    load_stored_centroid_posting_index,
    save_stored_centroid_posting_index,
)


def make_stored_index() raises -> StoredPackedIndex:
    return StoredPackedIndex(
        "mock://centroid-postings-store",
        "mock-model",
        VECTOR_SCALAR_NAME,
        pack_documents(
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                EncodedDocument("doc-b", [[1.0, 0.0], [0.5, 0.5]]),
            ]
        ),
    )


def unique_temp_root(prefix: String) -> Path:
    var candidate = Path(prefix)
    var suffix = 0

    while candidate.exists():
        suffix += 1
        candidate = Path(prefix + "-" + String(suffix))

    return candidate


def copy_storage_file(source_root: Path, target_root: Path, file_name: String) raises:
    var source = source_root / file_name
    var target = target_root / file_name

    if source.suffix() == ".bin":
        target.write_bytes(source.read_bytes())
        return

    target.write_text(source.read_text())


def write_legacy_centroid_postings_root(
    source_root: Path, target_root: Path, include_document_counts: Bool = False
) raises:
    makedirs(target_root, exist_ok=True)
    var legacy_manifest = (source_root / "manifest.tsv").read_text().replace(
        "posting_order_kind\t" + CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC + "\n", ""
    )
    legacy_manifest = legacy_manifest.replace("posting_cap\t0\n", "")
    (target_root / "manifest.tsv").write_text(legacy_manifest)
    copy_storage_file(source_root, target_root, "centroid_dims.tsv")
    copy_storage_file(source_root, target_root, "posting_offsets.tsv")
    copy_storage_file(source_root, target_root, "posting_doc_indices.tsv")
    copy_storage_file(source_root, target_root, "posting_weights.tsv")
    copy_storage_file(source_root, target_root, "centroid_vectors.bin")

    if include_document_counts:
        copy_storage_file(source_root, target_root, "centroid_document_counts.tsv")


def test_centroid_postings_roundtrip_preserves_summary_arrays() raises:
    var root = unique_temp_root("/tmp/kayak-centroid-postings-store-roundtrip")
    var stored = build_stored_centroid_posting_index(make_stored_index(), 0)
    save_stored_centroid_posting_index(root, stored)

    var loaded = load_stored_centroid_posting_index(root)

    assert_equal(
        loaded.posting_order_kind, CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC
    )
    assert_equal(loaded.posting_cap, 0)
    assert_equal(loaded.index.posting_doc_indices[0], 1)
    assert_equal(loaded.index.posting_weights[0], 2)
    assert_equal(loaded.index.posting_doc_indices[1], 0)
    assert_equal(loaded.index.posting_weights[1], 1)
    assert_equal(loaded.index.centroid_document_counts[0], 2)
    assert_equal(loaded.index.centroid_document_counts[1], 1)
    assert_equal(loaded.index.centroid_token_counts[0], 3)
    assert_equal(loaded.index.centroid_token_counts[1], 1)
    assert_equal(loaded.index.total_centroid_token_count, 4)
    assert_equal(
        loaded.artifact_byte_size, centroid_postings_storage_byte_size(root)
    )


def test_centroid_postings_loads_legacy_layout_without_summary_files() raises:
    var source_root = unique_temp_root(
        "/tmp/kayak-centroid-postings-store-legacy-source"
    )
    var root = unique_temp_root("/tmp/kayak-centroid-postings-store-legacy")
    var stored = build_stored_centroid_posting_index(make_stored_index(), 0)
    save_stored_centroid_posting_index(source_root, stored)
    write_legacy_centroid_postings_root(source_root, root)

    var loaded = load_stored_centroid_posting_index(root)

    assert_equal(loaded.posting_order_kind, CENTROID_POSTINGS_ORDER_UNSPECIFIED)
    assert_equal(loaded.posting_cap, 0)
    assert_equal(loaded.index.centroid_document_counts[0], 2)
    assert_equal(loaded.index.centroid_document_counts[1], 1)
    assert_equal(loaded.index.centroid_token_counts[0], 3)
    assert_equal(loaded.index.centroid_token_counts[1], 1)
    assert_equal(loaded.index.total_centroid_token_count, 4)


def test_centroid_postings_rejects_partial_summary_layout() raises:
    var source_root = unique_temp_root(
        "/tmp/kayak-centroid-postings-store-partial-source"
    )
    var root = unique_temp_root("/tmp/kayak-centroid-postings-store-partial-summary")
    var stored = build_stored_centroid_posting_index(make_stored_index(), 0)
    save_stored_centroid_posting_index(source_root, stored)
    write_legacy_centroid_postings_root(source_root, root, True)

    var raised = False
    try:
        _ = load_stored_centroid_posting_index(root)
    except:
        raised = True

    assert_equal(raised, True)


def test_ensure_centroid_postings_rewrites_legacy_layout_with_summaries() raises:
    var source_root = unique_temp_root(
        "/tmp/kayak-centroid-postings-store-ensure-source"
    )
    var root = unique_temp_root("/tmp/kayak-centroid-postings-store-ensure-upgrade")
    var stored_index = make_stored_index()
    save_stored_centroid_posting_index(
        source_root,
        build_stored_centroid_posting_index(stored_index.copy(), 0),
    )
    write_legacy_centroid_postings_root(source_root, root)

    var cache = ensure_stored_centroid_posting_index(root, stored_index, 0)

    assert_equal(cache.loaded_from_storage, False)
    assert_equal(
        cache.stored_index.posting_order_kind,
        CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC,
    )
    assert_equal((root / "centroid_document_counts.tsv").exists(), True)
    assert_equal((root / "centroid_token_counts.tsv").exists(), True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
