from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import EncodedDocument, StoredPackedIndex, VECTOR_SCALAR_NAME, pack_documents
from kayak.storage import (
    build_stored_centroid_heads_index,
    centroid_heads_storage_byte_size,
    ensure_stored_centroid_heads_index,
    load_stored_centroid_heads_index,
    save_stored_centroid_heads_index,
)


def make_stored_index() raises -> StoredPackedIndex:
    return StoredPackedIndex(
        "mock://centroid-heads-store",
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


def test_centroid_heads_roundtrip_persists_posting_cap_and_truncation() raises:
    var root = unique_temp_root("/tmp/kayak-centroid-heads-store-roundtrip")
    var stored = build_stored_centroid_heads_index(make_stored_index(), 0, 1)
    save_stored_centroid_heads_index(root, stored)

    var loaded = load_stored_centroid_heads_index(root)

    assert_equal(loaded.posting_cap, 1)
    assert_equal(loaded.index.total_posting_count, 2)
    assert_equal(loaded.index.posting_offsets[0], 0)
    assert_equal(loaded.index.posting_offsets[1], 1)
    assert_equal(loaded.index.posting_offsets[2], 2)
    assert_equal(loaded.index.posting_doc_indices[0], 1)
    assert_equal(loaded.index.posting_weights[0], 2)
    assert_equal(loaded.index.posting_doc_indices[1], 0)
    assert_equal(loaded.index.posting_weights[1], 1)
    assert_equal(loaded.index.centroid_document_counts[0], 1)
    assert_equal(loaded.index.centroid_document_counts[1], 1)
    assert_equal(loaded.index.centroid_token_counts[0], 2)
    assert_equal(loaded.index.centroid_token_counts[1], 1)
    assert_equal(loaded.artifact_byte_size, centroid_heads_storage_byte_size(root))


def test_ensure_centroid_heads_reuses_matching_sidecar() raises:
    var root = unique_temp_root("/tmp/kayak-centroid-heads-store-ensure")
    var stored_index = make_stored_index()

    var first = ensure_stored_centroid_heads_index(root, stored_index.copy(), 0, 1)
    var second = ensure_stored_centroid_heads_index(root, stored_index, 0, 1)

    assert_equal(first.loaded_from_storage, False)
    assert_equal(second.loaded_from_storage, True)
    assert_equal(second.stored_index.posting_cap, 1)
    assert_equal(second.stored_index.index.total_posting_count, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
