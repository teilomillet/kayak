from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    EncodedDocument,
    NamespaceId,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    pack_documents,
)
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.text import DocumentTextCorpus


def unique_collection_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def mirror_fixture_index() raises -> StoredPackedIndex:
    return StoredPackedIndex(
        "mock://mirror",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        pack_documents(
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                EncodedDocument("doc-b", [[0.0, 1.0], [1.0, 0.0]]),
            ]
        ),
    )


def test_collection_mirror_persists_optional_text_corpus() raises:
    var stored_index = mirror_fixture_index()
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-collection-mirror-text"),
        CollectionId("mirror-text"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        stored_index,
        DocumentTextCorpus(
            ["doc-a", "doc-b"],
            ["alpha clause evidence", "beta supporting passage"],
        ),
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )

    assert_equal(snapshot.segments[0].has_text_corpus, True)
    assert_equal(
        snapshot.segments[0].stored_text_corpus.corpus.doc_ids[1],
        "doc-b",
    )
    assert_equal(
        snapshot.segments[0].stored_text_corpus.corpus.texts[0],
        "alpha clause evidence",
    )


def test_collection_mirror_rejects_misaligned_text_corpus_doc_order() raises:
    var raised = False

    try:
        _ = ensure_one_segment_collection_mirror(
            unique_collection_root("kayak-collection-mirror-misaligned"),
            CollectionId("mirror-text"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            mirror_fixture_index(),
            DocumentTextCorpus(
                ["doc-b", "doc-a"],
                ["beta supporting passage", "alpha clause evidence"],
            ),
        )
    except _:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
