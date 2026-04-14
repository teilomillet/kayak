from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import EncodedDocument, VECTOR_SCALAR_NAME, pack_documents
from kayak.collections import (
    CollectionId,
    CollectionManifest,
    CollectionStats,
    DocumentMetadataEntry,
    DocumentMetadataMap,
    NamespaceId,
    SegmentId,
    SegmentStats,
    SealedSegmentManifest,
    SnapshotId,
    SnapshotManifest,
    StoredDocumentMetadataCorpus,
    StoredDocumentTextCorpus,
    TenantId,
    document_metadata_search_artifact,
    export_snapshot_bundle,
    import_snapshot_bundle,
    load_collection_manifest,
    load_resolved_collection_snapshot,
    save_collection_manifest,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
    save_stored_document_metadata_corpus,
    save_stored_document_text_corpus,
)
from kayak.storage import StoredPackedIndex, save_stored_packed_index
from kayak.text import DocumentTextCorpus


def unique_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def write_segment_payload(
    segment_root: Path,
    collection_id: String,
    model_name: String,
    documents: List[EncodedDocument],
) raises:
    save_stored_packed_index(
        segment_root / "packed_index",
        StoredPackedIndex(
            collection_id,
            model_name,
            VECTOR_SCALAR_NAME,
            pack_documents(documents),
        ),
    )


def build_source_collection(root: Path) raises:
    save_collection_manifest(
        root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            7,
            2,
        ),
    )

    var segment_root = root / "segments" / "segment-0001"
    write_segment_payload(
        segment_root,
        "collection://news",
        "colbertv2",
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[0.5, 0.5], [0.0, 1.0]]),
        ],
    )
    save_stored_document_text_corpus(
        segment_root / "text_corpus",
        StoredDocumentTextCorpus(
            CollectionId("news"),
            SegmentId("segment-0001"),
            DocumentTextCorpus(["doc-a", "doc-b"], ["alpha", "beta"]),
        ),
    )
    save_stored_document_metadata_corpus(
        segment_root / "document_metadata",
        StoredDocumentMetadataCorpus(
            CollectionId("news"),
            SegmentId("segment-0001"),
            ["doc-a", "doc-b"],
            [
                DocumentMetadataMap(
                    [DocumentMetadataEntry("source", "wire")]
                ),
                DocumentMetadataMap(),
            ],
        ),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            4,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            [document_metadata_search_artifact("document_metadata")],
            "text_corpus",
            SegmentStats(2, 4, 4, 1024),
        ),
    )
    save_snapshot_manifest(
        root / "snapshots" / "snapshot-0004",
        SnapshotManifest(
            SnapshotId("snapshot-0004"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            4,
            [SegmentId("segment-0001")],
            CollectionStats(1, 2, 4, 4, 1024),
        ),
    )


def test_snapshot_bundle_export_import_roundtrip() raises:
    var source_root = unique_root("kayak-snapshot-bundle-source")
    var bundle_root = unique_root("kayak-snapshot-bundle-export")
    var target_root = unique_root("kayak-snapshot-bundle-import")

    build_source_collection(source_root)

    var bundle = export_snapshot_bundle(
        source_root, SnapshotId("snapshot-0004"), bundle_root
    )
    var exported_collection = load_collection_manifest(bundle_root)

    assert_equal(bundle.snapshot_id.value, "snapshot-0004")
    assert_equal(bundle.segment_count, 1)
    assert_equal(exported_collection.latest_generation, 4)
    assert_equal(exported_collection.active_snapshot_id, "snapshot-0004")
    assert_equal(exported_collection.default_keep_latest_inactive_count, 2)
    assert_equal(
        len(exported_collection.search_artifact_build_policy.stage1_artifacts),
        2,
    )

    _ = import_snapshot_bundle(bundle_root, target_root)

    var imported_collection = load_collection_manifest(target_root)
    var resolved = load_resolved_collection_snapshot(
        target_root, SnapshotId("snapshot-0004")
    )

    assert_equal(imported_collection.latest_generation, 4)
    assert_equal(imported_collection.active_snapshot_id, "snapshot-0004")
    assert_equal(imported_collection.default_keep_latest_inactive_count, 2)
    assert_equal(
        len(imported_collection.search_artifact_build_policy.stage1_artifacts),
        2,
    )
    assert_equal(len(resolved.segments), 1)
    assert_equal(resolved.segments[0].stored_index.index.document_count, 2)
    assert_equal(resolved.segments[0].has_text_corpus, True)
    assert_equal(resolved.segments[0].stored_text_corpus.corpus.texts[1], "beta")
    assert_equal(
        resolved.segments[0].search_artifacts[0].stored_document_metadata_corpus
            .metadata_maps[0]
            .entries[0]
            .value,
        "wire",
    )


def test_snapshot_bundle_import_rejects_collection_mismatch() raises:
    var source_root = unique_root("kayak-snapshot-bundle-mismatch-source")
    var bundle_root = unique_root("kayak-snapshot-bundle-mismatch-export")
    var target_root = unique_root("kayak-snapshot-bundle-mismatch-target")

    build_source_collection(source_root)
    _ = export_snapshot_bundle(source_root, SnapshotId("snapshot-0004"), bundle_root)

    save_collection_manifest(
        target_root,
        CollectionManifest(
            CollectionId("other"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            0,
        ),
    )

    var raised = False
    try:
        _ = import_snapshot_bundle(bundle_root, target_root)
    except:
        raised = True

    assert_equal(raised, True)


def test_snapshot_bundle_import_rejects_model_name_mismatch() raises:
    var source_root = unique_root("kayak-snapshot-bundle-model-source")
    var bundle_root = unique_root("kayak-snapshot-bundle-model-export")
    var target_root = unique_root("kayak-snapshot-bundle-model-target")

    build_source_collection(source_root)
    _ = export_snapshot_bundle(source_root, SnapshotId("snapshot-0004"), bundle_root)

    save_collection_manifest(
        target_root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "bge-small-en-v1.5",
            VECTOR_SCALAR_NAME,
            2,
            0,
        ),
    )

    var raised = False
    try:
        _ = import_snapshot_bundle(bundle_root, target_root)
    except:
        raised = True

    assert_equal(raised, True)


def test_snapshot_bundle_import_rejects_vector_dim_mismatch() raises:
    var source_root = unique_root("kayak-snapshot-bundle-dim-source")
    var bundle_root = unique_root("kayak-snapshot-bundle-dim-export")
    var target_root = unique_root("kayak-snapshot-bundle-dim-target")

    build_source_collection(source_root)
    _ = export_snapshot_bundle(source_root, SnapshotId("snapshot-0004"), bundle_root)

    save_collection_manifest(
        target_root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            0,
        ),
    )

    var raised = False
    try:
        _ = import_snapshot_bundle(bundle_root, target_root)
    except:
        raised = True

    assert_equal(raised, True)


def test_snapshot_bundle_import_preserves_existing_collection_retention_default() raises:
    var source_root = unique_root("kayak-snapshot-bundle-retention-source")
    var bundle_root = unique_root("kayak-snapshot-bundle-retention-export")
    var target_root = unique_root("kayak-snapshot-bundle-retention-target")

    build_source_collection(source_root)
    _ = export_snapshot_bundle(source_root, SnapshotId("snapshot-0004"), bundle_root)

    save_collection_manifest(
        target_root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            0,
            "",
            5,
        ),
    )

    _ = import_snapshot_bundle(bundle_root, target_root)

    var imported_collection = load_collection_manifest(target_root)

    assert_equal(imported_collection.latest_generation, 4)
    assert_equal(imported_collection.active_snapshot_id, "snapshot-0004")
    assert_equal(imported_collection.default_keep_latest_inactive_count, 5)
    assert_equal(
        len(imported_collection.search_artifact_build_policy.stage1_artifacts),
        2,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
