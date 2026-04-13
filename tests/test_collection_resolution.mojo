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
    document_proxy_search_artifact,
    exact_only_snapshot_requirements,
    gem_graph_search_artifact,
    loaded_search_artifact_family,
    loaded_search_artifact_stored_document_metadata,
    loaded_search_artifact_stored_document_proxy_index,
    loaded_search_artifact_stored_gem_graph_index,
    loaded_segment_has_document_metadata,
    loaded_segment_stored_document_metadata,
    loaded_segment_has_document_proxy_index,
    loaded_segment_has_gem_graph_index,
    loaded_segment_has_search_artifact,
    loaded_segment_search_artifact,
    loaded_segment_search_artifact_families,
    loaded_segment_stored_gem_graph_index,
    load_collection_storage_report,
    load_resolved_collection_snapshot,
    save_collection_manifest,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
    save_stored_document_metadata_corpus,
    save_stored_document_text_corpus,
    search_artifact_snapshot_requirements,
)
from kayak.storage import (
    StoredPackedIndex,
    build_stored_document_proxy_index,
    build_stored_gem_graph_index,
    save_stored_document_proxy_index,
    save_stored_gem_graph_index,
    save_stored_packed_index,
)
from kayak.text import DocumentTextCorpus


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


def test_resolved_snapshot_loads_segments_and_text_sidecars() raises:
    var collection_root = Path("/tmp/kayak-resolved-collection")
    save_collection_manifest(
        collection_root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            4,
        ),
    )

    var segment_one_root = collection_root / "segments" / "segment-0001"
    var segment_two_root = collection_root / "segments" / "segment-0002"

    write_segment_payload(
        segment_one_root,
        "collection://news",
        "colbertv2",
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[0.5, 0.5], [0.0, 1.0]]),
        ],
    )
    write_segment_payload(
        segment_two_root,
        "collection://news",
        "colbertv2",
        [EncodedDocument("doc-c", [[1.0, 1.0], [0.0, 1.0]])],
    )

    save_stored_document_text_corpus(
        segment_one_root / "text_corpus",
        StoredDocumentTextCorpus(
            CollectionId("news"),
            SegmentId("segment-0001"),
            DocumentTextCorpus(
                ["doc-a", "doc-b"], ["alpha\nbeta", "gamma\tdelta"]
            ),
        ),
    )

    save_sealed_segment_manifest(
        segment_one_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            3,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            "",
            "text_corpus",
            SegmentStats(2, 4, 4, 1024),
        ),
    )
    save_sealed_segment_manifest(
        segment_two_root,
        SealedSegmentManifest(
            SegmentId("segment-0002"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            4,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            "",
            "",
            SegmentStats(1, 2, 2, 512),
        ),
    )
    save_snapshot_manifest(
        collection_root / "snapshots" / "snapshot-0004",
        SnapshotManifest(
            SnapshotId("snapshot-0004"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            4,
            [SegmentId("segment-0001"), SegmentId("segment-0002")],
            CollectionStats(2, 3, 6, 6, 1536),
        ),
    )

    var resolved = load_resolved_collection_snapshot(
        collection_root, SnapshotId("snapshot-0004")
    )
    var report = load_collection_storage_report(
        collection_root, SnapshotId("snapshot-0004")
    )

    assert_equal(resolved.snapshot.segment_ids[0].value, "segment-0001")
    assert_equal(len(resolved.segments), 2)
    assert_equal(resolved.segments[0].stored_index.index.document_count, 2)
    assert_equal(resolved.segments[0].has_text_corpus, True)
    assert_equal(
        resolved.segments[0].stored_text_corpus.corpus.texts[1], "gamma\tdelta"
    )
    assert_equal(resolved.segments[1].has_text_corpus, False)
    assert_equal(report.stats.document_count, 3)
    assert_equal(report.density.bytes_per_document, 512.0)
    assert_equal(report.segment_count_with_text, 1)
    assert_equal(report.segment_count_without_text, 1)
    assert_equal(len(report.segment_reports), 2)
    assert_equal(report.segment_reports[0].segment_id, "segment-0001")
    assert_equal(report.segment_reports[0].density.bytes_per_document, 512.0)


def test_resolved_snapshot_loads_document_metadata_sidecar() raises:
    var collection_root = Path("/tmp/kayak-resolved-collection-metadata")
    save_collection_manifest(
        collection_root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            1,
        ),
    )

    var segment_root = collection_root / "segments" / "segment-0001"
    write_segment_payload(
        segment_root,
        "collection://news",
        "colbertv2",
        [EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]])],
    )
    save_stored_document_metadata_corpus(
        segment_root / "document_metadata",
        StoredDocumentMetadataCorpus(
            CollectionId("news"),
            SegmentId("segment-0001"),
            ["doc-a"],
            [
                DocumentMetadataMap(
                    [DocumentMetadataEntry("source", "wire")]
                )
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
            1,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            [document_metadata_search_artifact("document_metadata")],
            "",
            SegmentStats(1, 2, 2, 512),
        ),
    )
    save_snapshot_manifest(
        collection_root / "snapshots" / "snapshot-0001",
        SnapshotManifest(
            SnapshotId("snapshot-0001"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            [SegmentId("segment-0001")],
            CollectionStats(1, 1, 2, 2, 512),
        ),
    )

    var resolved = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
        search_artifact_snapshot_requirements("document_metadata"),
    )

    assert_equal(loaded_segment_has_document_metadata(resolved.segments[0]), True)
    var metadata_artifact = loaded_segment_search_artifact(
        resolved.segments[0],
        "document_metadata",
    )
    assert_equal(loaded_search_artifact_family(metadata_artifact), "document_metadata")
    assert_equal(
        loaded_segment_stored_document_metadata(resolved.segments[0]).metadata_maps[0]
            .entries[0]
            .value,
        "wire",
    )
    assert_equal(
        loaded_search_artifact_stored_document_metadata(metadata_artifact)
            .metadata_maps[0]
            .entries[0]
            .value,
        "wire",
    )


def test_resolved_snapshot_rejects_segment_generation_ahead_of_snapshot() raises:
    var collection_root = Path("/tmp/kayak-resolved-collection-generation-mismatch")
    save_collection_manifest(
        collection_root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            4,
        ),
    )

    var segment_root = collection_root / "segments" / "segment-0001"
    write_segment_payload(
        segment_root,
        "collection://news",
        "colbertv2",
        [EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]])],
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            5,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            "",
            "",
            SegmentStats(1, 2, 2, 512),
        ),
    )
    save_snapshot_manifest(
        collection_root / "snapshots" / "snapshot-0004",
        SnapshotManifest(
            SnapshotId("snapshot-0004"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            4,
            [SegmentId("segment-0001")],
            CollectionStats(1, 1, 2, 2, 512),
        ),
    )

    var raised = False
    try:
        _ = load_resolved_collection_snapshot(
            collection_root, SnapshotId("snapshot-0004")
        )
    except:
        raised = True

    assert_equal(raised, True)


def test_segment_manifest_rejects_non_relative_artifact_roots() raises:
    var root = Path("/tmp/kayak-segment-manifest-invalid-root")

    var raised = False
    try:
        save_sealed_segment_manifest(
            root,
            SealedSegmentManifest(
                SegmentId("segment-0001"),
                CollectionId("news"),
                TenantId("tenant-a"),
                NamespaceId("search"),
                1,
                "colbertv2",
                VECTOR_SCALAR_NAME,
                2,
                "../packed_index",
                "",
                "",
                SegmentStats(1, 2, 2, 512),
            ),
        )
    except:
        raised = True

    assert_equal(raised, True)


def test_resolved_snapshot_loads_gem_graph_artifact_payload() raises:
    var collection_root = Path("/tmp/kayak-resolved-collection-gem-graph")
    save_collection_manifest(
        collection_root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            4,
        ),
    )

    var segment_root = collection_root / "segments" / "segment-0001"
    write_segment_payload(
        segment_root,
        "collection://news",
        "colbertv2",
        [EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]])],
    )
    var packed_index = StoredPackedIndex(
        "collection://news",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        pack_documents([EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]])]),
    )
    save_stored_gem_graph_index(
        segment_root / "gem_graph",
        build_stored_gem_graph_index(packed_index, 1, 1, 1, 1, 1),
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
            [gem_graph_search_artifact("gem_graph")],
            "",
            SegmentStats(1, 2, 2, 512),
        ),
    )
    save_snapshot_manifest(
        collection_root / "snapshots" / "snapshot-0004",
        SnapshotManifest(
            SnapshotId("snapshot-0004"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            4,
            [SegmentId("segment-0001")],
            CollectionStats(1, 1, 2, 2, 512),
        ),
    )

    var resolved = load_resolved_collection_snapshot(
        collection_root, SnapshotId("snapshot-0004")
    )

    assert_equal(loaded_segment_has_gem_graph_index(resolved.segments[0]), True)
    assert_equal(
        loaded_segment_search_artifact_families(resolved.segments[0])[0],
        "gem_graph",
    )
    var gem_artifact = loaded_segment_search_artifact(
        resolved.segments[0],
        "gem_graph",
    )
    assert_equal(loaded_search_artifact_family(gem_artifact), "gem_graph")
    var stored_gem_graph = loaded_segment_stored_gem_graph_index(resolved.segments[0])
    assert_equal(stored_gem_graph.document_count, 1)
    assert_equal(stored_gem_graph.cluster_count, 1)
    assert_equal(stored_gem_graph.index.document_count, 1)
    assert_equal(stored_gem_graph.index.doc_ids[0], "doc-a")
    assert_equal(
        loaded_search_artifact_stored_gem_graph_index(gem_artifact).document_count,
        1,
    )


def test_resolved_snapshot_can_skip_text_and_unrequested_sidecars() raises:
    var collection_root = Path("/tmp/kayak-resolved-collection-selective-load")
    save_collection_manifest(
        collection_root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            4,
        ),
    )

    var segment_root = collection_root / "segments" / "segment-0001"
    var documents = [EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]])]
    write_segment_payload(
        segment_root,
        "collection://news",
        "colbertv2",
        documents,
    )
    save_stored_document_proxy_index(
        segment_root / "document_proxy",
        build_stored_document_proxy_index(
            StoredPackedIndex(
                "collection://news",
                "colbertv2",
                VECTOR_SCALAR_NAME,
                pack_documents(documents),
            ),
            0,
        ),
    )
    save_stored_gem_graph_index(
        segment_root / "gem_graph",
        build_stored_gem_graph_index(
            StoredPackedIndex(
                "collection://news",
                "colbertv2",
                VECTOR_SCALAR_NAME,
                pack_documents(documents),
            ),
            1,
            1,
            1,
            1,
            1,
        ),
    )
    save_stored_document_text_corpus(
        segment_root / "text_corpus",
        StoredDocumentTextCorpus(
            CollectionId("news"),
            SegmentId("segment-0001"),
            DocumentTextCorpus(["doc-a"], ["alpha"]),
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
            [
                document_proxy_search_artifact("document_proxy"),
                gem_graph_search_artifact("gem_graph"),
            ],
            "text_corpus",
            SegmentStats(1, 2, 2, 512),
        ),
    )
    save_snapshot_manifest(
        collection_root / "snapshots" / "snapshot-0004",
        SnapshotManifest(
            SnapshotId("snapshot-0004"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            4,
            [SegmentId("segment-0001")],
            CollectionStats(1, 1, 2, 2, 512),
        ),
    )

    var exact_only = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0004"),
        exact_only_snapshot_requirements(),
    )
    assert_equal(exact_only.segments[0].has_text_corpus, False)
    assert_equal(
        loaded_segment_has_search_artifact(exact_only.segments[0], "document_proxy"),
        False,
    )
    assert_equal(
        loaded_segment_has_search_artifact(exact_only.segments[0], "gem_graph"),
        False,
    )

    var proxy_only = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0004"),
        search_artifact_snapshot_requirements("document_proxy"),
    )
    assert_equal(proxy_only.segments[0].has_text_corpus, False)
    assert_equal(loaded_segment_has_document_proxy_index(proxy_only.segments[0]), True)
    assert_equal(loaded_segment_has_gem_graph_index(proxy_only.segments[0]), False)
    assert_equal(len(loaded_segment_search_artifact_families(proxy_only.segments[0])), 1)
    assert_equal(
        loaded_segment_search_artifact_families(proxy_only.segments[0])[0],
        "document_proxy",
    )
    assert_equal(
        loaded_search_artifact_stored_document_proxy_index(
            loaded_segment_search_artifact(
                proxy_only.segments[0],
                "document_proxy",
            )
        ).index.document_count,
        1,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
