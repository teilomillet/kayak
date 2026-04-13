from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak.collections import (
    CollectionId,
    CollectionManifest,
    CollectionStats,
    DocumentMetadataEntry,
    DocumentMetadataMap,
    NamespaceId,
    SearchArtifactManifest,
    SearchArtifactBuildPolicy,
    SearchArtifactBuildSpec,
    SegmentId,
    SegmentStats,
    SealedSegmentManifest,
    SnapshotId,
    SnapshotManifest,
    StoredDocumentMetadataCorpus,
    StoredDocumentTextCorpus,
    TenantId,
    load_stored_document_metadata_corpus,
    document_proxy_search_artifact,
    gem_graph_search_artifact,
    load_collection_manifest,
    load_sealed_segment_manifest,
    load_snapshot_manifest,
    load_stored_document_text_corpus,
    save_collection_manifest,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
    save_stored_document_metadata_corpus,
    save_stored_document_text_corpus,
    same_search_artifact_build_policy,
)
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.text import DocumentTextCorpus


def test_collection_storage_roundtrip_preserves_manifests_and_text() raises:
    var root = Path("/tmp/kayak-collection-storage-roundtrip")
    var collection_root = root / "collection"
    var segment_root = root / "segment-0001"
    var snapshot_root = root / "snapshot-0001"
    var text_corpus_root = segment_root / "text_corpus"

    var tenant_id = TenantId("tenant-a")
    var namespace_id = NamespaceId("search")
    var collection_id = CollectionId("news")
    var segment_id = SegmentId("segment-0001")
    var build_policy = SearchArtifactBuildPolicy(
        [
            SearchArtifactBuildSpec("document_proxy", "proxy_sidecar"),
            SearchArtifactBuildSpec("centroid_postings", "postings_sidecar"),
        ]
    )

    save_collection_manifest(
        collection_root,
        CollectionManifest(
            collection_id.copy(),
            tenant_id.copy(),
            namespace_id.copy(),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            4,
            "snapshot-0001",
            2,
            build_policy,
        ),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            segment_id.copy(),
            collection_id.copy(),
            tenant_id.copy(),
            namespace_id.copy(),
            4,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            "packed_index",
            "",
            "text_corpus",
            SegmentStats(2, 11, 10, 4096),
        ),
    )
    save_snapshot_manifest(
        snapshot_root,
        SnapshotManifest(
            SnapshotId("snapshot-0001"),
            collection_id.copy(),
            tenant_id.copy(),
            namespace_id.copy(),
            4,
            [segment_id.copy()],
            CollectionStats(1, 2, 11, 10, 8192),
        ),
    )
    save_stored_document_text_corpus(
        text_corpus_root,
        StoredDocumentTextCorpus(
            collection_id,
            segment_id,
            DocumentTextCorpus(
                ["doc-a", "doc-b"],
                ["alpha\nbeta\tgamma", "delta\r\nepsilon"],
            ),
        ),
    )

    var loaded_collection = load_collection_manifest(collection_root)
    var loaded_segment = load_sealed_segment_manifest(segment_root)
    var loaded_snapshot = load_snapshot_manifest(snapshot_root)
    var loaded_text_corpus = load_stored_document_text_corpus(text_corpus_root)

    assert_equal(loaded_collection.latest_generation, 4)
    assert_equal(loaded_collection.active_snapshot_id, "snapshot-0001")
    assert_equal(loaded_collection.default_keep_latest_inactive_count, 2)
    assert_equal(
        same_search_artifact_build_policy(
            loaded_collection.search_artifact_build_policy,
            build_policy,
        ),
        True,
    )
    assert_equal(loaded_segment.packed_index_root, "packed_index")
    assert_equal(loaded_segment.text_corpus_root, "text_corpus")
    assert_equal(loaded_snapshot.stats.segment_count, 1)
    assert_equal(loaded_snapshot.segment_ids[0].value, "segment-0001")
    assert_equal(loaded_text_corpus.corpus.doc_ids[0], "doc-a")
    assert_equal(loaded_text_corpus.corpus.texts[0], "alpha\nbeta\tgamma")
    assert_equal(loaded_text_corpus.corpus.texts[1], "delta\r\nepsilon")


def test_snapshot_manifest_rejects_segment_count_mismatch() raises:
    var root = Path("/tmp/kayak-snapshot-storage-mismatch")
    save_snapshot_manifest(
        root,
        SnapshotManifest(
            SnapshotId("snapshot-0002"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            [SegmentId("segment-0001")],
            CollectionStats(1, 2, 11, 10, 8192),
        ),
    )

    var manifest_path = root / "manifest.tsv"
    manifest_path.write_text(
        manifest_path.read_text().replace("segment_count\t1", "segment_count\t2")
    )

    var raised = False
    try:
        _ = load_snapshot_manifest(root)
    except:
        raised = True

    assert_equal(raised, True)


def test_segment_manifest_roundtrip_preserves_search_artifact_registry() raises:
    var root = Path("/tmp/kayak-segment-search-artifact-roundtrip")

    save_sealed_segment_manifest(
        root,
        SealedSegmentManifest(
            SegmentId("segment-0004"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            7,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            "packed_index",
            [
                document_proxy_search_artifact("document_proxy"),
                gem_graph_search_artifact("gem_graph"),
            ],
            "text_corpus",
            SegmentStats(2, 11, 10, 4096),
        ),
    )

    var loaded_segment = load_sealed_segment_manifest(root)

    assert_equal(loaded_segment.packed_index_root, "packed_index")
    assert_equal(loaded_segment.text_corpus_root, "text_corpus")
    assert_equal(len(loaded_segment.search_artifacts), 2)
    assert_equal(loaded_segment.search_artifacts[0].family, "document_proxy")
    assert_equal(loaded_segment.search_artifacts[0].root, "document_proxy")
    assert_equal(loaded_segment.search_artifacts[1].family, "gem_graph")
    assert_equal(loaded_segment.search_artifacts[1].root, "gem_graph")


def test_document_text_corpus_rejects_missing_text_file() raises:
    var root = Path("/tmp/kayak-text-corpus-missing-file")
    save_stored_document_text_corpus(
        root,
        StoredDocumentTextCorpus(
            CollectionId("news"),
            SegmentId("segment-0003"),
            DocumentTextCorpus(["doc-a"], ["alpha"]),
        ),
    )

    var entries_path = root / "entries.tsv"
    entries_path.write_text(entries_path.read_text().replace("0.txt", "missing.txt"))

    var raised = False
    try:
        _ = load_stored_document_text_corpus(root)
    except:
        raised = True

    assert_equal(raised, True)


def test_document_metadata_corpus_roundtrip_preserves_entries() raises:
    var root = Path("/tmp/kayak-document-metadata-roundtrip")
    save_stored_document_metadata_corpus(
        root,
        StoredDocumentMetadataCorpus(
            CollectionId("news"),
            SegmentId("segment-0009"),
            ["doc-a", "doc-b"],
            [
                DocumentMetadataMap(
                    [DocumentMetadataEntry("source", "wire")]
                ),
                DocumentMetadataMap(),
            ],
        ),
    )

    var loaded = load_stored_document_metadata_corpus(root)

    assert_equal(loaded.doc_ids[0], "doc-a")
    assert_equal(loaded.metadata_maps[0].entries[0].key, "source")
    assert_equal(loaded.metadata_maps[0].entries[0].value, "wire")
    assert_equal(loaded.metadata_maps[1].is_empty(), True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
