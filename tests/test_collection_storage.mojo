from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak.collections import (
    COLLECTION_LAYOUT_FAMILY_SHARED_POOL,
    COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED,
    CollectionId,
    CollectionManifest,
    CollectionStats,
    DocumentMetadataEntry,
    DocumentMetadataMap,
    DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
    DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
    DOCUMENT_ENCODER_COMPRESSION_KIND_NONE,
    gem_graph_build_spec,
    NamespaceId,
    DOCUMENT_REPRESENTATION_TRANSFORM_KIND_PREFIX_PRUNING,
    DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
    SearchArtifactManifest,
    SearchArtifactBuildPolicy,
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
    document_encoder_compression_config_value,
    save_collection_manifest,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
    save_stored_document_metadata_corpus,
    save_stored_document_text_corpus,
    same_document_encoder_compression_manifest,
    same_document_representation_transforms,
    same_search_artifact_build_policy,
    memory_tokens_document_encoder_compression,
    prefix_pruning_document_representation_transform,
    sealed_segment_has_document_representation_transform_kind,
    sealed_segment_has_document_representation_transforms,
    token_pooling_document_representation_transform,
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
    var document_encoder_compression = memory_tokens_document_encoder_compression(16)
    var build_policy = SearchArtifactBuildPolicy(
        [
            gem_graph_build_spec(2, 3, 1, "gem_sidecar", 4, 5),
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
            COLLECTION_LAYOUT_FAMILY_SHARED_POOL,
            document_encoder_compression,
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
            [],
            document_encoder_compression,
            [],
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
    assert_equal(
        loaded_collection.collection_layout_family,
        COLLECTION_LAYOUT_FAMILY_SHARED_POOL,
    )
    assert_equal(loaded_collection.default_keep_latest_inactive_count, 2)
    assert_equal(
        same_search_artifact_build_policy(
            loaded_collection.search_artifact_build_policy,
            build_policy,
        ),
        True,
    )
    assert_equal(
        same_document_encoder_compression_manifest(
            loaded_collection.document_encoder_compression,
            document_encoder_compression,
        ),
        True,
    )
    assert_equal(
        loaded_collection.document_encoder_compression.kind,
        DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
    )
    assert_equal(
        document_encoder_compression_config_value(
            loaded_collection.document_encoder_compression,
            DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "16",
    )
    assert_equal(
        loaded_collection.search_artifact_build_policy.stage1_artifacts[0]
            .config[0]
            .key,
        "fine_cluster_count",
    )
    assert_equal(
        loaded_collection.search_artifact_build_policy.stage1_artifacts[0]
            .config[0]
            .value,
        "2",
    )
    assert_equal(loaded_segment.packed_index_root, "packed_index")
    assert_equal(loaded_segment.text_corpus_root, "text_corpus")
    assert_equal(
        same_document_encoder_compression_manifest(
            loaded_segment.document_encoder_compression,
            document_encoder_compression,
        ),
        True,
    )
    assert_equal(loaded_snapshot.stats.segment_count, 1)
    assert_equal(loaded_snapshot.segment_ids[0].value, "segment-0001")
    assert_equal(loaded_text_corpus.corpus.doc_ids[0], "doc-a")
    assert_equal(loaded_text_corpus.corpus.texts[0], "alpha\nbeta\tgamma")
    assert_equal(loaded_text_corpus.corpus.texts[1], "delta\r\nepsilon")


def test_collection_manifest_loader_defaults_missing_layout_family() raises:
    var root = Path("/tmp/kayak-collection-layout-backcompat")

    save_collection_manifest(
        root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            2,
            "snapshot-0002",
            1,
            SearchArtifactBuildPolicy(
                [gem_graph_build_spec(2, 3, 1, "gem_sidecar", 4, 5)]
            ),
            COLLECTION_LAYOUT_FAMILY_SHARED_POOL,
        ),
    )

    var manifest_path = root / "collection.manifest.tsv"
    manifest_path.write_text(
        manifest_path
            .read_text()
            .replace("collection_layout_family\tshared_pool\n", "")
    )

    var loaded_collection = load_collection_manifest(root)

    assert_equal(
        loaded_collection.collection_layout_family,
        COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED,
    )


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


def test_segment_manifest_roundtrip_preserves_document_representation_transforms() raises:
    var root = Path("/tmp/kayak-segment-transform-roundtrip")
    var transforms = [
        token_pooling_document_representation_transform(2),
        prefix_pruning_document_representation_transform(16),
    ]
    var document_encoder_compression = memory_tokens_document_encoder_compression(12)

    save_sealed_segment_manifest(
        root,
        SealedSegmentManifest(
            SegmentId("segment-0010"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            9,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            "packed_index",
            [document_proxy_search_artifact("document_proxy")],
            "text_corpus",
            SegmentStats(2, 11, 10, 4096),
            transforms,
            document_encoder_compression,
        ),
    )

    var loaded_segment = load_sealed_segment_manifest(root)

    assert_equal(
        same_document_representation_transforms(
            loaded_segment.document_representation_transforms,
            transforms,
        ),
        True,
    )
    assert_equal(
        sealed_segment_has_document_representation_transforms(loaded_segment),
        True,
    )
    assert_equal(
        sealed_segment_has_document_representation_transform_kind(
            loaded_segment,
            DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
        ),
        True,
    )
    assert_equal(
        sealed_segment_has_document_representation_transform_kind(
            loaded_segment,
            DOCUMENT_REPRESENTATION_TRANSFORM_KIND_PREFIX_PRUNING,
        ),
        True,
    )
    assert_equal(
        same_document_encoder_compression_manifest(
            loaded_segment.document_encoder_compression,
            document_encoder_compression,
        ),
        True,
    )
    assert_equal(
        document_encoder_compression_config_value(
            loaded_segment.document_encoder_compression,
            DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "12",
    )


def test_segment_manifest_loader_accepts_old_manifest_without_transform_entries() raises:
    var root = Path("/tmp/kayak-segment-transform-backcompat")

    save_sealed_segment_manifest(
        root,
        SealedSegmentManifest(
            SegmentId("segment-0011"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            10,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            "packed_index",
            [document_proxy_search_artifact("document_proxy")],
            "",
            SegmentStats(1, 2, 2, 512),
            [
                token_pooling_document_representation_transform(2),
            ],
        ),
    )

    var manifest_path = root / "manifest.tsv"
    manifest_path.write_text(
        manifest_path
            .read_text()
            .replace("document_representation_transform_count\t1\n", "")
            .replace(
                "document_representation_transform_0_kind\ttoken_pooling\n",
                "",
            )
            .replace(
                "document_representation_transform_0_config_count\t2\n",
                "",
            )
            .replace(
                "document_representation_transform_0_config_0_key\tpool_factor\n",
                "",
            )
            .replace(
                "document_representation_transform_0_config_0_value\t2\n",
                "",
            )
            .replace(
                "document_representation_transform_0_config_1_key\tpolicy\n",
                "",
            )
            .replace(
                "document_representation_transform_0_config_1_value\thierarchical\n",
                "",
            )
    )

    var loaded_segment = load_sealed_segment_manifest(root)

    assert_equal(
        sealed_segment_has_document_representation_transforms(loaded_segment),
        False,
    )
    assert_equal(
        loaded_segment.document_encoder_compression.kind,
        DOCUMENT_ENCODER_COMPRESSION_KIND_NONE,
    )


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
