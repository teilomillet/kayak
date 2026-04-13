from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    CollectionManifest,
    EncodedDocument,
    centroid_heads_build_spec,
    gem_graph_build_spec,
    DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
    load_stored_centroid_heads_index,
    load_stored_gem_graph_index,
    NamespaceId,
    SearchArtifactBuildPolicy,
    SearchArtifactBuildSpec,
    SegmentId,
    TenantId,
    VECTOR_SCALAR_NAME,
    seal_single_segment,
    sealed_segment_has_document_representation_transform_kind,
    token_pooling_document_representation_transform,
)


def make_documents() raises -> List[EncodedDocument]:
    return [
        EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
        EncodedDocument("doc-b", [[0.5, 0.5], [1.0, 0.0]]),
    ]


def test_seal_single_segment_supports_empty_stage1_policy() raises:
    var collection_root = Path("/tmp/kayak-segment-builder-empty-policy")
    var collection = CollectionManifest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
        0,
        SearchArtifactBuildPolicy([]),
    )

    var sealed = seal_single_segment(
        collection_root,
        collection,
        SegmentId("segment-0001"),
        1,
        make_documents(),
        ["alpha", "beta"],
    )

    assert_equal(len(sealed.search_artifacts), 0)
    assert_equal(
        (collection_root / "segments" / "segment-0001" / "document_proxy").exists(),
        False,
    )
    assert_equal(
        (
            collection_root / "segments" / "segment-0001" / "centroid_postings"
        ).exists(),
        False,
    )


def test_seal_single_segment_uses_configured_stage1_sidecar_roots() raises:
    var collection_root = Path("/tmp/kayak-segment-builder-custom-policy")
    var collection = CollectionManifest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
        0,
        SearchArtifactBuildPolicy(
            [
                SearchArtifactBuildSpec("document_proxy", "proxy_sidecar"),
                SearchArtifactBuildSpec("centroid_postings", "postings_sidecar"),
            ]
        ),
    )

    var sealed = seal_single_segment(
        collection_root,
        collection,
        SegmentId("segment-0001"),
        1,
        make_documents(),
        ["alpha", "beta"],
    )

    assert_equal(len(sealed.search_artifacts), 2)
    assert_equal(sealed.search_artifacts[0].root, "proxy_sidecar")
    assert_equal(sealed.search_artifacts[1].root, "postings_sidecar")
    assert_equal(
        (
            collection_root
            / "segments"
            / "segment-0001"
            / "proxy_sidecar"
            / "manifest.tsv"
        ).exists(),
        True,
    )


def test_seal_single_segment_builds_configured_centroid_heads_sidecar() raises:
    var collection_root = Path("/tmp/kayak-segment-builder-centroid-heads-policy")
    var collection = CollectionManifest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
        0,
        SearchArtifactBuildPolicy(
            [centroid_heads_build_spec(1, "heads_sidecar", 1)]
        ),
    )

    var sealed = seal_single_segment(
        collection_root,
        collection,
        SegmentId("segment-0001"),
        1,
        make_documents(),
        ["alpha", "beta"],
    )
    var stored = load_stored_centroid_heads_index(
        collection_root / "segments" / "segment-0001" / "heads_sidecar"
    )

    assert_equal(len(sealed.search_artifacts), 1)
    assert_equal(sealed.search_artifacts[0].family, "centroid_heads")
    assert_equal(stored.centroid_budget, 1)
    assert_equal(stored.posting_cap, 1)


def test_seal_single_segment_builds_configured_gem_graph_sidecar() raises:
    var collection_root = Path("/tmp/kayak-segment-builder-gem-graph-policy")
    var collection = CollectionManifest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
        0,
        SearchArtifactBuildPolicy(
            [gem_graph_build_spec(1, 1, 1, "gem_sidecar", 1, 1)]
        ),
    )

    var sealed = seal_single_segment(
        collection_root,
        collection,
        SegmentId("segment-0001"),
        1,
        make_documents(),
        ["alpha", "beta"],
    )
    var stored = load_stored_gem_graph_index(
        collection_root / "segments" / "segment-0001" / "gem_sidecar"
    )

    assert_equal(len(sealed.search_artifacts), 1)
    assert_equal(sealed.search_artifacts[0].family, "gem_graph")
    assert_equal(stored.cluster_cutoff, 1)
    assert_equal(stored.construction_neighbor_count, 1)
    assert_equal(stored.degree_limit, 1)
    assert_equal(
        (
            collection_root
            / "segments"
            / "segment-0001"
            / "gem_sidecar"
            / "manifest.tsv"
        ).exists(),
        True,
    )


def test_seal_single_segment_applies_token_pooling_transform() raises:
    var collection_root = Path("/tmp/kayak-segment-builder-token-pooling")
    var collection = CollectionManifest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
        0,
        SearchArtifactBuildPolicy([]),
    )

    var sealed = seal_single_segment(
        collection_root,
        collection,
        SegmentId("segment-0001"),
        1,
        [
            EncodedDocument(
                "doc-a",
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.1, 0.9],
                ],
            ),
            EncodedDocument(
                "doc-b",
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ],
            ),
        ],
        ["alpha", "beta"],
        [token_pooling_document_representation_transform(2)],
    )

    assert_equal(sealed.stats.document_count, 2)
    assert_equal(sealed.stats.total_vector_count, 4)
    assert_equal(
        sealed_segment_has_document_representation_transform_kind(
            sealed,
            DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
        ),
        True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
