from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    CollectionManifest,
    EncodedDocument,
    NamespaceId,
    SearchArtifactBuildPolicy,
    SearchArtifactBuildSpec,
    SegmentId,
    TenantId,
    VECTOR_SCALAR_NAME,
    seal_single_segment,
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
    assert_equal(
        (
            collection_root
            / "segments"
            / "segment-0001"
            / "postings_sidecar"
            / "manifest.tsv"
        ).exists(),
        True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
