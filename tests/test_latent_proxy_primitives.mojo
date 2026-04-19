from std.testing import TestSuite, assert_equal

from kayak import EncodedQuery
from kayak.index import (
    LATENT_PROXY_ACTIVATION_RELU,
    LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
    LatentProxyIndex,
    LatentQueryProjection,
    LatentQueryProjectionBlock,
    build_query_latent_proxy_vector,
)
from kayak.planning import (
    ProjectedLatentQuery,
    project_query_with_latent_proxy,
    score_projected_latent_query_against_document,
    segment_hits_for_projected_latent_query,
    sum_projected_latent_query_scores_against_index,
)


def test_project_query_with_latent_proxy_keeps_explicit_single_vector_shape() raises:
    var projection = LatentQueryProjection(
        2,
        2,
        2.0,
        [
            LatentQueryProjectionBlock(
                LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
                LATENT_PROXY_ACTIVATION_RELU,
                2,
                2,
                [[1.0, 0.0], [0.0, 1.0]],
                [0.0, 0.0],
                1.0,
                False,
                0.00001,
                [],
                [],
            )
        ],
    )
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var projected = project_query_with_latent_proxy(query, projection)
    var expected = build_query_latent_proxy_vector(query, projection)

    assert_equal(projected.vector_count, 1)
    assert_equal(projected.vector_dim, 2)
    assert_equal(len(projected.proxy_vector), len(expected))
    for index in range(len(expected)):
        assert_equal(
            abs(Float64(projected.proxy_vector[index]) - Float64(expected[index]))
                < 0.0000001,
            True,
        )


def test_projected_latent_query_scan_matches_manual_scores() raises:
    var projected = ProjectedLatentQuery([1.0, -1.0])
    var index = LatentProxyIndex(
        ["doc-a", "doc-b", "doc-c"],
        [[2.0, 0.0], [0.0, 1.0], [1.0, -2.0]],
        2,
    )

    assert_equal(
        abs(
            Float64(
                score_projected_latent_query_against_document(projected, index, 0)
            )
            - Float64(2.0)
        ) < 0.0000001,
        True,
    )
    assert_equal(
        abs(
            Float64(
                score_projected_latent_query_against_document(projected, index, 1)
            )
            - Float64(-1.0)
        ) < 0.0000001,
        True,
    )
    assert_equal(
        abs(
            Float64(sum_projected_latent_query_scores_against_index(projected, index))
            - Float64(4.0)
        ) < 0.0000001,
        True,
    )


def test_segment_hits_for_projected_latent_query_respects_allowlist_and_topk() raises:
    var projected = ProjectedLatentQuery([1.0, -1.0])
    var index = LatentProxyIndex(
        ["doc-a", "doc-b", "doc-c"],
        [[2.0, 0.0], [0.0, 1.0], [1.0, -2.0]],
        2,
    )
    var hits = segment_hits_for_projected_latent_query(
        projected,
        "segment-0001",
        7,
        index,
        2,
        [1, 0, 1],
    )

    assert_equal(len(hits), 2)
    assert_equal(hits[0].doc_id, "doc-c")
    assert_equal(hits[0].segment_id, "segment-0001")
    assert_equal(hits[0].segment_index, 7)
    assert_equal(hits[0].document_index, 2)
    assert_equal(
        abs(Float64(hits[0].score) - Float64(3.0)) < 0.0000001,
        True,
    )
    assert_equal(hits[1].doc_id, "doc-a")
    assert_equal(
        abs(Float64(hits[1].score) - Float64(2.0)) < 0.0000001,
        True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
