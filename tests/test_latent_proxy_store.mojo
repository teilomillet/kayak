from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import VECTOR_SCALAR_NAME
from kayak.index import (
    LATENT_PROXY_ACTIVATION_RELU,
    LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION,
    LatentProxyIndex,
    LatentQueryProjection,
    LatentQueryProjectionBlock,
)
from kayak.storage import (
    StoredLatentProxyIndex,
    latent_proxy_storage_byte_size,
    load_stored_latent_proxy_index,
    save_stored_latent_proxy_index,
)


def test_latent_proxy_store_roundtrip_preserves_projection_and_vectors() raises:
    var root = Path("/tmp/kayak-latent-proxy-store")
    var stored = StoredLatentProxyIndex(
        "dataset://tiny",
        "lemur-native",
        VECTOR_SCALAR_NAME,
        2,
        0,
        LatentQueryProjection(
            2,
            2,
            32.0,
            [
                LatentQueryProjectionBlock(
                    LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION,
                    LATENT_PROXY_ACTIVATION_RELU,
                    2,
                    2,
                    [[1.0, 0.0], [0.0, 1.0]],
                    [0.25, -0.5],
                    1.0,
                    True,
                    0.00001,
                    [1.5, 0.75],
                    [0.0, 0.125],
                )
            ],
        ),
        LatentProxyIndex(
            ["doc-a", "doc-b"],
            [[1.0, -1.0], [-1.0, 1.0]],
            2,
        ),
    )

    save_stored_latent_proxy_index(root, stored)
    var loaded = load_stored_latent_proxy_index(root)

    assert_equal(loaded.dataset_id, "dataset://tiny")
    assert_equal(loaded.model_name, "lemur-native")
    assert_equal(loaded.vector_scalar_name, VECTOR_SCALAR_NAME)
    assert_equal(loaded.input_vector_dim, 2)
    assert_equal(loaded.index.vector_dim, 2)
    assert_equal(loaded.index.document_count, 2)
    assert_equal(loaded.index.doc_ids[0], "doc-a")
    assert_equal(loaded.index.doc_ids[1], "doc-b")
    assert_equal(loaded.index.proxy_vectors[0][0], 1.0)
    assert_equal(loaded.index.proxy_vectors[0][1], -1.0)
    assert_equal(loaded.query_projection.query_divisor, 32.0)
    assert_equal(len(loaded.query_projection.blocks), 1)
    assert_equal(
        loaded.query_projection.blocks[0].order_kind,
        LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION,
    )
    assert_equal(
        loaded.query_projection.blocks[0].activation_kind,
        LATENT_PROXY_ACTIVATION_RELU,
    )
    assert_equal(loaded.query_projection.blocks[0].linear_rows[0][0], 1.0)
    assert_equal(loaded.query_projection.blocks[0].linear_rows[1][1], 1.0)
    assert_equal(loaded.query_projection.blocks[0].linear_bias[0], 0.25)
    assert_equal(loaded.query_projection.blocks[0].linear_bias[1], -0.5)
    assert_equal(loaded.query_projection.blocks[0].activation_output_scale, 1.0)
    assert_equal(loaded.query_projection.blocks[0].layer_norm_affine, True)
    assert_equal(loaded.query_projection.blocks[0].layer_norm_weight[0], 1.5)
    assert_equal(loaded.query_projection.blocks[0].layer_norm_weight[1], 0.75)
    assert_equal(loaded.query_projection.blocks[0].layer_norm_bias[1], 0.125)
    assert_equal(
        loaded.artifact_byte_size,
        latent_proxy_storage_byte_size(root),
    )
    assert_equal(loaded.artifact_byte_size > 0, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
