from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak.storage import (
    StoredGemGraphIndex,
    gem_graph_index_exists,
    load_stored_gem_graph_index,
    save_stored_gem_graph_index,
)
from kayak.numeric import VECTOR_SCALAR_NAME


def test_gem_graph_store_roundtrip_preserves_metadata() raises:
    var root = Path("/tmp/kayak-gem-graph-store-roundtrip")

    save_stored_gem_graph_index(
        root,
        StoredGemGraphIndex(
            "collection://news",
            "colbertv2",
            VECTOR_SCALAR_NAME,
            3,
            7,
            11,
            2,
            4,
            64,
            0,
        ),
    )

    var loaded = load_stored_gem_graph_index(root)

    assert_equal(gem_graph_index_exists(root), True)
    assert_equal(loaded.dataset_id, "collection://news")
    assert_equal(loaded.document_count, 3)
    assert_equal(loaded.cluster_count, 7)
    assert_equal(loaded.graph_edge_count, 11)
    assert_equal(loaded.shortcut_edge_count, 2)
    assert_equal(loaded.entry_point_count, 4)
    assert_equal(loaded.quantization_centroid_count, 64)
    assert_equal(loaded.artifact_byte_size > 0, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
