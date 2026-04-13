from std.testing import TestSuite, assert_equal

from kayak import (
    stage1_capabilities_for_candidate_generator_kind,
    stage1_required_search_artifact_families,
)


def test_exact_stage1_capabilities_require_no_sidecars() raises:
    var capabilities = stage1_capabilities_for_candidate_generator_kind(
        "exact_full_scan"
    )

    assert_equal(capabilities.generator_family, "exact")
    assert_equal(capabilities.stage1_is_exact, True)
    assert_equal(capabilities.supports_match_all_filter, True)
    assert_equal(capabilities.supports_structured_filter, True)
    assert_equal(len(capabilities.required_search_artifact_families), 0)


def test_centroid_and_graph_stage1_capabilities_remain_generic() raises:
    var centroid_capabilities = stage1_capabilities_for_candidate_generator_kind(
        "centroid_postings_head_auto"
    )
    var graph_capabilities = stage1_capabilities_for_candidate_generator_kind(
        "gem_graph"
    )

    assert_equal(centroid_capabilities.generator_family, "centroid")
    assert_equal(centroid_capabilities.stage1_is_exact, False)
    assert_equal(centroid_capabilities.supports_structured_filter, False)
    assert_equal(
        centroid_capabilities.required_search_artifact_families[0],
        "centroid_postings",
    )
    assert_equal(graph_capabilities.generator_family, "graph")
    assert_equal(
        stage1_required_search_artifact_families("gem_graph")[0],
        "gem_graph",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
