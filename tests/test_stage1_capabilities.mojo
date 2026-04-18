from std.testing import TestSuite, assert_equal

from kayak import (
    CandidateGenerator,
    SearchArtifactBuildPolicy,
    SearchArtifactBuildSpec,
    stage1_capabilities_for_candidate_generator_kind,
    stage1_requires_search_artifact_family,
    stage1_required_search_artifact_families,
    stage1_single_required_search_artifact_family,
    stage1_supported_by_search_artifact_build_policy,
)


def test_exact_stage1_capabilities_require_no_sidecars() raises:
    var capabilities = stage1_capabilities_for_candidate_generator_kind(
        "exact_full_scan"
    )

    assert_equal(capabilities.generator_family, "exact")
    assert_equal(capabilities.interaction_semantics, "exact_late_interaction")
    assert_equal(capabilities.alignment_granularity, "document_tokens")
    assert_equal(capabilities.score_kind, "exact_score")
    assert_equal(capabilities.stage1_is_exact, True)
    assert_equal(capabilities.supports_match_all_filter, True)
    assert_equal(capabilities.supports_exact_doc_id_filter, True)
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
    assert_equal(
        centroid_capabilities.interaction_semantics,
        "approximate_late_interaction",
    )
    assert_equal(centroid_capabilities.alignment_granularity, "centroid")
    assert_equal(
        centroid_capabilities.score_kind,
        "approximate_interaction_score",
    )
    assert_equal(centroid_capabilities.stage1_is_exact, False)
    assert_equal(centroid_capabilities.supports_exact_doc_id_filter, True)
    assert_equal(centroid_capabilities.supports_structured_filter, True)
    assert_equal(
        centroid_capabilities.required_search_artifact_families[0],
        "centroid_postings",
    )
    assert_equal(graph_capabilities.generator_family, "graph")
    assert_equal(graph_capabilities.supports_exact_doc_id_filter, False)
    assert_equal(
        graph_capabilities.interaction_semantics,
        "approximate_late_interaction",
    )
    assert_equal(graph_capabilities.alignment_granularity, "graph_node")
    assert_equal(
        stage1_required_search_artifact_families("gem_graph")[0],
        "gem_graph",
    )
    assert_equal(
        stage1_single_required_search_artifact_family(
            "centroid_postings_head_auto"
        ),
        "centroid_postings",
    )
    assert_equal(
        stage1_requires_search_artifact_family("gem_graph", "gem_graph"),
        True,
    )


def test_candidate_generator_carries_stage1_contract_metadata() raises:
    var generator = CandidateGenerator("document_proxy")

    assert_equal(generator.family, "proxy")
    assert_equal(generator.interaction_semantics, "none")
    assert_equal(generator.alignment_granularity, "document")
    assert_equal(generator.score_kind, "proxy_score")
    assert_equal(generator.artifact_family, "document_proxy")
    assert_equal(generator.required_search_artifact_families[0], "document_proxy")
    assert_equal(generator.is_exact, False)
    assert_equal(generator.supports_match_all_filter, True)
    assert_equal(generator.supports_exact_doc_id_filter, True)
    assert_equal(generator.supports_structured_filter, True)
    assert_equal(generator.requires_artifact_family("document_proxy"), True)
    assert_equal(generator.requires_artifact_family("gem_graph"), False)


def test_latent_proxy_candidate_generator_keeps_proxy_contract_shape() raises:
    var generator = CandidateGenerator("latent_proxy")

    assert_equal(generator.family, "proxy")
    assert_equal(generator.interaction_semantics, "none")
    assert_equal(generator.alignment_granularity, "document")
    assert_equal(generator.score_kind, "proxy_score")
    assert_equal(generator.artifact_family, "latent_proxy")
    assert_equal(generator.required_search_artifact_families[0], "latent_proxy")
    assert_equal(generator.is_exact, False)
    assert_equal(generator.requires_artifact_family("latent_proxy"), True)
    assert_equal(
        stage1_requires_search_artifact_family("latent_proxy", "latent_proxy"),
        True,
    )


def test_stage1_build_policy_support_check_is_requirement_driven() raises:
    var proxy_policy = SearchArtifactBuildPolicy(
        [SearchArtifactBuildSpec("document_proxy", "proxy_sidecar")]
    )
    var graph_policy = SearchArtifactBuildPolicy(
        [SearchArtifactBuildSpec("gem_graph", "graph_sidecar")]
    )

    assert_equal(
        stage1_supported_by_search_artifact_build_policy(
            proxy_policy,
            "document_proxy",
        ),
        True,
    )
    assert_equal(
        stage1_supported_by_search_artifact_build_policy(
            proxy_policy,
            "gem_graph",
        ),
        False,
    )
    assert_equal(
        stage1_supported_by_search_artifact_build_policy(
            graph_policy,
            "gem_graph",
        ),
        True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
