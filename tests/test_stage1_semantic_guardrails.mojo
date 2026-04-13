from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    CandidateGenerator,
    SearchPlan,
    best_effort_faithfulness_policy,
    exact_full_scan_search_plan,
    gem_graph_search_plan,
    registered_search_planner_candidate_generator_kinds,
    stage1_capabilities_for_candidate_generator_kind,
)
from kayak.benchmarks.search_plan_semantics_json import (
    append_search_plan_semantics_json_fields,
)


def assert_string_lists_equal(
    read actual: List[String], read expected: List[String]
) raises:
    assert_equal(len(actual), len(expected))
    for index in range(len(expected)):
        assert_equal(actual[index], expected[index])


def candidate_generator_for_guardrail(kind: String) raises -> CandidateGenerator:
    if kind == "gem_graph":
        return CandidateGenerator(kind, 4, 17)
    return CandidateGenerator(kind)


def assert_stage1_contract(
    kind: String,
    family: String,
    interaction_semantics: String,
    alignment_granularity: String,
    score_kind: String,
    read required_search_artifact_families: List[String],
    stage1_is_exact: Bool,
    supports_match_all_filter: Bool,
    supports_exact_doc_id_filter: Bool,
    supports_structured_filter: Bool,
) raises:
    var capabilities = stage1_capabilities_for_candidate_generator_kind(kind)

    assert_equal(capabilities.generator_kind, kind)
    assert_equal(capabilities.generator_family, family)
    assert_equal(capabilities.interaction_semantics, interaction_semantics)
    assert_equal(capabilities.alignment_granularity, alignment_granularity)
    assert_equal(capabilities.score_kind, score_kind)
    assert_string_lists_equal(
        capabilities.required_search_artifact_families,
        required_search_artifact_families,
    )
    assert_equal(capabilities.stage1_is_exact, stage1_is_exact)
    assert_equal(capabilities.supports_match_all_filter, supports_match_all_filter)
    assert_equal(
        capabilities.supports_exact_doc_id_filter,
        supports_exact_doc_id_filter,
    )
    assert_equal(
        capabilities.supports_structured_filter,
        supports_structured_filter,
    )

    var generator = candidate_generator_for_guardrail(kind)
    assert_equal(generator.kind, kind)
    assert_equal(generator.family, family)
    assert_equal(generator.interaction_semantics, interaction_semantics)
    assert_equal(generator.alignment_granularity, alignment_granularity)
    assert_equal(generator.score_kind, score_kind)
    assert_string_lists_equal(
        generator.required_search_artifact_families,
        required_search_artifact_families,
    )
    assert_equal(generator.is_exact, stage1_is_exact)
    assert_equal(generator.supports_match_all_filter, supports_match_all_filter)
    assert_equal(
        generator.supports_exact_doc_id_filter,
        supports_exact_doc_id_filter,
    )
    assert_equal(
        generator.supports_structured_filter,
        supports_structured_filter,
    )


def search_plan_semantics_json(read plan: SearchPlan) -> String:
    var buffer = "{"
    append_search_plan_semantics_json_fields(
        buffer,
        plan,
    )
    buffer += "}"
    return buffer


def test_registered_stage1_generators_keep_explicit_semantic_contracts() raises:
    var expected_registered_kinds = List[String]()
    expected_registered_kinds.append("exact_full_scan")
    expected_registered_kinds.append("document_proxy")
    expected_registered_kinds.append("centroid_postings_flat")
    expected_registered_kinds.append("centroid_postings_imputed_flat")
    expected_registered_kinds.append("centroid_postings")
    expected_registered_kinds.append("gem_graph")
    expected_registered_kinds.append("centroid_heads")
    expected_registered_kinds.append("centroid_postings_head")
    expected_registered_kinds.append("centroid_postings_head_auto")
    expected_registered_kinds.append("centroid_postings_blockmax")
    expected_registered_kinds.append("centroid_postings_imputed")
    assert_string_lists_equal(
        registered_search_planner_candidate_generator_kinds(),
        expected_registered_kinds,
    )

    assert_stage1_contract(
        "exact_full_scan",
        "exact",
        "exact_late_interaction",
        "document_tokens",
        "exact_score",
        [],
        True,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "document_proxy",
        "proxy",
        "none",
        "document",
        "proxy_score",
        ["document_proxy"],
        False,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "centroid_heads",
        "centroid",
        "approximate_late_interaction",
        "centroid",
        "approximate_interaction_score",
        ["centroid_heads"],
        False,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "centroid_postings",
        "centroid",
        "approximate_late_interaction",
        "centroid",
        "approximate_interaction_score",
        ["centroid_postings"],
        False,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "centroid_postings_flat",
        "centroid",
        "approximate_late_interaction",
        "centroid",
        "approximate_interaction_score",
        ["centroid_postings"],
        False,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "centroid_postings_head",
        "centroid",
        "approximate_late_interaction",
        "centroid",
        "approximate_interaction_score",
        ["centroid_postings"],
        False,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "centroid_postings_head_auto",
        "centroid",
        "approximate_late_interaction",
        "centroid",
        "approximate_interaction_score",
        ["centroid_postings"],
        False,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "centroid_postings_blockmax",
        "centroid",
        "approximate_late_interaction",
        "centroid",
        "approximate_interaction_score",
        ["centroid_postings"],
        False,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "centroid_postings_imputed",
        "centroid",
        "approximate_late_interaction",
        "centroid",
        "approximate_interaction_score",
        ["centroid_postings"],
        False,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "centroid_postings_imputed_flat",
        "centroid",
        "approximate_late_interaction",
        "centroid",
        "approximate_interaction_score",
        ["centroid_postings"],
        False,
        True,
        True,
        True,
    )
    assert_stage1_contract(
        "gem_graph",
        "graph",
        "approximate_late_interaction",
        "graph_node",
        "approximate_interaction_score",
        ["gem_graph"],
        False,
        True,
        False,
        False,
    )


def test_search_plan_semantics_json_keeps_stage1_fields_without_compatibility() raises:
    var exact_json = search_plan_semantics_json(exact_full_scan_search_plan(5, 5))
    assert_equal(
        exact_json.find("\"candidate_generator_kind\":\"exact_full_scan\"") != -1,
        True,
    )
    assert_equal(
        exact_json.find("\"candidate_generator_family\":\"exact\"") != -1,
        True,
    )
    assert_equal(
        exact_json.find("\"stage1_interaction_semantics\":\"exact_late_interaction\"")
            != -1,
        True,
    )
    assert_equal(
        exact_json.find("\"stage1_alignment_granularity\":\"document_tokens\"")
            != -1,
        True,
    )
    assert_equal(
        exact_json.find("\"stage1_score_kind\":\"exact_score\"") != -1,
        True,
    )
    assert_equal(
        exact_json.find("\"stage1_required_artifact_families\":[]") != -1,
        True,
    )
    assert_equal(
        exact_json.find("\"graph_cluster_top_k_per_query_token\":0") != -1,
        True,
    )
    assert_equal(exact_json.find("\"graph_beam_width\":0") != -1, True)
    assert_equal(exact_json.find("\"compatibility_stage2_kind\"") == -1, True)

    var graph_json = search_plan_semantics_json(
        gem_graph_search_plan(
            5,
            20,
            best_effort_faithfulness_policy(),
            4,
            17,
        ),
    )
    assert_equal(
        graph_json.find("\"candidate_generator_kind\":\"gem_graph\"") != -1,
        True,
    )
    assert_equal(
        graph_json.find("\"candidate_generator_family\":\"graph\"") != -1,
        True,
    )
    assert_equal(
        graph_json.find(
            "\"stage1_interaction_semantics\":\"approximate_late_interaction\""
        ) != -1,
        True,
    )
    assert_equal(
        graph_json.find("\"stage1_alignment_granularity\":\"graph_node\"") != -1,
        True,
    )
    assert_equal(
        graph_json.find(
            "\"stage1_score_kind\":\"approximate_interaction_score\""
        ) != -1,
        True,
    )
    assert_equal(
        graph_json.find("\"stage1_required_artifact_families\":[\"gem_graph\"]")
            != -1,
        True,
    )
    assert_equal(
        graph_json.find("\"graph_cluster_top_k_per_query_token\":4") != -1,
        True,
    )
    assert_equal(graph_json.find("\"graph_beam_width\":17") != -1, True)
    assert_equal(graph_json.find("\"compatibility_stage2_kind\"") == -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
