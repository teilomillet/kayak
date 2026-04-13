from std.testing import TestSuite, assert_equal

from kayak import (
    best_effort_faithfulness_policy,
    clause_text_stage3_verifier_operator,
    document_proxy_search_plan,
    exact_late_interaction_stage2_reference_operator,
    search_plan_compatibility_semantics,
    search_plan_with_stage_components,
    Stage2Operator,
    clause_text_stage2_operator,
    exact_late_interaction_clause_text_stage2_operator,
    exact_late_interaction_stage2_operator,
    noop_topk_stage2_operator,
)


def test_noop_stage2_operator_keeps_identity_contract() raises:
    var operator = noop_topk_stage2_operator()

    assert_equal(operator.kind, "noop_topk")
    assert_equal(operator.family, "identity")
    assert_equal(operator.execution_kind, "noop_topk")
    assert_equal(len(operator.required_artifact_families), 0)
    assert_equal(operator.requires_query_text, False)
    assert_equal(operator.is_exact_reference, False)
    assert_equal(operator.compatibility_exact_stage_kind, "none")
    assert_equal(operator.compatibility_reranker_kind, "none")


def test_exact_stage2_operator_keeps_late_interaction_contract() raises:
    var operator = exact_late_interaction_stage2_operator()

    assert_equal(operator.kind, "exact_late_interaction")
    assert_equal(operator.family, "late_interaction")
    assert_equal(operator.execution_kind, "exact_late_interaction")
    assert_equal(operator.required_artifact_families[0], "late_interaction")
    assert_equal(operator.requires_query_text, False)
    assert_equal(operator.is_exact_reference, True)
    assert_equal(
        operator.compatibility_exact_stage_kind,
        "exact_late_interaction",
    )
    assert_equal(operator.compatibility_reranker_kind, "none")


def test_hybrid_and_text_stage2_operators_keep_query_and_reranker_contracts() raises:
    var hybrid = exact_late_interaction_clause_text_stage2_operator()
    var text = clause_text_stage2_operator()

    assert_equal(hybrid.kind, "exact_late_interaction_clause_text")
    assert_equal(hybrid.family, "hybrid")
    assert_equal(hybrid.execution_kind, "exact_late_interaction_clause_text")
    assert_equal(len(hybrid.required_artifact_families), 2)
    assert_equal(hybrid.required_artifact_families[0], "late_interaction")
    assert_equal(hybrid.required_artifact_families[1], "document_text")
    assert_equal(hybrid.requires_query_text, True)
    assert_equal(hybrid.is_exact_reference, False)
    assert_equal(hybrid.compatibility_exact_stage_kind, "none")
    assert_equal(hybrid.compatibility_reranker_kind, "clause_text")

    assert_equal(text.kind, "clause_text")
    assert_equal(text.family, "text")
    assert_equal(text.execution_kind, "clause_text")
    assert_equal(text.required_artifact_families[0], "document_text")
    assert_equal(text.requires_query_text, True)
    assert_equal(text.is_exact_reference, False)
    assert_equal(text.compatibility_exact_stage_kind, "none")
    assert_equal(text.compatibility_reranker_kind, "clause_text")


def test_search_plan_compatibility_semantics_are_derived_from_explicit_components() raises:
    var plan = document_proxy_search_plan(
        1,
        2,
        best_effort_faithfulness_policy(),
        exact_late_interaction_clause_text_stage2_operator(),
    )
    var compatibility = search_plan_compatibility_semantics(plan)

    assert_equal(compatibility.stage2_kind, "exact_late_interaction_clause_text")
    assert_equal(compatibility.stage2_family, "hybrid")
    assert_equal(compatibility.stage2_requires_query_text, True)
    assert_equal(len(compatibility.stage2_required_artifact_families), 2)
    assert_equal(
        compatibility.stage2_required_artifact_families[0],
        "late_interaction",
    )
    assert_equal(
        compatibility.stage2_required_artifact_families[1],
        "document_text",
    )
    assert_equal(compatibility.exact_stage_kind, "exact_late_interaction")
    assert_equal(compatibility.reranker_kind, "clause_text")


def test_search_plan_with_stage_components_preserves_reference_semantics() raises:
    var plan = document_proxy_search_plan(
        1,
        2,
        best_effort_faithfulness_policy(),
    )
    var overridden = search_plan_with_stage_components(
        plan,
        exact_late_interaction_stage2_reference_operator(),
        clause_text_stage3_verifier_operator(),
    )
    var compatibility = search_plan_compatibility_semantics(overridden)

    assert_equal(
        overridden.reference_scoring_semantics.kind,
        plan.reference_scoring_semantics.kind,
    )
    assert_equal(
        overridden.stage2_reference_operator.kind,
        "exact_late_interaction",
    )
    assert_equal(overridden.stage3_verifier.kind, "clause_text")
    assert_equal(compatibility.exact_stage_kind, "exact_late_interaction")
    assert_equal(compatibility.reranker_kind, "clause_text")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
