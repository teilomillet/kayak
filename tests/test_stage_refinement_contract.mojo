from std.testing import TestSuite, assert_equal

from kayak import (
    SearchPlan,
    best_effort_faithfulness_policy,
    clause_text_stage3_verifier_operator,
    document_proxy_search_plan,
    exact_late_interaction_reference_scoring_semantics,
    exact_late_interaction_stage2_reference_operator,
    none_stage3_verifier_operator,
    noop_topk_stage2_reference_operator,
)


def test_noop_stage2_reference_operator_keeps_identity_contract() raises:
    var operator = noop_topk_stage2_reference_operator()

    assert_equal(operator.kind, "noop_topk")
    assert_equal(operator.family, "identity")
    assert_equal(len(operator.required_artifact_families), 0)
    assert_equal(operator.executes_reference_scoring, False)


def test_exact_stage2_reference_operator_keeps_late_interaction_contract() raises:
    var operator = exact_late_interaction_stage2_reference_operator()

    assert_equal(operator.kind, "exact_late_interaction")
    assert_equal(operator.family, "late_interaction")
    assert_equal(operator.required_artifact_families[0], "late_interaction")
    assert_equal(operator.executes_reference_scoring, True)


def test_stage3_verifiers_keep_identity_and_text_contracts() raises:
    var none_verifier = none_stage3_verifier_operator()
    var clause_verifier = clause_text_stage3_verifier_operator()

    assert_equal(none_verifier.kind, "none")
    assert_equal(none_verifier.family, "identity")
    assert_equal(len(none_verifier.required_artifact_families), 0)
    assert_equal(none_verifier.requires_query_text, False)

    assert_equal(clause_verifier.kind, "clause_text")
    assert_equal(clause_verifier.family, "text")
    assert_equal(clause_verifier.required_artifact_families[0], "document_text")
    assert_equal(clause_verifier.requires_query_text, True)


def test_document_proxy_plan_defaults_keep_explicit_stage_contracts() raises:
    var plan = document_proxy_search_plan(
        1,
        2,
        best_effort_faithfulness_policy(),
    )

    assert_equal(plan.reference_scoring_semantics.kind, "exact_late_interaction")
    assert_equal(plan.stage2_reference_operator.kind, "exact_late_interaction")
    assert_equal(plan.stage3_verifier.kind, "none")


def test_explicit_search_plan_construction_preserves_reference_semantics() raises:
    var plan = document_proxy_search_plan(
        1,
        2,
        best_effort_faithfulness_policy(),
    )
    var overridden = SearchPlan(
        plan.candidate_generator,
        plan.candidate_budget,
        plan.faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        clause_text_stage3_verifier_operator(),
    )

    assert_equal(
        overridden.reference_scoring_semantics.kind,
        "exact_late_interaction",
    )
    assert_equal(
        overridden.stage2_reference_operator.kind,
        "exact_late_interaction",
    )
    assert_equal(overridden.stage3_verifier.kind, "clause_text")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
