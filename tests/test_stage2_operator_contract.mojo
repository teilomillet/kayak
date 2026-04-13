from std.testing import TestSuite, assert_equal

from kayak import (
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
