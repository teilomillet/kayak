from std.testing import TestSuite, assert_equal

from kayak import (
    CENTROID_EXECUTION_SCORE_VARIANT_BLOCKMAX,
    CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED_FLAT,
    CENTROID_EXECUTION_SCORE_VARIANT_POSTINGS,
    CENTROID_EXECUTION_SHORTLIST_CANDIDATE_K,
    CENTROID_EXECUTION_SHORTLIST_FINAL_K,
    CENTROID_EXECUTION_SHORTLIST_NONE,
    CandidateGenerator,
    centroid_execution_contract,
)


def test_centroid_execution_contract_keeps_heads_and_postings_generic() raises:
    var heads = centroid_execution_contract(CandidateGenerator("centroid_heads"))
    var postings = centroid_execution_contract(
        CandidateGenerator("centroid_postings")
    )

    assert_equal(heads.score_variant, CENTROID_EXECUTION_SCORE_VARIANT_POSTINGS)
    assert_equal(heads.shortlist_budget_kind, CENTROID_EXECUTION_SHORTLIST_NONE)
    assert_equal(heads.requires_weight_sorted_postings, False)
    assert_equal(postings.score_variant, CENTROID_EXECUTION_SCORE_VARIANT_POSTINGS)
    assert_equal(
        postings.shortlist_budget_kind,
        CENTROID_EXECUTION_SHORTLIST_NONE,
    )


def test_centroid_execution_contract_marks_weight_sorted_variants() raises:
    var head = centroid_execution_contract(
        CandidateGenerator("centroid_postings_head")
    )
    var blockmax = centroid_execution_contract(
        CandidateGenerator("centroid_postings_blockmax")
    )

    assert_equal(head.requires_weight_sorted_postings, True)
    assert_equal(
        head.shortlist_budget_kind,
        CENTROID_EXECUTION_SHORTLIST_CANDIDATE_K,
    )
    assert_equal(blockmax.score_variant, CENTROID_EXECUTION_SCORE_VARIANT_BLOCKMAX)
    assert_equal(blockmax.requires_weight_sorted_postings, True)


def test_centroid_execution_contract_marks_imputed_flat_budget_and_query_shape() raises:
    var contract = centroid_execution_contract(
        CandidateGenerator("centroid_postings_imputed_flat")
    )

    assert_equal(
        contract.score_variant,
        CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED_FLAT,
    )
    assert_equal(
        contract.shortlist_budget_kind,
        CENTROID_EXECUTION_SHORTLIST_FINAL_K,
    )
    assert_equal(contract.consumes_full_query, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
