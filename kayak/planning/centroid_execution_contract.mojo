from kayak.collections import (
    SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
)
from kayak.collections.validation import require_non_empty_string

from .candidate_generator import CandidateGenerator


comptime CENTROID_EXECUTION_SCORE_VARIANT_POSTINGS = "postings"
comptime CENTROID_EXECUTION_SCORE_VARIANT_FLAT = "flat"
comptime CENTROID_EXECUTION_SCORE_VARIANT_HEAD = "head"
comptime CENTROID_EXECUTION_SCORE_VARIANT_HEAD_AUTO = "head_auto"
comptime CENTROID_EXECUTION_SCORE_VARIANT_BLOCKMAX = "blockmax"
comptime CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED = "imputed"
comptime CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED_FLAT = "imputed_flat"

comptime CENTROID_EXECUTION_SHORTLIST_NONE = "none"
comptime CENTROID_EXECUTION_SHORTLIST_CANDIDATE_K = "candidate_k"
comptime CENTROID_EXECUTION_SHORTLIST_FINAL_K = "final_k"


struct CentroidExecutionContract(Copyable):
    var generator_kind: String
    var artifact_family: String
    var score_variant: String
    var requires_weight_sorted_postings: Bool
    var shortlist_budget_kind: String
    var consumes_full_query: Bool

    def __init__(
        out self,
        var generator_kind: String,
        var artifact_family: String,
        var score_variant: String,
        requires_weight_sorted_postings: Bool,
        var shortlist_budget_kind: String,
        consumes_full_query: Bool,
    ) raises:
        self.generator_kind = require_non_empty_string(
            generator_kind, "centroid execution generator_kind"
        )
        self.artifact_family = require_non_empty_string(
            artifact_family, "centroid execution artifact_family"
        )
        self.score_variant = require_non_empty_string(
            score_variant, "centroid execution score_variant"
        )
        self.shortlist_budget_kind = require_non_empty_string(
            shortlist_budget_kind, "centroid execution shortlist_budget_kind"
        )
        self.requires_weight_sorted_postings = requires_weight_sorted_postings
        self.consumes_full_query = consumes_full_query

        if (
            self.artifact_family != SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS
            and self.artifact_family
            != SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS
        ):
            raise Error(
                "unsupported centroid execution artifact family: "
                + self.artifact_family
            )

        if (
            self.shortlist_budget_kind != CENTROID_EXECUTION_SHORTLIST_NONE
            and self.shortlist_budget_kind
            != CENTROID_EXECUTION_SHORTLIST_CANDIDATE_K
            and self.shortlist_budget_kind != CENTROID_EXECUTION_SHORTLIST_FINAL_K
        ):
            raise Error(
                "unsupported centroid execution shortlist budget kind: "
                + self.shortlist_budget_kind
            )

    def shortlist_budget(self, candidate_k: Int, final_k: Int) -> Int:
        if self.shortlist_budget_kind == CENTROID_EXECUTION_SHORTLIST_NONE:
            return 0
        if (
            self.shortlist_budget_kind
            == CENTROID_EXECUTION_SHORTLIST_CANDIDATE_K
        ):
            return candidate_k

        return final_k


def centroid_execution_contract(
    read candidate_generator: CandidateGenerator
) raises -> CentroidExecutionContract:
    if candidate_generator.kind == "centroid_heads":
        return CentroidExecutionContract(
            candidate_generator.kind.copy(),
            SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
            CENTROID_EXECUTION_SCORE_VARIANT_POSTINGS,
            False,
            CENTROID_EXECUTION_SHORTLIST_NONE,
            False,
        )

    if candidate_generator.kind == "centroid_postings":
        return CentroidExecutionContract(
            candidate_generator.kind.copy(),
            SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
            CENTROID_EXECUTION_SCORE_VARIANT_POSTINGS,
            False,
            CENTROID_EXECUTION_SHORTLIST_NONE,
            False,
        )

    if candidate_generator.kind == "centroid_postings_flat":
        return CentroidExecutionContract(
            candidate_generator.kind.copy(),
            SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
            CENTROID_EXECUTION_SCORE_VARIANT_FLAT,
            False,
            CENTROID_EXECUTION_SHORTLIST_NONE,
            True,
        )

    if candidate_generator.kind == "centroid_postings_head":
        return CentroidExecutionContract(
            candidate_generator.kind.copy(),
            SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
            CENTROID_EXECUTION_SCORE_VARIANT_HEAD,
            True,
            CENTROID_EXECUTION_SHORTLIST_CANDIDATE_K,
            False,
        )

    if candidate_generator.kind == "centroid_postings_head_auto":
        return CentroidExecutionContract(
            candidate_generator.kind.copy(),
            SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
            CENTROID_EXECUTION_SCORE_VARIANT_HEAD_AUTO,
            True,
            CENTROID_EXECUTION_SHORTLIST_CANDIDATE_K,
            False,
        )

    if candidate_generator.kind == "centroid_postings_blockmax":
        return CentroidExecutionContract(
            candidate_generator.kind.copy(),
            SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
            CENTROID_EXECUTION_SCORE_VARIANT_BLOCKMAX,
            True,
            CENTROID_EXECUTION_SHORTLIST_CANDIDATE_K,
            False,
        )

    if candidate_generator.kind == "centroid_postings_imputed":
        return CentroidExecutionContract(
            candidate_generator.kind.copy(),
            SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
            CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED,
            False,
            CENTROID_EXECUTION_SHORTLIST_FINAL_K,
            False,
        )

    if candidate_generator.kind == "centroid_postings_imputed_flat":
        return CentroidExecutionContract(
            candidate_generator.kind.copy(),
            SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
            CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED_FLAT,
            False,
            CENTROID_EXECUTION_SHORTLIST_FINAL_K,
            True,
        )

    raise Error(
        "candidate generator kind is not a supported centroid execution kind: "
        + candidate_generator.kind
    )
