from std.collections import List

from kayak.collections import (
    LoadedSealedSegment,
    SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    ResolvedCollectionSnapshot,
    loaded_search_artifact_stored_centroid_postings_index,
    loaded_segment_has_search_artifact,
    loaded_segment_search_artifact,
)
from kayak.contracts import EncodedQuery
from kayak.filters import FilterExpression, match_all_filter
from kayak.numeric import ScoreScalar
from kayak.runtime import ExactScoringBackend
from kayak.storage import CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC

from .candidate_set import CandidateSet
from .centroid_execution_contract import (
    CENTROID_EXECUTION_SCORE_VARIANT_BLOCKMAX,
    CENTROID_EXECUTION_SCORE_VARIANT_FLAT,
    CENTROID_EXECUTION_SCORE_VARIANT_HEAD,
    CENTROID_EXECUTION_SCORE_VARIANT_HEAD_AUTO,
    CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED,
    CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED_FLAT,
    CENTROID_EXECUTION_SCORE_VARIANT_POSTINGS,
    CentroidExecutionContract,
    centroid_execution_contract,
)
from .centroid_postings_blockmax_stage import (
    centroid_posting_blockmax_scores_for_segment,
)
from .centroid_postings_flat_stage import (
    centroid_posting_flat_scores_for_segment,
)
from .centroid_postings_head_auto_stage import (
    centroid_posting_head_auto_scores_for_segment,
)
from .centroid_postings_head_stage import (
    centroid_posting_head_scores_for_segment,
)
from .centroid_postings_imputed_flat_stage import (
    centroid_posting_imputed_flat_scores_for_segment,
)
from .centroid_postings_imputed_stage import (
    centroid_posting_imputed_scores_for_segment,
)
from .centroid_postings_stage import centroid_posting_scores_for_segment
from .collection_hit import CollectionHit
from .search_plan import SearchPlan
from .topk import insert_descending_collection_hit


def require_centroid_artifact_present(
    read segment: LoadedSealedSegment,
    read contract: CentroidExecutionContract,
) raises:
    if contract.artifact_family == SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS:
        if not loaded_segment_has_centroid_heads_index(segment):
            raise Error(
                "centroid_heads stage-1 requires a centroid heads sidecar for every segment"
            )
        return

    if not loaded_segment_has_centroid_postings_index(segment):
        raise Error(
            contract.generator_kind
            + " stage-1 requires a centroid postings sidecar for every segment"
        )


def insert_centroid_scores(
    mut hits: List[CollectionHit],
    segment_id: String,
    read doc_ids: List[String],
    read scores: List[ScoreScalar],
    candidate_k: Int,
):
    for document_index in range(len(scores)):
        insert_descending_collection_hit(
            hits,
            CollectionHit(
                segment_id.copy(),
                doc_ids[document_index].copy(),
                scores[document_index],
            ),
            candidate_k,
        )


def candidate_generation_for_centroid_family[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read filter_expression: FilterExpression = match_all_filter(),
) raises -> CandidateSet:
    _ = backend
    _ = filter_expression

    var hits = List[CollectionHit]()
    var vector_count = 0
    var token_count = 0
    var byte_size = 0
    var contract = centroid_execution_contract(plan.candidate_generator)

    for segment in snapshot.segments:
        require_centroid_artifact_present(segment, contract)

        var stored_centroid = loaded_segment_stored_centroid_postings_index(
            segment, contract.artifact_family
        )

        if contract.requires_weight_sorted_postings:
            if (
                stored_centroid.posting_order_kind
                != CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC
            ):
                raise Error(
                    contract.generator_kind
                    + " stage-1 requires weight-sorted centroid postings sidecars"
                )

        vector_count += stored_centroid.index.centroid_count
        token_count += stored_centroid.index.total_posting_count
        byte_size += stored_centroid.artifact_byte_size

        var shortlist_budget = contract.shortlist_budget(
            plan.candidate_budget.candidate_k,
            plan.candidate_budget.final_k,
        )
        var scores = centroid_posting_scores_for_segment(
            query.token_vectors,
            stored_centroid.index,
        )
        if contract.score_variant == CENTROID_EXECUTION_SCORE_VARIANT_FLAT:
            scores = centroid_posting_flat_scores_for_segment(
                query,
                stored_centroid.index,
            )
        elif contract.score_variant == CENTROID_EXECUTION_SCORE_VARIANT_HEAD:
            scores = centroid_posting_head_scores_for_segment(
                query.token_vectors,
                stored_centroid.index,
                shortlist_budget,
            )
        elif contract.score_variant == CENTROID_EXECUTION_SCORE_VARIANT_HEAD_AUTO:
            scores = centroid_posting_head_auto_scores_for_segment(
                query.token_vectors,
                stored_centroid.index,
                shortlist_budget,
            )
        elif contract.score_variant == CENTROID_EXECUTION_SCORE_VARIANT_BLOCKMAX:
            scores = centroid_posting_blockmax_scores_for_segment(
                query.token_vectors,
                stored_centroid.index,
                shortlist_budget,
            )
        elif contract.score_variant == CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED:
            scores = centroid_posting_imputed_scores_for_segment(
                query.token_vectors,
                stored_centroid.index,
                shortlist_budget,
            )
        elif (
            contract.score_variant == CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED_FLAT
        ):
            scores = centroid_posting_imputed_flat_scores_for_segment(
                query,
                stored_centroid.index,
                shortlist_budget,
            )
        elif contract.score_variant != CENTROID_EXECUTION_SCORE_VARIANT_POSTINGS:
            raise Error(
                "unsupported centroid execution score variant: "
                + contract.score_variant
            )

        insert_centroid_scores(
            hits,
            segment.manifest.segment_id.value,
            segment.stored_index.index.doc_ids,
            scores,
            plan.candidate_budget.candidate_k,
        )

    return CandidateSet(
        plan.candidate_generator.kind.copy(),
        hits^,
        snapshot.snapshot.stats.segment_count,
        snapshot.snapshot.stats.document_count,
        token_count,
        vector_count,
        byte_size,
    )
