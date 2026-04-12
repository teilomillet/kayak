from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.index import build_query_proxy_vector
from kayak.numeric import MetricScalar
from kayak.runtime import ExactScoringBackend
from kayak.scoring.dot import dot_product
from kayak.search import SearchHit
from kayak.storage import CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC

from .candidate_set import CandidateSet
from .centroid_postings_head_stage import (
    centroid_posting_head_scores_for_segment,
)
from .centroid_postings_imputed_stage import (
    centroid_posting_imputed_scores_for_segment,
)
from .centroid_postings_stage import centroid_posting_scores_for_segment
from .collection_hit import CollectionHit, to_search_hit
from .exact_stage import ExactStageResult, exact_rerank_candidates_for_plan
from .search_plan import SearchPlan
from .topk import insert_descending_collection_hit


def candidate_generation_for_plan[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> CandidateSet:
    var hits = List[CollectionHit]()
    if plan.candidate_generator.kind == "exact_full_scan":
        for segment in snapshot.segments:
            var scores = backend.score_all(query, segment.stored_index.index)

            for index in range(len(scores)):
                insert_descending_collection_hit(
                    hits,
                    CollectionHit(
                        segment.manifest.segment_id.value.copy(),
                        segment.stored_index.index.doc_ids[index].copy(),
                        scores[index],
                    ),
                    plan.candidate_budget.candidate_k,
                )

        return CandidateSet(
            plan.candidate_generator.kind.copy(),
            hits^,
            snapshot.snapshot.stats.segment_count,
            snapshot.snapshot.stats.document_count,
            snapshot.snapshot.stats.token_count,
            snapshot.snapshot.stats.total_vector_count,
            snapshot.snapshot.stats.byte_size,
        )

    if plan.candidate_generator.kind == "document_proxy":
        var query_proxy = build_query_proxy_vector(query, 0)
        var vector_count = 0
        var byte_size = 0

        for segment in snapshot.segments:
            if not segment.has_document_proxy_index:
                raise Error(
                    "document_proxy stage-1 requires a document proxy sidecar for every segment"
                )

            vector_count += (
                segment.stored_document_proxy_index.index.document_count
                * segment.stored_document_proxy_index.proxy_vector_count_per_document
            )
            byte_size += segment.stored_document_proxy_index.artifact_byte_size

            for document_index in range(
                segment.stored_document_proxy_index.index.document_count
            ):
                insert_descending_collection_hit(
                    hits,
                    CollectionHit(
                        segment.manifest.segment_id.value.copy(),
                        segment.stored_document_proxy_index.index.doc_ids[
                            document_index
                        ].copy(),
                        dot_product(
                            query_proxy,
                            segment.stored_document_proxy_index.index.proxy_vectors[
                                document_index
                            ],
                        ),
                    ),
                    plan.candidate_budget.candidate_k,
                )

        return CandidateSet(
            plan.candidate_generator.kind.copy(),
            hits^,
            snapshot.snapshot.stats.segment_count,
            snapshot.snapshot.stats.document_count,
            0,
            vector_count,
            byte_size,
        )

    if plan.candidate_generator.kind == "centroid_postings":
        var vector_count = 0
        var token_count = 0
        var byte_size = 0

        for segment in snapshot.segments:
            if not segment.has_centroid_postings_index:
                raise Error(
                    "centroid_postings stage-1 requires a centroid postings sidecar for every segment"
                )

            vector_count += segment.stored_centroid_postings_index.index.centroid_count
            token_count += (
                segment.stored_centroid_postings_index.index.total_posting_count
            )
            byte_size += segment.stored_centroid_postings_index.artifact_byte_size

            var scores = centroid_posting_scores_for_segment(
                query.token_vectors,
                segment.stored_centroid_postings_index.index,
            )

            for document_index in range(len(scores)):
                insert_descending_collection_hit(
                    hits,
                    CollectionHit(
                        segment.manifest.segment_id.value.copy(),
                        segment.stored_index.index.doc_ids[document_index].copy(),
                        scores[document_index],
                    ),
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

    if plan.candidate_generator.kind == "centroid_postings_head":
        var vector_count = 0
        var token_count = 0
        var byte_size = 0

        for segment in snapshot.segments:
            if not segment.has_centroid_postings_index:
                raise Error(
                    "centroid_postings_head stage-1 requires a centroid postings sidecar for every segment"
                )
            if (
                segment.stored_centroid_postings_index.posting_order_kind
                != CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC
            ):
                raise Error(
                    "centroid_postings_head stage-1 requires weight-sorted centroid postings sidecars"
                )

            vector_count += segment.stored_centroid_postings_index.index.centroid_count
            token_count += (
                segment.stored_centroid_postings_index.index.total_posting_count
            )
            byte_size += segment.stored_centroid_postings_index.artifact_byte_size

            var scores = centroid_posting_head_scores_for_segment(
                query.token_vectors,
                segment.stored_centroid_postings_index.index,
                plan.candidate_budget.candidate_k,
            )

            for document_index in range(len(scores)):
                insert_descending_collection_hit(
                    hits,
                    CollectionHit(
                        segment.manifest.segment_id.value.copy(),
                        segment.stored_index.index.doc_ids[document_index].copy(),
                        scores[document_index],
                    ),
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

    if plan.candidate_generator.kind == "centroid_postings_imputed":
        var vector_count = 0
        var token_count = 0
        var byte_size = 0

        for segment in snapshot.segments:
            if not segment.has_centroid_postings_index:
                raise Error(
                    "centroid_postings_imputed stage-1 requires a centroid postings sidecar for every segment"
                )

            vector_count += segment.stored_centroid_postings_index.index.centroid_count
            token_count += (
                segment.stored_centroid_postings_index.index.total_posting_count
            )
            byte_size += segment.stored_centroid_postings_index.artifact_byte_size

            var scores = centroid_posting_imputed_scores_for_segment(
                query.token_vectors,
                segment.stored_centroid_postings_index.index,
                plan.candidate_budget.final_k,
            )

            for document_index in range(len(scores)):
                insert_descending_collection_hit(
                    hits,
                    CollectionHit(
                        segment.manifest.segment_id.value.copy(),
                        segment.stored_index.index.doc_ids[document_index].copy(),
                        scores[document_index],
                    ),
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

    raise Error(
        "unsupported candidate generator kind: " + plan.candidate_generator.kind
    )


def collection_hit_matches(
    read lhs: CollectionHit, read rhs: CollectionHit
) -> Bool:
    return lhs.segment_id == rhs.segment_id and lhs.doc_id == rhs.doc_id


def candidate_recall_at_final_k(
    read candidate_set: CandidateSet, final_hits: List[CollectionHit]
) -> MetricScalar:
    if len(final_hits) == 0:
        return MetricScalar(0.0)

    var found_count = 0

    for final_hit in final_hits:
        for candidate_hit in candidate_set.hits:
            if collection_hit_matches(final_hit, candidate_hit):
                found_count += 1
                break

    return MetricScalar(found_count) / MetricScalar(len(final_hits))


def final_hits_for_plan[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_set: CandidateSet,
    read plan: SearchPlan,
) raises -> ExactStageResult:
    return exact_rerank_candidates_for_plan(
        backend,
        query,
        snapshot,
        candidate_set.hits,
        plan.candidate_budget.final_k,
    )


def final_hits_to_search_hits(
    read hits: List[CollectionHit]
) -> List[SearchHit]:
    var search_hits = List[SearchHit]()

    for hit in hits:
        search_hits.append(to_search_hit(hit))

    return search_hits^
