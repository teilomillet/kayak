# Prepared exact-stage scorer over prebuilt per-segment flat dim128 mirrors.
#
# This module owns the mirror scoring kernel and the narrow stage-2 helper used
# by explicit prepared snapshots. It does not own stage-1 candidate generation,
# prepared-snapshot policy, or stage-3 verification.

from std.algorithm.backend.cpu.parallelize import sync_parallelize
from std.collections import List
from std.sys.info import simd_width_of

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery, FlatQueryDim128, build_flat_query_dim128
from kayak.index import HybridFlatDim128Index, build_hybrid_flat_dim128_index
from kayak.numeric import (
    VECTOR_SCALAR_NAME,
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at
from kayak.scoring.maxsim import choose_parallel_work_item_count_for_shape

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .exact_stage import (
    ResolvedCandidateWindow,
    build_resolved_candidate_boundaries,
    resolve_candidate_window,
)
from .search_plan import SearchPlan
from .stage2_reference_operator import STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION
from .stage2_result import Stage2Result
from .stage_artifact_materialization import StageArtifactMaterialization
from .topk import insert_descending_collection_hit


def build_segment_hybrid_flat_dim128_mirrors(
    read snapshot: ResolvedCollectionSnapshot
) raises -> List[HybridFlatDim128Index]:
    var mirrors = List[HybridFlatDim128Index]()

    for segment in snapshot.segments:
        mirrors.append(build_hybrid_flat_dim128_index(segment.stored_index.index))

    return mirrors^


def segment_mirror_exact_rerank_supported(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read plan: SearchPlan,
) -> Bool:
    return (
        backend.scoring_config.enable_dim128_fast_path
        and VECTOR_SCALAR_NAME == "Float32"
        and query.vector_dim == COLBERT_VECTOR_DIM
        and query.vector_count == 32
        and plan.stage2_reference_operator.kind == "exact_late_interaction"
        and (
            plan.candidate_generator.kind == "centroid_postings_imputed"
            or plan.candidate_generator.kind == "centroid_postings_imputed_flat"
        )
    )


def exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4(
    read query: FlatQueryDim128,
    read hybrid_index: HybridFlatDim128Index,
    document_index: Int,
) -> ScoreScalar:
    if VECTOR_SCALAR_NAME != "Float32":
        return exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4_fallback(
            query,
            hybrid_index,
            document_index,
        )

    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        return exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4_fallback(
            query,
            hybrid_index,
            document_index,
        )

    var start_vector = hybrid_index.doc_offsets[document_index]
    var stop_vector = hybrid_index.doc_offsets[document_index + 1]
    var start_offset = start_vector * COLBERT_VECTOR_DIM
    var document_vector_count = stop_vector - start_vector
    var total = zero_score_scalar()
    var query_index = 0
    var query_values_ptr = query.token_values.unsafe_ptr()
    var token_values_ptr = hybrid_index.token_values.unsafe_ptr() + start_offset

    while query_index + 3 < query.vector_count:
        var query_offset0 = query_index * COLBERT_VECTOR_DIM
        var query_offset1 = (query_index + 1) * COLBERT_VECTOR_DIM
        var query_offset2 = (query_index + 2) * COLBERT_VECTOR_DIM
        var query_offset3 = (query_index + 3) * COLBERT_VECTOR_DIM
        var query_ptr0 = query_values_ptr + query_offset0
        var query_ptr1 = query_values_ptr + query_offset1
        var query_ptr2 = query_values_ptr + query_offset2
        var query_ptr3 = query_values_ptr + query_offset3
        var best_similarity0 = min_score_scalar()
        var best_similarity1 = min_score_scalar()
        var best_similarity2 = min_score_scalar()
        var best_similarity3 = min_score_scalar()

        for token_index in range(document_vector_count):
            var token_ptr = token_values_ptr + (token_index * COLBERT_VECTOR_DIM)
            var accum0 = SIMD[DType.float32, width](0.0)
            var accum1 = SIMD[DType.float32, width](0.0)
            var accum2 = SIMD[DType.float32, width](0.0)
            var accum3 = SIMD[DType.float32, width](0.0)

            for dim in range(0, COLBERT_VECTOR_DIM, width):
                var token_chunk = (token_ptr + dim).load[width=width]()
                accum0 += (query_ptr0 + dim).load[width=width]() * token_chunk
                accum1 += (query_ptr1 + dim).load[width=width]() * token_chunk
                accum2 += (query_ptr2 + dim).load[width=width]() * token_chunk
                accum3 += (query_ptr3 + dim).load[width=width]() * token_chunk

            var similarity0 = ScoreScalar(accum0.reduce_add()[0])
            var similarity1 = ScoreScalar(accum1.reduce_add()[0])
            var similarity2 = ScoreScalar(accum2.reduce_add()[0])
            var similarity3 = ScoreScalar(accum3.reduce_add()[0])

            if similarity0 > best_similarity0:
                best_similarity0 = similarity0
            if similarity1 > best_similarity1:
                best_similarity1 = similarity1
            if similarity2 > best_similarity2:
                best_similarity2 = similarity2
            if similarity3 > best_similarity3:
                best_similarity3 = similarity3

        total += best_similarity0
        total += best_similarity1
        total += best_similarity2
        total += best_similarity3
        query_index += 4

    while query_index < query.vector_count:
        var query_offset = query_index * COLBERT_VECTOR_DIM
        var best_similarity = min_score_scalar()

        for token_index in range(document_vector_count):
            var similarity = dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                hybrid_index.token_values,
                start_offset + (token_index * COLBERT_VECTOR_DIM),
            )
            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity
        query_index += 1

    return total


def exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4_fallback(
    read query: FlatQueryDim128,
    read hybrid_index: HybridFlatDim128Index,
    document_index: Int,
) -> ScoreScalar:
    var start_vector = hybrid_index.doc_offsets[document_index]
    var stop_vector = hybrid_index.doc_offsets[document_index + 1]
    var start_offset = start_vector * COLBERT_VECTOR_DIM
    var document_vector_count = stop_vector - start_vector
    var total = zero_score_scalar()

    for query_index in range(query.vector_count):
        var query_offset = query_index * COLBERT_VECTOR_DIM
        var best_similarity = min_score_scalar()

        for token_index in range(document_vector_count):
            var similarity = dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                hybrid_index.token_values,
                start_offset + (token_index * COLBERT_VECTOR_DIM),
            )
            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def score_resolved_candidate_window_for_cpu_with_segment_mirrors_dim128_tiled4(
    read backend: ExactCpuBackend,
    read query: FlatQueryDim128,
    read mirrors: List[HybridFlatDim128Index],
    read resolved: ResolvedCandidateWindow,
) raises -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for _ in range(len(resolved.documents)):
        scores.append(zero_score_scalar())

    var work_item_count = choose_parallel_work_item_count_for_shape(
        query.vector_count,
        len(resolved.documents),
        resolved.vector_count,
        backend.scoring_config,
    )
    var scores_ptr = scores.unsafe_ptr()

    if work_item_count <= 1:
        for document_index in range(len(resolved.documents)):
            var resolved_document = resolved.documents[document_index].copy()
            scores_ptr[document_index] = (
                exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4(
                    query,
                    mirrors[resolved_document.segment_index],
                    resolved_document.document_index,
                )
            )
        return scores^

    var boundaries = build_resolved_candidate_boundaries(
        resolved, work_item_count
    )

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            var resolved_document = resolved.documents[document_index].copy()
            scores_ptr[document_index] = (
                exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4(
                    query,
                    mirrors[resolved_document.segment_index],
                    resolved_document.document_index,
                )
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def reference_stage_output_k_with_segment_mirrors(
    read candidate_set: CandidateSet,
    read plan: SearchPlan,
) -> Int:
    if plan.stage3_verifier.kind == "none":
        return plan.candidate_budget.final_k

    var limit = plan.candidate_budget.candidate_k
    if limit > len(candidate_set.hits):
        limit = len(candidate_set.hits)
    return limit


def exact_rerank_candidates_for_plan_with_segment_mirrors(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read mirrors: List[HybridFlatDim128Index],
    read hits: List[CollectionHit],
    final_k: Int,
) raises -> Stage2Result:
    if len(hits) == 0:
        return Stage2Result([], 0, 0, 0, 0, 0)

    var resolved = resolve_candidate_window(snapshot, hits)
    var scores = (
        score_resolved_candidate_window_for_cpu_with_segment_mirrors_dim128_tiled4(
            backend,
            build_flat_query_dim128(query),
            mirrors,
            resolved,
        )
    )
    var final_hits = List[CollectionHit]()

    for document_index in range(len(scores)):
        var resolved_document = resolved.documents[document_index].copy()
        insert_descending_collection_hit(
            final_hits,
            CollectionHit(
                resolved_document.segment_id.copy(),
                resolved_document.doc_id.copy(),
                scores[document_index],
                resolved_document.segment_index,
                resolved_document.document_index,
            ),
            final_k,
        )

    return Stage2Result(
        final_hits^,
        [
            StageArtifactMaterialization(
                STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION,
                resolved.segment_count,
                len(resolved.documents),
                resolved.token_count,
                resolved.vector_count,
                resolved.byte_size,
            )
        ],
        resolved.segment_count,
        len(resolved.documents),
        resolved.token_count,
        resolved.vector_count,
        resolved.byte_size,
    )


def stage2_result_for_plan_with_segment_mirrors(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read mirrors: List[HybridFlatDim128Index],
    read candidate_set: CandidateSet,
    read plan: SearchPlan,
) raises -> Stage2Result:
    if plan.stage2_reference_operator.kind != "exact_late_interaction":
        raise Error(
            "segment mirrors only support exact_late_interaction stage-2, got "
            + plan.stage2_reference_operator.kind
        )

    return exact_rerank_candidates_for_plan_with_segment_mirrors(
        backend,
        query,
        snapshot,
        mirrors,
        candidate_set.hits,
        reference_stage_output_k_with_segment_mirrors(candidate_set, plan),
    )
