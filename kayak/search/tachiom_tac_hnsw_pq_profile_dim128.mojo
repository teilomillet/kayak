import std.benchmark as benchmark

from kayak.contracts import FlatQueryDim128
from kayak.numeric import ScoreScalar, zero_score_scalar

from .plaid_approx_dim128 import require_positive_int, top_positions_by_score
from .tachiom_tac_hnsw_pq_dim128 import (
    PreparedTachiomTacHnswPqIndex,
    tachiom_tac_hnsw_pq_candidate_positions_for_query,
    tachiom_tac_hnsw_pq_candidate_positions_for_query_with_pruning,
    tachiom_tac_hnsw_pq_search_positions_for_query,
    tachiom_tac_hnsw_pq_search_positions_for_query_with_pruning,
)
from .tachiom_tac_hnsw_pq_profile_helpers_dim128 import (
    accumulate_hnsw_pq_candidate_document_scores_from_centroids,
    build_hnsw_pq_centroid_positions_by_query_vector,
    hnsw_pq_profile_candidate_positions_from_scores,
    hnsw_pq_rerank_scores_for_candidates,
    hnsw_pq_winners_from_rerank_scores,
    regular_hnsw_pq_document_vector_count,
)
from .tachiom_tac_pq_dim128 import build_tachiom_pq_residual_score_table
from .tachiom_tac_profile_types_dim128 import TachiomTacHnswPqQueryProfile


# Benchmark-only profile boundaries for native streaming HNSW+PQ search. The
# production search module remains responsible for query execution.
def profile_tachiom_tac_hnsw_pq_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    ef_search: Int,
    candidate_pruning_alpha: ScoreScalar,
    measurement_iterations: Int,
) raises -> TachiomTacHnswPqQueryProfile:
    require_positive_int("centroids_per_query_vector", centroids_per_query_vector)
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)
    require_positive_int("ef_search", ef_search)
    require_positive_int("measurement_iterations", measurement_iterations)

    var centroid_positions_by_query_vector = (
        build_hnsw_pq_centroid_positions_by_query_vector(
            query,
            prepared_index,
            centroids_per_query_vector,
            ef_search,
        )
    )
    var accumulation = (
        accumulate_hnsw_pq_candidate_document_scores_from_centroids(
            query,
            prepared_index,
            centroid_positions_by_query_vector,
        )
    )
    var ranked_candidates = top_positions_by_score(
        accumulation.document_scores, candidate_k
    )
    var reference_candidates = hnsw_pq_profile_candidate_positions_from_scores(
        accumulation.document_scores,
        ranked_candidates,
        final_k,
        candidate_pruning_alpha,
    )
    var residual_score_table = build_tachiom_pq_residual_score_table(
        query, prepared_index.pq
    )
    var rerank_scores = hnsw_pq_rerank_scores_for_candidates(
        query,
        prepared_index,
        residual_score_table,
        reference_candidates,
    )
    var reference_final = hnsw_pq_winners_from_rerank_scores(
        reference_candidates, rerank_scores, final_k
    )

    var full_search_sink = 0

    def full_search_once() capturing raises:
        if candidate_pruning_alpha > zero_score_scalar():
            var positions = tachiom_tac_hnsw_pq_search_positions_for_query_with_pruning(
                query,
                prepared_index,
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
                candidate_pruning_alpha,
            )
            full_search_sink += len(positions)
        else:
            var positions = tachiom_tac_hnsw_pq_search_positions_for_query(
                query,
                prepared_index,
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
            )
            full_search_sink += len(positions)

    var candidate_generation_sink = 0

    def candidate_generation_once() capturing raises:
        if candidate_pruning_alpha > zero_score_scalar():
            var positions = (
                tachiom_tac_hnsw_pq_candidate_positions_for_query_with_pruning(
                    query,
                    prepared_index,
                    centroids_per_query_vector,
                    candidate_k,
                    final_k,
                    ef_search,
                    candidate_pruning_alpha,
                )
            )
            candidate_generation_sink += len(positions)
        else:
            var positions = tachiom_tac_hnsw_pq_candidate_positions_for_query(
                query,
                prepared_index,
                centroids_per_query_vector,
                candidate_k,
                ef_search,
            )
            candidate_generation_sink += len(positions)

    var hnsw_traversal_sink = 0

    def hnsw_traversal_once() capturing raises:
        var positions_by_query_vector = (
            build_hnsw_pq_centroid_positions_by_query_vector(
                query,
                prepared_index,
                centroids_per_query_vector,
                ef_search,
            )
        )
        for positions in positions_by_query_vector:
            hnsw_traversal_sink += len(positions)

    var candidate_score_accumulation_sink = zero_score_scalar()

    def candidate_score_accumulation_once() capturing:
        var profiled_accumulation = (
            accumulate_hnsw_pq_candidate_document_scores_from_centroids(
                query,
                prepared_index,
                centroid_positions_by_query_vector,
            )
        )
        candidate_score_accumulation_sink += profiled_accumulation.score_checksum

    var candidate_topk_sink = 0

    def candidate_topk_once() capturing raises:
        var positions = top_positions_by_score(
            accumulation.document_scores, candidate_k
        )
        candidate_topk_sink += len(positions)

    var candidate_pruning_sink = 0

    def candidate_pruning_once() capturing raises:
        var positions = hnsw_pq_profile_candidate_positions_from_scores(
            accumulation.document_scores,
            ranked_candidates,
            final_k,
            candidate_pruning_alpha,
        )
        candidate_pruning_sink += len(positions)

    var residual_score_table_sink = zero_score_scalar()

    def residual_score_table_once() capturing:
        var table = build_tachiom_pq_residual_score_table(
            query, prepared_index.pq
        )
        if len(table) > 0:
            residual_score_table_sink += table[0]

    var rerank_scoring_sink = zero_score_scalar()

    def rerank_scoring_once() capturing:
        var scores = hnsw_pq_rerank_scores_for_candidates(
            query,
            prepared_index,
            residual_score_table,
            reference_candidates,
        )
        for score in scores:
            rerank_scoring_sink += score

    var rerank_topk_sink = 0

    def rerank_topk_once() capturing raises:
        var positions = hnsw_pq_winners_from_rerank_scores(
            reference_candidates, rerank_scores, final_k
        )
        rerank_topk_sink += len(positions)

    var full_search_report = benchmark.run[full_search_once](
        max_iters=measurement_iterations
    )
    var candidate_generation_report = benchmark.run[candidate_generation_once](
        max_iters=measurement_iterations
    )
    var hnsw_traversal_report = benchmark.run[hnsw_traversal_once](
        max_iters=measurement_iterations
    )
    var candidate_score_accumulation_report = benchmark.run[
        candidate_score_accumulation_once
    ](max_iters=measurement_iterations)
    var candidate_topk_report = benchmark.run[candidate_topk_once](
        max_iters=measurement_iterations
    )

    var candidate_pruning_mean_seconds = 0.0
    if candidate_pruning_alpha > zero_score_scalar():
        var candidate_pruning_report = benchmark.run[candidate_pruning_once](
            max_iters=measurement_iterations
        )
        candidate_pruning_mean_seconds = candidate_pruning_report.mean()

    var residual_score_table_report = benchmark.run[
        residual_score_table_once
    ](max_iters=measurement_iterations)
    var rerank_scoring_report = benchmark.run[rerank_scoring_once](
        max_iters=measurement_iterations
    )
    var rerank_topk_report = benchmark.run[rerank_topk_once](
        max_iters=measurement_iterations
    )

    return TachiomTacHnswPqQueryProfile(
        full_search_report.mean(),
        candidate_generation_report.mean(),
        hnsw_traversal_report.mean(),
        candidate_score_accumulation_report.mean(),
        candidate_topk_report.mean(),
        candidate_pruning_mean_seconds,
        residual_score_table_report.mean(),
        rerank_scoring_report.mean(),
        rerank_topk_report.mean(),
        query.vector_count,
        prepared_index.pq.document_count,
        regular_hnsw_pq_document_vector_count(prepared_index),
        prepared_index.pq.total_vector_count,
        prepared_index.centroid_count,
        centroids_per_query_vector,
        candidate_k,
        final_k,
        ef_search,
        Float64(candidate_pruning_alpha),
        accumulation.selected_centroid_count,
        accumulation.posting_visit_count,
        accumulation.touched_document_count,
        accumulation.seen_document_count,
        len(ranked_candidates),
        len(reference_candidates),
        len(reference_final),
        measurement_iterations,
        Float64(full_search_sink)
        + Float64(candidate_generation_sink)
        + Float64(hnsw_traversal_sink)
        + Float64(candidate_topk_sink)
        + Float64(candidate_pruning_sink)
        + Float64(rerank_topk_sink)
        + Float64(candidate_score_accumulation_sink)
        + Float64(residual_score_table_sink)
        + Float64(rerank_scoring_sink),
    )
