import std.benchmark as benchmark
from std.collections import List

from kayak.contracts import FlatQueryDim128
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar

from .plaid_approx_dim128 import require_positive_int, top_positions_by_score
from .tachiom_tac_dim128 import (
    PreparedTachiomTacIndex,
    score_query_vector_against_tachiom_centroids,
    tachiom_tac_candidate_positions_for_query,
    tachiom_tac_rerank_candidates_for_query,
    tachiom_tac_search_positions_for_query,
)
from .tachiom_tac_profile_types_dim128 import (
    TachiomTacCandidateAccumulation,
    TachiomTacCandidateGenerationProfile,
)


# Benchmark-only breakdown for the TAC candidate generator. This mirrors the
# production substeps so the next optimization is chosen from measured cost.
def build_tac_centroid_scores_by_query_vector(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
) -> List[List[ScoreScalar]]:
    var scores_by_query_vector = List[List[ScoreScalar]]()
    scores_by_query_vector.reserve(query.vector_count)

    for query_vector_index in range(query.vector_count):
        scores_by_query_vector.append(
            score_query_vector_against_tachiom_centroids(
                query, query_vector_index, prepared_index
            )
        )

    return scores_by_query_vector^


def build_tac_centroid_positions_by_query_vector(
    read centroid_scores_by_query_vector: List[List[ScoreScalar]],
    centroids_per_query_vector: Int,
) raises -> List[List[Int]]:
    var positions_by_query_vector = List[List[Int]]()
    positions_by_query_vector.reserve(len(centroid_scores_by_query_vector))

    for query_vector_index in range(len(centroid_scores_by_query_vector)):
        positions_by_query_vector.append(
            top_positions_by_score(
                centroid_scores_by_query_vector[query_vector_index],
                centroids_per_query_vector,
            )
        )

    return positions_by_query_vector^


def accumulate_tac_candidate_document_scores_from_centroids(
    read prepared_index: PreparedTachiomTacIndex,
    read centroid_scores_by_query_vector: List[List[ScoreScalar]],
    read centroid_positions_by_query_vector: List[List[Int]],
) -> TachiomTacCandidateAccumulation:
    var document_scores = List[ScoreScalar]()
    var document_seen = List[Int]()
    for _ in range(prepared_index.index.document_count):
        document_scores.append(zero_score_scalar())
        document_seen.append(0)

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    for _ in range(prepared_index.index.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    var posting_visit_count = 0
    var selected_centroid_count = 0
    var touched_document_count = 0
    var touched_documents = List[Int]()
    touched_documents.reserve(prepared_index.index.document_count)

    for query_vector_index in range(len(centroid_positions_by_query_vector)):
        selected_centroid_count += len(
            centroid_positions_by_query_vector[query_vector_index]
        )
        var current_touched_document_count = 0

        for offset in range(
            len(centroid_positions_by_query_vector[query_vector_index])
        ):
            var centroid_position = centroid_positions_by_query_vector[
                query_vector_index
            ][offset]
            var centroid_score = centroid_scores_by_query_vector[
                query_vector_index
            ][centroid_position]
            var start_posting = prepared_index.centroid_doc_offsets[
                centroid_position
            ]
            var stop_posting = prepared_index.centroid_doc_offsets[
                centroid_position + 1
            ]
            posting_visit_count += stop_posting - start_posting
            for posting_index in range(start_posting, stop_posting):
                var document_index = prepared_index.centroid_doc_indices[
                    posting_index
                ]
                document_seen[document_index] = 1
                if token_seen[document_index] == 0:
                    token_best_scores[document_index] = centroid_score
                    token_seen[document_index] = 1
                    if current_touched_document_count == len(touched_documents):
                        touched_documents.append(document_index)
                    else:
                        touched_documents[
                            current_touched_document_count
                        ] = document_index
                    current_touched_document_count += 1
                elif centroid_score > token_best_scores[document_index]:
                    token_best_scores[document_index] = centroid_score

        touched_document_count += current_touched_document_count
        for touched_offset in range(current_touched_document_count):
            var document_index = touched_documents[touched_offset]
            document_scores[document_index] += token_best_scores[document_index]
            token_best_scores[document_index] = min_score_scalar()
            token_seen[document_index] = 0

    var score_checksum = zero_score_scalar()
    var seen_document_count = 0
    for document_index in range(prepared_index.index.document_count):
        if document_seen[document_index] == 0:
            document_scores[document_index] = min_score_scalar()
        else:
            seen_document_count += 1
            score_checksum += document_scores[document_index]

    return TachiomTacCandidateAccumulation(
        document_scores^,
        posting_visit_count,
        selected_centroid_count,
        touched_document_count,
        seen_document_count,
        score_checksum,
    )


def regular_tac_document_vector_count(
    read prepared_index: PreparedTachiomTacIndex,
) -> Int:
    if prepared_index.index.document_count <= 0:
        return 0

    var first_count = (
        prepared_index.index.doc_offsets[1]
        - prepared_index.index.doc_offsets[0]
    )
    for document_index in range(1, prepared_index.index.document_count):
        var count = (
            prepared_index.index.doc_offsets[document_index + 1]
            - prepared_index.index.doc_offsets[document_index]
        )
        if count != first_count:
            return -1

    return first_count


def profile_tachiom_tac_candidate_generation_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    measurement_iterations: Int,
) raises -> TachiomTacCandidateGenerationProfile:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)
    require_positive_int("measurement_iterations", measurement_iterations)

    var centroid_scores_by_query_vector = (
        build_tac_centroid_scores_by_query_vector(query, prepared_index)
    )
    var centroid_positions_by_query_vector = (
        build_tac_centroid_positions_by_query_vector(
            centroid_scores_by_query_vector, centroids_per_query_vector
        )
    )
    var accumulation = accumulate_tac_candidate_document_scores_from_centroids(
        prepared_index,
        centroid_scores_by_query_vector,
        centroid_positions_by_query_vector,
    )
    var reference_candidates = tachiom_tac_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    var reference_final = tachiom_tac_rerank_candidates_for_query(
        query, prepared_index, reference_candidates, final_k
    )

    var full_candidate_sink = 0

    def full_candidate_once() capturing raises:
        var positions = tachiom_tac_candidate_positions_for_query(
            query, prepared_index, centroids_per_query_vector, candidate_k
        )
        full_candidate_sink += len(positions)

    var full_search_sink = 0

    def full_search_once() capturing raises:
        var positions = tachiom_tac_search_positions_for_query(
            query,
            prepared_index,
            centroids_per_query_vector,
            candidate_k,
            final_k,
        )
        full_search_sink += len(positions)

    var centroid_scoring_sink = zero_score_scalar()

    def centroid_scoring_once() capturing:
        var scores_by_query_vector = build_tac_centroid_scores_by_query_vector(
            query, prepared_index
        )
        for scores in scores_by_query_vector:
            if len(scores) > 0:
                centroid_scoring_sink += scores[0]

    var centroid_selection_sink = 0

    def centroid_selection_once() capturing raises:
        var positions_by_query_vector = (
            build_tac_centroid_positions_by_query_vector(
                centroid_scores_by_query_vector, centroids_per_query_vector
            )
        )
        for positions in positions_by_query_vector:
            centroid_selection_sink += len(positions)

    var posting_accumulation_sink = zero_score_scalar()

    def posting_accumulation_once() capturing:
        var profiled_accumulation = (
            accumulate_tac_candidate_document_scores_from_centroids(
                prepared_index,
                centroid_scores_by_query_vector,
                centroid_positions_by_query_vector,
            )
        )
        posting_accumulation_sink += profiled_accumulation.score_checksum

    var final_topk_sink = 0

    def final_topk_once() capturing raises:
        var positions = top_positions_by_score(
            accumulation.document_scores, candidate_k
        )
        final_topk_sink += len(positions)

    var exact_rerank_sink = 0

    def exact_rerank_once() capturing raises:
        var positions = tachiom_tac_rerank_candidates_for_query(
            query, prepared_index, reference_candidates, final_k
        )
        exact_rerank_sink += len(positions)

    var full_candidate_report = benchmark.run[full_candidate_once](
        max_iters=measurement_iterations
    )
    var full_search_report = benchmark.run[full_search_once](
        max_iters=measurement_iterations
    )
    var centroid_scoring_report = benchmark.run[centroid_scoring_once](
        max_iters=measurement_iterations
    )
    var centroid_selection_report = benchmark.run[centroid_selection_once](
        max_iters=measurement_iterations
    )
    var posting_accumulation_report = benchmark.run[posting_accumulation_once](
        max_iters=measurement_iterations
    )
    var final_topk_report = benchmark.run[final_topk_once](
        max_iters=measurement_iterations
    )
    var exact_rerank_report = benchmark.run[exact_rerank_once](
        max_iters=measurement_iterations
    )

    return TachiomTacCandidateGenerationProfile(
        full_candidate_report.mean(),
        full_search_report.mean(),
        centroid_scoring_report.mean(),
        centroid_selection_report.mean(),
        posting_accumulation_report.mean(),
        final_topk_report.mean(),
        exact_rerank_report.mean(),
        query.vector_count,
        prepared_index.index.document_count,
        regular_tac_document_vector_count(prepared_index),
        prepared_index.index.total_vector_count,
        prepared_index.centroid_count,
        centroids_per_query_vector,
        candidate_k,
        final_k,
        accumulation.selected_centroid_count,
        accumulation.posting_visit_count,
        accumulation.touched_document_count,
        accumulation.seen_document_count,
        len(reference_candidates),
        len(reference_final),
        measurement_iterations,
        Float64(full_candidate_sink)
        + Float64(full_search_sink)
        + Float64(centroid_selection_sink)
        + Float64(final_topk_sink)
        + Float64(exact_rerank_sink)
        + Float64(centroid_scoring_sink)
        + Float64(posting_accumulation_sink),
    )
