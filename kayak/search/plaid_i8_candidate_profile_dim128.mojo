import std.benchmark as benchmark
from std.collections import List

from kayak.contracts import FlatQueryDim128
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar

from .plaid_approx_dim128 import (
    require_positive_int,
    top_positions_by_score,
    top_unordered_positions_by_score,
)
from .plaid_i8_approx_dim128 import (
    PreparedPlaidApproxI8Index,
    plaid_i8_candidate_positions_for_query,
    plaid_i8_candidate_positions_for_query_unordered,
    score_query_vector_against_i8_centroids,
)
from .plaid_i8_candidate_workspace_dim128 import (
    PlaidI8CandidateGenerationWorkspace,
    plaid_i8_candidate_positions_for_query_with_workspace,
)
from .plaid_i8_candidate_profile_types_dim128 import (
    PlaidI8CandidateAccumulation,
    PlaidI8CandidateGenerationProfile,
)


# Benchmark-only breakdown for the PLAID i8 candidate generator. This module
# does not own the production candidate path; it mirrors the main substeps so
# profiler output can explain which primitive should move or change next.


def build_i8_centroid_scores_by_query_vector(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
) -> List[List[ScoreScalar]]:
    var scores_by_query_vector = List[List[ScoreScalar]]()
    scores_by_query_vector.reserve(query.vector_count)

    for query_vector_index in range(query.vector_count):
        scores_by_query_vector.append(
            score_query_vector_against_i8_centroids(
                query, query_vector_index, prepared_index
            )
        )

    return scores_by_query_vector^


def build_i8_centroid_positions_by_query_vector(
    read centroid_scores_by_query_vector: List[List[ScoreScalar]],
    centroids_per_query_vector: Int,
) raises -> List[List[Int]]:
    return build_i8_centroid_positions_by_query_vector_with_order(
        centroid_scores_by_query_vector,
        centroids_per_query_vector,
        True,
    )


def build_i8_centroid_positions_by_query_vector_unordered(
    read centroid_scores_by_query_vector: List[List[ScoreScalar]],
    centroids_per_query_vector: Int,
) raises -> List[List[Int]]:
    return build_i8_centroid_positions_by_query_vector_with_order(
        centroid_scores_by_query_vector,
        centroids_per_query_vector,
        False,
    )


def build_i8_centroid_positions_by_query_vector_with_order(
    read centroid_scores_by_query_vector: List[List[ScoreScalar]],
    centroids_per_query_vector: Int,
    ordered_output: Bool,
) raises -> List[List[Int]]:
    var positions_by_query_vector = List[List[Int]]()
    positions_by_query_vector.reserve(len(centroid_scores_by_query_vector))

    for query_vector_index in range(len(centroid_scores_by_query_vector)):
        if ordered_output:
            positions_by_query_vector.append(
                top_positions_by_score(
                    centroid_scores_by_query_vector[query_vector_index],
                    centroids_per_query_vector,
                )
            )
        else:
            positions_by_query_vector.append(
                top_unordered_positions_by_score(
                    centroid_scores_by_query_vector[query_vector_index],
                    centroids_per_query_vector,
                )
            )

    return positions_by_query_vector^


def accumulate_i8_candidate_document_scores_from_centroids(
    read prepared_index: PreparedPlaidApproxI8Index,
    read centroid_scores_by_query_vector: List[List[ScoreScalar]],
    read centroid_positions_by_query_vector: List[List[Int]],
) -> PlaidI8CandidateAccumulation:
    var document_scores = List[ScoreScalar]()
    for _ in range(prepared_index.document_count):
        document_scores.append(zero_score_scalar())

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    for _ in range(prepared_index.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    var posting_visit_count = 0
    var selected_centroid_count = 0
    var touched_document_count = 0
    var touched_documents = List[Int]()
    touched_documents.reserve(prepared_index.document_count)

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
    for score in document_scores:
        score_checksum += score

    return PlaidI8CandidateAccumulation(
        document_scores^,
        posting_visit_count,
        selected_centroid_count,
        touched_document_count,
        score_checksum,
    )


def regular_document_vector_count(
    read prepared_index: PreparedPlaidApproxI8Index,
) -> Int:
    if prepared_index.document_count <= 0:
        return 0

    var first_count = (
        prepared_index.doc_offsets[1] - prepared_index.doc_offsets[0]
    )
    for document_index in range(1, prepared_index.document_count):
        var count = (
            prepared_index.doc_offsets[document_index + 1]
            - prepared_index.doc_offsets[document_index]
        )
        if count != first_count:
            return -1

    return first_count


def candidate_positions_equal(
    read left: List[Int], read right: List[Int]
) -> Bool:
    if len(left) != len(right):
        return False

    for index in range(len(left)):
        if left[index] != right[index]:
            return False

    return True


def candidate_position_sets_equal(
    read left: List[Int], read right: List[Int], document_count: Int
) -> Bool:
    if len(left) != len(right):
        return False

    var counts = List[Int]()
    counts.reserve(document_count)
    for _ in range(document_count):
        counts.append(0)

    for position in left:
        if position < 0 or position >= document_count:
            return False
        counts[position] += 1

    for position in right:
        if position < 0 or position >= document_count:
            return False
        counts[position] -= 1

    for count in counts:
        if count != 0:
            return False

    return True


def centroid_position_sets_equal(
    read left: List[List[Int]], read right: List[List[Int]], centroid_count: Int
) -> Bool:
    if len(left) != len(right):
        return False

    for query_vector_index in range(len(left)):
        if not candidate_position_sets_equal(
            left[query_vector_index],
            right[query_vector_index],
            centroid_count,
        ):
            return False

    return True


def profile_plaid_i8_candidate_generation_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    measurement_iterations: Int,
) raises -> PlaidI8CandidateGenerationProfile:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("measurement_iterations", measurement_iterations)

    var centroid_scores_by_query_vector = (
        build_i8_centroid_scores_by_query_vector(query, prepared_index)
    )
    var centroid_positions_by_query_vector = (
        build_i8_centroid_positions_by_query_vector(
            centroid_scores_by_query_vector, centroids_per_query_vector
        )
    )
    var unordered_centroid_positions_by_query_vector = (
        build_i8_centroid_positions_by_query_vector_unordered(
            centroid_scores_by_query_vector, centroids_per_query_vector
        )
    )
    if not centroid_position_sets_equal(
        centroid_positions_by_query_vector,
        unordered_centroid_positions_by_query_vector,
        prepared_index.centroid_count,
    ):
        raise Error("unordered centroid selection must match ordered set")
    var centroid_selection_set_agreement = Float64(1.0)
    var accumulation = accumulate_i8_candidate_document_scores_from_centroids(
        prepared_index,
        centroid_scores_by_query_vector,
        centroid_positions_by_query_vector,
    )
    var reference_candidates = plaid_i8_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    var workspace = PlaidI8CandidateGenerationWorkspace(
        prepared_index.document_count
    )
    var workspace_output_candidates = (
        plaid_i8_candidate_positions_for_query_with_workspace(
            query,
            prepared_index,
            centroids_per_query_vector,
            candidate_k,
            workspace,
        )
    )
    if not candidate_positions_equal(
        reference_candidates, workspace_output_candidates
    ):
        raise Error(
            "workspace candidate generation must match reference positions"
        )
    var workspace_candidate_position_agreement = Float64(1.0)
    var unordered_candidates = plaid_i8_candidate_positions_for_query_unordered(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    if not candidate_position_sets_equal(
        reference_candidates,
        unordered_candidates,
        prepared_index.document_count,
    ):
        raise Error("unordered candidate generation must match reference set")
    var unordered_candidate_set_agreement = Float64(1.0)

    var full_candidate_sink = 0

    def full_candidate_once() capturing raises:
        var positions = plaid_i8_candidate_positions_for_query(
            query, prepared_index, centroids_per_query_vector, candidate_k
        )
        full_candidate_sink += len(positions)

    var workspace_full_candidate_sink = 0

    def workspace_full_candidate_once() capturing raises:
        var positions = plaid_i8_candidate_positions_for_query_with_workspace(
            query,
            prepared_index,
            centroids_per_query_vector,
            candidate_k,
            workspace,
        )
        workspace_full_candidate_sink += len(positions)

    var unordered_candidate_sink = 0

    def unordered_candidate_once() capturing raises:
        var positions = plaid_i8_candidate_positions_for_query_unordered(
            query, prepared_index, centroids_per_query_vector, candidate_k
        )
        unordered_candidate_sink += len(positions)

    var centroid_scoring_sink = zero_score_scalar()

    def centroid_scoring_once() capturing:
        var scores_by_query_vector = build_i8_centroid_scores_by_query_vector(
            query, prepared_index
        )
        for scores in scores_by_query_vector:
            if len(scores) > 0:
                centroid_scoring_sink += scores[0]

    var centroid_selection_sink = 0

    def centroid_selection_once() capturing raises:
        var positions_by_query_vector = (
            build_i8_centroid_positions_by_query_vector(
                centroid_scores_by_query_vector, centroids_per_query_vector
            )
        )
        for positions in positions_by_query_vector:
            centroid_selection_sink += len(positions)

    var unordered_centroid_selection_sink = 0

    def unordered_centroid_selection_once() capturing raises:
        var positions_by_query_vector = (
            build_i8_centroid_positions_by_query_vector_unordered(
                centroid_scores_by_query_vector, centroids_per_query_vector
            )
        )
        for positions in positions_by_query_vector:
            unordered_centroid_selection_sink += len(positions)

    var posting_accumulation_sink = zero_score_scalar()

    def posting_accumulation_once() capturing:
        var profiled_accumulation = (
            accumulate_i8_candidate_document_scores_from_centroids(
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

    var unordered_final_topk_sink = 0

    def unordered_final_topk_once() capturing raises:
        var positions = top_unordered_positions_by_score(
            accumulation.document_scores, candidate_k
        )
        unordered_final_topk_sink += len(positions)

    var full_candidate_report = benchmark.run[full_candidate_once](
        max_iters=measurement_iterations
    )
    var workspace_full_candidate_report = benchmark.run[
        workspace_full_candidate_once
    ](max_iters=measurement_iterations)
    var unordered_candidate_report = benchmark.run[unordered_candidate_once](
        max_iters=measurement_iterations
    )
    var centroid_scoring_report = benchmark.run[centroid_scoring_once](
        max_iters=measurement_iterations
    )
    var centroid_selection_report = benchmark.run[centroid_selection_once](
        max_iters=measurement_iterations
    )
    var unordered_centroid_selection_report = benchmark.run[
        unordered_centroid_selection_once
    ](max_iters=measurement_iterations)
    var posting_accumulation_report = benchmark.run[posting_accumulation_once](
        max_iters=measurement_iterations
    )
    var final_topk_report = benchmark.run[final_topk_once](
        max_iters=measurement_iterations
    )
    var unordered_final_topk_report = benchmark.run[unordered_final_topk_once](
        max_iters=measurement_iterations
    )

    return PlaidI8CandidateGenerationProfile(
        full_candidate_report.mean(),
        workspace_full_candidate_report.mean(),
        workspace_candidate_position_agreement,
        unordered_candidate_report.mean(),
        unordered_candidate_set_agreement,
        centroid_scoring_report.mean(),
        centroid_selection_report.mean(),
        unordered_centroid_selection_report.mean(),
        centroid_selection_set_agreement,
        posting_accumulation_report.mean(),
        final_topk_report.mean(),
        unordered_final_topk_report.mean(),
        query.vector_count,
        prepared_index.document_count,
        regular_document_vector_count(prepared_index),
        prepared_index.total_vector_count,
        prepared_index.centroid_count,
        centroids_per_query_vector,
        candidate_k,
        accumulation.selected_centroid_count,
        accumulation.posting_visit_count,
        accumulation.touched_document_count,
        len(reference_candidates),
        measurement_iterations,
        Float64(full_candidate_sink)
        + Float64(workspace_full_candidate_sink)
        + Float64(unordered_candidate_sink)
        + Float64(centroid_selection_sink)
        + Float64(unordered_centroid_selection_sink)
        + Float64(final_topk_sink)
        + Float64(unordered_final_topk_sink)
        + Float64(centroid_scoring_sink)
        + Float64(posting_accumulation_sink),
    )
