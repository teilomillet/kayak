from std.collections import List

from kayak.contracts import FlatQueryDim128
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar

from .plaid_approx_dim128 import (
    insert_top_score_position,
    pop_worst_score_position,
    require_positive_int,
    top_positions_by_score,
)
from .plaid_i8_approx_dim128 import (
    PreparedPlaidApproxI8Index,
    score_query_vector_against_i8_centroids,
)


# Reusable scratch for benchmark-only i8 candidate generation experiments.
# The workspace does not own index/query lifetime and does not change public
# search behavior; it lets us test whether per-query scratch allocation is a
# real CPU bottleneck before promoting the path into serving code.
struct PlaidI8CandidateGenerationWorkspace:
    var document_scores: List[ScoreScalar]
    var document_score_generations: List[Int]
    var document_active_indices: List[Int]
    var document_active_count: Int
    var token_best_scores: List[ScoreScalar]
    var token_seen_generations: List[Int]
    var token_active_indices: List[Int]
    var token_active_count: Int
    var document_generation: Int
    var token_generation: Int

    def __init__(out self):
        self.document_scores = List[ScoreScalar]()
        self.document_score_generations = List[Int]()
        self.document_active_indices = List[Int]()
        self.document_active_count = 0
        self.token_best_scores = List[ScoreScalar]()
        self.token_seen_generations = List[Int]()
        self.token_active_indices = List[Int]()
        self.token_active_count = 0
        self.document_generation = 1
        self.token_generation = 1

    def __init__(out self, document_count: Int):
        self.document_scores = List[ScoreScalar]()
        self.document_score_generations = List[Int]()
        self.document_active_indices = List[Int]()
        self.document_active_count = 0
        self.token_best_scores = List[ScoreScalar]()
        self.token_seen_generations = List[Int]()
        self.token_active_indices = List[Int]()
        self.token_active_count = 0
        self.document_generation = 1
        self.token_generation = 1
        self.ensure_document_count(document_count)

    def ensure_document_count(mut self, document_count: Int):
        while len(self.document_scores) < document_count:
            self.document_scores.append(zero_score_scalar())
            self.document_score_generations.append(0)
            self.token_best_scores.append(min_score_scalar())
            self.token_seen_generations.append(0)

    def begin_query(mut self, document_count: Int):
        self.ensure_document_count(document_count)
        self.document_generation += 1
        self.document_active_count = 0

    def begin_token(mut self):
        self.token_generation += 1
        self.token_active_count = 0

    def append_document_active_index(mut self, document_index: Int):
        if self.document_active_count == len(self.document_active_indices):
            self.document_active_indices.append(document_index)
        else:
            self.document_active_indices[
                self.document_active_count
            ] = document_index
        self.document_active_count += 1

    def append_token_active_index(mut self, document_index: Int):
        if self.token_active_count == len(self.token_active_indices):
            self.token_active_indices.append(document_index)
        else:
            self.token_active_indices[self.token_active_count] = document_index
        self.token_active_count += 1

    def record_token_score(mut self, document_index: Int, score: ScoreScalar):
        if self.token_seen_generations[document_index] != self.token_generation:
            self.token_seen_generations[document_index] = self.token_generation
            self.token_best_scores[document_index] = score
            self.append_token_active_index(document_index)
        elif score > self.token_best_scores[document_index]:
            self.token_best_scores[document_index] = score

    def add_document_score(
        mut self, document_index: Int, score_delta: ScoreScalar
    ):
        if (
            self.document_score_generations[document_index]
            != self.document_generation
        ):
            self.document_score_generations[
                document_index
            ] = self.document_generation
            self.document_scores[document_index] = score_delta
            self.append_document_active_index(document_index)
            return

        self.document_scores[document_index] += score_delta

    def flush_token_scores(mut self):
        for active_index in range(self.token_active_count):
            var document_index = self.token_active_indices[active_index]
            self.add_document_score(
                document_index, self.token_best_scores[document_index]
            )

    def score_for_document(read self, document_index: Int) -> ScoreScalar:
        if (
            self.document_score_generations[document_index]
            != self.document_generation
        ):
            return zero_score_scalar()

        return self.document_scores[document_index]


def top_positions_by_workspace_scores(
    read workspace: PlaidI8CandidateGenerationWorkspace,
    document_count: Int,
    k: Int,
) raises -> List[Int]:
    require_positive_int("k", k)

    var selected = List[Int]()
    if document_count == 0:
        return selected^

    var limit = k
    if limit > document_count:
        limit = document_count

    var heap_positions = List[Int]()
    heap_positions.reserve(limit)
    var heap_scores = List[ScoreScalar]()
    heap_scores.reserve(limit)
    for document_index in range(document_count):
        insert_top_score_position(
            heap_positions,
            heap_scores,
            document_index,
            workspace.score_for_document(document_index),
            limit,
        )

    var ascending_positions = List[Int]()
    ascending_positions.reserve(limit)
    while len(heap_positions) > 0:
        ascending_positions.append(
            pop_worst_score_position(heap_positions, heap_scores)
        )

    selected.reserve(limit)
    for offset in range(len(ascending_positions)):
        selected.append(
            ascending_positions[len(ascending_positions) - offset - 1]
        )

    return selected^


def plaid_i8_candidate_positions_for_query_with_workspace(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    mut workspace: PlaidI8CandidateGenerationWorkspace,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)

    if candidate_k >= prepared_index.document_count:
        var all_positions = List[Int]()
        all_positions.reserve(prepared_index.document_count)
        for document_index in range(prepared_index.document_count):
            all_positions.append(document_index)
        return all_positions^

    workspace.begin_query(prepared_index.document_count)

    for query_vector_index in range(query.vector_count):
        var centroid_scores = score_query_vector_against_i8_centroids(
            query, query_vector_index, prepared_index
        )
        var centroid_positions = top_positions_by_score(
            centroid_scores, centroids_per_query_vector
        )

        workspace.begin_token()
        for centroid_position in centroid_positions:
            var centroid_score = centroid_scores[centroid_position]
            var start_posting = prepared_index.centroid_doc_offsets[
                centroid_position
            ]
            var stop_posting = prepared_index.centroid_doc_offsets[
                centroid_position + 1
            ]
            for posting_index in range(start_posting, stop_posting):
                workspace.record_token_score(
                    prepared_index.centroid_doc_indices[posting_index],
                    centroid_score,
                )

        workspace.flush_token_scores()

    return top_positions_by_workspace_scores(
        workspace, prepared_index.document_count, candidate_k
    )
