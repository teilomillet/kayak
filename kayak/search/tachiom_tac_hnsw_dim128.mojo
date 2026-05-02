from std.collections import List
from std.format import Writable, Writer

from kayak.contracts import FlatQueryDim128
from kayak.index import HybridFlatDim128Index
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar
from kayak.scoring import exact_score_for_hybrid_flat_document_dim128_with_flat_query
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .plaid_approx_dim128 import require_positive_int, top_positions_by_score
from .tachiom_tac_dim128 import pruned_top_positions_by_score


# Native search over a Python-built HNSW-style graph on TAC centroids. Graph
# construction remains in Python so this file owns only query-time traversal and
# exact candidate-window rerank.
struct PreparedTachiomTacHnswIndex(Movable, Writable):
    var index: HybridFlatDim128Index
    var centroid_values: List[ScoreScalar]
    var centroid_doc_offsets: List[Int]
    var centroid_doc_indices: List[Int]
    var graph_layer_node_offset_offsets: List[Int]
    var graph_node_offsets: List[Int]
    var graph_neighbor_indices: List[Int]
    var centroid_count: Int
    var level_count: Int
    var entry_point: Int

    def __init__(
        out self,
        var index: HybridFlatDim128Index,
        var centroid_values: List[ScoreScalar],
        var centroid_doc_offsets: List[Int],
        var centroid_doc_indices: List[Int],
        var graph_layer_node_offset_offsets: List[Int],
        var graph_node_offsets: List[Int],
        var graph_neighbor_indices: List[Int],
        entry_point: Int,
    ) raises:
        if len(centroid_values) % COLBERT_VECTOR_DIM != 0:
            raise Error("Tachiom HNSW centroid values must be aligned to dim128")
        var centroid_count = len(centroid_values) // COLBERT_VECTOR_DIM
        if len(centroid_doc_offsets) != centroid_count + 1:
            raise Error(
                "Tachiom HNSW centroid_doc_offsets length must be centroid_count + 1"
            )
        if len(graph_layer_node_offset_offsets) == 0:
            raise Error("Tachiom HNSW requires at least one graph layer")
        if entry_point < 0 or entry_point >= centroid_count:
            raise Error("Tachiom HNSW entry_point is out of bounds")
        for layer_index in range(len(graph_layer_node_offset_offsets)):
            var base = graph_layer_node_offset_offsets[layer_index]
            if base < 0 or base + centroid_count >= len(graph_node_offsets):
                raise Error("Tachiom HNSW graph layer offset is out of bounds")

        self.index = index^
        self.centroid_values = centroid_values^
        self.centroid_doc_offsets = centroid_doc_offsets^
        self.centroid_doc_indices = centroid_doc_indices^
        self.graph_layer_node_offset_offsets = graph_layer_node_offset_offsets^
        self.graph_node_offsets = graph_node_offsets^
        self.graph_neighbor_indices = graph_neighbor_indices^
        self.centroid_count = centroid_count
        self.level_count = len(self.graph_layer_node_offset_offsets)
        self.entry_point = entry_point

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedTachiomTacHnswIndex(document_count=",
            self.index.document_count,
            ", centroid_count=",
            self.centroid_count,
            ", level_count=",
            self.level_count,
            ")",
        )


def prepare_tachiom_tac_hnsw_hybrid_flat_dim128_index(
    var index: HybridFlatDim128Index,
    var centroid_values: List[ScoreScalar],
    var centroid_doc_offsets: List[Int],
    var centroid_doc_indices: List[Int],
    var graph_layer_node_offset_offsets: List[Int],
    var graph_node_offsets: List[Int],
    var graph_neighbor_indices: List[Int],
    entry_point: Int,
) raises -> PreparedTachiomTacHnswIndex:
    return PreparedTachiomTacHnswIndex(
        index^,
        centroid_values^,
        centroid_doc_offsets^,
        centroid_doc_indices^,
        graph_layer_node_offset_offsets^,
        graph_node_offsets^,
        graph_neighbor_indices^,
        entry_point,
    )


def tachiom_tac_hnsw_prepared_posting_count_value(
    read prepared_index: PreparedTachiomTacHnswIndex,
) -> Int:
    return len(prepared_index.centroid_doc_indices)


def tachiom_tac_hnsw_prepared_graph_edge_count_value(
    read prepared_index: PreparedTachiomTacHnswIndex,
) -> Int:
    return len(prepared_index.graph_neighbor_indices)


def hnsw_query_centroid_score(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswIndex,
    centroid_index: Int,
) -> ScoreScalar:
    return dot_product_dim128_flat_pair_at(
        query.token_values,
        query_vector_index * COLBERT_VECTOR_DIM,
        prepared_index.centroid_values,
        centroid_index * COLBERT_VECTOR_DIM,
    )


def hnsw_neighbor_start(
    read prepared_index: PreparedTachiomTacHnswIndex,
    layer_index: Int,
    node_index: Int,
) -> Int:
    return prepared_index.graph_node_offsets[
        prepared_index.graph_layer_node_offset_offsets[layer_index] + node_index
    ]


def hnsw_neighbor_stop(
    read prepared_index: PreparedTachiomTacHnswIndex,
    layer_index: Int,
    node_index: Int,
) -> Int:
    return prepared_index.graph_node_offsets[
        prepared_index.graph_layer_node_offset_offsets[layer_index] + node_index + 1
    ]


def hnsw_score_position_before(
    score: ScoreScalar,
    position: Int,
    other_score: ScoreScalar,
    other_position: Int,
) -> Bool:
    return score > other_score or (score == other_score and position < other_position)


def hnsw_insert_descending_position(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
    position: Int,
    score: ScoreScalar,
    limit: Int,
):
    var insert_at = 0
    while insert_at < len(scores) and not hnsw_score_position_before(
        score, position, scores[insert_at], positions[insert_at]
    ):
        insert_at += 1

    if insert_at >= limit:
        if len(scores) < limit:
            positions.append(position)
            scores.append(score)
        return

    if len(scores) < limit:
        positions.append(position)
        scores.append(score)

    var cursor = len(scores) - 1
    while cursor > insert_at:
        positions[cursor] = positions[cursor - 1]
        scores[cursor] = scores[cursor - 1]
        cursor -= 1

    positions[insert_at] = position
    scores[insert_at] = score


def hnsw_sparse_visited_table(size_hint: Int) -> List[Int]:
    var table_size = 1
    while table_size < size_hint:
        table_size *= 2
    if table_size < 8:
        table_size = 8

    var table = List[Int]()
    table.reserve(table_size)
    for _ in range(table_size):
        table.append(-1)
    return table^


def hnsw_sparse_visited_slot(read table: List[Int], position: Int) -> Int:
    return position % len(table)


def hnsw_sparse_visited_contains(
    read table: List[Int],
    position: Int,
) -> Bool:
    var slot = hnsw_sparse_visited_slot(table, position)
    while True:
        var existing = table[slot]
        if existing < 0:
            return False
        if existing == position:
            return True
        slot += 1
        if slot == len(table):
            slot = 0


def hnsw_sparse_visited_insert(mut table: List[Int], position: Int):
    var slot = hnsw_sparse_visited_slot(table, position)
    while True:
        var existing = table[slot]
        if existing == position:
            return
        if existing < 0:
            table[slot] = position
            return
        slot += 1
        if slot == len(table):
            slot = 0


def hnsw_sparse_visited_rehash(read table: List[Int]) -> List[Int]:
    var expanded = hnsw_sparse_visited_table(len(table) * 2)
    for offset in range(len(table)):
        var position = table[offset]
        if position >= 0:
            hnsw_sparse_visited_insert(expanded, position)
    return expanded^


def hnsw_greedy_layer_entry(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswIndex,
    layer_index: Int,
    entry_point: Int,
) -> Int:
    var current = entry_point
    var current_score = hnsw_query_centroid_score(
        query, query_vector_index, prepared_index, current
    )
    var improved = True
    while improved:
        improved = False
        var start = hnsw_neighbor_start(prepared_index, layer_index, current)
        var stop = hnsw_neighbor_stop(prepared_index, layer_index, current)
        for edge_index in range(start, stop):
            var neighbor = prepared_index.graph_neighbor_indices[edge_index]
            var score = hnsw_query_centroid_score(
                query, query_vector_index, prepared_index, neighbor
            )
            if hnsw_score_position_before(score, neighbor, current_score, current):
                current = neighbor
                current_score = score
                improved = True
    return current


def hnsw_search_layer_centroids(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswIndex,
    layer_index: Int,
    entry_point: Int,
    ef: Int,
) raises -> List[Int]:
    require_positive_int("ef", ef)
    var visited = hnsw_sparse_visited_table(ef * 64 + 16)
    var visited_count = 0

    var candidate_positions = List[Int]()
    var candidate_scores = List[ScoreScalar]()
    var candidate_expanded = List[Int]()
    var best_positions = List[Int]()
    var best_scores = List[ScoreScalar]()

    var entry_score = hnsw_query_centroid_score(
        query, query_vector_index, prepared_index, entry_point
    )
    hnsw_sparse_visited_insert(visited, entry_point)
    visited_count += 1
    candidate_positions.append(entry_point)
    candidate_scores.append(entry_score)
    candidate_expanded.append(0)
    hnsw_insert_descending_position(
        best_positions, best_scores, entry_point, entry_score, ef
    )

    while True:
        var best_candidate_offset = -1
        var best_candidate_score = min_score_scalar()
        var best_candidate_position = -1
        for offset in range(len(candidate_positions)):
            if candidate_expanded[offset] != 0:
                continue
            var position = candidate_positions[offset]
            var score = candidate_scores[offset]
            if best_candidate_offset < 0 or hnsw_score_position_before(
                score,
                position,
                best_candidate_score,
                best_candidate_position,
            ):
                best_candidate_offset = offset
                best_candidate_score = score
                best_candidate_position = position

        if best_candidate_offset < 0:
            break
        if len(best_scores) >= ef and best_candidate_score < best_scores[len(best_scores) - 1]:
            break
        candidate_expanded[best_candidate_offset] = 1

        var start = hnsw_neighbor_start(
            prepared_index, layer_index, best_candidate_position
        )
        var stop = hnsw_neighbor_stop(
            prepared_index, layer_index, best_candidate_position
        )
        for edge_index in range(start, stop):
            var neighbor = prepared_index.graph_neighbor_indices[edge_index]
            if hnsw_sparse_visited_contains(visited, neighbor):
                continue
            if visited_count * 2 >= len(visited):
                visited = hnsw_sparse_visited_rehash(visited)
            hnsw_sparse_visited_insert(visited, neighbor)
            visited_count += 1
            var score = hnsw_query_centroid_score(
                query, query_vector_index, prepared_index, neighbor
            )
            if len(best_scores) < ef or hnsw_score_position_before(
                score, neighbor, best_scores[len(best_scores) - 1], best_positions[len(best_positions) - 1]
            ):
                candidate_positions.append(neighbor)
                candidate_scores.append(score)
                candidate_expanded.append(0)
                hnsw_insert_descending_position(
                    best_positions, best_scores, neighbor, score, ef
                )

    return best_positions^


def tachiom_tac_hnsw_centroid_positions_for_query_vector(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswIndex,
    centroids_per_query_vector: Int,
    ef_search: Int,
) raises -> List[Int]:
    var entry = prepared_index.entry_point
    var layer_index = prepared_index.level_count - 1
    while layer_index > 0:
        entry = hnsw_greedy_layer_entry(
            query, query_vector_index, prepared_index, layer_index, entry
        )
        layer_index -= 1

    var ef = ef_search
    if ef < centroids_per_query_vector:
        ef = centroids_per_query_vector
    var candidates = hnsw_search_layer_centroids(
        query, query_vector_index, prepared_index, 0, entry, ef
    )
    var selected = List[Int]()
    var limit = centroids_per_query_vector
    if limit > len(candidates):
        limit = len(candidates)
    for offset in range(limit):
        selected.append(candidates[offset])
    return selected^


def tachiom_tac_hnsw_document_scores_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswIndex,
    centroids_per_query_vector: Int,
    ef_search: Int,
) raises -> List[ScoreScalar]:
    require_positive_int("centroids_per_query_vector", centroids_per_query_vector)
    require_positive_int("ef_search", ef_search)

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

    for query_vector_index in range(query.vector_count):
        var centroid_positions = tachiom_tac_hnsw_centroid_positions_for_query_vector(
            query,
            query_vector_index,
            prepared_index,
            centroids_per_query_vector,
            ef_search,
        )
        var touched_documents = List[Int]()
        touched_documents.reserve(prepared_index.index.document_count)

        for centroid_position in centroid_positions:
            var centroid_score = hnsw_query_centroid_score(
                query, query_vector_index, prepared_index, centroid_position
            )
            var start_posting = prepared_index.centroid_doc_offsets[
                centroid_position
            ]
            var stop_posting = prepared_index.centroid_doc_offsets[
                centroid_position + 1
            ]
            for posting_index in range(start_posting, stop_posting):
                var document_index = prepared_index.centroid_doc_indices[
                    posting_index
                ]
                if token_seen[document_index] == 0:
                    token_best_scores[document_index] = centroid_score
                    token_seen[document_index] = 1
                    touched_documents.append(document_index)
                    document_seen[document_index] = 1
                elif centroid_score > token_best_scores[document_index]:
                    token_best_scores[document_index] = centroid_score

        for document_index in touched_documents:
            document_scores[document_index] += token_best_scores[document_index]
            token_best_scores[document_index] = min_score_scalar()
            token_seen[document_index] = 0

    for document_index in range(prepared_index.index.document_count):
        if document_seen[document_index] == 0:
            document_scores[document_index] = min_score_scalar()

    return document_scores^


def tachiom_tac_hnsw_candidate_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    ef_search: Int,
) raises -> List[Int]:
    var document_scores = tachiom_tac_hnsw_document_scores_for_query(
        query, prepared_index, centroids_per_query_vector, ef_search
    )
    return top_positions_by_score(document_scores, candidate_k)


def tachiom_tac_hnsw_candidate_positions_for_query_with_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    ef_search: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    var document_scores = tachiom_tac_hnsw_document_scores_for_query(
        query, prepared_index, centroids_per_query_vector, ef_search
    )
    return pruned_top_positions_by_score(
        document_scores, candidate_k, final_k, candidate_pruning_alpha
    )


def tachiom_tac_hnsw_rerank_candidates_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswIndex,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[Int]:
    require_positive_int("final_k", final_k)
    var rerank_scores = List[ScoreScalar]()
    for document_index in candidate_positions:
        rerank_scores.append(
            exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                query, prepared_index.index, document_index
            )
        )
    var winner_offsets = top_positions_by_score(rerank_scores, final_k)
    var winners = List[Int]()
    for winner_offset in winner_offsets:
        winners.append(candidate_positions[winner_offset])
    return winners^


def tachiom_tac_hnsw_search_all_documents_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswIndex,
    final_k: Int,
) raises -> List[Int]:
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.index.document_count):
        scores.append(
            exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                query, prepared_index.index, document_index
            )
        )
    return top_positions_by_score(scores, final_k)


def tachiom_tac_hnsw_search_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    ef_search: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)
    if candidate_k >= prepared_index.index.document_count:
        return tachiom_tac_hnsw_search_all_documents_for_query(
            query, prepared_index, final_k
        )
    var candidate_positions = tachiom_tac_hnsw_candidate_positions_for_query_with_pruning(
        query,
        prepared_index,
        centroids_per_query_vector,
        candidate_k,
        final_k,
        ef_search,
        candidate_pruning_alpha,
    )
    return tachiom_tac_hnsw_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )
