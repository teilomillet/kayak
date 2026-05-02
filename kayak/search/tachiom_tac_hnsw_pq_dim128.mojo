from std.collections import List
from std.format import Writable, Writer
from std.sys.info import simd_width_of

from kayak.contracts import FlatQueryDim128
from kayak.numeric import ScoreScalar, VectorScalar, min_score_scalar, zero_score_scalar
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .plaid_approx_dim128 import require_positive_int, top_positions_by_score
from .tachiom_tac_dim128 import prune_tachiom_candidate_positions_by_score
from .tachiom_tac_pq_dim128 import (
    PreparedTachiomTacPqIndex,
    build_tachiom_pq_residual_score_table,
    prepare_tachiom_tac_pq_dim128_index,
)


# Native query-time path for streaming Tachiom-style HNSW candidate generation
# plus residual-PQ rerank. The reranker scores only candidate document tokens
# instead of building a full query x centroid table, which keeps centroid count
# from leaking back into the candidate-window rerank cost.
struct PreparedTachiomTacHnswPqIndex(Movable, Writable):
    var pq: PreparedTachiomTacPqIndex
    var graph_layer_node_offset_offsets: List[Int]
    var graph_node_offsets: List[Int]
    var graph_neighbor_indices: List[Int]
    var centroid_count: Int
    var level_count: Int
    var entry_point: Int

    def __init__(
        out self,
        var pq: PreparedTachiomTacPqIndex,
        var graph_layer_node_offset_offsets: List[Int],
        var graph_node_offsets: List[Int],
        var graph_neighbor_indices: List[Int],
        entry_point: Int,
    ) raises:
        if len(graph_layer_node_offset_offsets) == 0:
            raise Error("Tachiom HNSW+PQ requires at least one graph layer")
        if entry_point < 0 or entry_point >= pq.centroid_count:
            raise Error("Tachiom HNSW+PQ entry_point is out of bounds")
        for layer_index in range(len(graph_layer_node_offset_offsets)):
            var base = graph_layer_node_offset_offsets[layer_index]
            if base < 0 or base + pq.centroid_count >= len(graph_node_offsets):
                raise Error("Tachiom HNSW+PQ graph layer offset is out of bounds")

        self.centroid_count = pq.centroid_count
        self.level_count = len(graph_layer_node_offset_offsets)
        self.entry_point = entry_point
        self.pq = pq^
        self.graph_layer_node_offset_offsets = graph_layer_node_offset_offsets^
        self.graph_node_offsets = graph_node_offsets^
        self.graph_neighbor_indices = graph_neighbor_indices^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedTachiomTacHnswPqIndex(document_count=",
            self.pq.document_count,
            ", total_vector_count=",
            self.pq.total_vector_count,
            ", centroid_count=",
            self.centroid_count,
            ", level_count=",
            self.level_count,
            ", subspace_count=",
            self.pq.subspace_count,
            ")",
        )


def prepare_tachiom_tac_hnsw_pq_dim128_index(
    var doc_ids: List[String],
    var doc_offsets: List[Int],
    var centroid_values: List[ScoreScalar],
    var centroid_doc_offsets: List[Int],
    var centroid_doc_indices: List[Int],
    var token_centroid_positions: List[Int],
    var residual_norms: List[ScoreScalar],
    var pq_codes: List[Int],
    var pq_codebooks: List[ScoreScalar],
    subspace_count: Int,
    codebook_size: Int,
    var graph_layer_node_offset_offsets: List[Int],
    var graph_node_offsets: List[Int],
    var graph_neighbor_indices: List[Int],
    entry_point: Int,
) raises -> PreparedTachiomTacHnswPqIndex:
    var pq = prepare_tachiom_tac_pq_dim128_index(
        doc_ids^,
        doc_offsets^,
        centroid_values^,
        centroid_doc_offsets^,
        centroid_doc_indices^,
        token_centroid_positions^,
        residual_norms^,
        pq_codes^,
        pq_codebooks^,
        subspace_count,
        codebook_size,
    )
    return PreparedTachiomTacHnswPqIndex(
        pq^,
        graph_layer_node_offset_offsets^,
        graph_node_offsets^,
        graph_neighbor_indices^,
        entry_point,
    )


def tachiom_tac_hnsw_pq_prepared_posting_count_value(
    read prepared_index: PreparedTachiomTacHnswPqIndex,
) -> Int:
    return len(prepared_index.pq.centroid_doc_indices)


def tachiom_tac_hnsw_pq_prepared_graph_edge_count_value(
    read prepared_index: PreparedTachiomTacHnswPqIndex,
) -> Int:
    return len(prepared_index.graph_neighbor_indices)


def hnsw_pq_query_centroid_score(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    centroid_index: Int,
) -> ScoreScalar:
    return dot_product_dim128_flat_pair_at(
        query.token_values,
        query_vector_index * COLBERT_VECTOR_DIM,
        prepared_index.pq.centroid_values,
        centroid_index * COLBERT_VECTOR_DIM,
    )


def hnsw_pq_neighbor_start(
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    layer_index: Int,
    node_index: Int,
) -> Int:
    return prepared_index.graph_node_offsets[
        prepared_index.graph_layer_node_offset_offsets[layer_index] + node_index
    ]


def hnsw_pq_neighbor_stop(
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    layer_index: Int,
    node_index: Int,
) -> Int:
    return prepared_index.graph_node_offsets[
        prepared_index.graph_layer_node_offset_offsets[layer_index] + node_index + 1
    ]


def hnsw_pq_score_position_before(
    score: ScoreScalar,
    position: Int,
    other_score: ScoreScalar,
    other_position: Int,
) -> Bool:
    return score > other_score or (score == other_score and position < other_position)


def hnsw_pq_score_position_is_worse(
    score: ScoreScalar,
    position: Int,
    other_score: ScoreScalar,
    other_position: Int,
) -> Bool:
    if score < other_score:
        return True
    if score > other_score:
        return False
    return position > other_position


def hnsw_pq_swap_score_position_entries(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
    left: Int,
    right: Int,
):
    var position = positions[left]
    var score = scores[left]
    positions[left] = positions[right]
    scores[left] = scores[right]
    positions[right] = position
    scores[right] = score


def hnsw_pq_sift_up_worst_first_retained(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
    offset: Int,
):
    var current = offset
    while current > 0:
        var parent = (current - 1) // 2
        if not hnsw_pq_score_position_is_worse(
            scores[current],
            positions[current],
            scores[parent],
            positions[parent],
        ):
            break
        hnsw_pq_swap_score_position_entries(positions, scores, current, parent)
        current = parent


def hnsw_pq_sift_down_worst_first_retained(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
    offset: Int,
):
    var current = offset
    while True:
        var worst = current
        var left = current * 2 + 1
        var right = left + 1
        if left < len(scores) and hnsw_pq_score_position_is_worse(
            scores[left],
            positions[left],
            scores[worst],
            positions[worst],
        ):
            worst = left
        if right < len(scores) and hnsw_pq_score_position_is_worse(
            scores[right],
            positions[right],
            scores[worst],
            positions[worst],
        ):
            worst = right
        if worst == current:
            break
        hnsw_pq_swap_score_position_entries(positions, scores, current, worst)
        current = worst


def hnsw_pq_insert_retained_top_position(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
    position: Int,
    score: ScoreScalar,
    limit: Int,
):
    if len(scores) < limit:
        positions.append(position)
        scores.append(score)
        hnsw_pq_sift_up_worst_first_retained(positions, scores, len(scores) - 1)
        return
    if not hnsw_pq_score_position_before(score, position, scores[0], positions[0]):
        return
    positions[0] = position
    scores[0] = score
    hnsw_pq_sift_down_worst_first_retained(positions, scores, 0)


def hnsw_pq_pop_worst_retained_position(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
) -> Int:
    var worst_position = positions[0]
    var last_offset = len(positions) - 1
    positions[0] = positions[last_offset]
    scores[0] = scores[last_offset]
    _ = positions.pop()
    _ = scores.pop()
    if len(scores) > 0:
        hnsw_pq_sift_down_worst_first_retained(positions, scores, 0)
    return worst_position


def hnsw_pq_descending_positions_from_retained_heap(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
) -> List[Int]:
    var ascending_positions = List[Int]()
    ascending_positions.reserve(len(positions))
    while len(positions) > 0:
        ascending_positions.append(
            hnsw_pq_pop_worst_retained_position(positions, scores)
        )

    var descending_positions = List[Int]()
    descending_positions.reserve(len(ascending_positions))
    for offset in range(len(ascending_positions)):
        descending_positions.append(
            ascending_positions[len(ascending_positions) - offset - 1]
        )
    return descending_positions^


def hnsw_pq_sift_up_best_first_candidate(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
    offset: Int,
):
    var current = offset
    while current > 0:
        var parent = (current - 1) // 2
        if not hnsw_pq_score_position_before(
            scores[current],
            positions[current],
            scores[parent],
            positions[parent],
        ):
            break
        hnsw_pq_swap_score_position_entries(positions, scores, current, parent)
        current = parent


def hnsw_pq_sift_down_best_first_candidate(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
    offset: Int,
):
    var current = offset
    while True:
        var best = current
        var left = current * 2 + 1
        var right = left + 1
        if left < len(scores) and hnsw_pq_score_position_before(
            scores[left],
            positions[left],
            scores[best],
            positions[best],
        ):
            best = left
        if right < len(scores) and hnsw_pq_score_position_before(
            scores[right],
            positions[right],
            scores[best],
            positions[best],
        ):
            best = right
        if best == current:
            break
        hnsw_pq_swap_score_position_entries(positions, scores, current, best)
        current = best


def hnsw_pq_push_best_first_candidate(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
    position: Int,
    score: ScoreScalar,
):
    positions.append(position)
    scores.append(score)
    hnsw_pq_sift_up_best_first_candidate(positions, scores, len(scores) - 1)


def hnsw_pq_pop_best_first_candidate(
    mut positions: List[Int],
    mut scores: List[ScoreScalar],
):
    var last_offset = len(positions) - 1
    positions[0] = positions[last_offset]
    scores[0] = scores[last_offset]
    _ = positions.pop()
    _ = scores.pop()
    if len(scores) > 0:
        hnsw_pq_sift_down_best_first_candidate(positions, scores, 0)


def hnsw_pq_sparse_visited_table(size_hint: Int) -> List[Int]:
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


def hnsw_pq_sparse_visited_slot(
    read table: List[Int],
    position: Int,
) -> Int:
    return position % len(table)


def hnsw_pq_sparse_visited_contains(
    read table: List[Int],
    position: Int,
) -> Bool:
    var slot = hnsw_pq_sparse_visited_slot(table, position)
    while True:
        var existing = table[slot]
        if existing < 0:
            return False
        if existing == position:
            return True
        slot += 1
        if slot == len(table):
            slot = 0


def hnsw_pq_sparse_visited_insert(
    mut table: List[Int],
    position: Int,
):
    var slot = hnsw_pq_sparse_visited_slot(table, position)
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


def hnsw_pq_sparse_visited_rehash(
    read table: List[Int],
) -> List[Int]:
    var expanded = hnsw_pq_sparse_visited_table(len(table) * 2)
    for offset in range(len(table)):
        var position = table[offset]
        if position >= 0:
            hnsw_pq_sparse_visited_insert(expanded, position)
    return expanded^


def hnsw_pq_greedy_layer_entry(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    layer_index: Int,
    entry_point: Int,
) -> Int:
    var current = entry_point
    var current_score = hnsw_pq_query_centroid_score(
        query, query_vector_index, prepared_index, current
    )
    var improved = True
    while improved:
        improved = False
        var start = hnsw_pq_neighbor_start(prepared_index, layer_index, current)
        var stop = hnsw_pq_neighbor_stop(prepared_index, layer_index, current)
        for edge_index in range(start, stop):
            var neighbor = prepared_index.graph_neighbor_indices[edge_index]
            var score = hnsw_pq_query_centroid_score(
                query, query_vector_index, prepared_index, neighbor
            )
            if hnsw_pq_score_position_before(
                score, neighbor, current_score, current
            ):
                current = neighbor
                current_score = score
                improved = True
    return current


def hnsw_pq_search_layer_centroids(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    layer_index: Int,
    entry_point: Int,
    ef: Int,
    return_descending: Bool,
) raises -> List[Int]:
    require_positive_int("ef", ef)
    var visited = hnsw_pq_sparse_visited_table(ef * 64 + 16)
    var visited_count = 0

    var candidate_positions = List[Int]()
    var candidate_scores = List[ScoreScalar]()
    var best_positions = List[Int]()
    var best_scores = List[ScoreScalar]()

    var entry_score = hnsw_pq_query_centroid_score(
        query, query_vector_index, prepared_index, entry_point
    )
    hnsw_pq_sparse_visited_insert(visited, entry_point)
    visited_count += 1
    hnsw_pq_push_best_first_candidate(
        candidate_positions, candidate_scores, entry_point, entry_score
    )
    hnsw_pq_insert_retained_top_position(
        best_positions, best_scores, entry_point, entry_score, ef
    )

    while len(candidate_positions) > 0:
        var best_candidate_score = candidate_scores[0]
        var best_candidate_position = candidate_positions[0]
        if len(best_scores) >= ef and best_candidate_score < best_scores[0]:
            break
        hnsw_pq_pop_best_first_candidate(candidate_positions, candidate_scores)

        var start = hnsw_pq_neighbor_start(
            prepared_index, layer_index, best_candidate_position
        )
        var stop = hnsw_pq_neighbor_stop(
            prepared_index, layer_index, best_candidate_position
        )
        for edge_index in range(start, stop):
            var neighbor = prepared_index.graph_neighbor_indices[edge_index]
            if hnsw_pq_sparse_visited_contains(visited, neighbor):
                continue
            if visited_count * 2 >= len(visited):
                visited = hnsw_pq_sparse_visited_rehash(visited)
            hnsw_pq_sparse_visited_insert(visited, neighbor)
            visited_count += 1
            var score = hnsw_pq_query_centroid_score(
                query, query_vector_index, prepared_index, neighbor
            )
            if len(best_scores) < ef or hnsw_pq_score_position_before(
                score,
                neighbor,
                best_scores[0],
                best_positions[0],
            ):
                hnsw_pq_push_best_first_candidate(
                    candidate_positions, candidate_scores, neighbor, score
                )
                hnsw_pq_insert_retained_top_position(
                    best_positions, best_scores, neighbor, score, ef
                )

    if return_descending:
        return hnsw_pq_descending_positions_from_retained_heap(
            best_positions, best_scores
        )
    # Callers only skip ordering when every retained centroid is consumed.
    # Downstream document score accumulation is order-independent in that case.
    return best_positions^


def tachiom_tac_hnsw_pq_centroid_positions_for_query_vector(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    centroids_per_query_vector: Int,
    ef_search: Int,
) raises -> List[Int]:
    var entry = prepared_index.entry_point
    var layer_index = prepared_index.level_count - 1
    while layer_index > 0:
        entry = hnsw_pq_greedy_layer_entry(
            query, query_vector_index, prepared_index, layer_index, entry
        )
        layer_index -= 1

    var ef = ef_search
    if ef < centroids_per_query_vector:
        ef = centroids_per_query_vector
    var candidates = hnsw_pq_search_layer_centroids(
        query,
        query_vector_index,
        prepared_index,
        0,
        entry,
        ef,
        ef > centroids_per_query_vector,
    )
    var selected = List[Int]()
    var limit = centroids_per_query_vector
    if limit > len(candidates):
        limit = len(candidates)
    for offset in range(limit):
        selected.append(candidates[offset])
    return selected^


def tachiom_tac_hnsw_pq_document_scores_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    centroids_per_query_vector: Int,
    ef_search: Int,
) raises -> List[ScoreScalar]:
    require_positive_int("centroids_per_query_vector", centroids_per_query_vector)
    require_positive_int("ef_search", ef_search)

    var document_scores = List[ScoreScalar]()
    var document_seen = List[Int]()
    for _ in range(prepared_index.pq.document_count):
        document_scores.append(zero_score_scalar())
        document_seen.append(0)

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    for _ in range(prepared_index.pq.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    for query_vector_index in range(query.vector_count):
        var centroid_positions = tachiom_tac_hnsw_pq_centroid_positions_for_query_vector(
            query,
            query_vector_index,
            prepared_index,
            centroids_per_query_vector,
            ef_search,
        )
        var touched_documents = List[Int]()
        touched_documents.reserve(prepared_index.pq.document_count)

        for centroid_position in centroid_positions:
            var centroid_score = hnsw_pq_query_centroid_score(
                query, query_vector_index, prepared_index, centroid_position
            )
            var start_posting = prepared_index.pq.centroid_doc_offsets[
                centroid_position
            ]
            var stop_posting = prepared_index.pq.centroid_doc_offsets[
                centroid_position + 1
            ]
            for posting_index in range(start_posting, stop_posting):
                var document_index = prepared_index.pq.centroid_doc_indices[
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

    for document_index in range(prepared_index.pq.document_count):
        if document_seen[document_index] == 0:
            document_scores[document_index] = min_score_scalar()

    return document_scores^


def tachiom_tac_hnsw_pq_candidate_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    ef_search: Int,
) raises -> List[Int]:
    var document_scores = tachiom_tac_hnsw_pq_document_scores_for_query(
        query, prepared_index, centroids_per_query_vector, ef_search
    )
    return top_positions_by_score(document_scores, candidate_k)


def tachiom_tac_hnsw_pq_candidate_positions_for_query_with_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    ef_search: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    var document_scores = tachiom_tac_hnsw_pq_document_scores_for_query(
        query, prepared_index, centroids_per_query_vector, ef_search
    )
    var ranked_positions = top_positions_by_score(document_scores, candidate_k)
    return prune_tachiom_candidate_positions_by_score(
        ranked_positions, document_scores, final_k, candidate_pruning_alpha
    )


def dot_query_vector_with_tachiom_hnsw_pq_token_dim128(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    read residual_score_table: List[ScoreScalar],
    token_index: Int,
) -> ScoreScalar:
    var centroid_positions_ptr = prepared_index.pq.token_centroid_positions.unsafe_ptr()
    var pq_codes_ptr = prepared_index.pq.pq_codes.unsafe_ptr()
    var residual_score_ptr = residual_score_table.unsafe_ptr()
    var residual_norms_ptr = prepared_index.pq.residual_norms.unsafe_ptr()
    var centroid_position = centroid_positions_ptr[token_index]
    var centroid_score = hnsw_pq_query_centroid_score(
        query, query_vector_index, prepared_index, centroid_position
    )
    var residual_score = zero_score_scalar()
    var code_offset = token_index * prepared_index.pq.subspace_count
    var residual_base = query_vector_index * prepared_index.pq.subspace_count
    residual_base *= prepared_index.pq.codebook_size

    for subspace_index in range(prepared_index.pq.subspace_count):
        var code = pq_codes_ptr[code_offset + subspace_index]
        residual_score += residual_score_ptr[
            residual_base + subspace_index * prepared_index.pq.codebook_size + code
        ]

    return centroid_score + residual_norms_ptr[token_index] * residual_score


def best_tachiom_hnsw_pq_score_for_query_vector_in_document_dim128(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    read residual_score_table: List[ScoreScalar],
    start_token: Int,
    stop_token: Int,
) -> ScoreScalar:
    var best_score = min_score_scalar()

    for token_index in range(start_token, stop_token):
        var score = dot_query_vector_with_tachiom_hnsw_pq_token_dim128(
            query,
            query_vector_index,
            prepared_index,
            residual_score_table,
            token_index,
        )
        if score > best_score:
            best_score = score

    return best_score


def tachiom_tac_hnsw_pq_score_for_document(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    read residual_score_table: List[ScoreScalar],
    document_index: Int,
) -> ScoreScalar:
    var start_token = prepared_index.pq.doc_offsets[document_index]
    var stop_token = prepared_index.pq.doc_offsets[document_index + 1]
    var total = zero_score_scalar()

    for query_vector_index in range(query.vector_count):
        total += best_tachiom_hnsw_pq_score_for_query_vector_in_document_dim128(
            query,
            query_vector_index,
            prepared_index,
            residual_score_table,
            start_token,
            stop_token,
        )

    return total


def tachiom_tac_hnsw_pq_rerank_candidates_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[Int]:
    require_positive_int("final_k", final_k)
    var residual_score_table = build_tachiom_pq_residual_score_table(
        query, prepared_index.pq
    )

    var rerank_scores = List[ScoreScalar]()
    rerank_scores.reserve(len(candidate_positions))
    for document_index in candidate_positions:
        rerank_scores.append(
            tachiom_tac_hnsw_pq_score_for_document(
                query,
                prepared_index,
                residual_score_table,
                document_index,
            )
        )

    var winner_offsets = top_positions_by_score(rerank_scores, final_k)
    var winners = List[Int]()
    winners.reserve(len(winner_offsets))
    for winner_offset in winner_offsets:
        winners.append(candidate_positions[winner_offset])

    return winners^


def tachiom_tac_hnsw_pq_search_all_documents_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    final_k: Int,
) raises -> List[Int]:
    var residual_score_table = build_tachiom_pq_residual_score_table(
        query, prepared_index.pq
    )
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.pq.document_count):
        scores.append(
            tachiom_tac_hnsw_pq_score_for_document(
                query, prepared_index, residual_score_table, document_index
            )
        )

    return top_positions_by_score(scores, final_k)


def tachiom_tac_hnsw_pq_search_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    ef_search: Int,
) raises -> List[Int]:
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)
    if candidate_k >= prepared_index.pq.document_count:
        return tachiom_tac_hnsw_pq_search_all_documents_for_query(
            query, prepared_index, final_k
        )
    var candidate_positions = tachiom_tac_hnsw_pq_candidate_positions_for_query(
        query,
        prepared_index,
        centroids_per_query_vector,
        candidate_k,
        ef_search,
    )
    return tachiom_tac_hnsw_pq_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )


def tachiom_tac_hnsw_pq_search_positions_for_query_with_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    ef_search: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)
    if candidate_k >= prepared_index.pq.document_count:
        return tachiom_tac_hnsw_pq_search_all_documents_for_query(
            query, prepared_index, final_k
        )
    var candidate_positions = (
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
    return tachiom_tac_hnsw_pq_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )


struct PreparedTachiomTacHnswPqAddressIndex(Movable, Writable):
    var doc_offsets: UnsafePointer[Int64, MutAnyOrigin]
    var centroid_values: UnsafePointer[Float32, MutAnyOrigin]
    var centroid_doc_offsets: UnsafePointer[Int64, MutAnyOrigin]
    var centroid_doc_indices: UnsafePointer[UInt32, MutAnyOrigin]
    var token_centroid_positions: UnsafePointer[UInt32, MutAnyOrigin]
    var residual_norms: UnsafePointer[Float32, MutAnyOrigin]
    var pq_codes: UnsafePointer[UInt8, MutAnyOrigin]
    var pq_codebooks: UnsafePointer[Float32, MutAnyOrigin]
    var graph_layer_node_offset_offsets: UnsafePointer[Int64, MutAnyOrigin]
    var graph_node_offsets: UnsafePointer[Int64, MutAnyOrigin]
    var graph_neighbor_indices: UnsafePointer[UInt32, MutAnyOrigin]
    var document_count: Int
    var total_vector_count: Int
    var centroid_count: Int
    var posting_count: Int
    var subspace_count: Int
    var codebook_size: Int
    var subspace_dim: Int
    var level_count: Int
    var graph_edge_count: Int
    var entry_point: Int

    def __init__(
        out self,
        doc_offsets: UnsafePointer[Int64, MutAnyOrigin],
        document_count: Int,
        total_vector_count: Int,
        centroid_values: UnsafePointer[Float32, MutAnyOrigin],
        centroid_count: Int,
        centroid_doc_offsets: UnsafePointer[Int64, MutAnyOrigin],
        centroid_doc_indices: UnsafePointer[UInt32, MutAnyOrigin],
        posting_count: Int,
        token_centroid_positions: UnsafePointer[UInt32, MutAnyOrigin],
        residual_norms: UnsafePointer[Float32, MutAnyOrigin],
        pq_codes: UnsafePointer[UInt8, MutAnyOrigin],
        pq_codebooks: UnsafePointer[Float32, MutAnyOrigin],
        subspace_count: Int,
        codebook_size: Int,
        graph_layer_node_offset_offsets: UnsafePointer[Int64, MutAnyOrigin],
        graph_node_offsets: UnsafePointer[Int64, MutAnyOrigin],
        graph_neighbor_indices: UnsafePointer[UInt32, MutAnyOrigin],
        level_count: Int,
        graph_edge_count: Int,
        entry_point: Int,
    ) raises:
        require_positive_int("document_count", document_count)
        require_positive_int("total_vector_count", total_vector_count)
        require_positive_int("centroid_count", centroid_count)
        require_positive_int("posting_count", posting_count)
        require_positive_int("subspace_count", subspace_count)
        require_positive_int("codebook_size", codebook_size)
        require_positive_int("level_count", level_count)
        if COLBERT_VECTOR_DIM % subspace_count != 0:
            raise Error("Tachiom HNSW+PQ address subspace_count must divide dim128")
        if entry_point < 0 or entry_point >= centroid_count:
            raise Error("Tachiom HNSW+PQ address entry_point is out of bounds")

        self.doc_offsets = doc_offsets
        self.centroid_values = centroid_values
        self.centroid_doc_offsets = centroid_doc_offsets
        self.centroid_doc_indices = centroid_doc_indices
        self.token_centroid_positions = token_centroid_positions
        self.residual_norms = residual_norms
        self.pq_codes = pq_codes
        self.pq_codebooks = pq_codebooks
        self.graph_layer_node_offset_offsets = graph_layer_node_offset_offsets
        self.graph_node_offsets = graph_node_offsets
        self.graph_neighbor_indices = graph_neighbor_indices
        self.document_count = document_count
        self.total_vector_count = total_vector_count
        self.centroid_count = centroid_count
        self.posting_count = posting_count
        self.subspace_count = subspace_count
        self.codebook_size = codebook_size
        self.subspace_dim = COLBERT_VECTOR_DIM // subspace_count
        self.level_count = level_count
        self.graph_edge_count = graph_edge_count
        self.entry_point = entry_point

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedTachiomTacHnswPqAddressIndex(document_count=",
            self.document_count,
            ", total_vector_count=",
            self.total_vector_count,
            ", centroid_count=",
            self.centroid_count,
            ", level_count=",
            self.level_count,
            ", subspace_count=",
            self.subspace_count,
            ")",
        )


def prepare_tachiom_tac_hnsw_pq_dim128_address_index(
    doc_offsets_address: Int,
    document_count: Int,
    total_vector_count: Int,
    centroid_values_address: Int,
    centroid_count: Int,
    centroid_doc_offsets_address: Int,
    centroid_doc_indices_address: Int,
    posting_count: Int,
    token_centroid_positions_address: Int,
    residual_norms_address: Int,
    pq_codes_address: Int,
    pq_codebooks_address: Int,
    subspace_count: Int,
    codebook_size: Int,
    graph_layer_node_offset_offsets_address: Int,
    graph_node_offsets_address: Int,
    graph_neighbor_indices_address: Int,
    level_count: Int,
    graph_edge_count: Int,
    entry_point: Int,
) raises -> PreparedTachiomTacHnswPqAddressIndex:
    if doc_offsets_address == 0:
        raise Error("Tachiom HNSW+PQ address doc_offsets address must be non-zero")
    if centroid_values_address == 0:
        raise Error("Tachiom HNSW+PQ address centroids address must be non-zero")
    if centroid_doc_offsets_address == 0:
        raise Error("Tachiom HNSW+PQ address posting offsets address must be non-zero")
    if centroid_doc_indices_address == 0:
        raise Error("Tachiom HNSW+PQ address postings address must be non-zero")
    if token_centroid_positions_address == 0:
        raise Error("Tachiom HNSW+PQ address token centroids address must be non-zero")
    if residual_norms_address == 0:
        raise Error("Tachiom HNSW+PQ address residual norms address must be non-zero")
    if pq_codes_address == 0:
        raise Error("Tachiom HNSW+PQ address PQ codes address must be non-zero")
    if pq_codebooks_address == 0:
        raise Error("Tachiom HNSW+PQ address PQ codebooks address must be non-zero")
    if graph_layer_node_offset_offsets_address == 0:
        raise Error("Tachiom HNSW+PQ address graph layer address must be non-zero")
    if graph_node_offsets_address == 0:
        raise Error("Tachiom HNSW+PQ address graph offsets address must be non-zero")
    if graph_neighbor_indices_address == 0:
        raise Error("Tachiom HNSW+PQ address graph neighbors address must be non-zero")

    return PreparedTachiomTacHnswPqAddressIndex(
        UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=doc_offsets_address),
        document_count,
        total_vector_count,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=centroid_values_address),
        centroid_count,
        UnsafePointer[Int64, MutAnyOrigin](
            unsafe_from_address=centroid_doc_offsets_address
        ),
        UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=centroid_doc_indices_address
        ),
        posting_count,
        UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=token_centroid_positions_address
        ),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=residual_norms_address),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=pq_codes_address),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=pq_codebooks_address),
        subspace_count,
        codebook_size,
        UnsafePointer[Int64, MutAnyOrigin](
            unsafe_from_address=graph_layer_node_offset_offsets_address
        ),
        UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=graph_node_offsets_address),
        UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=graph_neighbor_indices_address
        ),
        level_count,
        graph_edge_count,
        entry_point,
    )


def tachiom_tac_hnsw_pq_address_prepared_posting_count_value(
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
) -> Int:
    return prepared_index.posting_count


def tachiom_tac_hnsw_pq_address_prepared_graph_edge_count_value(
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
) -> Int:
    return prepared_index.graph_edge_count


def hnsw_pq_address_query_centroid_score(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    centroid_index: Int,
) -> ScoreScalar:
    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        var score = zero_score_scalar()
        var query_offset = query_vector_index * COLBERT_VECTOR_DIM
        var centroid_offset = centroid_index * COLBERT_VECTOR_DIM
        for dim_index in range(COLBERT_VECTOR_DIM):
            score += (
                query.token_values[query_offset + dim_index]
                * prepared_index.centroid_values[centroid_offset + dim_index]
            )
        return score

    var accum = SIMD[DType.float32, width](0.0)
    var query_ptr = (
        query.token_values.unsafe_ptr() + query_vector_index * COLBERT_VECTOR_DIM
    )
    var centroid_ptr = (
        prepared_index.centroid_values + centroid_index * COLBERT_VECTOR_DIM
    )

    for dim_index in range(0, COLBERT_VECTOR_DIM, width):
        accum += (
            (query_ptr + dim_index).load[width=width]()
            * (centroid_ptr + dim_index).load[width=width]()
        )

    return ScoreScalar(accum.reduce_add()[0])


def hnsw_pq_address_query_centroid_score_scalar(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    centroid_index: Int,
) -> ScoreScalar:
    var score = zero_score_scalar()
    var query_offset = query_vector_index * COLBERT_VECTOR_DIM
    var centroid_offset = centroid_index * COLBERT_VECTOR_DIM
    for dim_index in range(COLBERT_VECTOR_DIM):
        score += (
            query.token_values[query_offset + dim_index]
            * prepared_index.centroid_values[centroid_offset + dim_index]
        )
    return score


def hnsw_pq_address_neighbor_start(
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    layer_index: Int,
    node_index: Int,
) -> Int:
    return Int(
        prepared_index.graph_node_offsets[
            prepared_index.graph_layer_node_offset_offsets[layer_index] + node_index
        ]
    )


def hnsw_pq_address_neighbor_stop(
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    layer_index: Int,
    node_index: Int,
) -> Int:
    return Int(
        prepared_index.graph_node_offsets[
            prepared_index.graph_layer_node_offset_offsets[layer_index]
            + node_index
            + 1
        ]
    )


def hnsw_pq_address_greedy_layer_entry(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    layer_index: Int,
    entry_point: Int,
) -> Int:
    var current = entry_point
    var current_score = hnsw_pq_address_query_centroid_score(
        query, query_vector_index, prepared_index, current
    )
    var improved = True
    while improved:
        improved = False
        var start = hnsw_pq_address_neighbor_start(
            prepared_index, layer_index, current
        )
        var stop = hnsw_pq_address_neighbor_stop(
            prepared_index, layer_index, current
        )
        for edge_index in range(start, stop):
            var neighbor = Int(prepared_index.graph_neighbor_indices[edge_index])
            var score = hnsw_pq_address_query_centroid_score(
                query, query_vector_index, prepared_index, neighbor
            )
            if hnsw_pq_score_position_before(
                score, neighbor, current_score, current
            ):
                current = neighbor
                current_score = score
                improved = True
    return current


def hnsw_pq_address_search_layer_centroids(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    layer_index: Int,
    entry_point: Int,
    ef: Int,
    return_descending: Bool,
) raises -> List[Int]:
    require_positive_int("ef", ef)
    var visited = hnsw_pq_sparse_visited_table(ef * 64 + 16)
    var visited_count = 0

    var candidate_positions = List[Int]()
    var candidate_scores = List[ScoreScalar]()
    var best_positions = List[Int]()
    var best_scores = List[ScoreScalar]()

    var entry_score = hnsw_pq_address_query_centroid_score(
        query, query_vector_index, prepared_index, entry_point
    )
    hnsw_pq_sparse_visited_insert(visited, entry_point)
    visited_count += 1
    hnsw_pq_push_best_first_candidate(
        candidate_positions, candidate_scores, entry_point, entry_score
    )
    hnsw_pq_insert_retained_top_position(
        best_positions, best_scores, entry_point, entry_score, ef
    )

    while len(candidate_positions) > 0:
        var best_candidate_score = candidate_scores[0]
        var best_candidate_position = candidate_positions[0]
        if len(best_scores) >= ef and best_candidate_score < best_scores[0]:
            break
        hnsw_pq_pop_best_first_candidate(candidate_positions, candidate_scores)

        var start = hnsw_pq_address_neighbor_start(
            prepared_index, layer_index, best_candidate_position
        )
        var stop = hnsw_pq_address_neighbor_stop(
            prepared_index, layer_index, best_candidate_position
        )
        for edge_index in range(start, stop):
            var neighbor = Int(prepared_index.graph_neighbor_indices[edge_index])
            if hnsw_pq_sparse_visited_contains(visited, neighbor):
                continue
            if visited_count * 2 >= len(visited):
                visited = hnsw_pq_sparse_visited_rehash(visited)
            hnsw_pq_sparse_visited_insert(visited, neighbor)
            visited_count += 1
            var score = hnsw_pq_address_query_centroid_score(
                query, query_vector_index, prepared_index, neighbor
            )
            if len(best_scores) < ef or hnsw_pq_score_position_before(
                score,
                neighbor,
                best_scores[0],
                best_positions[0],
            ):
                hnsw_pq_push_best_first_candidate(
                    candidate_positions, candidate_scores, neighbor, score
                )
                hnsw_pq_insert_retained_top_position(
                    best_positions, best_scores, neighbor, score, ef
                )

    if return_descending:
        return hnsw_pq_descending_positions_from_retained_heap(
            best_positions, best_scores
        )
    # Callers only skip ordering when every retained centroid is consumed.
    # Downstream document score accumulation is order-independent in that case.
    return best_positions^


def tachiom_tac_hnsw_pq_address_centroid_positions_for_query_vector(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    centroids_per_query_vector: Int,
    ef_search: Int,
) raises -> List[Int]:
    var entry = prepared_index.entry_point
    var layer_index = prepared_index.level_count - 1
    while layer_index > 0:
        entry = hnsw_pq_address_greedy_layer_entry(
            query, query_vector_index, prepared_index, layer_index, entry
        )
        layer_index -= 1

    var ef = ef_search
    if ef < centroids_per_query_vector:
        ef = centroids_per_query_vector
    var candidates = hnsw_pq_address_search_layer_centroids(
        query,
        query_vector_index,
        prepared_index,
        0,
        entry,
        ef,
        ef > centroids_per_query_vector,
    )
    var selected = List[Int]()
    var limit = centroids_per_query_vector
    if limit > len(candidates):
        limit = len(candidates)
    for offset in range(limit):
        selected.append(candidates[offset])
    return selected^


def tachiom_tac_hnsw_pq_address_document_scores_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    centroids_per_query_vector: Int,
    ef_search: Int,
) raises -> List[ScoreScalar]:
    require_positive_int("centroids_per_query_vector", centroids_per_query_vector)
    require_positive_int("ef_search", ef_search)

    var document_scores = List[ScoreScalar]()
    var document_seen = List[Int]()
    for _ in range(prepared_index.document_count):
        document_scores.append(zero_score_scalar())
        document_seen.append(0)

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    for _ in range(prepared_index.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    for query_vector_index in range(query.vector_count):
        var centroid_positions = (
            tachiom_tac_hnsw_pq_address_centroid_positions_for_query_vector(
                query,
                query_vector_index,
                prepared_index,
                centroids_per_query_vector,
                ef_search,
            )
        )
        var touched_documents = List[Int]()
        touched_documents.reserve(prepared_index.document_count)

        for centroid_position in centroid_positions:
            var centroid_score = hnsw_pq_address_query_centroid_score(
                query, query_vector_index, prepared_index, centroid_position
            )
            var start_posting = Int(
                prepared_index.centroid_doc_offsets[centroid_position]
            )
            var stop_posting = Int(
                prepared_index.centroid_doc_offsets[centroid_position + 1]
            )
            for posting_index in range(start_posting, stop_posting):
                var document_index = Int(
                    prepared_index.centroid_doc_indices[posting_index]
                )
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

    for document_index in range(prepared_index.document_count):
        if document_seen[document_index] == 0:
            document_scores[document_index] = min_score_scalar()

    return document_scores^


def tachiom_tac_hnsw_pq_address_candidate_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    ef_search: Int,
) raises -> List[Int]:
    var document_scores = tachiom_tac_hnsw_pq_address_document_scores_for_query(
        query, prepared_index, centroids_per_query_vector, ef_search
    )
    return top_positions_by_score(document_scores, candidate_k)


def tachiom_tac_hnsw_pq_address_candidate_positions_for_query_with_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    ef_search: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    var document_scores = tachiom_tac_hnsw_pq_address_document_scores_for_query(
        query, prepared_index, centroids_per_query_vector, ef_search
    )
    var ranked_positions = top_positions_by_score(document_scores, candidate_k)
    return prune_tachiom_candidate_positions_by_score(
        ranked_positions, document_scores, final_k, candidate_pruning_alpha
    )


def build_tachiom_hnsw_pq_address_residual_score_table(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
) -> List[ScoreScalar]:
    var table = List[ScoreScalar]()
    table.reserve(
        query.vector_count * prepared_index.subspace_count * prepared_index.codebook_size
    )

    for query_vector_index in range(query.vector_count):
        var query_offset = query_vector_index * COLBERT_VECTOR_DIM
        for subspace_index in range(prepared_index.subspace_count):
            var subspace_query_offset = (
                query_offset + subspace_index * prepared_index.subspace_dim
            )
            for code in range(prepared_index.codebook_size):
                var codebook_offset = (
                    (subspace_index * prepared_index.codebook_size + code)
                    * prepared_index.subspace_dim
                )
                var score = zero_score_scalar()
                for dim_index in range(prepared_index.subspace_dim):
                    score += (
                        query.token_values[subspace_query_offset + dim_index]
                        * prepared_index.pq_codebooks[codebook_offset + dim_index]
                    )
                table.append(score)

    return table^


def dot_query_vector_with_tachiom_hnsw_pq_address_token_dim128(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    read residual_score_table: List[ScoreScalar],
    token_index: Int,
) -> ScoreScalar:
    var centroid_position = Int(prepared_index.token_centroid_positions[token_index])
    var centroid_score = hnsw_pq_address_query_centroid_score(
        query, query_vector_index, prepared_index, centroid_position
    )
    var residual_score = zero_score_scalar()
    var code_offset = token_index * prepared_index.subspace_count
    var residual_base = query_vector_index * prepared_index.subspace_count
    residual_base *= prepared_index.codebook_size

    for subspace_index in range(prepared_index.subspace_count):
        var code = Int(prepared_index.pq_codes[code_offset + subspace_index])
        residual_score += residual_score_table[
            residual_base + subspace_index * prepared_index.codebook_size + code
        ]

    return prepared_index.residual_norms[token_index] * residual_score + centroid_score


def best_tachiom_hnsw_pq_address_score_for_query_vector_in_document_dim128(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    read residual_score_table: List[ScoreScalar],
    start_token: Int,
    stop_token: Int,
) -> ScoreScalar:
    var best_score = min_score_scalar()

    for token_index in range(start_token, stop_token):
        var score = dot_query_vector_with_tachiom_hnsw_pq_address_token_dim128(
            query,
            query_vector_index,
            prepared_index,
            residual_score_table,
            token_index,
        )
        if score > best_score:
            best_score = score

    return best_score


def tachiom_tac_hnsw_pq_address_score_for_document(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    read residual_score_table: List[ScoreScalar],
    document_index: Int,
) -> ScoreScalar:
    var start_token = Int(prepared_index.doc_offsets[document_index])
    var stop_token = Int(prepared_index.doc_offsets[document_index + 1])
    var total = zero_score_scalar()

    for query_vector_index in range(query.vector_count):
        total += best_tachiom_hnsw_pq_address_score_for_query_vector_in_document_dim128(
            query,
            query_vector_index,
            prepared_index,
            residual_score_table,
            start_token,
            stop_token,
        )

    return total


def tachiom_tac_hnsw_pq_address_rerank_candidates_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[Int]:
    require_positive_int("final_k", final_k)
    var residual_score_table = build_tachiom_hnsw_pq_address_residual_score_table(
        query, prepared_index
    )

    var rerank_scores = List[ScoreScalar]()
    rerank_scores.reserve(len(candidate_positions))
    for document_index in candidate_positions:
        rerank_scores.append(
            tachiom_tac_hnsw_pq_address_score_for_document(
                query, prepared_index, residual_score_table, document_index
            )
        )

    var winner_offsets = top_positions_by_score(rerank_scores, final_k)
    var winners = List[Int]()
    winners.reserve(len(winner_offsets))
    for winner_offset in winner_offsets:
        winners.append(candidate_positions[winner_offset])

    return winners^


def tachiom_tac_hnsw_pq_address_search_all_documents_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    final_k: Int,
) raises -> List[Int]:
    var residual_score_table = build_tachiom_hnsw_pq_address_residual_score_table(
        query, prepared_index
    )
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.document_count):
        scores.append(
            tachiom_tac_hnsw_pq_address_score_for_document(
                query, prepared_index, residual_score_table, document_index
            )
        )

    return top_positions_by_score(scores, final_k)


def tachiom_tac_hnsw_pq_address_search_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    ef_search: Int,
) raises -> List[Int]:
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)
    if candidate_k >= prepared_index.document_count:
        return tachiom_tac_hnsw_pq_address_search_all_documents_for_query(
            query, prepared_index, final_k
        )
    var candidate_positions = tachiom_tac_hnsw_pq_address_candidate_positions_for_query(
        query,
        prepared_index,
        centroids_per_query_vector,
        candidate_k,
        ef_search,
    )
    return tachiom_tac_hnsw_pq_address_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )


def tachiom_tac_hnsw_pq_address_search_positions_for_query_with_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqAddressIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    ef_search: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)
    if candidate_k >= prepared_index.document_count:
        return tachiom_tac_hnsw_pq_address_search_all_documents_for_query(
            query, prepared_index, final_k
        )
    var candidate_positions = (
        tachiom_tac_hnsw_pq_address_candidate_positions_for_query_with_pruning(
            query,
            prepared_index,
            centroids_per_query_vector,
            candidate_k,
            final_k,
            ef_search,
            candidate_pruning_alpha,
        )
    )
    return tachiom_tac_hnsw_pq_address_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )
