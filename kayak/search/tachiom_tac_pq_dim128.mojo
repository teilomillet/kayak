from std.collections import List
from std.format import Writable, Writer

from kayak.contracts import FlatQueryDim128
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .plaid_approx_dim128 import require_positive_int, top_positions_by_score
from .tachiom_tac_dim128 import prune_tachiom_candidate_positions_by_score


# TAC candidate generation with centroid-id plus normalized residual-PQ rerank.
# The PQ codebooks are built by Python; this file owns the native scoring loop.
struct PreparedTachiomTacPqIndex(Movable, Writable):
    var doc_ids: List[String]
    var doc_offsets: List[Int]
    var centroid_values: List[ScoreScalar]
    var centroid_doc_offsets: List[Int]
    var centroid_doc_indices: List[Int]
    var token_centroid_positions: List[Int]
    var residual_norms: List[ScoreScalar]
    var pq_codes: List[Int]
    var pq_codebooks: List[ScoreScalar]
    var vector_dim: Int
    var document_count: Int
    var total_vector_count: Int
    var centroid_count: Int
    var subspace_count: Int
    var codebook_size: Int
    var subspace_dim: Int

    def __init__(
        out self,
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
    ) raises:
        require_positive_int("subspace_count", subspace_count)
        require_positive_int("codebook_size", codebook_size)
        if COLBERT_VECTOR_DIM % subspace_count != 0:
            raise Error("Tachiom PQ subspace_count must divide dim128")
        if len(centroid_values) % COLBERT_VECTOR_DIM != 0:
            raise Error("Tachiom PQ centroid values must be aligned to dim128")
        var centroid_count = len(centroid_values) // COLBERT_VECTOR_DIM
        if len(doc_offsets) != len(doc_ids) + 1:
            raise Error("Tachiom PQ doc_offsets length must be document_count + 1")
        if len(doc_offsets) == 0 or doc_offsets[0] != 0:
            raise Error("Tachiom PQ doc_offsets must start at zero")
        var total_vector_count = doc_offsets[len(doc_offsets) - 1]
        if len(token_centroid_positions) != total_vector_count:
            raise Error(
                "Tachiom PQ token centroid positions must match token count"
            )
        if len(residual_norms) != total_vector_count:
            raise Error("Tachiom PQ residual norms must match token count")
        if len(pq_codes) != total_vector_count * subspace_count:
            raise Error("Tachiom PQ code count must match token/subspace count")
        var subspace_dim = COLBERT_VECTOR_DIM // subspace_count
        if len(pq_codebooks) != subspace_count * codebook_size * subspace_dim:
            raise Error("Tachiom PQ codebooks must match subspace/codebook shape")
        if len(centroid_doc_offsets) != centroid_count + 1:
            raise Error(
                "Tachiom PQ centroid_doc_offsets length must be centroid_count + 1"
            )
        if len(centroid_doc_offsets) == 0 or centroid_doc_offsets[0] != 0:
            raise Error("Tachiom PQ centroid_doc_offsets must start at zero")
        if centroid_doc_offsets[len(centroid_doc_offsets) - 1] != len(
            centroid_doc_indices
        ):
            raise Error("Tachiom PQ centroid_doc_offsets must end at posting count")

        self.doc_ids = doc_ids^
        self.doc_offsets = doc_offsets^
        self.centroid_values = centroid_values^
        self.centroid_doc_offsets = centroid_doc_offsets^
        self.centroid_doc_indices = centroid_doc_indices^
        self.token_centroid_positions = token_centroid_positions^
        self.residual_norms = residual_norms^
        self.pq_codes = pq_codes^
        self.pq_codebooks = pq_codebooks^
        self.vector_dim = COLBERT_VECTOR_DIM
        self.document_count = len(self.doc_ids)
        self.total_vector_count = total_vector_count
        self.centroid_count = centroid_count
        self.subspace_count = subspace_count
        self.codebook_size = codebook_size
        self.subspace_dim = subspace_dim

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedTachiomTacPqIndex(document_count=",
            self.document_count,
            ", total_vector_count=",
            self.total_vector_count,
            ", centroid_count=",
            self.centroid_count,
            ", subspace_count=",
            self.subspace_count,
            ", codebook_size=",
            self.codebook_size,
            ")",
        )


def prepare_tachiom_tac_pq_dim128_index(
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
) raises -> PreparedTachiomTacPqIndex:
    return PreparedTachiomTacPqIndex(
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


def tachiom_tac_pq_prepared_posting_count_value(
    read prepared_index: PreparedTachiomTacPqIndex,
) -> Int:
    return len(prepared_index.centroid_doc_indices)


def score_query_vector_against_tachiom_pq_centroids(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacPqIndex,
) -> List[ScoreScalar]:
    var centroid_scores = List[ScoreScalar]()
    centroid_scores.reserve(prepared_index.centroid_count)
    var query_offset = query_vector_index * COLBERT_VECTOR_DIM

    for centroid_index in range(prepared_index.centroid_count):
        centroid_scores.append(
            dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                prepared_index.centroid_values,
                centroid_index * COLBERT_VECTOR_DIM,
            )
        )

    return centroid_scores^


def tachiom_tac_pq_candidate_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
) raises -> List[Int]:
    var centroid_score_table = build_tachiom_pq_centroid_score_table(
        query, prepared_index
    )
    return tachiom_tac_pq_candidate_positions_for_query_with_centroid_table(
        query,
        prepared_index,
        centroid_score_table,
        centroids_per_query_vector,
        candidate_k,
    )


def tachiom_tac_pq_candidate_positions_for_query_with_centroid_table(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    read centroid_score_table: List[ScoreScalar],
    centroids_per_query_vector: Int,
    candidate_k: Int,
) raises -> List[Int]:
    var document_scores = tachiom_tac_pq_document_scores_for_query_with_centroid_table(
        query,
        prepared_index,
        centroid_score_table,
        centroids_per_query_vector,
    )
    return top_positions_by_score(document_scores, candidate_k)


def tachiom_tac_pq_candidate_positions_for_query_with_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    var centroid_score_table = build_tachiom_pq_centroid_score_table(
        query, prepared_index
    )
    return tachiom_tac_pq_candidate_positions_for_query_with_centroid_table_and_pruning(
        query,
        prepared_index,
        centroid_score_table,
        centroids_per_query_vector,
        candidate_k,
        final_k,
        candidate_pruning_alpha,
    )


def tachiom_tac_pq_candidate_positions_for_query_with_centroid_table_and_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    read centroid_score_table: List[ScoreScalar],
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    var document_scores = tachiom_tac_pq_document_scores_for_query_with_centroid_table(
        query,
        prepared_index,
        centroid_score_table,
        centroids_per_query_vector,
    )
    var ranked_positions = top_positions_by_score(document_scores, candidate_k)
    return prune_tachiom_candidate_positions_by_score(
        ranked_positions,
        document_scores,
        final_k,
        candidate_pruning_alpha,
    )


def tachiom_tac_pq_document_scores_for_query_with_centroid_table(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    read centroid_score_table: List[ScoreScalar],
    centroids_per_query_vector: Int,
) raises -> List[ScoreScalar]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )

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
        var centroid_scores = tachiom_pq_centroid_scores_for_query_vector_from_table(
            prepared_index,
            centroid_score_table,
            query_vector_index,
        )
        var centroid_positions = top_positions_by_score(
            centroid_scores, centroids_per_query_vector
        )
        var touched_documents = List[Int]()
        touched_documents.reserve(prepared_index.document_count)

        for centroid_position in centroid_positions:
            var centroid_score = centroid_scores[centroid_position]
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

    for document_index in range(prepared_index.document_count):
        if document_seen[document_index] == 0:
            document_scores[document_index] = min_score_scalar()

    return document_scores^


def tachiom_pq_centroid_scores_for_query_vector_from_table(
    read prepared_index: PreparedTachiomTacPqIndex,
    read centroid_score_table: List[ScoreScalar],
    query_vector_index: Int,
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    scores.reserve(prepared_index.centroid_count)
    var start = query_vector_index * prepared_index.centroid_count
    var score_ptr = centroid_score_table.unsafe_ptr()
    for centroid_index in range(prepared_index.centroid_count):
        scores.append(score_ptr[start + centroid_index])
    return scores^


def dot_query_vector_with_tachiom_pq_token_dim128(
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacPqIndex,
    read centroid_score_table: List[ScoreScalar],
    read residual_score_table: List[ScoreScalar],
    token_index: Int,
) -> ScoreScalar:
    var centroid_positions_ptr = prepared_index.token_centroid_positions.unsafe_ptr()
    var centroid_score_ptr = centroid_score_table.unsafe_ptr()
    var residual_score_ptr = residual_score_table.unsafe_ptr()
    var pq_codes_ptr = prepared_index.pq_codes.unsafe_ptr()
    var residual_norms_ptr = prepared_index.residual_norms.unsafe_ptr()
    var centroid_position = centroid_positions_ptr[token_index]
    var centroid_score = centroid_score_ptr[
        query_vector_index * prepared_index.centroid_count + centroid_position
    ]
    var residual_score = zero_score_scalar()
    var code_offset = token_index * prepared_index.subspace_count
    var residual_base = query_vector_index * prepared_index.subspace_count
    residual_base *= prepared_index.codebook_size

    for subspace_index in range(prepared_index.subspace_count):
        var code = pq_codes_ptr[code_offset + subspace_index]
        residual_score += residual_score_ptr[
            residual_base + subspace_index * prepared_index.codebook_size + code
        ]

    return centroid_score + residual_norms_ptr[token_index] * residual_score


def build_tachiom_pq_centroid_score_table(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
) -> List[ScoreScalar]:
    var table = List[ScoreScalar]()
    table.reserve(query.vector_count * prepared_index.centroid_count)

    for query_vector_index in range(query.vector_count):
        var query_offset = query_vector_index * COLBERT_VECTOR_DIM
        for centroid_index in range(prepared_index.centroid_count):
            table.append(
                dot_product_dim128_flat_pair_at(
                    query.token_values,
                    query_offset,
                    prepared_index.centroid_values,
                    centroid_index * COLBERT_VECTOR_DIM,
                )
            )

    return table^


def build_tachiom_pq_residual_score_table(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
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


def best_tachiom_pq_score_for_query_vector_in_document_dim128(
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacPqIndex,
    read centroid_score_table: List[ScoreScalar],
    read residual_score_table: List[ScoreScalar],
    start_token: Int,
    stop_token: Int,
) -> ScoreScalar:
    var best_score = min_score_scalar()

    for token_index in range(start_token, stop_token):
        var score = dot_query_vector_with_tachiom_pq_token_dim128(
            query_vector_index,
            prepared_index,
            centroid_score_table,
            residual_score_table,
            token_index,
        )
        if token_index == start_token or score > best_score:
            best_score = score

    return best_score


def tachiom_tac_pq_score_for_document(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    read centroid_score_table: List[ScoreScalar],
    read residual_score_table: List[ScoreScalar],
    document_index: Int,
) -> ScoreScalar:
    var start_token = prepared_index.doc_offsets[document_index]
    var stop_token = prepared_index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()

    for query_vector_index in range(query.vector_count):
        total += best_tachiom_pq_score_for_query_vector_in_document_dim128(
            query_vector_index,
            prepared_index,
            centroid_score_table,
            residual_score_table,
            start_token,
            stop_token,
        )

    return total


def tachiom_tac_pq_rerank_candidates_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[Int]:
    var centroid_score_table = build_tachiom_pq_centroid_score_table(
        query, prepared_index
    )
    return tachiom_tac_pq_rerank_candidates_for_query_with_centroid_table(
        query,
        prepared_index,
        candidate_positions,
        centroid_score_table,
        final_k,
    )


def tachiom_tac_pq_rerank_candidates_for_query_with_centroid_table(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    read candidate_positions: List[Int],
    read centroid_score_table: List[ScoreScalar],
    final_k: Int,
) raises -> List[Int]:
    require_positive_int("final_k", final_k)
    var residual_score_table = build_tachiom_pq_residual_score_table(
        query, prepared_index
    )

    var rerank_scores = List[ScoreScalar]()
    for document_index in candidate_positions:
        rerank_scores.append(
            tachiom_tac_pq_score_for_document(
                query,
                prepared_index,
                centroid_score_table,
                residual_score_table,
                document_index,
            )
        )

    var winner_offsets = top_positions_by_score(rerank_scores, final_k)
    var winners = List[Int]()
    for winner_offset in winner_offsets:
        winners.append(candidate_positions[winner_offset])

    return winners^


def tachiom_tac_pq_search_all_documents_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    final_k: Int,
) raises -> List[Int]:
    var centroid_score_table = build_tachiom_pq_centroid_score_table(
        query, prepared_index
    )
    var residual_score_table = build_tachiom_pq_residual_score_table(
        query, prepared_index
    )
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.document_count):
        scores.append(
            tachiom_tac_pq_score_for_document(
                query,
                prepared_index,
                centroid_score_table,
                residual_score_table,
                document_index,
            )
        )

    return top_positions_by_score(scores, final_k)


def tachiom_tac_pq_search_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)

    if candidate_k >= prepared_index.document_count:
        return tachiom_tac_pq_search_all_documents_for_query(
            query, prepared_index, final_k
        )

    var centroid_score_table = build_tachiom_pq_centroid_score_table(
        query, prepared_index
    )
    var candidate_positions = (
        tachiom_tac_pq_candidate_positions_for_query_with_centroid_table(
            query,
            prepared_index,
            centroid_score_table,
            centroids_per_query_vector,
            candidate_k,
        )
    )
    return tachiom_tac_pq_rerank_candidates_for_query_with_centroid_table(
        query,
        prepared_index,
        candidate_positions,
        centroid_score_table,
        final_k,
    )


def tachiom_tac_pq_search_positions_for_query_with_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacPqIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)

    if candidate_k >= prepared_index.document_count:
        return tachiom_tac_pq_search_all_documents_for_query(
            query, prepared_index, final_k
        )

    var centroid_score_table = build_tachiom_pq_centroid_score_table(
        query, prepared_index
    )
    var candidate_positions = (
        tachiom_tac_pq_candidate_positions_for_query_with_centroid_table_and_pruning(
            query,
            prepared_index,
            centroid_score_table,
            centroids_per_query_vector,
            candidate_k,
            final_k,
            candidate_pruning_alpha,
        )
    )
    return tachiom_tac_pq_rerank_candidates_for_query_with_centroid_table(
        query,
        prepared_index,
        candidate_positions,
        centroid_score_table,
        final_k,
    )
