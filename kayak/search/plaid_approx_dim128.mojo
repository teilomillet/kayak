from std.collections import List
from std.format import Writable, Writer

from kayak.contracts import FlatQueryDim128
from kayak.index import HybridFlatDim128Index
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar
from kayak.scoring import exact_score_for_hybrid_flat_document_dim128_with_flat_query
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .hit import SearchHit
from .topk import insert_descending, top_k_hits


# Owns the dim128 sampled-centroid approximation used by the FastPlaid speed
# track. It does not own Python binding, benchmark reporting, or public API
# policy.
struct PreparedPlaidApproxIndex(Movable, Writable):
    var index: HybridFlatDim128Index
    var centroid_token_indices: List[Int]
    var centroid_doc_offsets: List[Int]
    var centroid_doc_indices: List[Int]
    var centroid_count: Int

    def __init__(
        out self,
        var index: HybridFlatDim128Index,
        var centroid_token_indices: List[Int],
        var centroid_doc_offsets: List[Int],
        var centroid_doc_indices: List[Int],
    ):
        self.centroid_count = len(centroid_token_indices)
        self.index = index^
        self.centroid_token_indices = centroid_token_indices^
        self.centroid_doc_offsets = centroid_doc_offsets^
        self.centroid_doc_indices = centroid_doc_indices^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedPlaidApproxIndex(document_count=",
            self.index.document_count,
            ", total_vector_count=",
            self.index.total_vector_count,
            ", centroid_count=",
            self.centroid_count,
            ")",
        )


def require_positive_int(name: String, value: Int) raises:
    if value <= 0:
        raise Error(name + " must be positive")


def sampled_centroid_token_indices(
    total_vector_count: Int, requested_centroid_count: Int
) raises -> List[Int]:
    require_positive_int("centroid_count", requested_centroid_count)
    require_positive_int("total_vector_count", total_vector_count)

    var centroid_count = requested_centroid_count
    if centroid_count > total_vector_count:
        centroid_count = total_vector_count

    var token_indices = List[Int]()
    if centroid_count == 1:
        token_indices.append(0)
        return token_indices^

    var last_token_index = total_vector_count - 1
    var denominator = centroid_count - 1
    for centroid_index in range(centroid_count):
        token_indices.append((centroid_index * last_token_index) // denominator)

    return token_indices^


def nearest_sampled_centroid_for_token(
    read index: HybridFlatDim128Index,
    token_index: Int,
    read centroid_token_indices: List[Int],
) -> Int:
    var best_centroid_index = 0
    var best_score = min_score_scalar()
    var token_offset = token_index * COLBERT_VECTOR_DIM

    for centroid_index in range(len(centroid_token_indices)):
        var centroid_token_index = centroid_token_indices[centroid_index]
        var centroid_offset = centroid_token_index * COLBERT_VECTOR_DIM
        var score = dot_product_dim128_flat_pair_at(
            index.token_values,
            token_offset,
            index.token_values,
            centroid_offset,
        )
        if centroid_index == 0 or score > best_score:
            best_centroid_index = centroid_index
            best_score = score

    return best_centroid_index


def build_centroid_doc_presence(
    read index: HybridFlatDim128Index,
    read centroid_token_indices: List[Int],
) -> List[Int]:
    var presence = List[Int]()
    var presence_count = len(centroid_token_indices) * index.document_count
    for _ in range(presence_count):
        presence.append(0)

    for document_index in range(index.document_count):
        var start_token = index.doc_offsets[document_index]
        var stop_token = index.doc_offsets[document_index + 1]
        for token_index in range(start_token, stop_token):
            var centroid_index = nearest_sampled_centroid_for_token(
                index, token_index, centroid_token_indices
            )
            presence[(centroid_index * index.document_count) + document_index] = 1

    return presence^


def centroid_doc_offsets_from_presence(
    read presence: List[Int], centroid_count: Int, document_count: Int
) -> List[Int]:
    var offsets = List[Int]()
    var running_count = 0
    offsets.append(running_count)

    for centroid_index in range(centroid_count):
        var presence_offset = centroid_index * document_count
        for document_index in range(document_count):
            if presence[presence_offset + document_index] != 0:
                running_count += 1
        offsets.append(running_count)

    return offsets^


def centroid_doc_indices_from_presence(
    read presence: List[Int], centroid_count: Int, document_count: Int
) -> List[Int]:
    var doc_indices = List[Int]()

    for centroid_index in range(centroid_count):
        var presence_offset = centroid_index * document_count
        for document_index in range(document_count):
            if presence[presence_offset + document_index] != 0:
                doc_indices.append(document_index)

    return doc_indices^


def prepare_plaid_approx_hybrid_flat_dim128_index(
    var index: HybridFlatDim128Index,
    centroid_count: Int,
) raises -> PreparedPlaidApproxIndex:
    var centroid_token_indices = sampled_centroid_token_indices(
        index.total_vector_count, centroid_count
    )
    var centroid_doc_presence = build_centroid_doc_presence(
        index, centroid_token_indices
    )
    var centroid_doc_offsets = centroid_doc_offsets_from_presence(
        centroid_doc_presence,
        len(centroid_token_indices),
        index.document_count,
    )
    var centroid_doc_indices = centroid_doc_indices_from_presence(
        centroid_doc_presence,
        len(centroid_token_indices),
        index.document_count,
    )
    return PreparedPlaidApproxIndex(
        index^,
        centroid_token_indices^,
        centroid_doc_offsets^,
        centroid_doc_indices^,
    )


def plaid_approx_prepared_posting_count_value(
    read prepared_index: PreparedPlaidApproxIndex,
) -> Int:
    return len(prepared_index.centroid_doc_indices)


def top_positions_by_score(read scores: List[ScoreScalar], k: Int) raises -> List[Int]:
    require_positive_int("k", k)

    var selected = List[Int]()
    if len(scores) == 0:
        return selected^

    var used = List[Int]()
    for _ in range(len(scores)):
        used.append(0)

    var limit = k
    if limit > len(scores):
        limit = len(scores)

    for _ in range(limit):
        var best_index = -1
        var best_score = min_score_scalar()
        for index in range(len(scores)):
            if used[index] != 0:
                continue
            var score = scores[index]
            if best_index == -1 or score > best_score:
                best_index = index
                best_score = score

        if best_index == -1:
            break

        used[best_index] = 1
        selected.append(best_index)

    return selected^


def score_query_vector_against_sampled_centroids(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedPlaidApproxIndex,
) -> List[ScoreScalar]:
    var centroid_scores = List[ScoreScalar]()
    var query_offset = query_vector_index * COLBERT_VECTOR_DIM

    for centroid_index in range(prepared_index.centroid_count):
        var centroid_token_index = prepared_index.centroid_token_indices[centroid_index]
        var centroid_offset = centroid_token_index * COLBERT_VECTOR_DIM
        centroid_scores.append(
            dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                prepared_index.index.token_values,
                centroid_offset,
            )
        )

    return centroid_scores^


def plaid_candidate_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)

    var document_scores = List[ScoreScalar]()
    for _ in range(prepared_index.index.document_count):
        document_scores.append(zero_score_scalar())

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    for _ in range(prepared_index.index.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    for query_vector_index in range(query.vector_count):
        var centroid_scores = score_query_vector_against_sampled_centroids(
            query, query_vector_index, prepared_index
        )
        var centroid_positions = top_positions_by_score(
            centroid_scores, centroids_per_query_vector
        )
        var touched_documents = List[Int]()
        touched_documents.reserve(prepared_index.index.document_count)

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
                elif centroid_score > token_best_scores[document_index]:
                    token_best_scores[document_index] = centroid_score

        for document_index in touched_documents:
            document_scores[document_index] += token_best_scores[document_index]
            token_best_scores[document_index] = min_score_scalar()
            token_seen[document_index] = 0

    return top_positions_by_score(document_scores, candidate_k)


def plaid_rerank_candidates_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxIndex,
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


def plaid_rerank_candidate_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxIndex,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[SearchHit]:
    require_positive_int("final_k", final_k)

    var hits = List[SearchHit]()
    for document_index in candidate_positions:
        var score = exact_score_for_hybrid_flat_document_dim128_with_flat_query(
            query, prepared_index.index, document_index
        )
        insert_descending(
            hits,
            SearchHit(prepared_index.index.doc_ids[document_index].copy(), score),
            final_k,
        )

    return hits^


def plaid_search_all_documents_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxIndex,
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


def plaid_search_all_document_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxIndex,
    final_k: Int,
) raises -> List[SearchHit]:
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.index.document_count):
        scores.append(
            exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                query, prepared_index.index, document_index
            )
        )

    return top_k_hits(prepared_index.index.doc_ids, scores, final_k)


def plaid_search_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)

    if candidate_k >= prepared_index.index.document_count:
        return plaid_search_all_documents_for_query(
            query, prepared_index, final_k
        )

    var candidate_positions = plaid_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    return plaid_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )


def plaid_search_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
) raises -> List[SearchHit]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)

    if candidate_k >= prepared_index.index.document_count:
        return plaid_search_all_document_hits_for_query(
            query, prepared_index, final_k
        )

    var candidate_positions = plaid_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    return plaid_rerank_candidate_hits_for_query(
        query, prepared_index, candidate_positions, final_k
    )
