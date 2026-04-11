from std.algorithm.backend.cpu.parallelize import sync_parallelize
from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)
from kayak.scoring import ExactScoringConfig
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.maxsim import (
    build_vector_balanced_boundaries,
    choose_parallel_work_item_count,
)
from kayak.search.hit import SearchHit
from kayak.search.topk import top_k_hits

from benchmarks.structural.flat_dim128 import (
    dot_product_dim128_flat_at,
    flatten_document_tokens,
)


struct HybridFlatDim128Index(Copyable):
    var doc_offsets: List[Int]
    var token_values: List[VectorScalar]
    var document_count: Int

    def __init__(
        out self,
        var doc_offsets: List[Int],
        var token_values: List[VectorScalar],
        document_count: Int,
    ):
        self.doc_offsets = doc_offsets^
        self.token_values = token_values^
        self.document_count = document_count


def build_hybrid_flat_dim128_index(
    read index: PackedIndex
) -> HybridFlatDim128Index:
    return HybridFlatDim128Index(
        index.doc_offsets.copy(),
        flatten_document_tokens(index.token_vectors),
        index.document_count,
    )


def exact_score_for_hybrid_flat_document_dim128(
    read query: EncodedQuery,
    read hybrid_index: HybridFlatDim128Index,
    document_index: Int,
) -> ScoreScalar:
    var start_vector = hybrid_index.doc_offsets[document_index]
    var stop_vector = hybrid_index.doc_offsets[document_index + 1]
    var start_offset = start_vector * COLBERT_VECTOR_DIM
    var document_vector_count = stop_vector - start_vector
    var total = zero_score_scalar()

    for query_token in query.token_vectors:
        var best_similarity = min_score_scalar()

        for token_index in range(document_vector_count):
            var similarity = dot_product_dim128_flat_at(
                query_token,
                hybrid_index.token_values,
                start_offset + (token_index * COLBERT_VECTOR_DIM),
            )

            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def exact_scores_for_hybrid_flat_index_dim128(
    read query: EncodedQuery,
    read nested_index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
    read config: ExactScoringConfig,
) raises -> List[ScoreScalar]:
    if query.vector_dim != COLBERT_VECTOR_DIM:
        raise Error("hybrid dim128 scorer requires query vector_dim=128")
    if nested_index.vector_dim != COLBERT_VECTOR_DIM:
        raise Error("hybrid dim128 scorer requires index vector_dim=128")

    var document_count = nested_index.document_count
    var work_item_count = choose_parallel_work_item_count(
        query, nested_index, config
    )
    if work_item_count <= 1:
        var serial_scores = List[ScoreScalar]()
        for document_index in range(document_count):
            serial_scores.append(
                exact_score_for_hybrid_flat_document_dim128(
                    query, hybrid_index, document_index
                )
            )
        return serial_scores^

    var scores = List[ScoreScalar]()
    for _ in range(document_count):
        scores.append(zero_score_scalar())

    var boundaries = build_vector_balanced_boundaries(nested_index, work_item_count)
    var scores_ptr = scores.unsafe_ptr()

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            scores_ptr[document_index] = exact_score_for_hybrid_flat_document_dim128(
                query, hybrid_index, document_index
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def search_exact_hybrid_flat_dim128(
    read query: EncodedQuery,
    read nested_index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
    k: Int,
    read config: ExactScoringConfig,
) raises -> List[SearchHit]:
    var scores = exact_scores_for_hybrid_flat_index_dim128(
        query, nested_index, hybrid_index, config
    )
    return top_k_hits(nested_index.doc_ids, scores, k)
