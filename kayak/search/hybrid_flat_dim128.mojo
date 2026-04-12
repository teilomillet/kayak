from std.collections import List

from kayak.contracts import EncodedQuery, FlatQueryDim128
from kayak.index import HybridFlatDim128Index, PackedIndex
from kayak.scoring import (
    ExactScoringConfig,
    exact_scores_for_hybrid_flat_index_dim128,
    exact_scores_for_hybrid_flat_index_dim128_with_flat_query,
)

from .hit import SearchHit
from .topk import top_k_hits


# Owns top-k search over the optional flat dim128 document layout.
# It does not pick defaults; callers opt into this path explicitly.
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


def search_exact_hybrid_flat_dim128_with_flat_query(
    read query: FlatQueryDim128,
    read nested_index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
    k: Int,
    read config: ExactScoringConfig,
) raises -> List[SearchHit]:
    var scores = exact_scores_for_hybrid_flat_index_dim128_with_flat_query(
        query, nested_index, hybrid_index, config
    )
    return top_k_hits(nested_index.doc_ids, scores, k)
