from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.runtime import ExactScoringBackend

from .hit import SearchHit
from .topk import top_k_hits


def search_exact[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read index: PackedIndex,
    k: Int,
) raises -> List[SearchHit]:
    var scores = backend.score_all(query, index)
    return top_k_hits(index.doc_ids, scores, k)


def search_exact_all[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read index: PackedIndex,
) raises -> List[SearchHit]:
    var scores = backend.score_all(query, index)
    return top_k_hits(index.doc_ids, scores, len(index.doc_ids))
