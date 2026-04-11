from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.runtime import ExactCpuBackend

from .hit import SearchHit
from .topk import top_k_hits


def search_exact(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read index: PackedIndex,
    k: Int,
) raises -> List[SearchHit]:
    var scores = backend.score_all(query, index)
    return top_k_hits(index.doc_ids, scores, k)
