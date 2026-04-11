from std.collections import List

from kayak.search import SearchHit

from .hit_window import take_top_hits


def rerank_hits_noop(read hits: List[SearchHit], k: Int) raises -> List[SearchHit]:
    return take_top_hits(hits, k)
