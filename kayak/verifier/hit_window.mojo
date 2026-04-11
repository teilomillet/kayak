from std.collections import List

from kayak.search import SearchHit


def take_top_hits(read hits: List[SearchHit], k: Int) raises -> List[SearchHit]:
    if k < 0:
        raise Error("verifier top-k requires a non-negative k")

    var limited_hits = List[SearchHit]()
    var limit = k
    if limit > len(hits):
        limit = len(hits)

    for index in range(limit):
        limited_hits.append(hits[index].copy())

    return limited_hits^
