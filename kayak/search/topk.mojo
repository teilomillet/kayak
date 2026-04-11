from std.collections import List

from kayak.numeric import ScoreScalar

from .hit import SearchHit


def insert_descending(mut hits: List[SearchHit], var hit: SearchHit, k: Int):
    if k == 0:
        return

    var insert_at = 0
    while insert_at < len(hits) and hits[insert_at].score >= hit.score:
        insert_at += 1

    if insert_at >= k:
        if len(hits) < k:
            hits.append(hit^)
        return

    if len(hits) < k:
        hits.append(hit.copy())

    var current = len(hits) - 1
    while current > insert_at:
        hits[current] = hits[current - 1].copy()
        current -= 1

    hits[insert_at] = hit^


def top_k_hits(
    doc_ids: List[String], scores: List[ScoreScalar], k: Int
) raises -> List[SearchHit]:
    if len(doc_ids) != len(scores):
        raise Error("top-k input lengths must match")

    if k < 0:
        raise Error("top-k requires a non-negative k")

    var hits = List[SearchHit]()

    for index in range(len(scores)):
        insert_descending(
            hits, SearchHit(doc_ids[index].copy(), scores[index]), k
        )

    return hits^
