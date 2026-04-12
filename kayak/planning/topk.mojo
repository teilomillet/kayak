from std.collections import List

from .collection_hit import CollectionHit


def insert_descending_collection_hit(
    mut hits: List[CollectionHit], var hit: CollectionHit, k: Int
):
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
