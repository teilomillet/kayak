from std.collections import List
from std.math import min

from kayak.numeric import MetricScalar
from kayak.search import SearchHit


def is_relevant(doc_id: String, relevant_doc_ids: List[String]) -> Bool:
    for relevant_doc_id in relevant_doc_ids:
        if doc_id == relevant_doc_id:
            return True

    return False


def reciprocal_rank_at_k(
    hits: List[SearchHit], relevant_doc_ids: List[String], k: Int
) -> MetricScalar:
    var limit = min(k, len(hits))

    for index in range(limit):
        if is_relevant(hits[index].doc_id, relevant_doc_ids):
            return MetricScalar(1.0) / MetricScalar(index + 1)

    return MetricScalar(0.0)


def success_at_k(
    hits: List[SearchHit], relevant_doc_ids: List[String], k: Int
) -> MetricScalar:
    var limit = min(k, len(hits))

    for index in range(limit):
        if is_relevant(hits[index].doc_id, relevant_doc_ids):
            return 1.0

    return MetricScalar(0.0)


def recall_at_k(
    hits: List[SearchHit], relevant_doc_ids: List[String], k: Int
) -> MetricScalar:
    if len(relevant_doc_ids) == 0:
        return MetricScalar(0.0)

    var found_count = 0
    var limit = min(k, len(hits))

    for index in range(limit):
        if is_relevant(hits[index].doc_id, relevant_doc_ids):
            found_count += 1

    return MetricScalar(found_count) / MetricScalar(len(relevant_doc_ids))
