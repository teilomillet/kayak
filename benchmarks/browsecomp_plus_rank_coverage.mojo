from std.collections import List
from std.math import min

from kayak.numeric import ScoreScalar
from kayak.runtime import ExactCpuBackend
from kayak.search import SearchHit, search_exact_all
from kayak.storage import (
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_browsecomp_plus_real_subset_cache,
)


def is_relevant(doc_id: String, relevant_doc_ids: List[String]) -> Bool:
    for relevant_doc_id in relevant_doc_ids:
        if doc_id == relevant_doc_id:
            return True

    return False


def relevance_label(
    doc_id: String, evidence_doc_ids: List[String], gold_doc_ids: List[String]
) -> String:
    var in_evidence = is_relevant(doc_id, evidence_doc_ids)
    var in_gold = is_relevant(doc_id, gold_doc_ids)

    if in_evidence and in_gold:
        return "EG"

    if in_evidence:
        return "E"

    if in_gold:
        return "G"

    return "-"


def rank_of_doc_id(hits: List[SearchHit], doc_id: String) -> Int:
    for index in range(len(hits)):
        if hits[index].doc_id == doc_id:
            return index + 1

    return 0


def score_of_doc_id(
    hits: List[SearchHit], doc_id: String
) raises -> ScoreScalar:
    for index in range(len(hits)):
        if hits[index].doc_id == doc_id:
            return hits[index].score

    raise Error("doc id not found in ranked hits: " + doc_id)


def hit_count_at_k(
    hits: List[SearchHit], relevant_doc_ids: List[String], k: Int
) -> Int:
    var found_count = 0
    var limit = min(k, len(hits))

    for index in range(limit):
        if is_relevant(hits[index].doc_id, relevant_doc_ids):
            found_count += 1

    return found_count


def best_rank(hits: List[SearchHit], relevant_doc_ids: List[String]) -> Int:
    var current_best = 0

    for relevant_doc_id in relevant_doc_ids:
        var rank = rank_of_doc_id(hits, relevant_doc_id)
        if rank == 0:
            continue

        if current_best == 0 or rank < current_best:
            current_best = rank

    return current_best


def print_hit_curve(
    label: String, hits: List[SearchHit], relevant_doc_ids: List[String]
):
    print(label, " hit_curve:")
    for k in [5, 10, 20, 50]:
        print(
            "  hits@",
            k,
            "=",
            hit_count_at_k(hits, relevant_doc_ids, k),
            "/",
            len(relevant_doc_ids),
        )


def cutoff_score(hits: List[SearchHit], k: Int) -> ScoreScalar:
    if len(hits) == 0 or k <= 0:
        return 0.0

    return hits[min(k, len(hits)) - 1].score


def assert_same_ranking(
    evidence_hits: List[SearchHit], gold_hits: List[SearchHit]
) raises:
    if len(evidence_hits) != len(gold_hits):
        raise Error("evidence and gold rankings must have the same length")

    for index in range(len(evidence_hits)):
        if evidence_hits[index].doc_id != gold_hits[index].doc_id:
            raise Error(
                "evidence and gold rankings diverged at rank "
                + String(index + 1)
            )
        if evidence_hits[index].score != gold_hits[index].score:
            raise Error(
                "evidence and gold scores diverged at rank " + String(index + 1)
            )


def print_relevant_ranks(
    label: String,
    hits: List[SearchHit],
    relevant_doc_ids: List[String],
    k: Int,
) raises:
    print(label, " relevant_doc_ranks:")
    var top_k_cutoff = cutoff_score(hits, k)

    for relevant_doc_id in relevant_doc_ids:
        var rank = rank_of_doc_id(hits, relevant_doc_id)
        var score = score_of_doc_id(hits, relevant_doc_id)
        print(
            "  doc_id=",
            relevant_doc_id,
            " rank=",
            rank,
            " score=",
            score,
            " delta_to_top",
            k,
            "=",
            score - top_k_cutoff,
        )


def print_hits(
    label: String,
    hits: List[SearchHit],
    evidence_doc_ids: List[String],
    gold_doc_ids: List[String],
    max_hits: Int,
):
    print(label, ":")
    var limit = min(max_hits, len(hits))

    for index in range(limit):
        var hit = hits[index].copy()
        print(
            "  rank=",
            index + 1,
            " doc_id=",
            hit.doc_id,
            " score=",
            hit.score,
            " rel=",
            relevance_label(hit.doc_id, evidence_doc_ids, gold_doc_ids),
        )


def print_rank_window(
    label: String,
    hits: List[SearchHit],
    center_rank: Int,
    evidence_doc_ids: List[String],
    gold_doc_ids: List[String],
    radius: Int = 2,
):
    if center_rank == 0:
        return

    var start = center_rank - radius
    if start < 1:
        start = 1

    var end = center_rank + radius
    if end > len(hits):
        end = len(hits)

    print(label, " window:")
    for rank in range(start, end + 1):
        var hit = hits[rank - 1].copy()
        print(
            "  rank=",
            rank,
            " doc_id=",
            hit.doc_id,
            " score=",
            hit.score,
            " rel=",
            relevance_label(hit.doc_id, evidence_doc_ids, gold_doc_ids),
        )


def main() raises:
    print("loading BrowseComp-Plus evidence and gold slices with storage...")
    var evidence_cache = ensure_browsecomp_plus_real_subset_cache()
    var gold_cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var evidence_task = evidence_cache.stored_task.task.copy()
    var gold_task = gold_cache.stored_task.task.copy()
    var evidence_index = evidence_cache.stored_index.index.copy()
    var gold_index = gold_cache.stored_index.index.copy()
    var backend = ExactCpuBackend()

    if len(evidence_index.doc_ids) != len(gold_index.doc_ids):
        raise Error("BrowseComp-Plus evidence and gold indexes must have matching docs")

    for index in range(len(evidence_index.doc_ids)):
        if evidence_index.doc_ids[index] != gold_index.doc_ids[index]:
            raise Error(
                "BrowseComp-Plus evidence and gold indexes must have aligned doc ids"
            )

    print("family: ", evidence_task.family)
    print("evidence_slice: ", evidence_task.slice_name)
    print("gold_slice: ", gold_task.slice_name)
    print("queries: ", len(evidence_task.queries))
    print("documents: ", len(evidence_index.doc_ids))
    print("k: ", evidence_task.k)
    print("")
    print("query summaries:")

    var focus_found = False

    for index in range(len(evidence_task.queries)):
        var evidence_query = evidence_task.queries[index].copy()
        var gold_query = gold_task.queries[index].copy()

        if evidence_query.query_id != gold_query.query_id:
            raise Error("BrowseComp-Plus evidence and gold query ids must align")

        var evidence_hits = search_exact_all(
            backend, evidence_query.query, evidence_index
        )
        var gold_hits = search_exact_all(backend, gold_query.query, gold_index)
        assert_same_ranking(evidence_hits, gold_hits)

        print(
            "  query_id=",
            evidence_query.query_id,
            " evidence_hits@",
            evidence_task.k,
            "=",
            hit_count_at_k(
                evidence_hits, evidence_query.relevant_doc_ids, evidence_task.k
            ),
            "/",
            len(evidence_query.relevant_doc_ids),
            " gold_hits@",
            gold_task.k,
            "=",
            hit_count_at_k(gold_hits, gold_query.relevant_doc_ids, gold_task.k),
            "/",
            len(gold_query.relevant_doc_ids),
            " best_evidence_rank=",
            best_rank(evidence_hits, evidence_query.relevant_doc_ids),
            " best_gold_rank=",
            best_rank(gold_hits, gold_query.relevant_doc_ids),
        )

        if evidence_query.query_id != "772":
            continue

        focus_found = True
        print("")
        print("focused query: ", evidence_query.query_id)
        print("text: ", evidence_query.description)
        print("top", evidence_task.k, " cutoff score: ", cutoff_score(evidence_hits, evidence_task.k))
        print_hit_curve("evidence", evidence_hits, evidence_query.relevant_doc_ids)
        print_hit_curve("gold", gold_hits, gold_query.relevant_doc_ids)
        print_relevant_ranks(
            "evidence", evidence_hits, evidence_query.relevant_doc_ids, evidence_task.k
        )
        print_relevant_ranks(
            "gold", gold_hits, gold_query.relevant_doc_ids, gold_task.k
        )
        print_hits(
            "top hits",
            evidence_hits,
            evidence_query.relevant_doc_ids,
            gold_query.relevant_doc_ids,
            evidence_task.k,
        )

        for gold_doc_id in gold_query.relevant_doc_ids:
            print_rank_window(
                "gold doc " + gold_doc_id,
                gold_hits,
                rank_of_doc_id(gold_hits, gold_doc_id),
                evidence_query.relevant_doc_ids,
                gold_query.relevant_doc_ids,
            )
        print("")

    if not focus_found:
        print("focus query 772 not present in this slice")
