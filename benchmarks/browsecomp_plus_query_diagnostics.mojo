from std.collections import List
from std.math import min

from kayak.eval import evaluate_query_hits
from kayak.runtime import ExactCpuBackend
from kayak.search import SearchHit, search_exact
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


def print_hits(
    label: String,
    hits: List[SearchHit],
    evidence_doc_ids: List[String],
    gold_doc_ids: List[String],
    max_hits: Int = 5,
):
    print(label, " top_hits:")
    var limit = min(max_hits, len(hits))

    for index in range(limit):
        var hit = hits[index].copy()
        print(
            "  ",
            index + 1,
            ". ",
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

    if len(evidence_task.queries) != len(gold_task.queries):
        raise Error("BrowseComp-Plus evidence and gold tasks must have matching queries")

    print("family: ", evidence_task.family)
    print("evidence_slice: ", evidence_task.slice_name)
    print("gold_slice: ", gold_task.slice_name)
    print("queries: ", len(evidence_task.queries))
    print("documents: ", len(evidence_task.documents))
    print("query_vectors≈ ", evidence_task.nominal_query_vector_count)
    print("doc_vectors≈ ", evidence_task.nominal_document_vector_count)
    print("vector_dim: ", evidence_task.vector_dim)
    print("")

    for index in range(len(evidence_task.queries)):
        var evidence_query = evidence_task.queries[index].copy()
        var gold_query = gold_task.queries[index].copy()

        if evidence_query.query_id != gold_query.query_id:
            raise Error("BrowseComp-Plus evidence and gold query ids must align")

        var evidence_hits = search_exact(
            backend, evidence_query.query, evidence_index, evidence_task.k
        )
        var gold_hits = search_exact(
            backend, gold_query.query, gold_index, gold_task.k
        )
        var evidence_eval = evaluate_query_hits(
            evidence_query, evidence_hits, evidence_task.k, evidence_task.primary_metric
        )
        var gold_eval = evaluate_query_hits(
            gold_query, gold_hits, gold_task.k, gold_task.primary_metric
        )

        print("query_id: ", evidence_query.query_id)
        print("text: ", evidence_query.description)
        print(
            "evidence metrics: ndcg@",
            evidence_eval.k,
            "=",
            evidence_eval.ndcg_at_k,
            " mrr@",
            evidence_eval.k,
            "=",
            evidence_eval.reciprocal_rank_at_k,
            " recall@",
            evidence_eval.k,
            "=",
            evidence_eval.recall_at_k,
            " success@",
            evidence_eval.k,
            "=",
            evidence_eval.success_at_k,
        )
        print(
            "gold metrics: ndcg@",
            gold_eval.k,
            "=",
            gold_eval.ndcg_at_k,
            " mrr@",
            gold_eval.k,
            "=",
            gold_eval.reciprocal_rank_at_k,
            " recall@",
            gold_eval.k,
            "=",
            gold_eval.recall_at_k,
            " success@",
            gold_eval.k,
            "=",
            gold_eval.success_at_k,
        )
        print(
            "relevant counts: evidence=",
            evidence_eval.relevant_doc_count,
            " gold=",
            gold_eval.relevant_doc_count,
        )
        print_hits(
            "evidence",
            evidence_eval.hits,
            evidence_query.relevant_doc_ids,
            gold_query.relevant_doc_ids,
        )
        print_hits(
            "gold",
            gold_eval.hits,
            evidence_query.relevant_doc_ids,
            gold_query.relevant_doc_ids,
        )
        print("")
