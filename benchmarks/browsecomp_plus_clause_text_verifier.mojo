from std.collections import List

from kayak.benchmarks import (
    load_public_benchmark_dataset,
    require_public_benchmark_dataset_loaded_text_corpus,
)
from kayak.eval import JudgedTask, evaluate_query_hits
from kayak.index import PackedIndex
from kayak.numeric import MetricScalar
from kayak.runtime import ExactCpuBackend
from kayak.search import SearchHit, search_exact
from kayak.verifier import (
    default_clause_text_rerank_config,
    rerank_hits_clause_text,
    rescore_hits_clause_text,
)
from kayak.text import DocumentTextCorpus


def truncate_hits(hits: List[SearchHit], k: Int) -> List[SearchHit]:
    var limited = List[SearchHit]()
    var limit = k
    if limit > len(hits):
        limit = len(hits)

    for index in range(limit):
        limited.append(hits[index].copy())

    return limited^


def rank_of_doc_id(hits: List[SearchHit], doc_id: String) -> Int:
    for index in range(len(hits)):
        if hits[index].doc_id == doc_id:
            return index + 1

    return 0


def evaluate_reranked_task(
    label: String,
    read task: JudgedTask,
    read index: PackedIndex,
    read document_texts: DocumentTextCorpus,
    candidate_k: Int,
) raises:
    var backend = ExactCpuBackend()
    var config = default_clause_text_rerank_config()
    var ndcg_total = MetricScalar(0.0)
    var reciprocal_rank_total = MetricScalar(0.0)
    var recall_total = MetricScalar(0.0)
    var success_total = MetricScalar(0.0)

    print("==", label, "==")

    for query_index in range(len(task.queries)):
        var judged_query = task.queries[query_index].copy()
        var candidate_hits = search_exact(
            backend, judged_query.query, index, candidate_k
        )
        var baseline_hits = truncate_hits(candidate_hits, task.k)
        var reranked_hits = rerank_hits_clause_text(
            judged_query.description,
            candidate_hits,
            document_texts,
            task.k,
            config,
        )

        var baseline_eval = evaluate_query_hits(
            judged_query, baseline_hits, task.k, task.primary_metric
        )
        var reranked_eval = evaluate_query_hits(
            judged_query, reranked_hits, task.k, task.primary_metric
        )

        ndcg_total += reranked_eval.ndcg_at_k
        reciprocal_rank_total += reranked_eval.reciprocal_rank_at_k
        recall_total += reranked_eval.recall_at_k
        success_total += reranked_eval.success_at_k

        if judged_query.query_id == "772":
            var reranked_candidates = rescore_hits_clause_text(
                judged_query.description, candidate_hits, document_texts, config
            )
            var focus_doc_id = "93372"
            print("focus_query_id: ", judged_query.query_id)
            print("baseline_gold_rank: ", rank_of_doc_id(candidate_hits, focus_doc_id))
            print(
                "reranked_gold_rank: ",
                rank_of_doc_id(reranked_candidates, focus_doc_id),
            )
            print(
                "baseline_success@",
                task.k,
                ": ",
                baseline_eval.success_at_k,
                " reranked_success@",
                task.k,
                ": ",
                reranked_eval.success_at_k,
            )
            print(
                "baseline_ndcg@",
                task.k,
                ": ",
                baseline_eval.ndcg_at_k,
                " reranked_ndcg@",
                task.k,
                ": ",
                reranked_eval.ndcg_at_k,
            )
            print("reranked_top_hits:")
            for hit_index in range(len(reranked_hits)):
                var hit = reranked_hits[hit_index].copy()
                print(
                    "  rank=",
                    hit_index + 1,
                    " doc_id=",
                    hit.doc_id,
                    " score=",
                    hit.score,
                )

    var query_count = MetricScalar(len(task.queries))
    print("candidate_k: ", candidate_k)
    print("ndcg@", task.k, " = ", ndcg_total / query_count)
    print("mrr@", task.k, " = ", reciprocal_rank_total / query_count)
    print("recall@", task.k, " = ", recall_total / query_count)
    print("success@", task.k, " = ", success_total / query_count)
    print("")


def main() raises:
    print("loading BrowseComp-Plus evidence and gold slices with text sidecar...")
    var evidence_dataset = load_public_benchmark_dataset(
        "browsecomp_plus_real_subset",
        load_text_corpus=True,
    )
    var gold_dataset = load_public_benchmark_dataset(
        "browsecomp_plus_gold",
        load_text_corpus=True,
    )
    var evidence_task = evidence_dataset.stored_task.task.copy()
    var gold_task = gold_dataset.stored_task.task.copy()
    var evidence_index = evidence_dataset.stored_index.index.copy()
    var gold_index = gold_dataset.stored_index.index.copy()
    var evidence_document_texts = require_public_benchmark_dataset_loaded_text_corpus(
        evidence_dataset
    )
    var gold_document_texts = require_public_benchmark_dataset_loaded_text_corpus(
        gold_dataset
    )

    evaluate_reranked_task(
        "BrowseComp evidence clause-text rerank",
        evidence_task,
        evidence_index,
        evidence_document_texts,
        20,
    )
    evaluate_reranked_task(
        "BrowseComp gold clause-text rerank",
        gold_task,
        gold_index,
        gold_document_texts,
        20,
    )
