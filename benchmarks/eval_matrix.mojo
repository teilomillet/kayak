from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import default_proxy_tasks
from kayak.benchmarks import (
    ProxyTaskEvaluationSummary,
    build_proxy_task_evaluation_summary,
    proxy_task_evaluation_summaries_json,
)
from kayak.eval import TaskEvaluation, evaluate_task
from kayak.runtime import ExactCpuBackend


def print_result(result: TaskEvaluation):
    print("== ", result.family, ".", result.slice_name, " ==")
    print("primary: ", result.primary_metric, "@", result.k, " = ", result.primary_value)
    print("queries: ", result.query_count, " documents: ", result.document_count)
    print("ndcg@", result.k, " = ", result.mean_ndcg_at_k)
    print("mrr@", result.k, " = ", result.mean_reciprocal_rank)
    print("recall@", result.k, " = ", result.mean_recall_at_k)
    print("success@", result.k, " = ", result.success_rate_at_k)
    print("")


def main() raises:
    var backend = ExactCpuBackend()
    var summaries = List[ProxyTaskEvaluationSummary]()

    for task in default_proxy_tasks():
        var result = evaluate_task(backend, task)
        print_result(result)
        summaries.append(build_proxy_task_evaluation_summary(result))

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "proxy_eval_matrix.json"
    output_path.write_text(proxy_task_evaluation_summaries_json(summaries))
    print("wrote ", String(output_path))
