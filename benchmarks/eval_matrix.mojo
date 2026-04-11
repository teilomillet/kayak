from kayak.benchmarks import default_proxy_tasks
from kayak.eval import TaskEvaluation, evaluate_task
from kayak.runtime import ExactCpuBackend


def print_result(result: TaskEvaluation):
    print("== ", result.family, ".", result.slice_name, " ==")
    print("primary: ", result.primary_metric, "@", result.k, " = ", result.primary_value)
    print("queries: ", result.query_count, " documents: ", result.document_count)
    print("mrr@", result.k, " = ", result.mean_reciprocal_rank)
    print("recall@", result.k, " = ", result.mean_recall_at_k)
    print("success@", result.k, " = ", result.success_rate_at_k)
    print("")


def main() raises:
    var backend = ExactCpuBackend()

    for task in default_proxy_tasks():
        print_result(evaluate_task(backend, task))
