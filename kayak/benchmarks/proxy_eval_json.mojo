from std.collections import List

from kayak.eval import TaskEvaluation

from .json_common import json_escape


struct ProxyTaskEvaluationSummary(Copyable):
    var family: String
    var slice_name: String
    var primary_metric: String
    var primary_value: Float64
    var k: Int
    var query_count: Int
    var document_count: Int
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64

    def __init__(
        out self,
        var family: String,
        var slice_name: String,
        var primary_metric: String,
        primary_value: Float64,
        k: Int,
        query_count: Int,
        document_count: Int,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
    ):
        self.family = family^
        self.slice_name = slice_name^
        self.primary_metric = primary_metric^
        self.primary_value = primary_value
        self.k = k
        self.query_count = query_count
        self.document_count = document_count
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k


def build_proxy_task_evaluation_summary(
    read evaluation: TaskEvaluation
) -> ProxyTaskEvaluationSummary:
    return ProxyTaskEvaluationSummary(
        evaluation.family.copy(),
        evaluation.slice_name.copy(),
        evaluation.primary_metric.copy(),
        Float64(evaluation.primary_value),
        evaluation.k,
        evaluation.query_count,
        evaluation.document_count,
        Float64(evaluation.mean_ndcg_at_k),
        Float64(evaluation.mean_reciprocal_rank),
        Float64(evaluation.mean_recall_at_k),
        Float64(evaluation.success_rate_at_k),
    )


def append_proxy_task_evaluation_summary_json(
    mut buffer: String, read summary: ProxyTaskEvaluationSummary
):
    buffer += "{"
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"primary_metric\":\"" + json_escape(summary.primary_metric) + "\","
    buffer += "\"primary_value\":" + String(summary.primary_value) + ","
    buffer += "\"k\":" + String(summary.k) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":"
    buffer += String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k)
    buffer += "}"


def proxy_task_evaluation_summaries_json(
    read summaries: List[ProxyTaskEvaluationSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","

        append_proxy_task_evaluation_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
