from kayak.numeric import MetricScalar


def choose_primary_value(
    primary_metric: String,
    ndcg_at_k: MetricScalar,
    reciprocal_rank_at_k: MetricScalar,
    recall_at_k: MetricScalar,
    success_at_k: MetricScalar,
) raises -> MetricScalar:
    if primary_metric == "ndcg":
        return ndcg_at_k

    if primary_metric == "mrr":
        return reciprocal_rank_at_k

    if primary_metric == "recall":
        return recall_at_k

    if primary_metric == "success":
        return success_at_k

    raise Error("unknown primary metric: " + primary_metric)
