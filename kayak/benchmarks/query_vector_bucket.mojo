from std.collections import List

from kayak.contracts import EncodedDocument
from kayak.eval import JudgedQuery, JudgedTask
from kayak.storage import StoredJudgedTask


# Query-vector buckets are a benchmark-side slicing primitive.
# They let us inspect whether a plan remains strong as query fan-out grows
# without changing search semantics or storage layout.
struct QueryVectorBucket(Copyable):
    var key: String
    var label: String
    var min_vector_count: Int
    var max_vector_count: Int

    def __init__(
        out self,
        var key: String,
        var label: String,
        min_vector_count: Int,
        max_vector_count: Int,
    ) raises:
        if min_vector_count <= 0:
            raise Error("query vector bucket min_vector_count must be positive")
        if max_vector_count < min_vector_count:
            raise Error(
                "query vector bucket max_vector_count must be >= min_vector_count"
            )

        self.key = key^
        self.label = label^
        self.min_vector_count = min_vector_count
        self.max_vector_count = max_vector_count


def make_query_vector_bucket(
    min_vector_count: Int, max_vector_count: Int
) raises -> QueryVectorBucket:
    var key = (
        "qv_"
        + String(min_vector_count)
        + "_"
        + String(max_vector_count)
    )
    var label = String(min_vector_count)
    if min_vector_count != max_vector_count:
        label += "-" + String(max_vector_count)
    return QueryVectorBucket(
        key,
        label,
        min_vector_count,
        max_vector_count,
    )


def standard_query_vector_buckets(
    max_query_vector_count: Int
) raises -> List[QueryVectorBucket]:
    var buckets = List[QueryVectorBucket]()
    if max_query_vector_count <= 0:
        return buckets^

    var lower_bound = 1
    for raw_upper_bound in [4, 8, 16, 32, 64, 128, max_query_vector_count]:
        var upper_bound = raw_upper_bound
        if upper_bound > max_query_vector_count:
            upper_bound = max_query_vector_count
        if upper_bound < lower_bound:
            continue
        buckets.append(
            make_query_vector_bucket(lower_bound, upper_bound)
        )
        lower_bound = upper_bound + 1
        if lower_bound > max_query_vector_count:
            break

    return buckets^


def query_is_in_vector_bucket(
    read judged_query: JudgedQuery, read bucket: QueryVectorBucket
) -> Bool:
    var vector_count = judged_query.query.vector_count
    if vector_count < bucket.min_vector_count:
        return False
    if vector_count > bucket.max_vector_count:
        return False
    return True


def query_vector_bucket_query_count(
    read task: JudgedTask, read bucket: QueryVectorBucket
) -> Int:
    var matching_query_count = 0
    for judged_query in task.queries:
        if query_is_in_vector_bucket(judged_query, bucket):
            matching_query_count += 1
    return matching_query_count


def max_query_vector_count_in_bucket(
    read task: JudgedTask, read bucket: QueryVectorBucket
) -> Int:
    var max_vector_count = 0
    for judged_query in task.queries:
        if query_is_in_vector_bucket(judged_query, bucket):
            if judged_query.query.vector_count > max_vector_count:
                max_vector_count = judged_query.query.vector_count
    return max_vector_count


def non_empty_standard_query_vector_buckets(
    read task: JudgedTask
) raises -> List[QueryVectorBucket]:
    var buckets = List[QueryVectorBucket]()
    var max_query_vector_count = 0
    for judged_query in task.queries:
        if judged_query.query.vector_count > max_query_vector_count:
            max_query_vector_count = judged_query.query.vector_count

    for bucket in standard_query_vector_buckets(max_query_vector_count):
        if query_vector_bucket_query_count(task, bucket) != 0:
            buckets.append(bucket.copy())

    return buckets^


def subset_judged_task_queries_to_query_vector_bucket(
    read task: JudgedTask, read bucket: QueryVectorBucket
) raises -> JudgedTask:
    var bucket_queries = List[JudgedQuery]()
    for judged_query in task.queries:
        if query_is_in_vector_bucket(judged_query, bucket):
            bucket_queries.append(judged_query.copy())

    var bucket_nominal_query_vector_count = max_query_vector_count_in_bucket(
        task,
        bucket,
    )
    if bucket_nominal_query_vector_count == 0:
        raise Error(
            "query vector bucket "
            + bucket.key
            + " does not match any judged queries"
        )

    # Stage-aware public benchmarks search against the mirrored snapshot rather
    # than task.documents, so keeping documents empty avoids O(bucket_count)
    # deep copies of the full encoded corpus when slicing only by queries.
    return JudgedTask(
        task.family.copy(),
        task.slice_name + "__" + bucket.key,
        task.why
        + " Restricted to queries with vector_count in ["
        + String(bucket.min_vector_count)
        + ", "
        + String(bucket.max_vector_count)
        + "].",
        task.primary_metric.copy(),
        task.k,
        bucket_nominal_query_vector_count,
        task.nominal_document_vector_count,
        task.vector_dim,
        List[EncodedDocument](),
        bucket_queries^,
    )


def subset_stored_judged_task_queries_to_query_vector_bucket(
    read stored_task: StoredJudgedTask, read bucket: QueryVectorBucket
) raises -> StoredJudgedTask:
    return StoredJudgedTask(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        subset_judged_task_queries_to_query_vector_bucket(
            stored_task.task,
            bucket,
        ),
    )
