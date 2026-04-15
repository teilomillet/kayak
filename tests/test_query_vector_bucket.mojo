from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak.benchmarks import (
    QueryBucketStageAwareSearchSummary,
    QueryVectorBucket,
    StageAwareSearchSummary,
    StageDensitySummary,
    make_query_vector_bucket,
    non_empty_standard_query_vector_buckets,
    query_bucket_stage_aware_search_summary_json,
    query_vector_bucket_query_count,
    standard_query_vector_buckets,
    subset_stored_judged_task_queries_to_query_vector_bucket,
)
from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.eval import JudgedQuery, JudgedTask
from kayak.planning import (
    best_effort_faithfulness_policy,
    document_proxy_search_plan,
)
from kayak.storage import StoredJudgedTask


def encoded_query(vector_count: Int) raises -> EncodedQuery:
    var token_vectors = List[List[Float32]]()
    for vector_index in range(vector_count):
        token_vectors.append(
            [
                Float32(vector_index + 1),
                Float32(vector_index + 2),
            ]
        )
    return EncodedQuery(token_vectors^)


def document(doc_id: String) raises -> EncodedDocument:
    return EncodedDocument(
        doc_id,
        [[1.0, 0.0], [0.0, 1.0]],
    )


def sample_stored_task() raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "mock-dataset",
        "mock-model",
        "float32",
        JudgedTask(
            "beir",
            "mock_slice",
            "mock why",
            "ndcg@10",
            2,
            16,
            32,
            2,
            [document("doc-a"), document("doc-b")],
            [
                JudgedQuery(
                    "q-2",
                    "two vectors",
                    encoded_query(2),
                    ["doc-a"],
                ),
                JudgedQuery(
                    "q-6",
                    "six vectors",
                    encoded_query(6),
                    ["doc-b"],
                ),
                JudgedQuery(
                    "q-12",
                    "twelve vectors",
                    encoded_query(12),
                    ["doc-a"],
                ),
            ],
        ),
    )


def sample_stage_aware_summary() raises -> StageAwareSearchSummary:
    var density = StageDensitySummary(
        2,
        4,
        4,
        64,
        32.0,
        16.0,
    )
    return StageAwareSearchSummary(
        "mock-dataset",
        "mock-model",
        "beir",
        "mock_slice__qv_1_4",
        "mock_collection",
        "snapshot-0001",
        document_proxy_search_plan(
            2,
            4,
            best_effort_faithfulness_policy(),
        ),
        "ndcg@10",
        0.5,
        0.5,
        0.5,
        0.5,
        1.0,
        0.5,
        0.01,
        2,
        4,
        1,
        4,
        32,
        2,
        4,
        4,
        64,
        32.0,
        16.0,
        density,
        False,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        List[String](),
        density,
        List[String](),
        density,
        density,
        2,
    )


def test_standard_query_vector_buckets_cap_last_bucket() raises:
    var buckets = standard_query_vector_buckets(19)

    assert_equal(len(buckets), 4)
    assert_equal(buckets[0].key, "qv_1_4")
    assert_equal(buckets[1].key, "qv_5_8")
    assert_equal(buckets[2].key, "qv_9_16")
    assert_equal(buckets[3].key, "qv_17_19")


def test_subset_stored_task_query_bucket_slices_queries_without_copying_docs() raises:
    var stored_task = sample_stored_task()
    var bucket = make_query_vector_bucket(5, 8)

    assert_equal(query_vector_bucket_query_count(stored_task.task, bucket), 1)

    var bucketed = subset_stored_judged_task_queries_to_query_vector_bucket(
        stored_task,
        bucket,
    )

    assert_equal(len(bucketed.task.documents), 0)
    assert_equal(len(bucketed.task.queries), 1)
    assert_equal(bucketed.task.queries[0].query_id, "q-6")
    assert_equal(bucketed.task.nominal_query_vector_count, 6)
    assert_equal(bucketed.task.slice_name, "mock_slice__qv_5_8")


def test_non_empty_standard_query_vector_buckets_skips_empty_ranges() raises:
    var stored_task = sample_stored_task()
    var buckets = non_empty_standard_query_vector_buckets(stored_task.task)

    assert_equal(len(buckets), 3)
    assert_equal(buckets[0].key, "qv_1_4")
    assert_equal(buckets[1].key, "qv_5_8")
    assert_equal(buckets[2].key, "qv_9_12")


def test_query_bucket_stage_aware_json_contains_bucket_metadata() raises:
    var summary = QueryBucketStageAwareSearchSummary(
        QueryVectorBucket("qv_1_4", "1-4", 1, 4),
        sample_stage_aware_summary(),
    )
    var json = query_bucket_stage_aware_search_summary_json(summary)

    assert_equal(json.find("\"query_vector_bucket_key\":\"qv_1_4\"") != -1, True)
    assert_equal(json.find("\"query_vector_bucket_min\":1") != -1, True)
    assert_equal(json.find("\"measured\":") != -1, True)
    assert_equal(json.find("\"slice_name\":\"mock_slice__qv_1_4\"") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
