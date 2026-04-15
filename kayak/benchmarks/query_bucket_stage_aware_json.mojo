from std.collections import List

from .json_common import json_escape
from .query_vector_bucket import QueryVectorBucket
from .stage_aware_json import (
    StageAwareSearchSummary,
    stage_aware_search_summary_json,
)


struct QueryBucketStageAwareSearchSummary(Copyable):
    var query_vector_bucket: QueryVectorBucket
    var measured: StageAwareSearchSummary

    def __init__(
        out self,
        read query_vector_bucket: QueryVectorBucket,
        read measured: StageAwareSearchSummary,
    ):
        self.query_vector_bucket = query_vector_bucket.copy()
        self.measured = measured.copy()


def build_query_bucket_stage_aware_search_summary(
    read query_vector_bucket: QueryVectorBucket,
    read measured: StageAwareSearchSummary,
) -> QueryBucketStageAwareSearchSummary:
    return QueryBucketStageAwareSearchSummary(
        query_vector_bucket,
        measured,
    )


def append_query_bucket_stage_aware_search_summary_json(
    mut buffer: String,
    read summary: QueryBucketStageAwareSearchSummary,
):
    buffer += "{"
    buffer += "\"query_vector_bucket_key\":\""
    buffer += json_escape(summary.query_vector_bucket.key) + "\","
    buffer += "\"query_vector_bucket_label\":\""
    buffer += json_escape(summary.query_vector_bucket.label) + "\","
    buffer += "\"query_vector_bucket_min\":"
    buffer += String(summary.query_vector_bucket.min_vector_count) + ","
    buffer += "\"query_vector_bucket_max\":"
    buffer += String(summary.query_vector_bucket.max_vector_count) + ","
    buffer += "\"measured\":"
    buffer += stage_aware_search_summary_json(summary.measured)
    buffer += "}"


def query_bucket_stage_aware_search_summary_json(
    read summary: QueryBucketStageAwareSearchSummary
) -> String:
    var buffer = String()
    append_query_bucket_stage_aware_search_summary_json(buffer, summary)
    return buffer^


def query_bucket_stage_aware_search_summaries_json(
    read summaries: List[QueryBucketStageAwareSearchSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_query_bucket_stage_aware_search_summary_json(
            buffer,
            summaries[index],
        )

    buffer += "]"
    return buffer^
