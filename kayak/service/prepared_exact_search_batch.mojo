# Explicit batch execution for repeated exact search on one prepared snapshot.
#
# This module owns outer-request parallel batch execution for a pinned published
# snapshot. It does not own transport queues, request admission, or hidden
# snapshot cache policy.

from std.algorithm.backend.cpu.parallelize import sync_parallelize
from std.collections import List

from kayak.planning import CollectionHit
from kayak.scoring import ExactScoringConfig

from .prepared_exact_search_executor import (
    exact_cpu_backend_for_scoring_config,
)
from .prepared_snapshot_runtime import (
    PreparedSearchSnapshot,
    execute_search_with_prepared_snapshot,
)
from .search_contracts import SearchRequest, SearchResponse


struct PreparedExactSearchBatchConfig(Copyable):
    var worker_count: Int
    var scoring_config: ExactScoringConfig

    def __init__(out self, worker_count: Int) raises:
        self = PreparedExactSearchBatchConfig(
            worker_count,
            ExactScoringConfig(),
        )

    def __init__(
        out self,
        worker_count: Int,
        read scoring_config: ExactScoringConfig,
    ) raises:
        if worker_count < 1:
            raise Error("prepared exact batch worker_count must be positive")

        self.worker_count = worker_count
        self.scoring_config = scoring_config.copy()


def effective_worker_count(
    request_count: Int, requested_worker_count: Int
) raises -> Int:
    if requested_worker_count < 1:
        raise Error("prepared exact batch worker_count must be positive")

    if request_count < 1:
        return 0

    if requested_worker_count > request_count:
        return request_count

    return requested_worker_count


def build_request_boundaries(
    request_count: Int, worker_count: Int
) raises -> List[Int]:
    var boundaries = List[Int]()
    boundaries.reserve(worker_count + 1)

    for work_item in range(worker_count):
        boundaries.append((request_count * work_item) // worker_count)

    boundaries.append(request_count)
    return boundaries^


def make_placeholder_search_responses(
    read requests: List[SearchRequest]
) raises -> List[SearchResponse]:
    var responses = List[SearchResponse]()
    responses.reserve(len(requests))

    for request in requests:
        responses.append(
            SearchResponse(
                request.collection_id,
                request.tenant_id,
                request.namespace_id,
                request.snapshot_id,
                request.plan,
                List[CollectionHit](),
            )
        )

    return responses^


def execute_search_batch_with_prepared_snapshot(
    read prepared: PreparedSearchSnapshot,
    read requests: List[SearchRequest],
    read batch_config: PreparedExactSearchBatchConfig,
) raises -> List[SearchResponse]:
    var worker_count = effective_worker_count(
        len(requests),
        batch_config.worker_count,
    )
    if worker_count == 0:
        return List[SearchResponse]()

    var responses = make_placeholder_search_responses(requests)
    var responses_ptr = responses.unsafe_ptr()

    if worker_count <= 1:
        var backend = exact_cpu_backend_for_scoring_config(
            batch_config.scoring_config
        )
        for request_index in range(len(requests)):
            responses_ptr[request_index] = execute_search_with_prepared_snapshot(
                backend,
                prepared,
                requests[request_index],
            )

        return responses^

    var boundaries = build_request_boundaries(len(requests), worker_count)
    var scoring_config = batch_config.scoring_config.copy()

    @parameter
    def execute_partition(work_item: Int) raises:
        var backend = exact_cpu_backend_for_scoring_config(scoring_config)
        var start_request = boundaries[work_item]
        var stop_request = boundaries[work_item + 1]

        for request_index in range(start_request, stop_request):
            responses_ptr[request_index] = execute_search_with_prepared_snapshot(
                backend,
                prepared,
                requests[request_index],
            )

    sync_parallelize[execute_partition](worker_count)
    return responses^


def execute_search_batch_with_prepared_snapshot(
    read prepared: PreparedSearchSnapshot,
    read requests: List[SearchRequest],
    worker_count: Int,
) raises -> List[SearchResponse]:
    return execute_search_batch_with_prepared_snapshot(
        prepared,
        requests,
        PreparedExactSearchBatchConfig(worker_count),
    )
