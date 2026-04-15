import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.runtime.asyncrt import parallelism_level

from kayak import (
    CollectionHit,
    CollectionId,
    ExactScoringConfig,
    NamespaceId,
    PreparedExactSearchBatchConfig,
    SearchRequest,
    SearchResponse,
    SnapshotId,
    TenantId,
    default_exact_search_request,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    exact_cpu_backend_for_scoring_config,
    execute_search_batch_with_prepared_snapshot,
    execute_search_with_prepared_snapshot,
    PreparedSearchSnapshot,
    prepare_service_search_snapshot,
)
from kayak.eval import JudgedTask
from kayak.service.paths import service_collection_root

# Keep the request pool large enough to expose sustained outer-request
# concurrency, but bounded so quiet-wrapper reruns remain practical.
comptime REQUEST_POOL_COUNT = 32
comptime BATCH_MAX_ITERS = 64


struct BatchMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var scoring_mode: String
    var worker_count: Int
    var request_pool_count: Int
    var distinct_query_count: Int
    var document_count: Int
    var query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int
    var enable_parallel_scoring: Bool
    var enable_parallel_work_item_oversubscription: Bool
    var parallel_work_item_count_override: Int
    var mean_batch_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var scoring_mode: String,
        worker_count: Int,
        request_pool_count: Int,
        distinct_query_count: Int,
        document_count: Int,
        query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
        enable_parallel_scoring: Bool,
        enable_parallel_work_item_oversubscription: Bool,
        parallel_work_item_count_override: Int,
        mean_batch_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.scoring_mode = scoring_mode^
        self.worker_count = worker_count
        self.request_pool_count = request_pool_count
        self.distinct_query_count = distinct_query_count
        self.document_count = document_count
        self.query_vector_count = query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.enable_parallel_scoring = enable_parallel_scoring
        self.enable_parallel_work_item_oversubscription = (
            enable_parallel_work_item_oversubscription
        )
        self.parallel_work_item_count_override = (
            parallel_work_item_count_override
        )
        self.mean_batch_seconds = mean_batch_seconds


def append_measurement_tsv_line(mut out: String, read measurement: BatchMeasurement):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.scoring_mode
    out += "\t"
    out += String(measurement.worker_count)
    out += "\t"
    out += String(measurement.request_pool_count)
    out += "\t"
    out += String(measurement.distinct_query_count)
    out += "\t"
    out += String(measurement.document_count)
    out += "\t"
    out += String(measurement.query_vector_count)
    out += "\t"
    out += String(measurement.nominal_document_vector_count)
    out += "\t"
    out += String(measurement.vector_dim)
    out += "\t"
    out += String(measurement.enable_parallel_scoring)
    out += "\t"
    out += String(measurement.enable_parallel_work_item_oversubscription)
    out += "\t"
    out += String(measurement.parallel_work_item_count_override)
    out += "\t"
    out += String(measurement.mean_batch_seconds)
    out += "\t"
    out += String(
        Float64(measurement.request_pool_count) / measurement.mean_batch_seconds
    )
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[BatchMeasurement]
) raises:
    var lines = String()
    lines += "dataset_name\tslice_name\tscoring_mode\tworker_count\trequest_pool_count\tdistinct_query_count\tdocument_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\tenable_parallel_scoring\tenable_parallel_work_item_oversubscription\tparallel_work_item_count_override\tmean_batch_seconds\tthroughput_queries_per_second\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def profile_service_root() -> Path:
    return Path(
        ".cache/kayak/profile_prepared_snapshot_parallel_exact_batch_browsecomp_gold"
    )


def ensure_profile_service_collection() raises -> Path:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var service_root = profile_service_root()
    var collection_root = service_collection_root(
        service_root,
        TenantId("public"),
        NamespaceId("benchmark"),
        CollectionId("browsecomp-plus-gold-service"),
    )
    makedirs(service_root, exist_ok=True)
    _ = ensure_one_segment_collection_mirror(
        collection_root,
        CollectionId("browsecomp-plus-gold-service"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        cache.stored_index,
    )
    return service_root


def build_exact_requests(
    read task: JudgedTask, model_name: String
) raises -> List[SearchRequest]:
    var requests = List[SearchRequest]()
    for judged_query in task.queries:
        requests.append(
            default_exact_search_request(
                CollectionId("browsecomp-plus-gold-service"),
                TenantId("public"),
                NamespaceId("benchmark"),
                SnapshotId("snapshot-0001"),
                judged_query.query,
                task.k,
                model_name,
            )
        )

    return requests^


def build_request_pool(
    read base_requests: List[SearchRequest], request_pool_count: Int
) raises -> List[SearchRequest]:
    var requests = List[SearchRequest]()
    requests.reserve(request_pool_count)

    for request_index in range(request_pool_count):
        requests.append(base_requests[request_index % len(base_requests)].copy())

    return requests^


def make_default_scoring_config() -> ExactScoringConfig:
    return ExactScoringConfig()


def make_serial_scoring_config() -> ExactScoringConfig:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return config^


def make_fixed2_scoring_config() -> ExactScoringConfig:
    var config = ExactScoringConfig()
    config.enable_parallel_work_item_oversubscription = False
    config.parallel_work_item_count_override = 2
    return config^


def make_shared_core_budget_scoring_config(worker_count: Int) -> ExactScoringConfig:
    var config = ExactScoringConfig()
    config.enable_parallel_work_item_oversubscription = False
    var budget = parallelism_level() // worker_count
    if budget < 1:
        budget = 1
    config.parallel_work_item_count_override = budget
    return config^


def scoring_config_for_mode(
    scoring_mode: String, worker_count: Int
) raises -> ExactScoringConfig:
    if scoring_mode == "default_auto":
        return make_default_scoring_config()
    if scoring_mode == "serial_inner":
        return make_serial_scoring_config()
    if scoring_mode == "fixed2_inner":
        return make_fixed2_scoring_config()
    if scoring_mode == "shared_core_budget":
        return make_shared_core_budget_scoring_config(worker_count)

    raise Error("unknown scoring mode: " + scoring_mode)


def scoring_modes() -> List[String]:
    return [
        "default_auto",
        "serial_inner",
        "fixed2_inner",
        "shared_core_budget",
    ]


def worker_counts() -> List[Int]:
    var max_workers = parallelism_level()
    if max_workers < 1:
        max_workers = 1
    if max_workers > REQUEST_POOL_COUNT:
        max_workers = REQUEST_POOL_COUNT

    var counts = List[Int]()
    counts.append(1)
    if max_workers >= 2:
        counts.append(2)
    if max_workers >= 4:
        counts.append(4)
    if max_workers >= 8:
        counts.append(8)
    if counts[len(counts) - 1] != max_workers:
        counts.append(max_workers)

    return counts^


def assert_search_responses_equal(
    scoring_mode: String,
    worker_count: Int,
    read expected: List[SearchResponse],
    read observed: List[SearchResponse],
) raises:
    if len(expected) != len(observed):
        raise Error(
            scoring_mode
            + " worker_count="
            + String(worker_count)
            + " response count mismatch"
        )

    for response_index in range(len(expected)):
        if (
            expected[response_index].snapshot_id.value
            != observed[response_index].snapshot_id.value
        ):
            raise Error(
                scoring_mode
                + " worker_count="
                + String(worker_count)
                + " snapshot mismatch at response "
                + String(response_index)
            )

        if len(expected[response_index].hits) != len(observed[response_index].hits):
            raise Error(
                scoring_mode
                + " worker_count="
                + String(worker_count)
                + " hit count mismatch at response "
                + String(response_index)
            )

        for hit_index in range(len(expected[response_index].hits)):
            var expected_hit = expected[response_index].hits[hit_index].copy()
            var observed_hit = observed[response_index].hits[hit_index].copy()
            if expected_hit.doc_id != observed_hit.doc_id:
                raise Error(
                    scoring_mode
                    + " worker_count="
                    + String(worker_count)
                    + " doc_id mismatch at response "
                    + String(response_index)
                    + " hit "
                    + String(hit_index)
                )

            if expected_hit.score != observed_hit.score:
                raise Error(
                    scoring_mode
                    + " worker_count="
                    + String(worker_count)
                    + " score mismatch at response "
                    + String(response_index)
                    + " hit "
                    + String(hit_index)
                )


def validate_batch_matches_serial(
    scoring_mode: String,
    worker_count: Int,
    read prepared: PreparedSearchSnapshot,
    read requests: List[SearchRequest],
    read scoring_config: ExactScoringConfig,
) raises:
    var backend = exact_cpu_backend_for_scoring_config(scoring_config)
    var expected = List[SearchResponse]()
    for request in requests:
        expected.append(
            execute_search_with_prepared_snapshot(backend, prepared, request)
        )

    var observed = execute_search_batch_with_prepared_snapshot(
        prepared,
        requests,
        PreparedExactSearchBatchConfig(worker_count, scoring_config),
    )
    assert_search_responses_equal(scoring_mode, worker_count, expected, observed)


def benchmark_batch_mean_seconds(
    read prepared: PreparedSearchSnapshot,
    read requests: List[SearchRequest],
    worker_count: Int,
    scoring_mode: String,
    read scoring_config: ExactScoringConfig,
) raises -> Float64:
    var batch_config = PreparedExactSearchBatchConfig(worker_count, scoring_config)

    print(
        "mode=",
        scoring_mode,
        " worker_count=",
        worker_count,
        " override=",
        batch_config.scoring_config.parallel_work_item_count_override,
        " parallel=",
        batch_config.scoring_config.enable_parallel_scoring,
        " oversub=",
        batch_config.scoring_config.enable_parallel_work_item_oversubscription,
    )

    def search_once() capturing raises:
        bench_compiler.keep(
            execute_search_batch_with_prepared_snapshot(
                prepared,
                requests,
                batch_config,
            )
        )

    var report = benchmark.run[search_once](max_iters=BATCH_MAX_ITERS)
    report.print()
    print("")
    return report.mean()


def main() raises:
    print("loading BrowseComp+ gold real-subset cache...")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var service_root = ensure_profile_service_collection()
    var base_requests = build_exact_requests(task, cache.stored_task.model_name)
    var request_pool = build_request_pool(base_requests, REQUEST_POOL_COUNT)
    var prepared = prepare_service_search_snapshot(
        service_root,
        CollectionId("browsecomp-plus-gold-service"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        False,
    )
    var measurements = List[BatchMeasurement]()
    var concurrency = parallelism_level()
    if concurrency < 1:
        concurrency = 1

    print("dataset: ", cache.stored_task.dataset_id)
    print("slice: ", task.slice_name)
    print("distinct queries: ", len(base_requests))
    print("request pool: ", len(request_pool))
    print("documents: ", len(task.documents))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)
    print("parallelism_level: ", concurrency)
    print("service_root: ", String(service_root))
    print("")

    for worker_count in worker_counts():
        for scoring_mode in scoring_modes():
            var scoring_config = scoring_config_for_mode(
                scoring_mode,
                worker_count,
            )
            validate_batch_matches_serial(
                scoring_mode,
                worker_count,
                prepared,
                request_pool,
                scoring_config,
            )

            var mean_batch_seconds = benchmark_batch_mean_seconds(
                prepared,
                request_pool,
                worker_count,
                scoring_mode,
                scoring_config,
            )

            measurements.append(
                BatchMeasurement(
                    cache.stored_task.dataset_id.copy(),
                    task.slice_name.copy(),
                    scoring_mode,
                    worker_count,
                    len(request_pool),
                    len(base_requests),
                    len(task.documents),
                    task.nominal_query_vector_count,
                    task.nominal_document_vector_count,
                    task.vector_dim,
                    scoring_config.enable_parallel_scoring,
                    scoring_config.enable_parallel_work_item_oversubscription,
                    scoring_config.parallel_work_item_count_override,
                    mean_batch_seconds,
                )
            )

    var out_path = Path(
        ".cache/kayak/prepared_snapshot_parallel_exact_batch_browsecomp_gold.tsv"
    )
    write_measurements_tsv(out_path, measurements)
    print("wrote ", String(out_path))
