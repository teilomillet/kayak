import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.runtime.asyncrt import parallelism_level

from kayak import (
    CollectionId,
    ExactCpuBackend,
    ExactScoringConfig,
    NamespaceId,
    PreparedExactSearchBatchConfig,
    PreparedSearchSnapshot,
    SearchRequest,
    SearchResponse,
    SnapshotId,
    TenantId,
    best_effort_faithfulness_policy,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_imputed_search_plan,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    exact_cpu_backend_for_scoring_config,
    execute_search_batch_with_prepared_snapshot,
    execute_search_with_prepared_snapshot,
    match_all_filter,
    prepare_service_search_snapshot,
)
from kayak.eval import JudgedTask
from kayak.service.paths import service_collection_root


comptime REQUEST_POOL_COUNT = 32
comptime PREPARE_MAX_ITERS = 128
comptime PREPARED_SEARCH_MAX_ITERS = 256
comptime BATCH_MAX_ITERS = 64
comptime CENTROID_HEAD_POSTING_CAP = 16


struct PreparedCentroidMirrorMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var candidate_generator_kind: String
    var benchmark_kind: String
    var worker_count: Int
    var request_pool_count: Int
    var distinct_query_count: Int
    var document_count: Int
    var query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var candidate_generator_kind: String,
        var benchmark_kind: String,
        worker_count: Int,
        request_pool_count: Int,
        distinct_query_count: Int,
        document_count: Int,
        query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
        mean_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.candidate_generator_kind = candidate_generator_kind^
        self.benchmark_kind = benchmark_kind^
        self.worker_count = worker_count
        self.request_pool_count = request_pool_count
        self.distinct_query_count = distinct_query_count
        self.document_count = document_count
        self.query_vector_count = query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(
    mut out: String, read measurement: PreparedCentroidMirrorMeasurement
):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.candidate_generator_kind
    out += "\t"
    out += measurement.benchmark_kind
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
    out += String(measurement.mean_seconds)
    if measurement.request_pool_count > 0:
        out += "\t"
        out += String(
            Float64(measurement.request_pool_count) / measurement.mean_seconds
        )
    else:
        out += "\t"
        out += String(Float64(1.0) / measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[PreparedCentroidMirrorMeasurement]
) raises:
    var lines = String()
    lines += "dataset_name\tslice_name\tcandidate_generator_kind\tbenchmark_kind\tworker_count\trequest_pool_count\tdistinct_query_count\tdocument_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\tmean_seconds\tthroughput_queries_per_second\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def stage1_candidate_k(task: JudgedTask) -> Int:
    var candidate_k = task.k * 4
    if candidate_k > len(task.documents):
        return len(task.documents)

    return candidate_k


def profile_service_root() -> Path:
    return Path(
        ".cache/kayak/profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold"
    )


def ensure_profile_service_collection() raises -> Path:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var service_root = profile_service_root()
    var collection_root = service_collection_root(
        service_root,
        TenantId("public"),
        NamespaceId("benchmark"),
        CollectionId("browsecomp-plus-gold-centroid-service"),
    )
    makedirs(service_root, exist_ok=True)
    _ = ensure_one_segment_collection_mirror(
        collection_root,
        CollectionId("browsecomp-plus-gold-centroid-service"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        cache.stored_index,
        0,
        0,
        CENTROID_HEAD_POSTING_CAP,
    )
    return service_root


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


def build_requests(
    read task: JudgedTask,
    model_name: String,
    candidate_generator_kind: String,
) raises -> List[SearchRequest]:
    var requests = List[SearchRequest]()
    var candidate_k = stage1_candidate_k(task)

    for judged_query in task.queries:
        if candidate_generator_kind == "centroid_postings_imputed":
            requests.append(
                SearchRequest(
                    CollectionId("browsecomp-plus-gold-centroid-service"),
                    TenantId("public"),
                    NamespaceId("benchmark"),
                    SnapshotId("snapshot-0001"),
                    judged_query.query,
                    model_name,
                    match_all_filter(),
                    centroid_postings_imputed_search_plan(
                        task.k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                    False,
                )
            )
            continue

        if candidate_generator_kind == "centroid_postings_imputed_flat":
            requests.append(
                SearchRequest(
                    CollectionId("browsecomp-plus-gold-centroid-service"),
                    TenantId("public"),
                    NamespaceId("benchmark"),
                    SnapshotId("snapshot-0001"),
                    judged_query.query,
                    model_name,
                    match_all_filter(),
                    centroid_postings_imputed_flat_search_plan(
                        task.k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                    False,
                )
            )
            continue

        raise Error(
            "unsupported candidate_generator_kind: " + candidate_generator_kind
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


def prepare_snapshot(
    service_root: Path,
    load_dim128_segment_mirrors: Bool,
) raises -> PreparedSearchSnapshot:
    return prepare_service_search_snapshot(
        service_root,
        CollectionId("browsecomp-plus-gold-centroid-service"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        False,
        load_dim128_segment_mirrors,
    )


def assert_search_responses_equal(
    context: String,
    read expected: List[SearchResponse],
    read observed: List[SearchResponse],
) raises:
    if len(expected) != len(observed):
        raise Error(context + " response count mismatch")

    for response_index in range(len(expected)):
        if (
            expected[response_index].snapshot_id.value
            != observed[response_index].snapshot_id.value
        ):
            raise Error(
                context + " snapshot mismatch at response " + String(response_index)
            )

        if len(expected[response_index].hits) != len(observed[response_index].hits):
            raise Error(
                context + " hit-count mismatch at response " + String(response_index)
            )

        for hit_index in range(len(expected[response_index].hits)):
            if (
                expected[response_index].hits[hit_index].doc_id
                != observed[response_index].hits[hit_index].doc_id
            ):
                raise Error(
                    context
                    + " doc_id mismatch at response "
                    + String(response_index)
                    + " hit "
                    + String(hit_index)
                )

            if (
                expected[response_index].hits[hit_index].score
                != observed[response_index].hits[hit_index].score
            ):
                raise Error(
                    context
                    + " score mismatch at response "
                    + String(response_index)
                    + " hit "
                    + String(hit_index)
                )


def validate_mirror_matches_baseline(
    candidate_generator_kind: String,
    read base_prepared: PreparedSearchSnapshot,
    read mirrored_prepared: PreparedSearchSnapshot,
    read requests: List[SearchRequest],
    read scoring_config: ExactScoringConfig,
) raises:
    var backend = exact_cpu_backend_for_scoring_config(scoring_config)
    var expected = List[SearchResponse]()
    var observed = List[SearchResponse]()

    for request in requests:
        expected.append(
            execute_search_with_prepared_snapshot(backend, base_prepared, request)
        )
        observed.append(
            execute_search_with_prepared_snapshot(
                backend,
                mirrored_prepared,
                request,
            )
        )

    assert_search_responses_equal(
        candidate_generator_kind + " mirrored_vs_baseline",
        expected,
        observed,
    )


def validate_batch_matches_serial(
    candidate_generator_kind: String,
    mirror_mode: String,
    read prepared: PreparedSearchSnapshot,
    read requests: List[SearchRequest],
    worker_count: Int,
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
    assert_search_responses_equal(
        candidate_generator_kind
            + " "
            + mirror_mode
            + " worker_count="
            + String(worker_count),
        expected,
        observed,
    )


def benchmark_prepare_snapshot_mean_seconds(
    service_root: Path,
    load_dim128_segment_mirrors: Bool,
) raises -> Float64:
    def prepare_once() capturing raises:
        var prepared = prepare_snapshot(service_root, load_dim128_segment_mirrors)
        bench_compiler.keep(len(prepared.snapshot.segments))

    var report = benchmark.run[prepare_once](max_iters=PREPARE_MAX_ITERS)
    report.print()
    print("")
    return report.mean()


def benchmark_direct_mean_seconds(
    read backend: ExactCpuBackend,
    read prepared: PreparedSearchSnapshot,
    read requests: List[SearchRequest],
) raises -> Float64:
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(
            execute_search_with_prepared_snapshot(
                backend,
                prepared,
                requests[query_index],
            )
        )
        query_index += 1
        if query_index == len(requests):
            query_index = 0

    var report = benchmark.run[search_once](max_iters=PREPARED_SEARCH_MAX_ITERS)
    report.print()
    print("")
    return report.mean()


def benchmark_batch_mean_seconds(
    read prepared: PreparedSearchSnapshot,
    read requests: List[SearchRequest],
    worker_count: Int,
    read scoring_config: ExactScoringConfig,
) raises -> Float64:
    var batch_config = PreparedExactSearchBatchConfig(worker_count, scoring_config)

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


def append_measurement(
    mut measurements: List[PreparedCentroidMirrorMeasurement],
    dataset_name: String,
    slice_name: String,
    candidate_generator_kind: String,
    benchmark_kind: String,
    worker_count: Int,
    request_pool_count: Int,
    distinct_query_count: Int,
    document_count: Int,
    query_vector_count: Int,
    nominal_document_vector_count: Int,
    vector_dim: Int,
    mean_seconds: Float64,
):
    measurements.append(
        PreparedCentroidMirrorMeasurement(
            dataset_name,
            slice_name,
            candidate_generator_kind,
            benchmark_kind,
            worker_count,
            request_pool_count,
            distinct_query_count,
            document_count,
            query_vector_count,
            nominal_document_vector_count,
            vector_dim,
            mean_seconds,
        )
    )


def main() raises:
    print("loading BrowseComp+ gold real-subset cache...")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var service_root = ensure_profile_service_collection()
    var scoring_config = ExactScoringConfig()
    var backend = ExactCpuBackend()
    var measurements = List[PreparedCentroidMirrorMeasurement]()
    var candidate_generator_kinds = [
        "centroid_postings_imputed",
        "centroid_postings_imputed_flat",
    ]

    print("dataset: ", cache.stored_task.dataset_id)
    print("slice: ", task.slice_name)
    print("documents: ", len(task.documents))
    print("distinct queries: ", len(task.queries))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)
    print("parallelism_level: ", parallelism_level())
    print("service_root: ", String(service_root))
    print("")

    print("== prepare prepared snapshot without mirrors ==")
    append_measurement(
        measurements,
        cache.stored_task.dataset_id.copy(),
        task.slice_name.copy(),
        "shared_prepared_snapshot",
        "prepare_snapshot",
        0,
        0,
        len(task.queries),
        len(task.documents),
        task.nominal_query_vector_count,
        task.nominal_document_vector_count,
        task.vector_dim,
        benchmark_prepare_snapshot_mean_seconds(service_root, False),
    )

    print("== prepare prepared snapshot with mirrors ==")
    append_measurement(
        measurements,
        cache.stored_task.dataset_id.copy(),
        task.slice_name.copy(),
        "shared_prepared_snapshot",
        "prepare_snapshot_with_mirrors",
        0,
        0,
        len(task.queries),
        len(task.documents),
        task.nominal_query_vector_count,
        task.nominal_document_vector_count,
        task.vector_dim,
        benchmark_prepare_snapshot_mean_seconds(service_root, True),
    )

    for candidate_generator_kind in candidate_generator_kinds:
        var base_requests = build_requests(
            task,
            cache.stored_task.model_name,
            candidate_generator_kind,
        )
        var request_pool = build_request_pool(base_requests, REQUEST_POOL_COUNT)
        var base_prepared = prepare_snapshot(service_root, False)
        var mirrored_prepared = prepare_snapshot(service_root, True)

        validate_mirror_matches_baseline(
            candidate_generator_kind,
            base_prepared,
            mirrored_prepared,
            base_requests,
            scoring_config,
        )

        print("== ", candidate_generator_kind, " direct prepared search ==")
        append_measurement(
            measurements,
            cache.stored_task.dataset_id.copy(),
            task.slice_name.copy(),
            candidate_generator_kind.copy(),
            "direct_prepared_search",
            1,
            1,
            len(base_requests),
            len(task.documents),
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            benchmark_direct_mean_seconds(backend, base_prepared, request_pool),
        )

        print("== ", candidate_generator_kind, " direct prepared search with mirrors ==")
        append_measurement(
            measurements,
            cache.stored_task.dataset_id.copy(),
            task.slice_name.copy(),
            candidate_generator_kind.copy(),
            "direct_prepared_search_with_mirrors",
            1,
            1,
            len(base_requests),
            len(task.documents),
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            benchmark_direct_mean_seconds(backend, mirrored_prepared, request_pool),
        )

        for worker_count in worker_counts():
            validate_batch_matches_serial(
                candidate_generator_kind,
                "baseline",
                base_prepared,
                request_pool,
                worker_count,
                scoring_config,
            )
            validate_batch_matches_serial(
                candidate_generator_kind,
                "mirrors",
                mirrored_prepared,
                request_pool,
                worker_count,
                scoring_config,
            )

            print(
                "== ",
                candidate_generator_kind,
                " batch prepared search worker_count=",
                worker_count,
                " ==",
            )
            append_measurement(
                measurements,
                cache.stored_task.dataset_id.copy(),
                task.slice_name.copy(),
                candidate_generator_kind.copy(),
                "batch_prepared_search",
                worker_count,
                len(request_pool),
                len(base_requests),
                len(task.documents),
                task.nominal_query_vector_count,
                task.nominal_document_vector_count,
                task.vector_dim,
                benchmark_batch_mean_seconds(
                    base_prepared,
                    request_pool,
                    worker_count,
                    scoring_config,
                ),
            )

            print(
                "== ",
                candidate_generator_kind,
                " batch prepared search with mirrors worker_count=",
                worker_count,
                " ==",
            )
            append_measurement(
                measurements,
                cache.stored_task.dataset_id.copy(),
                task.slice_name.copy(),
                candidate_generator_kind.copy(),
                "batch_prepared_search_with_mirrors",
                worker_count,
                len(request_pool),
                len(base_requests),
                len(task.documents),
                task.nominal_query_vector_count,
                task.nominal_document_vector_count,
                task.vector_dim,
                benchmark_batch_mean_seconds(
                    mirrored_prepared,
                    request_pool,
                    worker_count,
                    scoring_config,
                ),
            )

    var out_path = Path(
        ".cache/kayak/prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.tsv"
    )
    write_measurements_tsv(out_path, measurements)
    print("wrote ", String(out_path))
