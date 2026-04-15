import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactScoringConfig,
    NamespaceId,
    SearchRequest,
    SnapshotId,
    TenantId,
    default_exact_search_request,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    execute_search_with_prepared_snapshot,
    prepare_service_exact_search_executor,
    prepare_service_exact_search_executor_with_config,
    prepare_service_search_snapshot,
)
from kayak.eval import JudgedTask
from kayak.runtime import ExactCpuBackend
from kayak.service.paths import service_collection_root

# Bound each section so forced quiet-wrapper reruns remain comparable.
comptime PREPARE_MAX_ITERS = 256
comptime SEARCH_MAX_ITERS = 1024


struct ExecutorMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var benchmark_kind: String
    var document_count: Int
    var query_count: Int
    var query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var benchmark_kind: String,
        document_count: Int,
        query_count: Int,
        query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
        mean_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.benchmark_kind = benchmark_kind^
        self.document_count = document_count
        self.query_count = query_count
        self.query_vector_count = query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(
    mut out: String, read measurement: ExecutorMeasurement
):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.benchmark_kind
    out += "\t"
    out += String(measurement.document_count)
    out += "\t"
    out += String(measurement.query_count)
    out += "\t"
    out += String(measurement.query_vector_count)
    out += "\t"
    out += String(measurement.nominal_document_vector_count)
    out += "\t"
    out += String(measurement.vector_dim)
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(path: Path, measurements: List[ExecutorMeasurement]) raises:
    var lines = String()
    lines += "dataset_name\tslice_name\tbenchmark_kind\tdocument_count\tquery_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\tmean_seconds\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def profile_service_root() -> Path:
    return Path(".cache/kayak/profile_prepared_exact_search_executor_browsecomp_gold")


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


def build_exact_requests(read task: JudgedTask, model_name: String) raises -> List[SearchRequest]:
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


def benchmark_prepare_snapshot_mean_seconds(service_root: Path) raises -> Float64:
    def prepare_once() capturing raises:
        var prepared = prepare_service_search_snapshot(
            service_root,
            CollectionId("browsecomp-plus-gold-service"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            False,
        )
        bench_compiler.keep(len(prepared.snapshot.segments))

    var report = benchmark.run[prepare_once](max_iters=PREPARE_MAX_ITERS)
    report.print()
    print("")
    return report.mean()


def benchmark_prepare_executor_default_mean_seconds(service_root: Path) raises -> Float64:
    def prepare_once() capturing raises:
        var executor = prepare_service_exact_search_executor(
            service_root,
            CollectionId("browsecomp-plus-gold-service"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            False,
        )
        bench_compiler.keep(len(executor.prepared.segments))

    var report = benchmark.run[prepare_once](max_iters=PREPARE_MAX_ITERS)
    report.print()
    print("")
    return report.mean()


def benchmark_search_prepared_snapshot_direct_mean_seconds(
    service_root: Path,
    read requests: List[SearchRequest],
) raises -> Float64:
    var backend = ExactCpuBackend()
    var prepared = prepare_service_search_snapshot(
        service_root,
        CollectionId("browsecomp-plus-gold-service"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        False,
    )
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

    var report = benchmark.run[search_once](max_iters=SEARCH_MAX_ITERS)
    report.print()
    print("")
    return report.mean()


def benchmark_search_executor_default_mean_seconds(
    service_root: Path,
    read requests: List[SearchRequest],
) raises -> Float64:
    var executor = prepare_service_exact_search_executor(
        service_root,
        CollectionId("browsecomp-plus-gold-service"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        False,
    )
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(executor.execute_search(requests[query_index]))
        query_index += 1
        if query_index == len(requests):
            query_index = 0

    var report = benchmark.run[search_once](max_iters=SEARCH_MAX_ITERS)
    report.print()
    print("")
    return report.mean()


def benchmark_search_executor_serial_mean_seconds(
    service_root: Path,
    read requests: List[SearchRequest],
) raises -> Float64:
    var serial_config = ExactScoringConfig()
    serial_config.enable_parallel_scoring = False
    var executor = prepare_service_exact_search_executor_with_config(
        service_root,
        CollectionId("browsecomp-plus-gold-service"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        serial_config,
        False,
    )
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(executor.execute_search(requests[query_index]))
        query_index += 1
        if query_index == len(requests):
            query_index = 0

    var report = benchmark.run[search_once](max_iters=SEARCH_MAX_ITERS)
    report.print()
    print("")
    return report.mean()


def benchmark_search_executor_parallel2_mean_seconds(
    service_root: Path,
    read requests: List[SearchRequest],
) raises -> Float64:
    var config = ExactScoringConfig()
    config.parallel_work_item_count_override = 2
    var executor = prepare_service_exact_search_executor_with_config(
        service_root,
        CollectionId("browsecomp-plus-gold-service"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        config,
        False,
    )
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(executor.execute_search(requests[query_index]))
        query_index += 1
        if query_index == len(requests):
            query_index = 0

    var report = benchmark.run[search_once](max_iters=SEARCH_MAX_ITERS)
    report.print()
    print("")
    return report.mean()


def append_measurement(
    mut measurements: List[ExecutorMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    benchmark_kind: String,
    mean_seconds: Float64,
):
    measurements.append(
        ExecutorMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            benchmark_kind,
            len(task.documents),
            len(task.queries),
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            mean_seconds,
        )
    )


def main() raises:
    print("loading BrowseComp+ gold real-subset cache...")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var service_root = ensure_profile_service_collection()
    var requests = build_exact_requests(task, cache.stored_task.model_name)
    var measurements = List[ExecutorMeasurement]()

    print("dataset: ", cache.stored_task.dataset_id)
    print("slice: ", task.slice_name)
    print("queries: ", len(task.queries))
    print("documents: ", len(task.documents))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)
    print("service_root: ", String(service_root))
    print("")

    print("== prepare_snapshot ==")
    append_measurement(
        measurements,
        cache.stored_task.dataset_id,
        task,
        "prepare_snapshot",
        benchmark_prepare_snapshot_mean_seconds(service_root),
    )

    print("== prepare_executor_default ==")
    append_measurement(
        measurements,
        cache.stored_task.dataset_id,
        task,
        "prepare_executor_default",
        benchmark_prepare_executor_default_mean_seconds(service_root),
    )

    print("== search_prepared_snapshot_direct ==")
    append_measurement(
        measurements,
        cache.stored_task.dataset_id,
        task,
        "search_prepared_snapshot_direct",
        benchmark_search_prepared_snapshot_direct_mean_seconds(
            service_root,
            requests,
        ),
    )

    print("== search_executor_default ==")
    append_measurement(
        measurements,
        cache.stored_task.dataset_id,
        task,
        "search_executor_default",
        benchmark_search_executor_default_mean_seconds(service_root, requests),
    )

    print("== search_executor_serial ==")
    append_measurement(
        measurements,
        cache.stored_task.dataset_id,
        task,
        "search_executor_serial",
        benchmark_search_executor_serial_mean_seconds(service_root, requests),
    )

    print("== search_executor_parallel2 ==")
    append_measurement(
        measurements,
        cache.stored_task.dataset_id,
        task,
        "search_executor_parallel2",
        benchmark_search_executor_parallel2_mean_seconds(service_root, requests),
    )

    var out_path = Path(
        ".cache/kayak/prepared_exact_search_executor_browsecomp_gold.tsv"
    )
    write_measurements_tsv(out_path, measurements)
    print("wrote ", String(out_path))
