import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactCpuBackend,
    NamespaceId,
    PlannedSearchRequest,
    SearchPlanSelectionRequest,
    SearchRequest,
    SnapshotId,
    TenantId,
    best_effort_faithfulness_policy,
    default_exact_search_request,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    execute_planned_search,
    execute_planned_search_with_prepared_snapshot,
    execute_search,
    execute_search_with_prepared_snapshot,
    match_all_filter,
    prepare_service_search_snapshot,
)
from kayak.eval import JudgedTask
from kayak.service.paths import service_collection_root


struct PreparedSnapshotMeasurement(Copyable):
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
    mut out: String, read measurement: PreparedSnapshotMeasurement
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


def write_measurements_tsv(
    path: Path, measurements: List[PreparedSnapshotMeasurement]
) raises:
    var lines = String()
    lines += "dataset_name\tslice_name\tbenchmark_kind\tdocument_count\tquery_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\tmean_seconds\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def stage1_candidate_k(task: JudgedTask) -> Int:
    var candidate_k = task.k * 4
    if candidate_k > len(task.documents):
        return len(task.documents)

    return candidate_k


def exact_preferred_candidate_generator_kinds() -> List[String]:
    var kinds = List[String]()
    kinds.append("exact_full_scan")
    return kinds^


def profile_service_root() -> Path:
    return Path(".cache/kayak/profile_service_prepared_snapshot_reuse_browsecomp_gold")


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


def build_planned_requests(
    read task: JudgedTask, model_name: String
) raises -> List[PlannedSearchRequest]:
    var requests = List[PlannedSearchRequest]()
    var preferred_candidate_generator_kinds = (
        exact_preferred_candidate_generator_kinds()
    )

    for judged_query in task.queries:
        requests.append(
            PlannedSearchRequest(
                CollectionId("browsecomp-plus-gold-service"),
                TenantId("public"),
                NamespaceId("benchmark"),
                SnapshotId("snapshot-0001"),
                judged_query.query,
                model_name,
                match_all_filter(),
                SearchPlanSelectionRequest(
                    task.k,
                    stage1_candidate_k(task),
                    best_effort_faithfulness_policy(),
                    match_all_filter(),
                    "balanced",
                    preferred_candidate_generator_kinds,
                ),
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

    var report = benchmark.run[prepare_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_exact_search_mean_seconds(
    read backend: ExactCpuBackend,
    service_root: Path,
    read requests: List[SearchRequest],
) raises -> Float64:
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(execute_search(backend, service_root, requests[query_index]))
        query_index += 1
        if query_index == len(requests):
            query_index = 0

    var report = benchmark.run[search_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_exact_search_with_prepared_snapshot_mean_seconds(
    read backend: ExactCpuBackend,
    service_root: Path,
    read requests: List[SearchRequest],
) raises -> Float64:
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

    var report = benchmark.run[search_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_planned_search_mean_seconds(
    read backend: ExactCpuBackend,
    service_root: Path,
    read requests: List[PlannedSearchRequest],
) raises -> Float64:
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(
            execute_planned_search(backend, service_root, requests[query_index])
        )
        query_index += 1
        if query_index == len(requests):
            query_index = 0

    var report = benchmark.run[search_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_planned_search_with_prepared_snapshot_mean_seconds(
    read backend: ExactCpuBackend,
    service_root: Path,
    read requests: List[PlannedSearchRequest],
) raises -> Float64:
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
            execute_planned_search_with_prepared_snapshot(
                backend,
                prepared,
                requests[query_index],
            )
        )
        query_index += 1
        if query_index == len(requests):
            query_index = 0

    var report = benchmark.run[search_once]()
    report.print()
    print("")
    return report.mean()


def main() raises:
    print("loading BrowseComp+ gold real-subset cache...")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var service_root = ensure_profile_service_collection()
    var exact_requests = build_exact_requests(task, cache.stored_task.model_name)
    var planned_requests = build_planned_requests(task, cache.stored_task.model_name)
    var backend = ExactCpuBackend()
    var measurements = List[PreparedSnapshotMeasurement]()

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
    measurements.append(
        PreparedSnapshotMeasurement(
            cache.stored_task.dataset_id.copy(),
            task.slice_name.copy(),
            "prepare_snapshot",
            len(task.documents),
            len(task.queries),
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            benchmark_prepare_snapshot_mean_seconds(service_root),
        )
    )

    print("== exact_search_stateless ==")
    measurements.append(
        PreparedSnapshotMeasurement(
            cache.stored_task.dataset_id.copy(),
            task.slice_name.copy(),
            "exact_search_stateless",
            len(task.documents),
            len(task.queries),
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            benchmark_exact_search_mean_seconds(
                backend,
                service_root,
                exact_requests,
            ),
        )
    )

    print("== exact_search_prepared ==")
    measurements.append(
        PreparedSnapshotMeasurement(
            cache.stored_task.dataset_id.copy(),
            task.slice_name.copy(),
            "exact_search_prepared",
            len(task.documents),
            len(task.queries),
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            benchmark_exact_search_with_prepared_snapshot_mean_seconds(
                backend,
                service_root,
                exact_requests,
            ),
        )
    )

    print("== planned_search_stateless_exact_preferred ==")
    measurements.append(
        PreparedSnapshotMeasurement(
            cache.stored_task.dataset_id.copy(),
            task.slice_name.copy(),
            "planned_search_stateless_exact_preferred",
            len(task.documents),
            len(task.queries),
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            benchmark_planned_search_mean_seconds(
                backend,
                service_root,
                planned_requests,
            ),
        )
    )

    print("== planned_search_prepared_exact_preferred ==")
    measurements.append(
        PreparedSnapshotMeasurement(
            cache.stored_task.dataset_id.copy(),
            task.slice_name.copy(),
            "planned_search_prepared_exact_preferred",
            len(task.documents),
            len(task.queries),
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            benchmark_planned_search_with_prepared_snapshot_mean_seconds(
                backend,
                service_root,
                planned_requests,
            ),
        )
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "service_prepared_snapshot_reuse_browsecomp_gold.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
