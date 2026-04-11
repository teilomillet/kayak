import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.runtime.asyncrt import parallelism_level

from kayak import ExactCpuBackend, ExactScoringConfig
from kayak.benchmarks import ExactSearchProfile, make_exact_search_fixture_for_profiled_search
from kayak.search import search_exact


struct USLWorkload(Copyable):
    var name: String
    var profile: ExactSearchProfile

    def __init__(out self, var name: String, var profile: ExactSearchProfile):
        self.name = name^
        self.profile = profile^


struct USLMeasurement(Copyable):
    var benchmark_name: String
    var workload_name: String
    var document_count: Int
    var document_vector_count: Int
    var query_vector_count: Int
    var vector_dim: Int
    var top_k: Int
    var work_items: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var benchmark_name: String,
        var workload_name: String,
        document_count: Int,
        document_vector_count: Int,
        query_vector_count: Int,
        vector_dim: Int,
        top_k: Int,
        work_items: Int,
        mean_seconds: Float64,
    ):
        self.benchmark_name = benchmark_name^
        self.workload_name = workload_name^
        self.document_count = document_count
        self.document_vector_count = document_vector_count
        self.query_vector_count = query_vector_count
        self.vector_dim = vector_dim
        self.top_k = top_k
        self.work_items = work_items
        self.mean_seconds = mean_seconds


def default_usl_workloads() -> List[USLWorkload]:
    return [
        USLWorkload("medium", ExactSearchProfile(512, 32, 16, 128, 10)),
        USLWorkload("large", ExactSearchProfile(1024, 64, 32, 128, 10)),
    ]


def make_backend_for_work_items(work_items: Int) -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.parallel_work_item_count_override = work_items
    return ExactCpuBackend(config^)


def append_tsv_line(
    mut out: String,
    benchmark_name: String,
    workload_name: String,
    document_count: Int,
    document_vector_count: Int,
    query_vector_count: Int,
    vector_dim: Int,
    top_k: Int,
    work_items: Int,
    mean_seconds: Float64,
    throughput_per_second: Float64,
    relative_capacity: Float64,
):
    out += benchmark_name
    out += "\t"
    out += workload_name
    out += "\t"
    out += String(document_count)
    out += "\t"
    out += String(document_vector_count)
    out += "\t"
    out += String(query_vector_count)
    out += "\t"
    out += String(vector_dim)
    out += "\t"
    out += String(top_k)
    out += "\t"
    out += String(work_items)
    out += "\t"
    out += String(mean_seconds)
    out += "\t"
    out += String(throughput_per_second)
    out += "\t"
    out += String(relative_capacity)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[USLMeasurement]
) raises:
    var lines = String()
    lines += "benchmark_name\tworkload_name\tdocument_count\tdocument_vector_count\tquery_vector_count\tvector_dim\ttop_k\twork_items\tmean_seconds\tthroughput_per_second\trelative_capacity\n"

    for measurement in measurements:
        var throughput_per_second = 1.0 / measurement.mean_seconds
        var relative_capacity = Float64(1.0)

        for baseline in measurements:
            if (
                baseline.benchmark_name == measurement.benchmark_name
                and baseline.workload_name == measurement.workload_name
                and baseline.work_items == 1
            ):
                relative_capacity = throughput_per_second / (
                    1.0 / baseline.mean_seconds
                )
                break

        append_tsv_line(
            lines,
            measurement.benchmark_name,
            measurement.workload_name,
            measurement.document_count,
            measurement.document_vector_count,
            measurement.query_vector_count,
            measurement.vector_dim,
            measurement.top_k,
            measurement.work_items,
            measurement.mean_seconds,
            throughput_per_second,
            relative_capacity,
        )

    path.write_text(lines)


def benchmark_score_all_scaling(
    mut measurements: List[USLMeasurement],
    workload: USLWorkload,
    max_work_items: Int,
) raises:
    var fixture = make_exact_search_fixture_for_profiled_search(workload.profile)

    for work_items in range(1, max_work_items + 1):
        var backend = make_backend_for_work_items(work_items)

        print(
            "score_all",
            " workload=",
            workload.name,
            " work_items=",
            work_items,
        )

        def score_once() capturing raises:
            bench_compiler.keep(backend.score_all(fixture.query, fixture.index))

        var report = benchmark.run[score_once]()
        var mean_seconds = report.mean()
        report.print()
        print("")

        measurements.append(
            USLMeasurement(
                "score_all",
                workload.name.copy(),
                workload.profile.document_count,
                workload.profile.document_vector_count,
                workload.profile.query_vector_count,
                workload.profile.vector_dim,
                workload.profile.top_k,
                work_items,
                mean_seconds,
            )
        )


def benchmark_search_exact_scaling(
    mut measurements: List[USLMeasurement],
    workload: USLWorkload,
    max_work_items: Int,
) raises:
    var fixture = make_exact_search_fixture_for_profiled_search(workload.profile)

    for work_items in range(1, max_work_items + 1):
        var backend = make_backend_for_work_items(work_items)

        print(
            "search_exact",
            " workload=",
            workload.name,
            " work_items=",
            work_items,
        )

        def score_once() capturing raises:
            bench_compiler.keep(
                search_exact(backend, fixture.query, fixture.index, fixture.top_k)
            )

        var report = benchmark.run[score_once]()
        var mean_seconds = report.mean()
        report.print()
        print("")

        measurements.append(
            USLMeasurement(
                "search_exact",
                workload.name.copy(),
                workload.profile.document_count,
                workload.profile.document_vector_count,
                workload.profile.query_vector_count,
                workload.profile.vector_dim,
                workload.profile.top_k,
                work_items,
                mean_seconds,
            )
        )


def main() raises:
    var root = Path(".cache/kayak")
    makedirs(root, exist_ok=True)
    var output_path = root / "profile_exact_cpu_usl.tsv"
    var measurements = List[USLMeasurement]()
    var max_work_items = parallelism_level()
    if max_work_items < 1:
        max_work_items = 1

    print("writing USL sweep to ", String(output_path))
    print("max_work_items=", max_work_items)
    print("")

    for workload in default_usl_workloads():
        benchmark_score_all_scaling(measurements, workload, max_work_items)
        benchmark_search_exact_scaling(measurements, workload, max_work_items)

    write_measurements_tsv(output_path, measurements)
    print("saved ", String(output_path))
