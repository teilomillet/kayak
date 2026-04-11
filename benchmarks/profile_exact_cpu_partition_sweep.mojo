import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import ExactCpuBackend, ExactScoringConfig
from kayak.benchmarks import ExactSearchProfile, make_exact_search_fixture_for_profiled_search


struct PartitionSweepWorkload(Copyable):
    var name: String
    var profile: ExactSearchProfile

    def __init__(out self, var name: String, var profile: ExactSearchProfile):
        self.name = name^
        self.profile = profile^


struct PartitionSweepMeasurement(Copyable):
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
        var workload_name: String,
        document_count: Int,
        document_vector_count: Int,
        query_vector_count: Int,
        vector_dim: Int,
        top_k: Int,
        work_items: Int,
        mean_seconds: Float64,
    ):
        self.workload_name = workload_name^
        self.document_count = document_count
        self.document_vector_count = document_vector_count
        self.query_vector_count = query_vector_count
        self.vector_dim = vector_dim
        self.top_k = top_k
        self.work_items = work_items
        self.mean_seconds = mean_seconds


def default_partition_sweep_workloads() -> List[PartitionSweepWorkload]:
    return [
        PartitionSweepWorkload("small", ExactSearchProfile(128, 16, 8, 128, 10)),
        PartitionSweepWorkload("medium", ExactSearchProfile(512, 32, 16, 128, 10)),
        PartitionSweepWorkload("large", ExactSearchProfile(1024, 64, 32, 128, 10)),
    ]


def candidate_work_items() -> List[Int]:
    return [1, 2, 4, 5, 6, 7, 8, 12, 16, 24, 32]


def make_backend_for_work_items(work_items: Int) -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.parallel_work_item_count_override = work_items
    return ExactCpuBackend(config^)


def write_measurements_tsv(
    path: Path, measurements: List[PartitionSweepMeasurement]
) raises:
    var lines = String()
    lines += "workload_name\tdocument_count\tdocument_vector_count\tquery_vector_count\tvector_dim\ttop_k\twork_items\tmean_seconds\tthroughput_per_second\n"

    for measurement in measurements:
        lines += measurement.workload_name
        lines += "\t"
        lines += String(measurement.document_count)
        lines += "\t"
        lines += String(measurement.document_vector_count)
        lines += "\t"
        lines += String(measurement.query_vector_count)
        lines += "\t"
        lines += String(measurement.vector_dim)
        lines += "\t"
        lines += String(measurement.top_k)
        lines += "\t"
        lines += String(measurement.work_items)
        lines += "\t"
        lines += String(measurement.mean_seconds)
        lines += "\t"
        lines += String(1.0 / measurement.mean_seconds)
        lines += "\n"

    path.write_text(lines)


def benchmark_partition_sweep() raises -> List[PartitionSweepMeasurement]:
    var measurements = List[PartitionSweepMeasurement]()

    for workload in default_partition_sweep_workloads():
        var fixture = make_exact_search_fixture_for_profiled_search(workload.profile)

        for work_items in candidate_work_items():
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
                PartitionSweepMeasurement(
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

    return measurements^


def main() raises:
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_exact_cpu_partition_sweep.tsv"
    print("writing partition sweep to ", String(output_path))
    print("")

    write_measurements_tsv(output_path, benchmark_partition_sweep())
    print("saved ", String(output_path))
