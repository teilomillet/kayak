import std.benchmark as benchmark
from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    WorkloadProfile,
    WorkloadBenchmarkSummary,
    build_workload_benchmark_summary,
    default_workload_profiles,
    make_exact_search_fixture_for_profile,
    workload_benchmark_summaries_json,
)
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact


def benchmark_profile(profile: WorkloadProfile) raises -> WorkloadBenchmarkSummary:
    var fixture = make_exact_search_fixture_for_profile(profile)
    var backend = ExactCpuBackend()

    print("== ", profile.family, ".", profile.slice_name, " ==")
    print("why: ", profile.why)
    print(
        "shape: docs=",
        profile.document_count,
        " doc_vecs=",
        profile.document_vector_count,
        " query_vecs=",
        profile.query_vector_count,
        " dim=",
        profile.vector_dim,
        " top_k=",
        profile.top_k,
    )

    def score_once() capturing raises:
        _ = search_exact(backend, fixture.query, fixture.index, fixture.top_k)

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return build_workload_benchmark_summary(profile, Float64(report.mean()))


def main() raises:
    var summaries = List[WorkloadBenchmarkSummary]()
    for profile in default_workload_profiles():
        summaries.append(benchmark_profile(profile))

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "workload_matrix.json"
    output_path.write_text(workload_benchmark_summaries_json(summaries))
    print("wrote ", String(output_path))
