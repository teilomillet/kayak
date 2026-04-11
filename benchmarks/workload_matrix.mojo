import std.benchmark as benchmark

from kayak.benchmarks import (
    WorkloadProfile,
    default_workload_profiles,
    make_exact_search_fixture_for_profile,
)
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact


def benchmark_profile(profile: WorkloadProfile) raises:
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


def main() raises:
    for profile in default_workload_profiles():
        benchmark_profile(profile)
