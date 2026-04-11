import std.benchmark as benchmark

from kayak.benchmarks import make_exact_search_fixture
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact


def main() raises:
    var fixture = make_exact_search_fixture()
    var backend = ExactCpuBackend()

    def score_once() capturing raises:
        _ = search_exact(backend, fixture.query, fixture.index, fixture.top_k)

    var report = benchmark.run[score_once]()
    report.print()
