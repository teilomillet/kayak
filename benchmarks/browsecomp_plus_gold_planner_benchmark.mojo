from kayak.benchmarks import (
    PlannerBenchmarkRunOptions,
    write_public_planner_benchmark,
)


def main() raises:
    var output_path = write_public_planner_benchmark(
        PlannerBenchmarkRunOptions(
            "browsecomp_plus_gold_planner_benchmark.json",
            "browsecomp_plus_gold_planner_collection",
            True,
            0,
            False,
            False,
            False,
            False,
            True,
            True,
        )
    )
    print("wrote ", String(output_path))
