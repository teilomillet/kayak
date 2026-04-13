from kayak.benchmarks import (
    smoke_planner_benchmark_run_options,
    write_public_planner_benchmark,
)


def main() raises:
    var output_path = write_public_planner_benchmark(
        smoke_planner_benchmark_run_options()
    )
    print("wrote ", String(output_path))
