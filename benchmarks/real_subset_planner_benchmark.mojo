from kayak.benchmarks import (
    default_planner_benchmark_run_options,
    write_public_planner_benchmark,
)


def main() raises:
    var output_path = write_public_planner_benchmark(
        default_planner_benchmark_run_options()
    )
    print("wrote ", String(output_path))
