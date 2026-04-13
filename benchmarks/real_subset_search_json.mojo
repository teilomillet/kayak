from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    RealSliceBenchmarkSummary,
    build_real_slice_benchmark_summary,
    load_default_public_benchmark_datasets,
    real_slice_benchmark_summaries_json,
)
from kayak.runtime import ExactCpuBackend


def main() raises:
    var backend = ExactCpuBackend()
    var summaries = List[RealSliceBenchmarkSummary]()

    for dataset in load_default_public_benchmark_datasets():
        summaries.append(
            build_real_slice_benchmark_summary(
                backend,
                dataset.stored_task,
                dataset.stored_index,
                dataset.loaded_task_from_storage,
                dataset.loaded_index_from_storage,
            )
        )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "public_real_slice_benchmarks.json").write_text(
        real_slice_benchmark_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "public_real_slice_benchmarks.json"))
