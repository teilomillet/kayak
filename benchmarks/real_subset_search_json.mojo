from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    RealSliceBenchmarkSummary,
    build_real_slice_benchmark_summary,
    real_slice_benchmark_summaries_json,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import (
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_browsecomp_plus_real_subset_cache,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_scifact_real_subset_cache,
)


def main() raises:
    var backend = ExactCpuBackend()
    var summaries = List[RealSliceBenchmarkSummary]()

    var scifact_cache = ensure_scifact_real_subset_cache()
    summaries.append(
        build_real_slice_benchmark_summary(
            backend,
            scifact_cache.stored_task,
            scifact_cache.stored_index,
            scifact_cache.loaded_task_from_storage,
            scifact_cache.loaded_index_from_storage,
        )
    )

    var fiqa_cache = ensure_fiqa_real_subset_cache()
    summaries.append(
        build_real_slice_benchmark_summary(
            backend,
            fiqa_cache.stored_task,
            fiqa_cache.stored_index,
            fiqa_cache.loaded_task_from_storage,
            fiqa_cache.loaded_index_from_storage,
        )
    )

    var limit_small_cache = ensure_limit_small_real_subset_cache()
    summaries.append(
        build_real_slice_benchmark_summary(
            backend,
            limit_small_cache.stored_task,
            limit_small_cache.stored_index,
            limit_small_cache.loaded_task_from_storage,
            limit_small_cache.loaded_index_from_storage,
        )
    )

    var browsecomp_cache = ensure_browsecomp_plus_real_subset_cache()
    summaries.append(
        build_real_slice_benchmark_summary(
            backend,
            browsecomp_cache.stored_task,
            browsecomp_cache.stored_index,
            browsecomp_cache.loaded_task_from_storage,
            browsecomp_cache.loaded_index_from_storage,
        )
    )

    var browsecomp_gold_cache = ensure_browsecomp_plus_gold_real_subset_cache()
    summaries.append(
        build_real_slice_benchmark_summary(
            backend,
            browsecomp_gold_cache.stored_task,
            browsecomp_gold_cache.stored_index,
            browsecomp_gold_cache.loaded_task_from_storage,
            browsecomp_gold_cache.loaded_index_from_storage,
        )
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "public_real_slice_benchmarks.json").write_text(
        real_slice_benchmark_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "public_real_slice_benchmarks.json"))
