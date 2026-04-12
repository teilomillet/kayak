from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    StorageEncodingSummary,
    build_storage_encoding_summary,
    storage_encoding_summaries_json,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import (
    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    ensure_browsecomp_plus_gold_real_subset_cache,
)


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var backend = ExactCpuBackend()
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var benchmark_root = output_root / "browsecomp_plus_gold_storage_encoding"
    makedirs(benchmark_root, exist_ok=True)
    var summaries = List[StorageEncodingSummary]()

    for encoding_kind in [
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
        VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    ]:
        var summary = build_storage_encoding_summary(
            backend,
            cache.stored_task,
            cache.stored_index,
            benchmark_root / encoding_kind,
            encoding_kind,
        )
        print(
            "encoding=",
            summary.encoding_kind,
            " bytes/vector=",
            summary.artifact_bytes_per_vector,
            " build_s=",
            summary.mean_build_seconds,
            " load_s=",
            summary.mean_load_seconds,
            " search_s=",
            summary.mean_search_seconds,
            " ndcg=",
            summary.mean_ndcg_at_k,
        )
        summaries.append(summary.copy())
    (output_root / "browsecomp_plus_gold_storage_encoding.json").write_text(
        storage_encoding_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "browsecomp_plus_gold_storage_encoding.json"))
