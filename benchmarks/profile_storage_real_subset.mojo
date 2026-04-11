import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler
from std.pathlib import Path

from kayak import StoredJudgedTask, StoredPackedIndex
from kayak.storage import (
    ensure_fiqa_real_subset_cache,
    ensure_scifact_real_subset_cache,
    load_stored_judged_task,
    load_stored_packed_index,
    save_stored_judged_task,
    save_stored_packed_index,
)
from benchmarks.storage_legacy_text import (
    judged_task_storage_byte_size,
    packed_index_storage_byte_size,
    write_v1_stored_judged_task,
    write_v1_stored_packed_index,
)


struct PreparedStorageFormats(Copyable):
    var v1_task_root: Path
    var v1_index_root: Path
    var v2_task_root: Path
    var v2_index_root: Path

    def __init__(
        out self,
        var v1_task_root: Path,
        var v1_index_root: Path,
        var v2_task_root: Path,
        var v2_index_root: Path,
    ):
        self.v1_task_root = v1_task_root^
        self.v1_index_root = v1_index_root^
        self.v2_task_root = v2_task_root^
        self.v2_index_root = v2_index_root^


def print_size_line(label: String, byte_count: Int):
    print(label, ": ", byte_count, " bytes")


def benchmark_load_task(label: String, root: Path) raises:
    print("== load_stored_judged_task(", label, ") ==")

    def load_once() capturing raises:
        bench_compiler.keep(load_stored_judged_task(root))

    benchmark.run[load_once]().print()
    print("")


def benchmark_load_index(label: String, root: Path) raises:
    print("== load_stored_packed_index(", label, ") ==")

    def load_once() capturing raises:
        bench_compiler.keep(load_stored_packed_index(root))

    benchmark.run[load_once]().print()
    print("")


def prepare_storage_formats(
    root: Path,
    stored_task: StoredJudgedTask,
    stored_index: StoredPackedIndex,
) raises -> PreparedStorageFormats:
    var v1_task_root = root / "v1" / "judged_task"
    var v1_index_root = root / "v1" / "packed_index"
    var v2_task_root = root / "v2" / "judged_task"
    var v2_index_root = root / "v2" / "packed_index"

    write_v1_stored_judged_task(v1_task_root, stored_task.copy())
    write_v1_stored_packed_index(v1_index_root, stored_index.copy())
    save_stored_judged_task(v2_task_root, stored_task.copy())
    save_stored_packed_index(v2_index_root, stored_index.copy())

    return PreparedStorageFormats(
        v1_task_root^,
        v1_index_root^,
        v2_task_root^,
        v2_index_root^,
    )


def benchmark_dataset(
    dataset_name: String,
    root: Path,
    stored_task: StoredJudgedTask,
    stored_index: StoredPackedIndex,
) raises:
    print("dataset: ", dataset_name)
    var roots = prepare_storage_formats(root, stored_task, stored_index)

    print_size_line(
        "judged_task v1 size", judged_task_storage_byte_size(roots.v1_task_root)
    )
    print_size_line(
        "judged_task v2 size", judged_task_storage_byte_size(roots.v2_task_root)
    )
    print_size_line(
        "packed_index v1 size", packed_index_storage_byte_size(roots.v1_index_root)
    )
    print_size_line(
        "packed_index v2 size", packed_index_storage_byte_size(roots.v2_index_root)
    )
    print("")

    benchmark_load_task("v1", roots.v1_task_root)
    benchmark_load_task("v2", roots.v2_task_root)
    benchmark_load_index("v1", roots.v1_index_root)
    benchmark_load_index("v2", roots.v2_index_root)


def benchmark_scifact_real_subset() raises:
    var cache = ensure_scifact_real_subset_cache()
    benchmark_dataset(
        "SciFact",
        Path("/tmp/kayak-storage-profile/scifact"),
        cache.stored_task,
        cache.stored_index,
    )


def benchmark_fiqa_real_subset() raises:
    var cache = ensure_fiqa_real_subset_cache()
    benchmark_dataset(
        "FIQA",
        Path("/tmp/kayak-storage-profile/fiqa"),
        cache.stored_task,
        cache.stored_index,
    )


def main() raises:
    benchmark_scifact_real_subset()
    benchmark_fiqa_real_subset()
