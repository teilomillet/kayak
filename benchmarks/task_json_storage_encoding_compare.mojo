from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.python import Python, PythonObject

from kayak.benchmarks import (
    StorageEncodingSummary,
    build_storage_encoding_summary,
    storage_encoding_summaries_json,
)
from kayak.index import pack_documents
from kayak.interop.python_task_decoder import decode_judged_task
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.runtime import ExactCpuBackend
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
)


comptime INPUT_TASK_PATH = ".cache/kayak/task_json_storage_encoding_compare_input.json"
comptime OUTPUT_SUMMARIES_PATH = (
    ".cache/kayak/task_json_storage_encoding_compare_output.json"
)
comptime ARTIFACT_ROOT = ".cache/kayak/task_json_storage_encoding_compare_artifact"


def load_raw_python_task(path: String) raises -> PythonObject:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.json_task_loader")
    return module.load_task_json(path)


def main() raises:
    var py_task = load_raw_python_task(INPUT_TASK_PATH)
    var task = decode_judged_task(py_task)
    var dataset_id = String(py=py_task["dataset_id"])
    var model_name = String(py=py_task["model_name"])
    var stored_task = StoredJudgedTask(
        dataset_id.copy(),
        model_name.copy(),
        VECTOR_SCALAR_NAME,
        task.copy(),
    )
    var stored_index = StoredPackedIndex(
        dataset_id.copy(),
        model_name.copy(),
        VECTOR_SCALAR_NAME,
        pack_documents(task.documents),
    )
    var backend = ExactCpuBackend()
    var benchmark_root = Path(ARTIFACT_ROOT)
    var summaries = List[StorageEncodingSummary]()

    for encoding_kind in [
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
        VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    ]:
        var summary = build_storage_encoding_summary(
            backend,
            stored_task,
            stored_index,
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

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = Path(OUTPUT_SUMMARIES_PATH)
    output_path.write_text(storage_encoding_summaries_json(summaries))
    print("wrote ", String(output_path))
