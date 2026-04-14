from std.os import makedirs
from std.pathlib import Path
from std.python import Python, PythonObject

from kayak.benchmarks import (
    build_storage_encoding_summary,
    storage_encoding_summary_json,
)
from kayak.index import pack_documents
from kayak.interop.python_task_decoder import decode_judged_task
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.runtime import ExactCpuBackend
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
)


comptime INPUT_TASK_PATH = ".cache/kayak/task_json_storage_encoding_input.json"
comptime OUTPUT_SUMMARY_PATH = ".cache/kayak/task_json_storage_encoding_output.json"
comptime ARTIFACT_ROOT = ".cache/kayak/task_json_storage_encoding_artifact"


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
    var summary = build_storage_encoding_summary(
        ExactCpuBackend(),
        stored_task,
        stored_index,
        Path(ARTIFACT_ROOT),
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = Path(OUTPUT_SUMMARY_PATH)
    output_path.write_text(storage_encoding_summary_json(summary))

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
    print("wrote ", String(output_path))
