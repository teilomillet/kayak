from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.python import Python, PythonObject

from kayak.index import pack_documents
from kayak.interop.python_task_decoder import decode_judged_task
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.storage import (
    StoredPackedIndex,
    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    save_stored_packed_index_with_encoding,
)


comptime INPUT_TASK_PATH = ".cache/kayak/task_json_storage_cold_workflow_input.json"
comptime OUTPUT_SUMMARY_PATH = (
    ".cache/kayak/task_json_storage_artifact_prepare_output.json"
)
comptime ARTIFACT_ROOT = ".cache/kayak/task_json_storage_cold_workflow_artifact"


def load_raw_python_task(path: String) raises -> PythonObject:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.json_task_loader")
    return module.load_task_json(path)


def summary_json(
    dataset_id: String,
    model_name: String,
    family: String,
    slice_name: String,
    document_count: Int,
    vector_count: Int,
    vector_dim: Int,
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"dataset_id\":\"" + dataset_id + "\","
    buffer += "\"model_name\":\"" + model_name + "\","
    buffer += "\"family\":\"" + family + "\","
    buffer += "\"slice_name\":\"" + slice_name + "\","
    buffer += "\"artifact_root\":\"" + String(Path(ARTIFACT_ROOT)) + "\","
    buffer += "\"document_count\":" + String(document_count) + ","
    buffer += "\"vector_count\":" + String(vector_count) + ","
    buffer += "\"vector_dim\":" + String(vector_dim)
    buffer += "}"
    return buffer^


def main() raises:
    var py_task = load_raw_python_task(INPUT_TASK_PATH)
    var task = decode_judged_task(py_task)
    var dataset_id = String(py=py_task["dataset_id"])
    var model_name = String(py=py_task["model_name"])
    var stored_index = StoredPackedIndex(
        dataset_id.copy(),
        model_name.copy(),
        VECTOR_SCALAR_NAME,
        pack_documents(task.documents),
    )
    var artifact_root = Path(ARTIFACT_ROOT)
    makedirs(artifact_root, exist_ok=True)

    for encoding_kind in [
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
        VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    ]:
        save_stored_packed_index_with_encoding(
            artifact_root / encoding_kind,
            stored_index.copy(),
            encoding_kind,
        )
    var output_path = Path(OUTPUT_SUMMARY_PATH)
    output_path.write_text(
        summary_json(
            dataset_id,
            model_name,
            task.family.copy(),
            task.slice_name.copy(),
            len(task.documents),
            stored_index.index.total_vector_count,
            task.vector_dim,
        )
    )
    print("wrote ", String(output_path))
