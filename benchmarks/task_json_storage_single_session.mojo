from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.pathlib import Path
from std.python import Python, PythonObject

from kayak.interop.python_task_decoder import decode_judged_task
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import load_stored_packed_index


comptime INPUT_TASK_PATH = ".cache/kayak/task_json_storage_cold_workflow_input.json"
comptime INPUT_PARAMS_PATH = (
    ".cache/kayak/task_json_storage_single_session_input.json"
)
comptime OUTPUT_SUMMARY_PATH = (
    ".cache/kayak/task_json_storage_single_session_output.json"
)
comptime ARTIFACT_ROOT = ".cache/kayak/task_json_storage_cold_workflow_artifact"


struct SingleSessionConfig(Copyable):
    var encoding_kind: String
    var queries_per_load: Int
    var query_start: Int

    def __init__(
        out self,
        var encoding_kind: String,
        queries_per_load: Int,
        query_start: Int,
    ) raises:
        if queries_per_load <= 0:
            raise Error("queries_per_load must be positive")
        if query_start < 0:
            raise Error("query_start must be non-negative")
        self.encoding_kind = encoding_kind^
        self.queries_per_load = queries_per_load
        self.query_start = query_start


def load_raw_python_task(path: String) raises -> PythonObject:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.json_task_loader")
    return module.load_task_json(path)


def load_single_session_config(path: String) raises -> SingleSessionConfig:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.task_storage_cold_workflow")
    var payload = module.load_single_session_config_json(path)
    return SingleSessionConfig(
        String(py=payload["encoding_kind"]),
        Int(py=payload["queries_per_load"]),
        Int(py=payload["query_start"]),
    )


def summary_json(
    dataset_id: String,
    model_name: String,
    family: String,
    slice_name: String,
    encoding_kind: String,
    queries_per_load: Int,
    query_start: Int,
    session_seconds: Float64,
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
    buffer += "\"encoding_kind\":\"" + encoding_kind + "\","
    buffer += "\"queries_per_load\":" + String(queries_per_load) + ","
    buffer += "\"query_start\":" + String(query_start) + ","
    buffer += "\"session_seconds\":" + String(session_seconds) + ","
    buffer += "\"document_count\":" + String(document_count) + ","
    buffer += "\"vector_count\":" + String(vector_count) + ","
    buffer += "\"vector_dim\":" + String(vector_dim)
    buffer += "}"
    return buffer^


def main() raises:
    var py_task = load_raw_python_task(INPUT_TASK_PATH)
    var task = decode_judged_task(py_task)
    if len(task.queries) == 0:
        raise Error("cannot benchmark a task with zero queries")
    var config = load_single_session_config(INPUT_PARAMS_PATH)
    var query_start = config.query_start
    while query_start >= len(task.queries):
        query_start -= len(task.queries)
    var backend = ExactCpuBackend()
    var root = Path(ARTIFACT_ROOT) / config.encoding_kind

    def session_once() capturing raises:
        var loaded_index = load_stored_packed_index(root)
        var query_index = query_start
        for _ in range(config.queries_per_load):
            bench_compiler.keep(
                search_exact(
                    backend,
                    task.queries[query_index].query,
                    loaded_index.index,
                    task.k,
                )
            )
            query_index += 1
            if query_index == len(task.queries):
                query_index = 0

    var report = run[session_once](
        num_warmup_iters=0,
        max_iters=1,
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    var loaded_index = load_stored_packed_index(root)
    var output_path = Path(OUTPUT_SUMMARY_PATH)
    output_path.write_text(
        summary_json(
            String(py=py_task["dataset_id"]),
            String(py=py_task["model_name"]),
            task.family.copy(),
            task.slice_name.copy(),
            config.encoding_kind.copy(),
            config.queries_per_load,
            query_start,
            Float64(report.mean()),
            loaded_index.index.document_count,
            loaded_index.index.total_vector_count,
            loaded_index.index.vector_dim,
        )
    )
    print("wrote ", String(output_path))
