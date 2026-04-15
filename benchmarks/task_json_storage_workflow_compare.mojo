from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.python import Python, PythonObject

from kayak.index import pack_documents
from kayak.interop.python_task_decoder import decode_judged_task
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    load_stored_packed_index,
    save_stored_packed_index_with_encoding,
)


comptime INPUT_TASK_PATH = ".cache/kayak/task_json_storage_workflow_compare_input.json"
comptime INPUT_QUERY_COUNTS_PATH = (
    ".cache/kayak/task_json_storage_workflow_compare_query_counts.json"
)
comptime OUTPUT_SUMMARIES_PATH = (
    ".cache/kayak/task_json_storage_workflow_compare_output.json"
)
comptime ARTIFACT_ROOT = ".cache/kayak/task_json_storage_workflow_compare_artifact"
comptime SESSION_BENCH_MIN_SECONDS = 0.05
comptime SESSION_BENCH_MAX_SECONDS = 0.3
comptime SESSION_BENCH_MAX_ITERS = 50


struct StorageWorkflowSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var encoding_kind: String
    var queries_per_load: Int
    var mean_session_seconds: Float64
    var mean_session_seconds_per_query: Float64
    var query_count: Int
    var document_count: Int
    var vector_count: Int
    var vector_dim: Int

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var encoding_kind: String,
        queries_per_load: Int,
        mean_session_seconds: Float64,
        mean_session_seconds_per_query: Float64,
        query_count: Int,
        document_count: Int,
        vector_count: Int,
        vector_dim: Int,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.encoding_kind = encoding_kind^
        self.queries_per_load = queries_per_load
        self.mean_session_seconds = mean_session_seconds
        self.mean_session_seconds_per_query = mean_session_seconds_per_query
        self.query_count = query_count
        self.document_count = document_count
        self.vector_count = vector_count
        self.vector_dim = vector_dim


def load_raw_python_task(path: String) raises -> PythonObject:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.json_task_loader")
    return module.load_task_json(path)


def load_queries_per_load(path: String) raises -> List[Int]:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.task_storage_workflow")
    var py_counts = module.load_queries_per_load_json(path)
    var counts = List[Int]()
    for py_value in py_counts:
        counts.append(Int(py=py_value))
    return counts^


def append_storage_workflow_summary_json(
    mut buffer: String, read summary: StorageWorkflowSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + summary.dataset_id + "\","
    buffer += "\"model_name\":\"" + summary.model_name + "\","
    buffer += "\"family\":\"" + summary.family + "\","
    buffer += "\"slice_name\":\"" + summary.slice_name + "\","
    buffer += "\"encoding_kind\":\"" + summary.encoding_kind + "\","
    buffer += "\"queries_per_load\":" + String(summary.queries_per_load) + ","
    buffer += "\"mean_session_seconds\":"
    buffer += String(summary.mean_session_seconds) + ","
    buffer += "\"mean_session_seconds_per_query\":"
    buffer += String(summary.mean_session_seconds_per_query) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"vector_count\":" + String(summary.vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim)
    buffer += "}"


def storage_workflow_summaries_json(
    read summaries: List[StorageWorkflowSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_storage_workflow_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^


def build_storage_workflow_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    root: Path,
    encoding_kind: String,
    queries_per_load: Int,
) raises -> StorageWorkflowSummary:
    if queries_per_load <= 0:
        raise Error("queries_per_load must be positive")

    var task = stored_task.task.copy()
    if len(task.queries) == 0:
        raise Error("cannot benchmark workflow with zero queries")

    var session_query_start = 0

    def session_once() capturing raises:
        var loaded_index = load_stored_packed_index(root)
        var query_index = session_query_start
        for _ in range(queries_per_load):
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

        session_query_start += queries_per_load
        while session_query_start >= len(task.queries):
            session_query_start -= len(task.queries)

    var report = run[session_once](
        num_warmup_iters=1,
        max_iters=SESSION_BENCH_MAX_ITERS,
        min_runtime_secs=SESSION_BENCH_MIN_SECONDS,
        max_runtime_secs=SESSION_BENCH_MAX_SECONDS,
    )
    var loaded_index = load_stored_packed_index(root)

    return StorageWorkflowSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        encoding_kind.copy(),
        queries_per_load,
        Float64(report.mean()),
        Float64(report.mean()) / Float64(queries_per_load),
        len(task.queries),
        loaded_index.index.document_count,
        loaded_index.index.total_vector_count,
        loaded_index.index.vector_dim,
    )


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
    var queries_per_loads = load_queries_per_load(INPUT_QUERY_COUNTS_PATH)
    var benchmark_root = Path(ARTIFACT_ROOT)
    var backend = ExactCpuBackend()
    var summaries = List[StorageWorkflowSummary]()

    for encoding_kind in [
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
        VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    ]:
        var encoding_root = benchmark_root / encoding_kind
        makedirs(encoding_root, exist_ok=True)
        save_stored_packed_index_with_encoding(
            encoding_root,
            stored_index.copy(),
            encoding_kind,
        )
        for queries_per_load in queries_per_loads:
            var summary = build_storage_workflow_summary(
                backend,
                stored_task,
                encoding_root,
                encoding_kind,
                queries_per_load,
            )
            print(
                "encoding=",
                summary.encoding_kind,
                " queries_per_load=",
                summary.queries_per_load,
                " session_s=",
                summary.mean_session_seconds,
                " session_s_per_query=",
                summary.mean_session_seconds_per_query,
            )
            summaries.append(summary.copy())

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = Path(OUTPUT_SUMMARIES_PATH)
    output_path.write_text(storage_workflow_summaries_json(summaries))
    print("wrote ", String(output_path))
