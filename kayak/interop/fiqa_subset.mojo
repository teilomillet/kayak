from std.python import Python

from kayak.eval import JudgedTask

from .python_task_decoder import decode_judged_task


def load_fiqa_real_subset(
    query_limit: Int = 6,
    negative_doc_limit: Int = 64,
    model_name: String = "colbert-ir/colbertv2.0",
) raises -> JudgedTask:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.fiqa_subset")
    var py_task = module.build_fiqa_colbert_subset(
        query_limit, negative_doc_limit, model_name
    )
    return decode_judged_task(py_task)
