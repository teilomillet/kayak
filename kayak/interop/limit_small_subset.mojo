from std.python import Python

from kayak.eval import JudgedTask

from .python_task_decoder import decode_judged_task


def load_limit_small_real_subset(
    query_limit: Int = 32,
    model_name: String = "colbert-ir/colbertv2.0",
) raises -> JudgedTask:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.limit_subset")
    var py_task = module.build_limit_small_colbert_subset(query_limit, model_name)
    return decode_judged_task(py_task)
