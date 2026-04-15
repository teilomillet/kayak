from std.python import Python

from kayak.eval import JudgedTask

from .python_task_decoder import decode_judged_task


def load_r2med_biology_real_subset(
    query_limit: Int = 8,
    negative_doc_limit: Int = 128,
    model_name: String = "colbert-ir/colbertv2.0",
) raises -> JudgedTask:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.r2med_biology_subset")
    var py_task = module.build_r2med_biology_colbert_subset(
        query_limit, negative_doc_limit, model_name
    )
    return decode_judged_task(py_task)
