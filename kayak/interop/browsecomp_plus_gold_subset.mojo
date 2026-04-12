from std.python import Python

from kayak.eval import JudgedTask

from .json_task import load_task_json
from .python_task_decoder import decode_judged_task


def load_browsecomp_plus_gold_real_subset(
    query_limit: Int = 4,
    negative_doc_limit: Int = 16,
    model_name: String = "colbert-ir/colbertv2.0",
) raises -> JudgedTask:
    if (
        query_limit == 4
        and negative_doc_limit == 16
        and model_name == "colbert-ir/colbertv2.0"
    ):
        return load_task_json(
            ".cache/kayak/browsecomp_plus_real_subset/python_task_gold.json"
        )

    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.browsecomp_plus_subset")
    var py_task = module.build_browsecomp_plus_gold_colbert_subset(
        query_limit, negative_doc_limit, model_name
    )
    return decode_judged_task(py_task)
