from std.python import Python

from kayak.eval import JudgedTask

from .python_task_decoder import decode_judged_task


def load_task_json(path: String) raises -> JudgedTask:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.json_task_loader")
    return decode_judged_task(module.load_task_json(path))
