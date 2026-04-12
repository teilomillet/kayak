from std.collections import List
from std.python import Python, PythonObject

from kayak.eval import JudgedTask
from kayak.text import DocumentTextCorpus

from .python_task_decoder import decode_judged_task


def load_task_json(path: String) raises -> JudgedTask:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.json_task_loader")
    return decode_judged_task(module.load_task_json(path))


def decode_document_text_corpus(py_documents: PythonObject) raises -> DocumentTextCorpus:
    var doc_ids = List[String]()
    var texts = List[String]()

    for index in range(len(py_documents)):
        var py_document = py_documents[index]
        doc_ids.append(String(py=py_document["doc_id"]))
        texts.append(String(py=py_document["text"]))

    return DocumentTextCorpus(doc_ids^, texts^)


def load_document_text_corpus_json(path: String) raises -> DocumentTextCorpus:
    Python.add_to_path("python")
    var module = Python.import_module("kayak_bridge.json_task_loader")
    var py_task = module.load_task_json(path)
    return decode_document_text_corpus(py_task["documents"])
