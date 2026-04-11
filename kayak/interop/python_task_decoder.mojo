from std.collections import List
from std.python import PythonObject

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.eval import JudgedQuery, JudgedTask
from kayak.numeric import VectorScalar


def decode_string_list(py_values: PythonObject) raises -> List[String]:
    var values = List[String]()

    for index in range(len(py_values)):
        values.append(String(py=py_values[index]))

    return values^


def decode_float_vector(py_values: PythonObject) raises -> List[VectorScalar]:
    var values = List[VectorScalar]()

    for index in range(len(py_values)):
        values.append(VectorScalar(py=py_values[index]))

    return values^


def decode_float_vectors(py_values: PythonObject) raises -> List[List[VectorScalar]]:
    var vectors = List[List[VectorScalar]]()

    for index in range(len(py_values)):
        vectors.append(decode_float_vector(py_values[index]))

    return vectors^


def decode_documents(py_documents: PythonObject) raises -> List[EncodedDocument]:
    var documents = List[EncodedDocument]()

    for index in range(len(py_documents)):
        var py_document = py_documents[index]
        documents.append(
            EncodedDocument(
                String(py=py_document["doc_id"]),
                decode_float_vectors(py_document["vectors"]),
            )
        )

    return documents^


def decode_queries(py_queries: PythonObject) raises -> List[JudgedQuery]:
    var queries = List[JudgedQuery]()

    for index in range(len(py_queries)):
        var py_query = py_queries[index]
        queries.append(
            JudgedQuery(
                String(py=py_query["query_id"]),
                String(py=py_query["text"]),
                EncodedQuery(decode_float_vectors(py_query["vectors"])),
                decode_string_list(py_query["relevant_doc_ids"]),
            )
        )

    return queries^


def decode_judged_task(py_task: PythonObject) raises -> JudgedTask:
    return JudgedTask(
        String(py=py_task["family"]),
        String(py=py_task["slice_name"]),
        String(py=py_task["why"]),
        String(py=py_task["primary_metric"]),
        Int(py=py_task["k"]),
        Int(py=py_task["nominal_query_vector_count"]),
        Int(py=py_task["nominal_document_vector_count"]),
        Int(py=py_task["vector_dim"]),
        decode_documents(py_task["documents"]),
        decode_queries(py_task["queries"]),
    )
