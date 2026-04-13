from std.collections import List


def json_escape(text: String) -> String:
    var escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\"", "\\\"")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


def append_json_string_list(mut buffer: String, read values: List[String]):
    buffer += "["
    for index in range(len(values)):
        if index > 0:
            buffer += ","
        buffer += "\""
        buffer += json_escape(values[index])
        buffer += "\""
    buffer += "]"
