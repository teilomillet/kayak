from std.collections import List
from std.pathlib import Path

from kayak.numeric import VectorScalar


def normalize_inline_text(text: String) -> String:
    var normalized = text.replace("\r", " ")
    normalized = normalized.replace("\n", " ")
    normalized = normalized.replace("\t", " ")
    return normalized


def append_line(mut buffer: String, line: String):
    buffer += line
    buffer += "\n"


def read_non_empty_lines(path: Path) raises -> List[String]:
    var lines = List[String]()

    # Mojo's splitlines() currently tokenizes TSV payloads on tab characters.
    # Storage artifacts are newline-delimited, so split only on "\n".
    for line in path.read_text().split("\n"):
        var text = String(line).replace("\r", "")
        if text.byte_length() == 0:
            continue

        lines.append(text^)

    return lines^


def split_fields(line: String, delimiter: String) -> List[String]:
    var fields = List[String]()
    for field in line.split(delimiter):
        fields.append(String(field))
    return fields^


def split_tab_fields(
    line: String, expected_field_count: Int, owner: String
) raises -> List[String]:
    var fields = split_fields(line, "\t")
    if len(fields) != expected_field_count:
        raise Error(
            owner
            + " expected "
            + String(expected_field_count)
            + " tab-separated fields"
        )

    return fields^


def parse_int(text: String, owner: String) raises -> Int:
    try:
        return Int(text)
    except:
        raise Error(owner + " is not a valid integer: " + text)


def parse_vector_scalar(text: String, owner: String) raises -> VectorScalar:
    try:
        return VectorScalar(atof(text))
    except:
        raise Error(owner + " is not a valid vector scalar: " + text)


def encode_vector_line(vector: List[VectorScalar]) -> String:
    var line = String()

    for index in range(len(vector)):
        if index > 0:
            line += ","

        line += String(vector[index])

    return line^


def decode_vector_line(line: String) raises -> List[VectorScalar]:
    var values = List[VectorScalar]()

    for field in split_fields(line, ","):
        values.append(parse_vector_scalar(field, "vector entry"))

    if len(values) == 0:
        raise Error("vector line must contain at least one value")

    return values^
