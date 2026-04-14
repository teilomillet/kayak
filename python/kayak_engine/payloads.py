"""Parses JSON payloads for the hosted-engine HTTP service."""

from __future__ import annotations

from typing import Any


class PayloadError(ValueError):
    """Raised when an HTTP payload is malformed."""


def require_string(
    payload: dict[str, Any],
    key: str,
    *,
    default: str | None = None,
) -> str:
    if key not in payload:
        if default is not None:
            return default
        raise PayloadError(f"missing required field: {key}")
    value = payload[key]
    if not isinstance(value, str):
        raise PayloadError(f"field must be a string: {key}")
    if default is None and value == "":
        raise PayloadError(f"field must not be empty: {key}")
    return value


def require_int(
    payload: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    if key not in payload:
        if default is not None:
            return default
        raise PayloadError(f"missing required field: {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PayloadError(f"field must be an integer: {key}")
    return value


def require_bool(
    payload: dict[str, Any],
    key: str,
    *,
    default: bool | None = None,
) -> bool:
    if key not in payload:
        if default is not None:
            return default
        raise PayloadError(f"missing required field: {key}")
    value = payload[key]
    if not isinstance(value, bool):
        raise PayloadError(f"field must be a boolean: {key}")
    return value


def require_list(
    payload: dict[str, Any],
    key: str,
    *,
    default: list[Any] | None = None,
) -> list[Any]:
    if key not in payload:
        if default is not None:
            return default
        raise PayloadError(f"missing required field: {key}")
    value = payload[key]
    if not isinstance(value, list):
        raise PayloadError(f"field must be a list: {key}")
    return value


def require_query_vectors(payload: dict[str, Any]) -> list[list[float]]:
    query = require_list(payload, "query")
    if len(query) == 0:
        raise PayloadError("query must contain at least one vector")
    for row in query:
        if not isinstance(row, list) or len(row) == 0:
            raise PayloadError("query vectors must be non-empty lists")
        for value in row:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PayloadError("query vector values must be numeric")
    return query


def filter_payload_parts(
    payload: dict[str, Any],
) -> tuple[list[list[str]], list[list[str]], list[list[list[str]]]]:
    expression = payload.get("filter_expression", {"clauses": []})
    if not isinstance(expression, dict):
        raise PayloadError("filter_expression must be an object")
    clauses = expression.get("clauses", [])
    if not isinstance(clauses, list):
        raise PayloadError("filter_expression.clauses must be a list")

    clause_fields: list[list[str]] = []
    clause_operators: list[list[str]] = []
    clause_values: list[list[list[str]]] = []
    for clause in clauses:
        if not isinstance(clause, dict):
            raise PayloadError("filter_expression clauses must be objects")
        terms = clause.get("terms", [])
        if not isinstance(terms, list) or len(terms) == 0:
            raise PayloadError("filter_expression clauses must contain terms")
        fields: list[str] = []
        operators: list[str] = []
        values: list[list[str]] = []
        for term in terms:
            if not isinstance(term, dict):
                raise PayloadError("filter terms must be objects")
            field = term.get("field")
            operator = term.get("operator")
            term_values = term.get("values")
            if not isinstance(field, str) or field == "":
                raise PayloadError("filter term field must be a non-empty string")
            if not isinstance(operator, str) or operator == "":
                raise PayloadError(
                    "filter term operator must be a non-empty string"
                )
            if not isinstance(term_values, list) or len(term_values) == 0:
                raise PayloadError("filter term values must be a non-empty list")
            normalized_values: list[str] = []
            for value in term_values:
                if not isinstance(value, str) or value == "":
                    raise PayloadError(
                        "filter term values must be non-empty strings"
                    )
                normalized_values.append(value)
            fields.append(field)
            operators.append(operator)
            values.append(normalized_values)
        clause_fields.append(fields)
        clause_operators.append(operators)
        clause_values.append(values)
    return clause_fields, clause_operators, clause_values


def document_payload_parts(
    payload: dict[str, Any],
) -> tuple[
    list[str],
    list[list[list[float]]],
    list[str],
    list[list[str]],
    list[list[str]],
]:
    documents = require_list(payload, "documents")
    if len(documents) == 0:
        raise PayloadError("documents must contain at least one entry")

    doc_ids: list[str] = []
    document_vectors: list[list[list[float]]] = []
    texts: list[str] = []
    metadata_keys: list[list[str]] = []
    metadata_values: list[list[str]] = []
    for entry in documents:
        if not isinstance(entry, dict):
            raise PayloadError("each document must be an object")
        doc_id = require_string(entry, "doc_id")
        vectors = require_list(entry, "vectors")
        if len(vectors) == 0:
            raise PayloadError("document vectors must contain at least one row")
        for row in vectors:
            if not isinstance(row, list) or len(row) == 0:
                raise PayloadError("document vector rows must be non-empty lists")
            for value in row:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise PayloadError("document vector values must be numeric")
        text = require_string(entry, "text", default="")
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            raise PayloadError("document metadata must be an object")

        row_keys: list[str] = []
        row_values: list[str] = []
        for key, value in metadata.items():
            if not isinstance(key, str) or key == "":
                raise PayloadError("document metadata keys must be non-empty strings")
            if not isinstance(value, str):
                raise PayloadError("document metadata values must be strings")
            row_keys.append(key)
            row_values.append(value)

        doc_ids.append(doc_id)
        document_vectors.append(vectors)
        texts.append(text)
        metadata_keys.append(row_keys)
        metadata_values.append(row_values)

    return doc_ids, document_vectors, texts, metadata_keys, metadata_values
