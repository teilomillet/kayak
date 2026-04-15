"""Parses JSON payloads for the hosted-engine HTTP service."""

from __future__ import annotations

from typing import Any

from .prepared_exact_types import (
    ExactScoringOptions,
    PreparedExactSearchRuntimeConfig,
    _runtime_config,
)


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


def require_object(
    payload: dict[str, Any],
    key: str,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if key not in payload:
        if default is not None:
            return default
        raise PayloadError(f"missing required field: {key}")
    value = payload[key]
    if not isinstance(value, dict):
        raise PayloadError(f"field must be an object: {key}")
    return value


def require_string_list(
    payload: dict[str, Any],
    key: str,
    *,
    default: list[str] | None = None,
) -> list[str]:
    values = require_list(payload, key, default=default)
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or value == "":
            raise PayloadError(f"field must contain non-empty strings: {key}")
        normalized.append(value)
    return normalized


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


def collection_identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "collection_id": require_string(payload, "collection_id"),
        "tenant_id": require_string(payload, "tenant_id"),
        "namespace_id": require_string(payload, "namespace_id"),
    }


def policy_override_payload(
    payload: dict[str, Any],
) -> tuple[bool, int, list[str]]:
    override = payload.get("policy_override")
    if override is None:
        return False, 0, []
    if not isinstance(override, dict):
        raise PayloadError("policy_override must be an object")
    return (
        True,
        require_int(override, "keep_latest_inactive_count"),
        require_string_list(override, "pinned_snapshot_ids", default=[]),
    )


def delete_documents_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **collection_identity_payload(payload),
        "doc_ids": require_string_list(payload, "doc_ids"),
    }


def retention_update_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **collection_identity_payload(payload),
        "default_keep_latest_inactive_count": require_int(
            payload,
            "default_keep_latest_inactive_count",
        ),
    }


def export_snapshot_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    bundle_uri = require_string(payload, "bundle_uri")
    if not bundle_uri.startswith("file://"):
        raise PayloadError("bundle_uri must use the file:// scheme")
    return {
        **collection_identity_payload(payload),
        "snapshot_id": require_string(payload, "snapshot_id"),
        "bundle_root": bundle_uri.removeprefix("file://"),
    }


def import_snapshot_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    source_uri = require_string(payload, "source_uri")
    if not source_uri.startswith("file://"):
        raise PayloadError("source_uri must use the file:// scheme")
    return {
        **collection_identity_payload(payload),
        "snapshot_id": require_string(payload, "snapshot_id"),
        "source_uri": source_uri,
    }


def lifecycle_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    (
        has_policy_override,
        keep_latest_inactive_count,
        pinned_snapshot_ids,
    ) = policy_override_payload(payload)
    return {
        **collection_identity_payload(payload),
        "has_policy_override": has_policy_override,
        "policy_override_keep_latest_inactive_count": keep_latest_inactive_count,
        "policy_override_pinned_snapshot_ids": pinned_snapshot_ids,
    }


def exact_search_request_payload(
    payload: dict[str, Any],
    *,
    debug_mode_default: bool = False,
) -> dict[str, Any]:
    clause_fields, clause_operators, clause_values = filter_payload_parts(payload)
    return {
        **collection_identity_payload(payload),
        "snapshot_id": require_string(payload, "snapshot_id"),
        "query": require_query_vectors(payload),
        "query_model_name": require_string(payload, "query_model_name"),
        "query_text": require_string(payload, "query_text", default=""),
        "final_k": require_int(payload, "final_k"),
        "debug_mode": require_bool(
            payload,
            "debug_mode",
            default=debug_mode_default,
        ),
        "clause_fields": clause_fields,
        "clause_operators": clause_operators,
        "clause_values": clause_values,
    }


def exact_scoring_options_payload(
    payload: dict[str, Any],
    key: str,
) -> ExactScoringOptions:
    scoring = require_object(payload, key, default={})
    return ExactScoringOptions(
        enable_parallel_scoring=require_bool(
            scoring,
            "enable_parallel_scoring",
            default=True,
        ),
        enable_dim128_fast_path=require_bool(
            scoring,
            "enable_dim128_fast_path",
            default=True,
        ),
        enable_parallel_work_item_oversubscription=require_bool(
            scoring,
            "enable_parallel_work_item_oversubscription",
            default=True,
        ),
        parallel_work_item_count_override=require_int(
            scoring,
            "parallel_work_item_count_override",
            default=0,
        ),
    )


def prepared_exact_runtime_config_payload(
    payload: dict[str, Any],
    key: str = "config",
) -> PreparedExactSearchRuntimeConfig:
    config = require_object(payload, key, default={})
    return _runtime_config(
        PreparedExactSearchRuntimeConfig(
            execution_backend=require_string(
                config,
                "execution_backend",
                default="process",
            ),
            concurrency_lane_count=require_int(
                config,
                "concurrency_lane_count",
                default=1,
            ),
            worker_count=require_int(
                config,
                "worker_count",
                default=1,
            ),
            max_batch_size=require_int(
                config,
                "max_batch_size",
                default=32,
            ),
            max_batch_wait_ms=require_int(
                config,
                "max_batch_wait_ms",
                default=1,
            ),
            max_outstanding_request_count=require_int(
                config,
                "max_outstanding_request_count",
                default=0,
            ),
            scoring=exact_scoring_options_payload(config, "scoring"),
        )
    )


def _reclaim_decision_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PayloadError("reclaim plan decisions must be objects")
    return {
        "snapshot_id": require_string(value, "snapshot_id"),
        "generation": require_int(value, "generation"),
        "segment_count": require_int(value, "segment_count"),
        "byte_size": require_int(value, "byte_size"),
        "retain": require_bool(value, "retain"),
        "reason": require_string(value, "reason"),
    }


def require_reclaim_plan_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    plan = require_object(payload, key)
    decisions = require_list(plan, "decisions")
    return {
        "collection_id": require_string(plan, "collection_id"),
        "tenant_id": require_string(plan, "tenant_id"),
        "namespace_id": require_string(plan, "namespace_id"),
        "active_snapshot_id": require_string(plan, "active_snapshot_id"),
        "total_snapshot_count": require_int(plan, "total_snapshot_count"),
        "inactive_snapshot_count": require_int(plan, "inactive_snapshot_count"),
        "retained_inactive_snapshot_count": require_int(
            plan,
            "retained_inactive_snapshot_count",
        ),
        "reclaimable_snapshot_count": require_int(
            plan,
            "reclaimable_snapshot_count",
        ),
        "reclaimable_unique_segment_count": require_int(
            plan,
            "reclaimable_unique_segment_count",
        ),
        "reclaimable_unique_byte_size": require_int(
            plan,
            "reclaimable_unique_byte_size",
        ),
        "decisions": [_reclaim_decision_payload(decision) for decision in decisions],
        "reclaimable_unique_segment_ids": require_string_list(
            plan,
            "reclaimable_unique_segment_ids",
            default=[],
        ),
    }


def execute_reclaim_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **collection_identity_payload(payload),
        "dry_run": require_bool(payload, "dry_run", default=True),
        "plan": require_reclaim_plan_payload(payload, "plan"),
    }
