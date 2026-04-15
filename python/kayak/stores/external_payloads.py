"""Shared payload helpers for external vector-database store adapters.

This module owns:
- stable reserved field names used inside third-party stores
- JSON round-tripping of full metadata rows
- a conservative set of metadata fields that can be pushed down as native
  scalar filters in backends that support them

It does not own backend-specific query syntax or collection configuration.
"""

from __future__ import annotations

import json


INTERNAL_DOC_ID_KEY = "__kayak_doc_id"
INTERNAL_TEXT_KEY = "__kayak_text"
INTERNAL_METADATA_JSON_KEY = "__kayak_metadata_json"
INTERNAL_VECTOR_COUNT_KEY = "__kayak_vector_count"
INTERNAL_VECTOR_DIM_KEY = "__kayak_vector_dim"
INTERNAL_TOKEN_MATRIX_JSON_KEY = "__kayak_token_matrix_json"


def encode_metadata_json(metadata: dict[str, object] | None) -> str | None:
    if metadata is None:
        return None
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def decode_metadata_json(payload: object) -> dict[str, object] | None:
    if payload is None:
        return None
    if not isinstance(payload, str):
        raise ValueError("stored metadata payload must be a JSON string")
    return dict(json.loads(payload))


def filterable_metadata_items(
    metadata: dict[str, object] | None,
) -> dict[str, str | bool | int | float]:
    if metadata is None:
        return {}

    pushed: dict[str, str | bool | int | float] = {}
    for key, value in metadata.items():
        if key.startswith("__kayak_"):
            continue
        if isinstance(value, bool):
            pushed[key] = value
            continue
        if isinstance(value, (str, int, float)):
            pushed[key] = value
    return pushed


def is_filter_pushdown_safe(where: dict[str, object] | None) -> bool:
    if where is None:
        return True
    for key, value in where.items():
        if key.startswith("__kayak_"):
            return False
        if isinstance(value, bool):
            continue
        if not isinstance(value, (str, int, float)):
            return False
    return True
