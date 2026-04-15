"""Normalizes metadata rows and exact-match metadata filters for stores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json


def normalize_metadata_rows(
    metadata: object | None,
    *,
    expected_length: int,
) -> tuple[dict[str, object], ...] | None:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        raise ValueError("metadata must align with documents, not be one mapping")
    if not isinstance(metadata, Sequence):
        raise ValueError("metadata must be a sequence of mappings")

    rows = tuple(_normalize_metadata_row(row) for row in metadata)
    if len(rows) != expected_length:
        raise ValueError("metadata rows must align with documents")
    return rows


def normalize_metadata_filter(
    where: object | None,
) -> dict[str, object] | None:
    if where is None:
        return None
    if not isinstance(where, Mapping):
        raise ValueError("metadata filter must be a mapping")
    return _normalize_metadata_row(where)


def matches_metadata_filter(
    metadata: dict[str, object] | None,
    where: dict[str, object] | None,
) -> bool:
    if where is None:
        return True
    if metadata is None:
        return False
    for key, expected in where.items():
        if metadata.get(key) != expected:
            return False
    return True


def _normalize_metadata_row(row: object) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("metadata rows must be mappings")
    normalized = {str(key): value for key, value in row.items()}
    try:
        payload = json.dumps(normalized, sort_keys=True)
    except TypeError as exc:
        raise ValueError("metadata rows must be JSON serializable") from exc
    return dict(json.loads(payload))
