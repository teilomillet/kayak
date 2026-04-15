"""Owns the public pgvector-backed late-store adapter for the Kayak SDK."""

from __future__ import annotations

from collections.abc import Mapping
import json

import numpy as np

from kayak_bridge import LateDocuments, LateIndex
from kayak_bridge.array_conversions import to_doc_ids

from .base import LateStoreStats, StoreCapabilities
from .client_types import PsycopgConnectionLike
from .external_payloads import encode_metadata_json, is_filter_pushdown_safe
from .external_records import ExternalStoredDocument, packed_index_from_records
from .metadata import (
    matches_metadata_filter,
    normalize_metadata_filter,
    normalize_metadata_rows,
)
from .pgvector_imports import require_pgvector


class PgVectorLateStore:
    """Persist late-interaction documents in a Postgres table with pgvector.

    Parameters
    ----------
    dsn:
        Connection string used when Kayak should open its own psycopg
        connection.
    connection:
        Existing psycopg connection. Pass this when your application already
        owns transaction or connection lifecycle.
    table_name:
        Table that stores one row per document.
    schema_name:
        Database schema containing ``table_name``.
    ensure_extension:
        Whether Kayak should run ``CREATE EXTENSION IF NOT EXISTS vector``
        during initialization.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        connection: PsycopgConnectionLike | None = None,
        table_name: str = "late_documents",
        schema_name: str = "public",
        ensure_extension: bool = True,
    ) -> None:
        if connection is not None and dsn is not None:
            raise ValueError("pass either `dsn` or `connection`, not both")
        if connection is None and dsn is None:
            raise ValueError("pgvector store requires either `dsn` or `connection`")

        psycopg, register_vector = require_pgvector()
        self._table_name = _normalized_identifier(table_name, "table_name")
        self._schema_name = _normalized_identifier(schema_name, "schema_name")
        self._qualified_table_name = (
            f"{_sql_identifier(self._schema_name)}.{_sql_identifier(self._table_name)}"
        )
        self._owns_connection = connection is None
        self._connection = (
            connection if connection is not None else psycopg.connect(str(dsn))
        )
        self._register_vector = register_vector
        self._ensure_extension = bool(ensure_extension)
        self._is_ready = False

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            kind="pgvector",
            persistent=True,
            supports_metadata_filter=True,
            supports_document_subset_load=True,
            supported_layouts=("packed", "hybrid_flat_dim128"),
        )

    def close(self) -> None:
        if not self._owns_connection:
            return
        close_fn = getattr(self._connection, "close", None)
        if callable(close_fn):
            close_fn()

    def __enter__(self) -> "PgVectorLateStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def stats(self) -> LateStoreStats:
        self._ensure_ready()
        if not self._table_exists():
            return LateStoreStats(
                kind="pgvector",
                document_count=0,
                total_vector_count=0,
                vector_dim=None,
                has_texts=False,
                has_metadata=False,
                storage_byte_size=None,
            )

        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    COUNT(*),
                    COALESCE(SUM(cardinality(token_matrix)), 0),
                    COALESCE(BOOL_OR(text IS NOT NULL), FALSE),
                    COALESCE(BOOL_OR(metadata_json IS NOT NULL), FALSE)
                FROM {self._qualified_table_name}
                """
            )
            aggregate_row = cursor.fetchone()
            cursor.execute(
                f"""
                SELECT token_matrix
                FROM {self._qualified_table_name}
                LIMIT 1
                """
            )
            first_row = cursor.fetchone()

        assert aggregate_row is not None
        vector_dim = (
            None
            if first_row is None
            else _vector_dim_from_token_matrix(first_row[0])
        )
        return LateStoreStats(
            kind="pgvector",
            document_count=int(aggregate_row[0]),
            total_vector_count=int(aggregate_row[1]),
            vector_dim=vector_dim,
            has_texts=bool(aggregate_row[2]),
            has_metadata=bool(aggregate_row[3]),
            storage_byte_size=None,
        )

    def upsert(
        self,
        documents: LateDocuments,
        *,
        metadata: object | None = None,
    ) -> None:
        self._ensure_ready()
        metadata_rows = normalize_metadata_rows(
            metadata,
            expected_length=documents.document_count,
        )
        self._ensure_table(documents.vector_dim)

        rows: list[tuple[object, ...]] = []
        for index, doc_id in enumerate(documents.doc_ids):
            metadata_row = None if metadata_rows is None else metadata_rows[index]
            rows.append(
                (
                    doc_id,
                    _pgvector_token_matrix(documents.token_matrices[index]),
                    None if documents.texts is None else documents.texts[index],
                    None if metadata_row is None else _metadata_json(metadata_row),
                )
            )

        try:
            with self._connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self._qualified_table_name} (
                        doc_id,
                        token_matrix,
                        text,
                        metadata_json
                    )
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        token_matrix = EXCLUDED.token_matrix,
                        text = EXCLUDED.text,
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    rows,
                )
            self._commit()
        except Exception:
            self._rollback()
            raise

    def delete(self, doc_ids: object) -> None:
        self._ensure_ready()
        selected_doc_ids = to_doc_ids(doc_ids, "store delete doc_ids")
        if not selected_doc_ids or not self._table_exists():
            return

        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._qualified_table_name} WHERE doc_id = ANY(%s)",
                    (list(selected_doc_ids),),
                )
            self._commit()
        except Exception:
            self._rollback()
            raise

    def load_index(
        self,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool = False,
        layout: str = "packed",
    ) -> LateIndex:
        self._ensure_ready()
        if not self._table_exists():
            raise ValueError("store selection did not produce any documents")

        requested_doc_ids = (
            None if doc_ids is None else to_doc_ids(doc_ids, "store load doc_ids")
        )
        where_filter = normalize_metadata_filter(where)
        records = self._selected_records(
            requested_doc_ids=requested_doc_ids,
            where_filter=where_filter,
            include_text=include_text,
        )
        if not records:
            raise ValueError("store selection did not produce any documents")

        return packed_index_from_records(
            records,
            include_text=include_text,
            layout=layout,
        )

    def _selected_records(
        self,
        *,
        requested_doc_ids: tuple[str, ...] | None,
        where_filter: dict[str, object] | None,
        include_text: bool,
    ) -> tuple[ExternalStoredDocument, ...]:
        rows = self._fetch_rows(
            requested_doc_ids=requested_doc_ids,
            where_filter=where_filter,
            include_text=include_text,
        )

        decoded: list[ExternalStoredDocument] = []
        for row in rows:
            metadata = _decode_metadata_payload(row[3])
            if not matches_metadata_filter(metadata, where_filter):
                continue

            token_matrix = np.asarray(row[1], dtype=np.float32)
            token_matrix.setflags(write=False)
            decoded.append(
                ExternalStoredDocument(
                    doc_id=str(row[0]),
                    token_matrix=token_matrix,
                    text=_optional_string(row[2]) if include_text else None,
                    metadata=metadata,
                )
            )

        if requested_doc_ids is None:
            return tuple(decoded)

        order = {doc_id: index for index, doc_id in enumerate(requested_doc_ids)}
        decoded.sort(key=lambda record: order[record.doc_id])
        return tuple(decoded)

    def _fetch_rows(
        self,
        *,
        requested_doc_ids: tuple[str, ...] | None,
        where_filter: dict[str, object] | None,
        include_text: bool,
    ) -> list[tuple[object, ...]]:
        select_text = "text" if include_text else "NULL"
        where_clauses: list[str] = []
        params: list[object] = []

        if requested_doc_ids is not None:
            where_clauses.append("doc_id = ANY(%s)")
            params.append(list(requested_doc_ids))
        if where_filter is not None and is_filter_pushdown_safe(where_filter):
            where_clauses.append("metadata_json @> %s::jsonb")
            params.append(_metadata_json(where_filter))

        query = (
            f"SELECT doc_id, token_matrix, {select_text}, metadata_json "
            f"FROM {self._qualified_table_name}"
        )
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        if requested_doc_ids is None:
            query += " ORDER BY doc_id"

        with self._connection.cursor() as cursor:
            cursor.execute(query, tuple(params))
            fetched = cursor.fetchall()
        return [tuple(row) for row in fetched]

    def _ensure_ready(self) -> None:
        if self._is_ready:
            return

        try:
            if self._ensure_extension:
                with self._connection.cursor() as cursor:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self._commit()
            self._register_vector(self._connection)
        except Exception as exc:
            self._rollback()
            raise RuntimeError(
                "PgVectorLateStore could not initialize pgvector support. "
                "Make sure the `vector` extension is installed in the target "
                "database, or create it yourself and pass `ensure_extension=False`."
            ) from exc

        self._is_ready = True

    def _ensure_table(self, vector_dim: int) -> None:
        if self._table_exists():
            stored_vector_dim = self._stored_vector_dim()
            if stored_vector_dim is not None and stored_vector_dim != vector_dim:
                raise ValueError(
                    "pgvector store vector dimension must stay stable across upserts"
                )
            return

        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._qualified_table_name} (
                        doc_id TEXT PRIMARY KEY,
                        token_matrix vector({int(vector_dim)})[] NOT NULL,
                        text TEXT,
                        metadata_json JSONB
                    )
                    """
                )
            self._commit()
        except Exception:
            self._rollback()
            raise

    def _table_exists(self) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass(%s)",
                (_regclass_name(self._schema_name, self._table_name),),
            )
            row = cursor.fetchone()
        return bool(row and row[0] is not None)

    def _stored_vector_dim(self) -> int | None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT token_matrix
                FROM {self._qualified_table_name}
                LIMIT 1
                """
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _vector_dim_from_token_matrix(row[0])

    def _commit(self) -> None:
        commit_fn = getattr(self._connection, "commit", None)
        if callable(commit_fn):
            commit_fn()

    def _rollback(self) -> None:
        rollback_fn = getattr(self._connection, "rollback", None)
        if callable(rollback_fn):
            rollback_fn()


def _normalized_identifier(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if normalized == "":
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _regclass_name(schema_name: str, table_name: str) -> str:
    return f"{_sql_identifier(schema_name)}.{_sql_identifier(table_name)}"


def _pgvector_token_matrix(token_matrix: np.ndarray) -> list[np.ndarray]:
    matrix = np.asarray(token_matrix, dtype=np.float32)
    return [np.asarray(row, dtype=np.float32) for row in matrix]


def _vector_dim_from_token_matrix(token_matrix: object) -> int:
    matrix = np.asarray(token_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] <= 0:
        raise ValueError("stored pgvector token matrix must be a 2D matrix")
    return int(matrix.shape[1])


def _metadata_json(metadata: dict[str, object]) -> str:
    payload = encode_metadata_json(metadata)
    assert payload is not None
    return payload


def _decode_metadata_payload(payload: object) -> dict[str, object] | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return dict(json.loads(payload))
    if isinstance(payload, Mapping):
        return dict(json.loads(json.dumps(dict(payload), sort_keys=True)))
    raise ValueError("stored pgvector metadata payload must be JSON-compatible")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
