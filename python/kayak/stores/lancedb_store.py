"""Owns the public LanceDB-backed late-store adapter for the Kayak SDK."""

from __future__ import annotations

from pathlib import Path

from kayak_bridge import LateDocuments, LateIndex
from kayak_bridge.array_conversions import to_doc_ids

from .base import LateStoreStats, StoreCapabilities
from .lancedb_arrow import (
    DOC_ID_COLUMN,
    arrow_table_to_packed_index,
    documents_to_arrow_table,
    filter_arrow_table_by_metadata,
    lancedb_schema,
    reorder_arrow_table_by_doc_ids,
    vector_dim_from_schema,
)
from .lancedb_imports import require_lancedb
from .metadata import normalize_metadata_filter, normalize_metadata_rows


class LanceDBLateStore:
    """Persists late-interaction documents in LanceDB row storage."""

    def __init__(
        self,
        path: str | Path,
        *,
        table_name: str = "late_documents",
    ) -> None:
        require_lancedb()
        self._root = Path(path)
        self._table_name = str(table_name)

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            kind="lancedb",
            persistent=True,
            supports_metadata_filter=True,
            supports_document_subset_load=True,
            supported_layouts=("packed", "hybrid_flat_dim128"),
        )

    def stats(self) -> LateStoreStats:
        table = self._open_table()
        if table is None:
            return LateStoreStats(
                kind="lancedb",
                document_count=0,
                total_vector_count=0,
                vector_dim=None,
                has_texts=False,
                has_metadata=False,
                storage_byte_size=0,
            )

        arrow_table = table.to_arrow()
        token_column = arrow_table.column("token_matrix").combine_chunks()
        metadata_values = arrow_table.column("metadata_json").combine_chunks().to_pylist()
        text_values = arrow_table.column("text").combine_chunks().to_pylist()

        return LateStoreStats(
            kind="lancedb",
            document_count=int(arrow_table.num_rows),
            total_vector_count=int(token_column.offsets[-1].as_py()),
            vector_dim=vector_dim_from_schema(table.schema),
            has_texts=any(value is not None for value in text_values),
            has_metadata=any(value is not None for value in metadata_values),
            storage_byte_size=_directory_byte_size(self._root),
        )

    def upsert(
        self,
        documents: LateDocuments,
        *,
        metadata: object | None = None,
    ) -> None:
        _, pa = require_lancedb()
        metadata_rows = normalize_metadata_rows(
            metadata,
            expected_length=documents.document_count,
        )
        arrow_table = documents_to_arrow_table(
            pa,
            documents,
            metadata_rows=metadata_rows,
        )

        table = self._open_table()
        if table is None:
            self._db().create_table(
                self._table_name,
                data=arrow_table,
                schema=lancedb_schema(pa, vector_dim=documents.vector_dim),
                mode="overwrite",
            )
            return

        existing_vector_dim = vector_dim_from_schema(table.schema)
        if existing_vector_dim != documents.vector_dim:
            raise ValueError(
                "LanceDB store vector dimension must stay stable across upserts"
            )

        table.merge_insert(DOC_ID_COLUMN).when_matched_update_all().when_not_matched_insert_all().execute(
            arrow_table
        )

    def delete(self, doc_ids: object) -> None:
        table = self._open_table()
        if table is None:
            return

        selected_doc_ids = to_doc_ids(doc_ids, "store delete doc_ids")
        table.delete(_doc_id_where_clause(selected_doc_ids))

    def load_index(
        self,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool = False,
        layout: str = "packed",
    ) -> LateIndex:
        _, pa = require_lancedb()
        table = self._open_table()
        if table is None:
            raise ValueError("store selection did not produce any documents")

        requested_doc_ids = (
            None if doc_ids is None else to_doc_ids(doc_ids, "store load doc_ids")
        )
        metadata_filter = normalize_metadata_filter(where)
        arrow_table = self._load_arrow_table(requested_doc_ids=requested_doc_ids)
        arrow_table = filter_arrow_table_by_metadata(
            pa,
            arrow_table,
            where=metadata_filter,
        )
        if requested_doc_ids is not None:
            arrow_table = reorder_arrow_table_by_doc_ids(
                pa,
                arrow_table,
                doc_ids=requested_doc_ids,
            )

        if arrow_table.num_rows <= 0:
            raise ValueError("store selection did not produce any documents")

        return arrow_table_to_packed_index(
            arrow_table,
            include_text=include_text,
        ).to_layout(layout)

    def _load_arrow_table(
        self,
        *,
        requested_doc_ids: tuple[str, ...] | None,
    ) -> object:
        table = self._open_table()
        if table is None:
            raise ValueError("store selection did not produce any documents")

        if requested_doc_ids is None:
            return table.to_arrow()
        return table.search().where(_doc_id_where_clause(requested_doc_ids)).to_arrow()

    def _db(self) -> object:
        lancedb, _ = require_lancedb()
        self._root.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(self._root))

    def _open_table(self) -> object | None:
        db = self._db()
        table_names = set(str(name) for name in db.list_tables().tables)
        if self._table_name not in table_names:
            return None
        return db.open_table(self._table_name)


def _doc_id_where_clause(doc_ids: tuple[str, ...]) -> str:
    quoted = ", ".join(_sql_quote(doc_id) for doc_id in doc_ids)
    return f"{DOC_ID_COLUMN} IN ({quoted})"


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _directory_byte_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
