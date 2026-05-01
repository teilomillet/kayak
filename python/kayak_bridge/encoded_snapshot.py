"""Sharded binary storage for large encoded late-interaction collections.

This module owns the file format and byte accounting for encoded document
snapshots. It does not own corpus parsing or model inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np


FORMAT_VERSION = 1
SHARD_DIR_PATTERN = re.compile(r"^\d{6}$")
VECTOR_DTYPES: dict[str, np.dtype] = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
}
TOKEN_ID_DTYPES: dict[str, np.dtype] = {
    "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
}


@dataclass(frozen=True, slots=True)
class SnapshotByteEstimate:
    vector_bytes: int
    token_id_bytes: int
    doc_offset_bytes: int
    payload_bytes: int

    def to_json_ready(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SnapshotShardSummary:
    shard_index: int
    document_count: int
    vector_count: int
    vector_dim: int
    vector_dtype: str
    token_id_dtype: str
    first_doc_id: str | None
    last_doc_id: str | None
    vector_bytes: int
    token_id_bytes: int
    doc_offset_bytes: int

    @property
    def payload_bytes(self) -> int:
        return self.vector_bytes + self.token_id_bytes + self.doc_offset_bytes

    def to_json_ready(self) -> dict[str, object]:
        row = asdict(self)
        row["payload_bytes"] = self.payload_bytes
        return row


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    dataset_id: str
    model_name: str
    vector_dim: int
    vector_dtype: str
    token_id_dtype: str
    document_count: int
    document_vector_count: int
    nominal_document_vector_count: int
    shard_count: int
    payload_bytes: int
    created_at_utc: str
    source: dict[str, object]
    shards: list[dict[str, object]]
    query_manifest: dict[str, object] | None = None
    paper_scale_estimate: dict[str, int] | None = None

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


class EncodedDocumentShardWriter:
    """Streams one encoded-document shard without keeping all vectors in memory."""

    def __init__(
        self,
        *,
        snapshot_root: Path,
        shard_index: int,
        vector_dim: int,
        vector_dtype: str,
        token_id_dtype: str,
    ) -> None:
        self.snapshot_root = snapshot_root
        self.shard_index = shard_index
        self.vector_dim = vector_dim
        self.vector_dtype_name = vector_dtype
        self.token_id_dtype_name = token_id_dtype
        self.vector_dtype = _require_dtype(vector_dtype, VECTOR_DTYPES, "vector_dtype")
        self.token_id_dtype = _require_dtype(
            token_id_dtype,
            TOKEN_ID_DTYPES,
            "token_id_dtype",
        )
        self.shard_dir = snapshot_root / "shards" / f"{shard_index:06d}"
        self.tmp_dir = snapshot_root / "shards" / f".{shard_index:06d}.tmp"
        if self.shard_dir.exists():
            raise FileExistsError(f"snapshot shard already exists: {self.shard_dir}")
        if self.tmp_dir.exists():
            raise FileExistsError(f"incomplete snapshot shard exists: {self.tmp_dir}")

        self.tmp_dir.mkdir(parents=True)
        self.vector_file = (self.tmp_dir / f"vectors.{_dtype_suffix(vector_dtype)}").open(
            "wb"
        )
        self.token_id_file = (
            self.tmp_dir / f"token_ids.{_dtype_suffix(token_id_dtype)}"
        ).open("wb")
        self.doc_offsets: list[int] = [0]
        self.doc_ids: list[str] = []
        self.closed = False

    @property
    def document_count(self) -> int:
        return len(self.doc_ids)

    @property
    def vector_count(self) -> int:
        return self.doc_offsets[-1]

    def append(self, *, doc_id: str, vectors: np.ndarray, token_ids: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a closed snapshot shard")
        vector_rows = np.asarray(vectors)
        token_row = np.asarray(token_ids)
        if vector_rows.ndim != 2 or int(vector_rows.shape[1]) != self.vector_dim:
            raise ValueError("encoded document vectors must be tokens x vector_dim")
        if int(token_row.reshape(-1).shape[0]) != int(vector_rows.shape[0]):
            raise ValueError("document token ids must align with vector rows")
        if np.any(token_row < 0) and self.token_id_dtype == np.dtype(np.uint32):
            raise ValueError("uint32 token ids cannot encode negative ids")

        np.ascontiguousarray(vector_rows, dtype=self.vector_dtype).tofile(
            self.vector_file
        )
        np.ascontiguousarray(token_row.reshape(-1), dtype=self.token_id_dtype).tofile(
            self.token_id_file
        )
        self.doc_ids.append(str(doc_id))
        self.doc_offsets.append(self.doc_offsets[-1] + int(vector_rows.shape[0]))

    def close(self) -> SnapshotShardSummary:
        if self.closed:
            raise RuntimeError("snapshot shard is already closed")
        self.vector_file.close()
        self.token_id_file.close()
        summary = self._summary()
        np.save(
            self.tmp_dir / "doc_offsets.u64.npy",
            np.asarray(self.doc_offsets, dtype=np.uint64),
        )
        (self.tmp_dir / "doc_ids.txt").write_text(
            "".join(f"{doc_id}\n" for doc_id in self.doc_ids),
            encoding="utf-8",
        )
        (self.tmp_dir / "stats.json").write_text(
            json.dumps(summary.to_json_ready(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.tmp_dir.rename(self.shard_dir)
        self.closed = True
        return summary

    def _summary(self) -> SnapshotShardSummary:
        estimate = estimate_snapshot_payload_bytes(
            document_count=self.document_count,
            vector_count=self.vector_count,
            vector_dim=self.vector_dim,
            vector_dtype=self.vector_dtype_name,
            token_id_dtype=self.token_id_dtype_name,
        )
        return SnapshotShardSummary(
            shard_index=self.shard_index,
            document_count=self.document_count,
            vector_count=self.vector_count,
            vector_dim=self.vector_dim,
            vector_dtype=self.vector_dtype_name,
            token_id_dtype=self.token_id_dtype_name,
            first_doc_id=self.doc_ids[0] if self.doc_ids else None,
            last_doc_id=self.doc_ids[-1] if self.doc_ids else None,
            vector_bytes=estimate.vector_bytes,
            token_id_bytes=estimate.token_id_bytes,
            doc_offset_bytes=estimate.doc_offset_bytes,
        )


def estimate_snapshot_payload_bytes(
    *,
    document_count: int,
    vector_count: int,
    vector_dim: int,
    vector_dtype: str,
    token_id_dtype: str,
) -> SnapshotByteEstimate:
    vector_np_dtype = _require_dtype(vector_dtype, VECTOR_DTYPES, "vector_dtype")
    token_np_dtype = _require_dtype(token_id_dtype, TOKEN_ID_DTYPES, "token_id_dtype")
    vector_bytes = int(vector_count) * int(vector_dim) * vector_np_dtype.itemsize
    token_id_bytes = int(vector_count) * token_np_dtype.itemsize
    doc_offset_bytes = (int(document_count) + 1) * np.dtype(np.uint64).itemsize
    return SnapshotByteEstimate(
        vector_bytes=vector_bytes,
        token_id_bytes=token_id_bytes,
        doc_offset_bytes=doc_offset_bytes,
        payload_bytes=vector_bytes + token_id_bytes + doc_offset_bytes,
    )


def load_completed_shards(snapshot_root: Path) -> list[SnapshotShardSummary]:
    shard_root = snapshot_root / "shards"
    if not shard_root.exists():
        return []
    summaries: list[SnapshotShardSummary] = []
    for shard_dir in sorted(shard_root.iterdir()):
        if not shard_dir.is_dir() or not SHARD_DIR_PATTERN.match(shard_dir.name):
            continue
        stats_path = shard_dir / "stats.json"
        if not stats_path.exists():
            continue
        row = json.loads(stats_path.read_text(encoding="utf-8"))
        summaries.append(
            SnapshotShardSummary(
                shard_index=int(row["shard_index"]),
                document_count=int(row["document_count"]),
                vector_count=int(row["vector_count"]),
                vector_dim=int(row["vector_dim"]),
                vector_dtype=str(row["vector_dtype"]),
                token_id_dtype=str(row["token_id_dtype"]),
                first_doc_id=(
                    None if row.get("first_doc_id") is None else str(row["first_doc_id"])
                ),
                last_doc_id=(
                    None if row.get("last_doc_id") is None else str(row["last_doc_id"])
                ),
                vector_bytes=int(row["vector_bytes"]),
                token_id_bytes=int(row["token_id_bytes"]),
                doc_offset_bytes=int(row["doc_offset_bytes"]),
            )
        )
    return summaries


def write_snapshot_manifest(
    *,
    snapshot_root: Path,
    dataset_id: str,
    model_name: str,
    vector_dim: int,
    vector_dtype: str,
    token_id_dtype: str,
    shards: list[SnapshotShardSummary],
    source: Mapping[str, object],
    query_manifest: Mapping[str, object] | None = None,
    paper_scale_estimate: Mapping[str, int] | None = None,
) -> SnapshotManifest:
    document_count = sum(shard.document_count for shard in shards)
    document_vector_count = sum(shard.vector_count for shard in shards)
    nominal_document_vector_count = (
        round(document_vector_count / float(document_count))
        if document_count
        else 0
    )
    manifest = SnapshotManifest(
        dataset_id=dataset_id,
        model_name=model_name,
        vector_dim=vector_dim,
        vector_dtype=vector_dtype,
        token_id_dtype=token_id_dtype,
        document_count=document_count,
        document_vector_count=document_vector_count,
        nominal_document_vector_count=nominal_document_vector_count,
        shard_count=len(shards),
        payload_bytes=sum(shard.payload_bytes for shard in shards),
        created_at_utc=datetime.now(UTC).isoformat(),
        source=dict(source),
        shards=[shard.to_json_ready() for shard in shards],
        query_manifest=None if query_manifest is None else dict(query_manifest),
        paper_scale_estimate=(
            None if paper_scale_estimate is None else dict(paper_scale_estimate)
        ),
    )
    (snapshot_root / "manifest.json").write_text(
        json.dumps(manifest.to_json_ready(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _require_dtype(
    dtype_name: str,
    allowed: Mapping[str, np.dtype],
    field_name: str,
) -> np.dtype:
    if dtype_name not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return allowed[dtype_name]


def _dtype_suffix(dtype_name: str) -> str:
    if dtype_name == "float16":
        return "f16"
    if dtype_name == "float32":
        return "f32"
    if dtype_name == "uint32":
        return "u32"
    if dtype_name == "int64":
        return "i64"
    raise ValueError(f"unsupported dtype suffix: {dtype_name}")
