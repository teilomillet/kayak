"""Streams MS MARCO ColBERTv2 embeddings into a sharded binary snapshot.

This module owns corpus-specific parsing and model batching for MS MARCO
passage materialization. It does not own search or Tachiom index construction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .colbert_encoder import (
    DEFAULT_MODEL_NAME,
    encode_document_texts_with_token_id_tensors,
    encode_query_texts_as_tensors,
)
from .encoded_snapshot import (
    EncodedDocumentShardWriter,
    SnapshotManifest,
    SnapshotShardSummary,
    estimate_snapshot_payload_bytes,
    load_completed_shards,
    write_snapshot_manifest,
)
from .msmarco_passage_task import (
    MsmarcoPassagePaths,
    choose_query_ids,
    load_positive_doc_ids_by_query,
    load_queries_by_id,
    parse_tsv_id_text_line,
)


PAPER_MSMARCO_PASSAGE_COUNT = 8_841_823
PAPER_MSMARCO_TOKEN_VECTOR_COUNT = 598_000_000
DEFAULT_DATASET_ID = "msmarco-passage/dev/small"

DocumentBatchEncoder = Callable[
    [list[str], str, int],
    Sequence[tuple[np.ndarray, np.ndarray]],
]
QueryBatchEncoder = Callable[
    [list[str], str, int],
    Sequence[np.ndarray],
]


def build_msmarco_colbert_snapshot(
    *,
    paths: MsmarcoPassagePaths,
    output: Path,
    document_limit: int | None = None,
    query_limit: int | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    vector_dim: int = 128,
    vector_dtype: str = "float16",
    token_id_dtype: str = "uint32",
    document_batch_size: int = 16,
    query_batch_size: int = 64,
    shard_max_vectors: int = 4_000_000,
    include_query_positives: bool = False,
    resume: bool = False,
    encode_document_batch: DocumentBatchEncoder | None = None,
    encode_query_batch: QueryBatchEncoder | None = None,
) -> SnapshotManifest:
    _validate_limits(
        document_limit=document_limit,
        query_limit=query_limit,
        document_batch_size=document_batch_size,
        query_batch_size=query_batch_size,
        shard_max_vectors=shard_max_vectors,
    )
    output = output.resolve()
    _prepare_output_root(output, resume=resume)
    required_positive_doc_ids = _positive_doc_ids_for_selected_queries(
        paths,
        query_limit=query_limit,
    ) if include_query_positives else set()
    if resume and required_positive_doc_ids:
        raise ValueError("resume is not supported with include_query_positives")
    document_encoder = encode_document_batch or _default_encode_document_batch
    query_encoder = encode_query_batch or _default_encode_query_batch

    completed_shards = load_completed_shards(output) if resume else []
    _validate_completed_shards(
        completed_shards,
        vector_dim=vector_dim,
        vector_dtype=vector_dtype,
        token_id_dtype=token_id_dtype,
    )
    new_shards = _materialize_document_shards(
        paths=paths,
        output=output,
        completed_shards=completed_shards,
        document_limit=document_limit,
        required_doc_ids=required_positive_doc_ids,
        model_name=model_name,
        vector_dim=vector_dim,
        vector_dtype=vector_dtype,
        token_id_dtype=token_id_dtype,
        document_batch_size=document_batch_size,
        shard_max_vectors=shard_max_vectors,
        encode_document_batch=document_encoder,
    )
    query_manifest = _materialize_queries(
        paths=paths,
        output=output,
        query_limit=query_limit,
        model_name=model_name,
        vector_dim=vector_dim,
        query_batch_size=query_batch_size,
        resume=resume,
        encode_query_batch=query_encoder,
    )
    shards = completed_shards + new_shards
    return write_snapshot_manifest(
        snapshot_root=output,
        dataset_id=dataset_id,
        model_name=model_name,
        vector_dim=vector_dim,
        vector_dtype=vector_dtype,
        token_id_dtype=token_id_dtype,
        shards=shards,
        source={
            "collection": str(paths.collection),
            "queries": str(paths.queries),
            "qrels": str(paths.qrels),
            "document_limit": document_limit,
            "query_limit": query_limit,
            "document_batch_size": document_batch_size,
            "query_batch_size": query_batch_size,
            "shard_max_vectors": shard_max_vectors,
            "include_query_positives": include_query_positives,
            "required_positive_doc_count": len(required_positive_doc_ids),
            "resumable": True,
        },
        query_manifest=query_manifest,
        paper_scale_estimate=estimate_paper_msmarco_snapshot_bytes(
            vector_dim=vector_dim,
            vector_dtype=vector_dtype,
            token_id_dtype=token_id_dtype,
        ),
    )


def estimate_paper_msmarco_snapshot_bytes(
    *,
    vector_dim: int = 128,
    vector_dtype: str = "float16",
    token_id_dtype: str = "uint32",
) -> dict[str, int]:
    estimate = estimate_snapshot_payload_bytes(
        document_count=PAPER_MSMARCO_PASSAGE_COUNT,
        vector_count=PAPER_MSMARCO_TOKEN_VECTOR_COUNT,
        vector_dim=vector_dim,
        vector_dtype=vector_dtype,
        token_id_dtype=token_id_dtype,
    )
    row = estimate.to_json_ready()
    row["paper_document_count"] = PAPER_MSMARCO_PASSAGE_COUNT
    row["paper_document_vector_count"] = PAPER_MSMARCO_TOKEN_VECTOR_COUNT
    return row


def _positive_doc_ids_for_selected_queries(
    paths: MsmarcoPassagePaths,
    *,
    query_limit: int | None,
) -> set[str]:
    positives_by_query = load_positive_doc_ids_by_query(paths.qrels)
    queries_by_id = load_queries_by_id(paths.queries)
    query_ids = choose_query_ids(
        positives_by_query,
        queries_by_id,
        query_limit=query_limit,
    )
    return {
        str(doc_id)
        for query_id in query_ids
        for doc_id in positives_by_query[query_id]
    }


def _materialize_document_shards(
    *,
    paths: MsmarcoPassagePaths,
    output: Path,
    completed_shards: Sequence[SnapshotShardSummary],
    document_limit: int | None,
    required_doc_ids: set[str],
    model_name: str,
    vector_dim: int,
    vector_dtype: str,
    token_id_dtype: str,
    document_batch_size: int,
    shard_max_vectors: int,
    encode_document_batch: DocumentBatchEncoder,
) -> list[SnapshotShardSummary]:
    completed_document_count = sum(shard.document_count for shard in completed_shards)
    if document_limit is not None and completed_document_count >= document_limit:
        return []

    next_shard_index = (
        max((shard.shard_index for shard in completed_shards), default=-1) + 1
    )
    writer: EncodedDocumentShardWriter | None = None
    new_shards: list[SnapshotShardSummary] = []
    batch_doc_ids: list[str] = []
    batch_texts: list[str] = []
    source_document_count = 0
    selected_document_count = 0
    seen_required_doc_ids: set[str] = set()

    def flush_batch() -> None:
        nonlocal writer, next_shard_index
        if not batch_texts:
            return
        encoded_batch = encode_document_batch(
            batch_texts,
            model_name,
            document_batch_size,
        )
        if len(encoded_batch) != len(batch_texts):
            raise ValueError("document encoder returned the wrong batch length")
        for doc_id, (vectors, token_ids) in zip(
            batch_doc_ids,
            encoded_batch,
            strict=True,
        ):
            vector_count = int(np.asarray(vectors).shape[0])
            if writer is None:
                writer = EncodedDocumentShardWriter(
                    snapshot_root=output,
                    shard_index=next_shard_index,
                    vector_dim=vector_dim,
                    vector_dtype=vector_dtype,
                    token_id_dtype=token_id_dtype,
                )
            would_cross = (
                writer.document_count > 0
                and writer.vector_count + vector_count > shard_max_vectors
            )
            if would_cross:
                new_shards.append(writer.close())
                next_shard_index += 1
                writer = EncodedDocumentShardWriter(
                    snapshot_root=output,
                    shard_index=next_shard_index,
                    vector_dim=vector_dim,
                    vector_dtype=vector_dtype,
                    token_id_dtype=token_id_dtype,
                )
            writer.append(doc_id=doc_id, vectors=vectors, token_ids=token_ids)
        batch_doc_ids.clear()
        batch_texts.clear()

    with paths.collection.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if (
                document_limit is not None
                and source_document_count >= document_limit
                and seen_required_doc_ids >= required_doc_ids
            ):
                break
            doc_id, text = parse_tsv_id_text_line(line, row_name="collection")
            source_document_count += 1
            in_prefix = (
                document_limit is None or source_document_count <= document_limit
            )
            is_required_positive = doc_id in required_doc_ids
            if not in_prefix and not is_required_positive:
                continue
            if is_required_positive:
                seen_required_doc_ids.add(doc_id)
            selected_document_count += 1
            if selected_document_count <= completed_document_count:
                continue
            batch_doc_ids.append(doc_id)
            batch_texts.append(text)
            if len(batch_texts) == document_batch_size:
                flush_batch()
    flush_batch()
    if writer is not None and writer.document_count > 0:
        new_shards.append(writer.close())
    missing_required_doc_ids = required_doc_ids - seen_required_doc_ids
    if missing_required_doc_ids:
        preview = ", ".join(sorted(missing_required_doc_ids)[:5])
        raise ValueError(
            "collection did not contain all selected query positives; "
            f"missing {len(missing_required_doc_ids)} doc ids, first: {preview}"
        )
    return new_shards


def _materialize_queries(
    *,
    paths: MsmarcoPassagePaths,
    output: Path,
    query_limit: int | None,
    model_name: str,
    vector_dim: int,
    query_batch_size: int,
    resume: bool,
    encode_query_batch: QueryBatchEncoder,
) -> dict[str, object]:
    query_root = output / "queries"
    manifest_path = query_root / "manifest.json"
    if manifest_path.exists() and resume:
        return dict(json_load(manifest_path))
    if query_root.exists():
        raise FileExistsError(f"query snapshot already exists: {query_root}")

    tmp_root = output / ".queries.tmp"
    if tmp_root.exists():
        raise FileExistsError(f"incomplete query snapshot exists: {tmp_root}")
    tmp_root.mkdir(parents=True)

    positives_by_query = load_positive_doc_ids_by_query(paths.qrels)
    queries_by_id = load_queries_by_id(paths.queries)
    query_ids = choose_query_ids(
        positives_by_query,
        queries_by_id,
        query_limit=query_limit,
    )
    query_vectors_path = tmp_root / "query_vectors.f32"
    query_offsets = [0]
    with (
        query_vectors_path.open("wb") as vector_file,
        (tmp_root / "queries.jsonl").open("w", encoding="utf-8") as query_file,
    ):
        for start in range(0, len(query_ids), query_batch_size):
            batch_ids = query_ids[start : start + query_batch_size]
            batch_texts = [queries_by_id[query_id] for query_id in batch_ids]
            encoded_batch = encode_query_batch(batch_texts, model_name, query_batch_size)
            if len(encoded_batch) != len(batch_ids):
                raise ValueError("query encoder returned the wrong batch length")
            for query_id, text, vectors in zip(
                batch_ids,
                batch_texts,
                encoded_batch,
                strict=True,
            ):
                matrix = np.asarray(vectors, dtype=np.float32)
                if matrix.ndim != 2 or int(matrix.shape[1]) != vector_dim:
                    raise ValueError("query vectors must be tokens x vector_dim")
                np.ascontiguousarray(matrix, dtype=np.float32).tofile(vector_file)
                query_offsets.append(query_offsets[-1] + int(matrix.shape[0]))
                query_file.write(
                    _json_line(
                        {
                            "query_id": query_id,
                            "text": text,
                            "relevant_doc_ids": positives_by_query[query_id],
                        }
                    )
                )
    np.save(tmp_root / "query_offsets.u64.npy", np.asarray(query_offsets, dtype=np.uint64))
    query_vector_count = query_offsets[-1]
    manifest = {
        "query_count": len(query_ids),
        "query_vector_count": query_vector_count,
        "nominal_query_vector_count": (
            round(query_vector_count / float(len(query_ids))) if query_ids else 0
        ),
        "vector_dim": vector_dim,
        "vector_dtype": "float32",
        "primary_metric": "mrr",
        "k": 10,
        "files": {
            "query_vectors": "query_vectors.f32",
            "query_offsets": "query_offsets.u64.npy",
            "queries": "queries.jsonl",
        },
    }
    (tmp_root / "manifest.json").write_text(
        _json_pretty(manifest),
        encoding="utf-8",
    )
    tmp_root.rename(query_root)
    return manifest


def _default_encode_document_batch(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    encoded = encode_document_texts_with_token_id_tensors(
        texts,
        model_name,
        batch_size=batch_size,
    )
    return tuple(
        (vectors.numpy(), token_ids.numpy())
        for vectors, token_ids in encoded
    )


def _default_encode_query_batch(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, ...]:
    encoded = encode_query_texts_as_tensors(
        texts,
        model_name,
        batch_size=batch_size,
    )
    return tuple(vectors.numpy() for vectors in encoded)


def _prepare_output_root(output: Path, *, resume: bool) -> None:
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"snapshot output is not empty; use --resume or choose a new path: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)


def _validate_limits(
    *,
    document_limit: int | None,
    query_limit: int | None,
    document_batch_size: int,
    query_batch_size: int,
    shard_max_vectors: int,
) -> None:
    if document_limit is not None and document_limit <= 0:
        raise ValueError("document_limit must be positive when provided")
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive when provided")
    if document_batch_size <= 0:
        raise ValueError("document_batch_size must be positive")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if shard_max_vectors <= 0:
        raise ValueError("shard_max_vectors must be positive")


def _validate_completed_shards(
    shards: Sequence[SnapshotShardSummary],
    *,
    vector_dim: int,
    vector_dtype: str,
    token_id_dtype: str,
) -> None:
    for shard in shards:
        if shard.vector_dim != vector_dim:
            raise ValueError("completed shard vector_dim does not match requested run")
        if shard.vector_dtype != vector_dtype:
            raise ValueError("completed shard vector_dtype does not match requested run")
        if shard.token_id_dtype != token_id_dtype:
            raise ValueError(
                "completed shard token_id_dtype does not match requested run"
            )


def _json_pretty(row: dict[str, Any]) -> str:
    import json

    return json.dumps(row, indent=2, sort_keys=True) + "\n"


def _json_line(row: dict[str, Any]) -> str:
    import json

    return json.dumps(row, sort_keys=True) + "\n"


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
