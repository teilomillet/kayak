"""Streaming TAC/PQ index construction over encoded snapshot shards.

This module owns the bounded-memory paper-scale build path. It trains
token-aware centroids from deterministic per-token samples, then streams the
snapshot again to write centroid postings and residual-PQ payloads. It does not
own query-time search over the resulting on-disk artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, TOKEN_ID_DTYPE, VECTOR_DTYPE
from .encoded_snapshot_loader import (
    iter_snapshot_document_shards,
    load_snapshot_manifest,
)
from .tachiom_clustering import _kmeans_deterministic, _squared_distances
from .tachiom_pq import TachiomResidualPqConfig
from .tachiom_types import TachiomTacConfig


@dataclass(frozen=True, slots=True)
class StreamingTachiomBuildConfig:
    tac: TachiomTacConfig
    pq: TachiomResidualPqConfig | None = None
    centroid_samples_per_centroid: int = 4
    kmeans_max_centroids_per_token: int = 256
    kmeans_max_samples_per_token: int = 4096
    assignment_vector_chunk_size: int = 2048
    assignment_centroid_chunk_size: int = 4096
    posting_partition_count: int = 128

    def validate(self, *, vector_dim: int) -> None:
        self.tac.validate(final_k=1)
        if self.pq is not None:
            self.pq.validate(vector_dim=vector_dim)
        if self.centroid_samples_per_centroid <= 0:
            raise ValueError("centroid_samples_per_centroid must be positive")
        if self.kmeans_max_centroids_per_token <= 0:
            raise ValueError("kmeans_max_centroids_per_token must be positive")
        if self.kmeans_max_samples_per_token <= 0:
            raise ValueError("kmeans_max_samples_per_token must be positive")
        if self.assignment_vector_chunk_size <= 0:
            raise ValueError("assignment_vector_chunk_size must be positive")
        if self.assignment_centroid_chunk_size <= 0:
            raise ValueError("assignment_centroid_chunk_size must be positive")
        if self.posting_partition_count <= 0:
            raise ValueError("posting_partition_count must be positive")

    def to_json_ready(self) -> dict[str, object]:
        row: dict[str, object] = {
            "tac": asdict(self.tac),
            "centroid_samples_per_centroid": self.centroid_samples_per_centroid,
            "kmeans_max_centroids_per_token": self.kmeans_max_centroids_per_token,
            "kmeans_max_samples_per_token": self.kmeans_max_samples_per_token,
            "assignment_vector_chunk_size": self.assignment_vector_chunk_size,
            "assignment_centroid_chunk_size": self.assignment_centroid_chunk_size,
            "posting_partition_count": self.posting_partition_count,
        }
        if self.pq is not None:
            row["pq"] = asdict(self.pq)
        return row


@dataclass(frozen=True, slots=True)
class StreamingTachiomBuildSummary:
    snapshot_root: str
    output_root: str
    dataset_id: str
    model_name: str
    document_count: int
    document_vector_count: int
    vector_dim: int
    centroid_count: int
    posting_count: int
    token_type_count: int
    sample_vector_count: int
    pq_enabled: bool
    index_payload_bytes: int
    build_payload_bytes: int
    created_at_utc: str
    config: dict[str, object]
    files: dict[str, object]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class _TokenStats:
    count: int
    vector_sum: np.ndarray
    squared_norm_sum: float


def build_streaming_tachiom_index(
    *,
    snapshot_root: Path,
    output_root: Path,
    config: StreamingTachiomBuildConfig,
    overwrite: bool = False,
) -> StreamingTachiomBuildSummary:
    manifest = load_snapshot_manifest(snapshot_root)
    vector_dim = int(manifest["vector_dim"])
    config.validate(vector_dim=vector_dim)
    output_root = output_root.resolve()
    _prepare_output_root(output_root, overwrite=overwrite)

    stats, document_count, vector_count = _collect_token_stats(
        snapshot_root,
        vector_dim=vector_dim,
    )
    allocation = _allocate_from_stats(stats, config.tac)
    sample_plan = _sample_plan_from_allocation(
        stats,
        allocation=allocation,
        config=config,
    )
    sample_offsets = _sample_offsets(sample_plan)
    sample_vectors = _collect_centroid_samples(
        snapshot_root,
        output_root=output_root,
        vector_dim=vector_dim,
        sample_offsets=sample_offsets,
        sample_plan=sample_plan,
    )
    centroids, centroid_token_ids, centroid_ranges = _train_and_write_centroids(
        output_root=output_root,
        vector_dim=vector_dim,
        allocation=allocation,
        sample_vectors=sample_vectors,
        sample_offsets=sample_offsets,
        config=config,
    )
    _write_document_metadata(snapshot_root, output_root)
    pq_codebooks = None
    if config.pq is not None:
        pq_codebooks = _train_and_write_pq_codebooks(
            output_root=output_root,
            centroids=centroids,
            centroid_ranges=centroid_ranges,
            sample_vectors=sample_vectors,
            sample_offsets=sample_offsets,
            allocation=allocation,
            config=config.pq,
        )
    posting_count = _assign_and_write_payloads(
        snapshot_root=snapshot_root,
        output_root=output_root,
        centroids=centroids,
        centroid_ranges=centroid_ranges,
        config=config,
        pq_codebooks=pq_codebooks,
        vector_count=vector_count,
        document_count=document_count,
    )
    summary = _write_manifest(
        snapshot_root=snapshot_root,
        output_root=output_root,
        source_manifest=manifest,
        config=config,
        document_count=document_count,
        vector_count=vector_count,
        centroid_count=int(centroids.shape[0]),
        posting_count=posting_count,
        token_type_count=len(allocation),
        sample_vector_count=int(sample_vectors.shape[0]),
    )
    return summary


def _collect_token_stats(
    snapshot_root: Path,
    *,
    vector_dim: int,
) -> tuple[dict[int, _TokenStats], int, int]:
    stats: dict[int, _TokenStats] = {}
    document_count = 0
    vector_count = 0
    for _shard, vectors, token_ids, offsets, _doc_ids in iter_snapshot_document_shards(
        snapshot_root
    ):
        document_count += int(offsets.shape[0]) - 1
        vector_count += int(vectors.shape[0])
        shard_token_ids = np.asarray(token_ids)
        for token_id in np.unique(shard_token_ids):
            positions = np.flatnonzero(shard_token_ids == token_id)
            values = np.asarray(vectors[positions], dtype=VECTOR_DTYPE)
            token = int(token_id)
            row = stats.get(token)
            if row is None:
                row = _TokenStats(
                    count=0,
                    vector_sum=np.zeros(vector_dim, dtype=np.float64),
                    squared_norm_sum=0.0,
                )
                stats[token] = row
            row.count += int(values.shape[0])
            row.vector_sum += values.sum(axis=0, dtype=np.float64)
            row.squared_norm_sum += float(np.sum(values * values, dtype=np.float64))
    if not stats:
        raise ValueError("snapshot contains no document token vectors")
    return stats, document_count, vector_count


def _allocate_from_stats(
    stats: Mapping[int, _TokenStats],
    config: TachiomTacConfig,
) -> dict[int, int]:
    total_vectors = sum(row.count for row in stats.values())
    effective_budget = min(config.centroid_count, total_vectors)
    allocation: dict[int, int] = {}
    active_tokens: list[int] = []
    weights: dict[int, float] = {}
    caps: dict[int, int] = {}
    for token in sorted(stats):
        count = stats[token].count
        if count < config.micro_token_threshold:
            allocation[token] = 1
            continue
        if count < config.small_token_threshold:
            allocation[token] = min(2, count)
            continue
        active_tokens.append(token)
        mean = stats[token].vector_sum / float(count)
        mean_squared_norm = float(np.dot(mean, mean))
        spread = max(stats[token].squared_norm_sum / float(count) - mean_squared_norm, 0.0)
        weights[token] = float(np.sqrt(max(count * spread, 1e-12)))
        caps[token] = max(
            config.active_token_floor,
            min(count, count // config.min_vectors_per_centroid),
        )

    fixed_total = sum(allocation.values())
    if fixed_total >= effective_budget:
        for token in active_tokens:
            allocation[token] = min(config.active_token_floor, caps[token])
        return _trim_allocation_to_budget(
            allocation,
            target=effective_budget,
            token_counts={token: row.count for token, row in stats.items()},
        )
    remaining_budget = effective_budget - fixed_total
    if not active_tokens:
        return _reconcile_allocations(
            allocation,
            target=effective_budget,
            weights={token: float(stats[token].count) for token in allocation},
            caps={token: stats[token].count for token in allocation},
            floors={token: 1 for token in allocation},
        )
    if remaining_budget < len(active_tokens) * config.active_token_floor:
        for token in sorted(
            active_tokens,
            key=lambda item: (weights.get(item, 0.0), -item),
            reverse=True,
        )[:remaining_budget]:
            allocation[token] = 1
        return allocation

    weight_sum = sum(weights[token] for token in active_tokens)
    floors: dict[int, int] = {}
    fractional: dict[int, float] = {}
    for token in active_tokens:
        raw = (
            remaining_budget * weights[token] / weight_sum
            if weight_sum > 0.0
            else remaining_budget / len(active_tokens)
        )
        floor = min(caps[token], max(config.active_token_floor, int(np.floor(raw))))
        allocation[token] = floor
        floors[token] = min(config.active_token_floor, caps[token])
        fractional[token] = raw - np.floor(raw)
    return _reconcile_allocations(
        allocation,
        target=effective_budget,
        weights={
            token: fractional.get(token, weights.get(token, 0.0))
            for token in allocation
        },
        caps={token: caps.get(token, stats[token].count) for token in allocation},
        floors={token: floors.get(token, 1) for token in allocation},
    )


def _sample_plan_from_allocation(
    stats: Mapping[int, _TokenStats],
    *,
    allocation: Mapping[int, int],
    config: StreamingTachiomBuildConfig,
) -> dict[int, int]:
    plan: dict[int, int] = {}
    for token, centroid_count in allocation.items():
        count = stats[token].count
        if _token_uses_kmeans(centroid_count, count, config):
            sample_count = min(
                count,
                max(
                    centroid_count,
                    min(
                        centroid_count * config.centroid_samples_per_centroid,
                        config.kmeans_max_samples_per_token,
                    ),
                ),
            )
        else:
            sample_count = min(count, centroid_count)
        plan[token] = int(sample_count)
    return plan


def _sample_offsets(sample_plan: Mapping[int, int]) -> dict[int, tuple[int, int]]:
    offsets: dict[int, tuple[int, int]] = {}
    current = 0
    for token in sorted(sample_plan):
        start = current
        current += sample_plan[token]
        offsets[token] = (start, current)
    return offsets


def _collect_centroid_samples(
    snapshot_root: Path,
    *,
    output_root: Path,
    vector_dim: int,
    sample_offsets: Mapping[int, tuple[int, int]],
    sample_plan: Mapping[int, int],
) -> np.memmap:
    total_samples = max((stop for _start, stop in sample_offsets.values()), default=0)
    sample_vectors = np.memmap(
        output_root / "centroid_training_samples.f32",
        dtype=VECTOR_DTYPE,
        mode="w+",
        shape=(total_samples, vector_dim),
    )
    seen = {token: 0 for token in sample_plan}
    written = {token: 0 for token in sample_plan}
    counts = _token_counts_from_snapshot(snapshot_root, sample_plan.keys())
    for _shard, vectors, token_ids, _offsets, _doc_ids in iter_snapshot_document_shards(
        snapshot_root
    ):
        shard_ids = np.asarray(token_ids)
        for token in np.unique(shard_ids):
            token_int = int(token)
            if token_int not in sample_plan:
                continue
            positions = np.flatnonzero(shard_ids == token)
            for position in positions:
                occurrence_index = seen[token_int]
                seen[token_int] += 1
                if not _select_occurrence(
                    occurrence_index=occurrence_index,
                    total_count=counts[token_int],
                    sample_count=sample_plan[token_int],
                ):
                    continue
                start, _stop = sample_offsets[token_int]
                output_index = start + written[token_int]
                sample_vectors[output_index] = np.asarray(
                    vectors[int(position)],
                    dtype=VECTOR_DTYPE,
                )
                written[token_int] += 1
    for token, expected in sample_plan.items():
        if written[token] != expected:
            raise RuntimeError(
                f"sample collection for token {token} wrote {written[token]} "
                f"of {expected} rows"
            )
    sample_vectors.flush()
    return sample_vectors


def _train_and_write_centroids(
    *,
    output_root: Path,
    vector_dim: int,
    allocation: Mapping[int, int],
    sample_vectors: np.memmap,
    sample_offsets: Mapping[int, tuple[int, int]],
    config: StreamingTachiomBuildConfig,
) -> tuple[np.memmap, np.memmap, dict[int, tuple[int, int]]]:
    centroid_count = sum(allocation.values())
    centroids = np.memmap(
        output_root / "centroids.f32",
        dtype=VECTOR_DTYPE,
        mode="w+",
        shape=(centroid_count, vector_dim),
    )
    centroid_token_ids = np.memmap(
        output_root / "centroid_token_ids.i64",
        dtype=TOKEN_ID_DTYPE,
        mode="w+",
        shape=(centroid_count,),
    )
    centroid_ranges: dict[int, tuple[int, int]] = {}
    output_offset = 0
    for token in sorted(allocation):
        k = allocation[token]
        sample_start, sample_stop = sample_offsets[token]
        samples = np.asarray(sample_vectors[sample_start:sample_stop], dtype=VECTOR_DTYPE)
        if _token_uses_kmeans(k, int(samples.shape[0]), config):
            token_centroids, _assignments = _kmeans_deterministic(
                samples,
                k=k,
                iterations=config.tac.kmeans_iterations,
            )
        else:
            token_centroids = samples[:k]
        next_offset = output_offset + k
        centroids[output_offset:next_offset] = token_centroids
        centroid_token_ids[output_offset:next_offset] = token
        centroid_ranges[token] = (output_offset, next_offset)
        output_offset = next_offset
    centroids.flush()
    centroid_token_ids.flush()
    return centroids, centroid_token_ids, centroid_ranges


def _train_and_write_pq_codebooks(
    *,
    output_root: Path,
    centroids: np.memmap,
    centroid_ranges: Mapping[int, tuple[int, int]],
    sample_vectors: np.memmap,
    sample_offsets: Mapping[int, tuple[int, int]],
    allocation: Mapping[int, int],
    config: TachiomResidualPqConfig,
) -> np.memmap:
    vector_dim = int(centroids.shape[1])
    subspace_dim = vector_dim // config.subspace_count
    sample_positions = _training_sample_positions(
        total_count=int(sample_vectors.shape[0]),
        sample_count=config.training_sample_count,
    )
    residual_samples = np.empty((len(sample_positions), vector_dim), dtype=VECTOR_DTYPE)
    token_for_sample_ranges = _token_for_sample_ranges(sample_offsets)
    for output_index, sample_position in enumerate(sample_positions):
        token = _token_for_sample_position(token_for_sample_ranges, int(sample_position))
        token_centroids = centroids[slice(*centroid_ranges[token])]
        value = np.asarray(sample_vectors[int(sample_position)], dtype=VECTOR_DTYPE)
        centroid_position = _nearest_positions_chunked(
            value.reshape(1, -1),
            token_centroids,
            value_chunk_size=1,
            centroid_chunk_size=max(1, min(4096, int(token_centroids.shape[0]))),
        )[0]
        residual = value - token_centroids[int(centroid_position)]
        norm = np.linalg.norm(residual).astype(VECTOR_DTYPE)
        safe_norm = max(float(norm), config.residual_norm_floor)
        residual_samples[output_index] = residual / np.asarray(safe_norm, dtype=VECTOR_DTYPE)

    effective_codebook_size = min(config.codebook_size, int(residual_samples.shape[0]))
    codebooks = np.memmap(
        output_root / "pq_codebooks.f32",
        dtype=VECTOR_DTYPE,
        mode="w+",
        shape=(config.subspace_count, effective_codebook_size, subspace_dim),
    )
    for subspace_index in range(config.subspace_count):
        start = subspace_index * subspace_dim
        stop = start + subspace_dim
        subspace_centroids, _ = _kmeans_deterministic(
            residual_samples[:, start:stop],
            k=effective_codebook_size,
            iterations=config.kmeans_iterations,
        )
        codebooks[subspace_index] = subspace_centroids
    codebooks.flush()
    return codebooks


def _assign_and_write_payloads(
    *,
    snapshot_root: Path,
    output_root: Path,
    centroids: np.memmap,
    centroid_ranges: Mapping[int, tuple[int, int]],
    config: StreamingTachiomBuildConfig,
    pq_codebooks: np.memmap | None,
    vector_count: int,
    document_count: int,
) -> int:
    partitions_root = output_root / "posting_pair_partitions"
    partitions_root.mkdir()
    partition_handles = [
        (partitions_root / f"pairs_{index:04d}.u32").open("wb")
        for index in range(config.posting_partition_count)
    ]
    token_centroid_writer = None
    residual_norm_writer = None
    pq_code_writer = None
    if pq_codebooks is not None:
        token_centroid_writer = (output_root / "token_centroid_positions.u32").open("wb")
        residual_norm_writer = (output_root / "residual_norms.f32").open("wb")
        pq_code_writer = (output_root / "pq_codes.u8").open("wb")

    global_doc_offset = 0
    try:
        for _shard, vectors, token_ids, offsets, _doc_ids in iter_snapshot_document_shards(
            snapshot_root
        ):
            local_doc_positions = np.repeat(
                np.arange(int(offsets.shape[0]) - 1, dtype=np.uint32) + global_doc_offset,
                np.diff(offsets).astype(np.int64),
            )
            shard_ids = np.asarray(token_ids)
            for start in range(0, int(vectors.shape[0]), config.assignment_vector_chunk_size):
                stop = min(start + config.assignment_vector_chunk_size, int(vectors.shape[0]))
                values = np.asarray(vectors[start:stop], dtype=VECTOR_DTYPE)
                chunk_ids = shard_ids[start:stop]
                chunk_docs = local_doc_positions[start:stop].astype(np.uint32, copy=False)
                global_assignments = np.zeros(int(values.shape[0]), dtype=np.uint32)
                posting_mask = np.zeros(int(values.shape[0]), dtype=bool)
                for token in np.unique(chunk_ids):
                    token_int = int(token)
                    local_positions = np.flatnonzero(chunk_ids == token)
                    local_values = values[local_positions]
                    if token_int in centroid_ranges:
                        centroid_start, centroid_stop = centroid_ranges[token_int]
                        local_centroids = centroids[centroid_start:centroid_stop]
                        assignments = _nearest_positions_chunked(
                            local_values,
                            local_centroids,
                            value_chunk_size=config.assignment_vector_chunk_size,
                            centroid_chunk_size=config.assignment_centroid_chunk_size,
                        ).astype(np.uint32)
                        global_assignments[local_positions] = assignments + np.uint32(
                            centroid_start
                        )
                        posting_mask[local_positions] = True
                    elif pq_codebooks is not None:
                        assignments = _nearest_positions_chunked(
                            local_values,
                            centroids,
                            value_chunk_size=config.assignment_vector_chunk_size,
                            centroid_chunk_size=config.assignment_centroid_chunk_size,
                        ).astype(np.uint32)
                        global_assignments[local_positions] = assignments
                if np.any(posting_mask):
                    _write_posting_pairs(
                        global_assignments[posting_mask],
                        chunk_docs[posting_mask],
                        partition_handles,
                        centroid_count=int(centroids.shape[0]),
                    )
                if pq_codebooks is not None:
                    _write_pq_payload_chunk(
                        values=values,
                        global_assignments=global_assignments,
                        centroids=centroids,
                        pq_codebooks=pq_codebooks,
                        token_centroid_writer=token_centroid_writer,
                        residual_norm_writer=residual_norm_writer,
                        pq_code_writer=pq_code_writer,
                        residual_norm_floor=config.pq.residual_norm_floor,
                    )
            global_doc_offset += int(offsets.shape[0]) - 1
    finally:
        for handle in partition_handles:
            handle.close()
        for handle in (token_centroid_writer, residual_norm_writer, pq_code_writer):
            if handle is not None:
                handle.close()

    posting_count = _build_posting_csr_from_partitions(
        output_root=output_root,
        partitions_root=partitions_root,
        centroid_count=int(centroids.shape[0]),
        partition_count=config.posting_partition_count,
    )
    _validate_payload_lengths(
        output_root=output_root,
        pq_enabled=pq_codebooks is not None,
        vector_count=vector_count,
        subspace_count=0 if config.pq is None else config.pq.subspace_count,
    )
    return posting_count


def _write_pq_payload_chunk(
    *,
    values: np.ndarray,
    global_assignments: np.ndarray,
    centroids: np.memmap,
    pq_codebooks: np.memmap,
    token_centroid_writer: Any,
    residual_norm_writer: Any,
    pq_code_writer: Any,
    residual_norm_floor: float,
) -> None:
    assigned_centroids = np.asarray(centroids[global_assignments], dtype=VECTOR_DTYPE)
    residuals = np.ascontiguousarray(values - assigned_centroids, dtype=VECTOR_DTYPE)
    residual_norms = np.linalg.norm(residuals, axis=1).astype(VECTOR_DTYPE)
    safe_norms = np.maximum(
        residual_norms,
        np.asarray(residual_norm_floor, dtype=VECTOR_DTYPE),
    )
    normalized = np.ascontiguousarray(residuals / safe_norms[:, None], dtype=VECTOR_DTYPE)
    zero_rows = residual_norms <= np.asarray(residual_norm_floor, dtype=VECTOR_DTYPE)
    if np.any(zero_rows):
        normalized[zero_rows] = np.asarray(0.0, dtype=VECTOR_DTYPE)
    codes = _pq_codes_for_normalized_residuals(normalized, pq_codebooks)
    np.ascontiguousarray(global_assignments, dtype=np.uint32).tofile(
        token_centroid_writer
    )
    np.ascontiguousarray(residual_norms, dtype=VECTOR_DTYPE).tofile(
        residual_norm_writer
    )
    np.ascontiguousarray(codes, dtype=np.uint8).tofile(pq_code_writer)


def _pq_codes_for_normalized_residuals(
    normalized: np.ndarray,
    codebooks: np.ndarray,
) -> np.ndarray:
    subspace_count = int(codebooks.shape[0])
    subspace_dim = int(codebooks.shape[2])
    codes = np.empty((int(normalized.shape[0]), subspace_count), dtype=np.uint8)
    for subspace_index in range(subspace_count):
        start = subspace_index * subspace_dim
        stop = start + subspace_dim
        assignments = np.argmin(
            _squared_distances(normalized[:, start:stop], codebooks[subspace_index]),
            axis=1,
        )
        codes[:, subspace_index] = assignments.astype(np.uint8, copy=False)
    return codes


def _write_posting_pairs(
    centroid_positions: np.ndarray,
    doc_positions: np.ndarray,
    partition_handles: Sequence[Any],
    *,
    centroid_count: int,
) -> None:
    partition_count = len(partition_handles)
    partitions = np.asarray(
        np.minimum(
            (centroid_positions.astype(np.uint64) * np.uint64(partition_count))
            // np.uint64(centroid_count),
            np.uint64(partition_count - 1),
        ),
        dtype=np.int64,
    )
    for partition in np.unique(partitions):
        mask = partitions == int(partition)
        pairs = np.empty((int(np.count_nonzero(mask)), 2), dtype=np.uint32)
        pairs[:, 0] = centroid_positions[mask]
        pairs[:, 1] = doc_positions[mask]
        pairs.tofile(partition_handles[int(partition)])


def _build_posting_csr_from_partitions(
    *,
    output_root: Path,
    partitions_root: Path,
    centroid_count: int,
    partition_count: int,
) -> int:
    posting_offsets = np.zeros(centroid_count + 1, dtype=np.uint64)
    postings_path = output_root / "centroid_doc_postings.u32"
    posting_count = 0
    cursor = 0
    with postings_path.open("wb") as postings_file:
        for partition in range(partition_count):
            pair_path = partitions_root / f"pairs_{partition:04d}.u32"
            raw = np.fromfile(pair_path, dtype=np.uint32)
            if raw.size == 0:
                pair_path.unlink()
                continue
            pairs = raw.reshape(raw.shape[0] // 2, 2)
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            ordered = pairs[order]
            unique_mask = np.ones(int(ordered.shape[0]), dtype=bool)
            unique_mask[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
            unique_pairs = ordered[unique_mask]
            for centroid in np.unique(unique_pairs[:, 0]):
                centroid_int = int(centroid)
                while cursor < centroid_int:
                    posting_offsets[cursor + 1] = posting_count
                    cursor += 1
                docs = unique_pairs[unique_pairs[:, 0] == centroid, 1]
                np.ascontiguousarray(docs, dtype=np.uint32).tofile(postings_file)
                posting_count += int(docs.shape[0])
                posting_offsets[centroid_int + 1] = posting_count
                cursor = centroid_int + 1
            pair_path.unlink()
        while cursor < centroid_count:
            posting_offsets[cursor + 1] = posting_count
            cursor += 1
    partitions_root.rmdir()
    np.save(output_root / "centroid_doc_posting_offsets.u64.npy", posting_offsets)
    return posting_count


def _nearest_positions_chunked(
    values: np.ndarray,
    centroids: np.ndarray,
    *,
    value_chunk_size: int,
    centroid_chunk_size: int,
) -> np.ndarray:
    assignments = np.empty(int(values.shape[0]), dtype=np.int64)
    for start in range(0, int(values.shape[0]), value_chunk_size):
        stop = min(start + value_chunk_size, int(values.shape[0]))
        chunk = values[start:stop]
        best_distances = np.full(int(chunk.shape[0]), np.inf, dtype=VECTOR_DTYPE)
        best_positions = np.zeros(int(chunk.shape[0]), dtype=np.int64)
        for centroid_start in range(0, int(centroids.shape[0]), centroid_chunk_size):
            centroid_stop = min(
                centroid_start + centroid_chunk_size,
                int(centroids.shape[0]),
            )
            distances = _squared_distances(chunk, centroids[centroid_start:centroid_stop])
            local_positions = np.argmin(distances, axis=1)
            local_distances = distances[np.arange(int(chunk.shape[0])), local_positions]
            improved = local_distances < best_distances
            best_distances[improved] = local_distances[improved]
            best_positions[improved] = centroid_start + local_positions[improved]
        assignments[start:stop] = best_positions
    return assignments


def _write_document_metadata(snapshot_root: Path, output_root: Path) -> None:
    doc_offsets = [0]
    with (output_root / "doc_ids.txt").open("w", encoding="utf-8") as doc_id_file:
        for _shard, _vectors, _token_ids, offsets, doc_ids in iter_snapshot_document_shards(
            snapshot_root
        ):
            base = doc_offsets[-1]
            local_offsets = np.asarray(offsets, dtype=np.uint64)
            for offset in local_offsets[1:]:
                doc_offsets.append(base + int(offset))
            for doc_id in doc_ids:
                doc_id_file.write(f"{doc_id}\n")
    np.save(output_root / "doc_offsets.u64.npy", np.asarray(doc_offsets, dtype=np.uint64))


def _write_manifest(
    *,
    snapshot_root: Path,
    output_root: Path,
    source_manifest: Mapping[str, Any],
    config: StreamingTachiomBuildConfig,
    document_count: int,
    vector_count: int,
    centroid_count: int,
    posting_count: int,
    token_type_count: int,
    sample_vector_count: int,
) -> StreamingTachiomBuildSummary:
    files: dict[str, object] = {
        "centroids": "centroids.f32",
        "centroid_token_ids": "centroid_token_ids.i64",
        "centroid_doc_postings": "centroid_doc_postings.u32",
        "centroid_doc_posting_offsets": "centroid_doc_posting_offsets.u64.npy",
        "doc_ids": "doc_ids.txt",
        "doc_offsets": "doc_offsets.u64.npy",
        "centroid_training_samples": "centroid_training_samples.f32",
    }
    if config.pq is not None:
        files["pq"] = {
            "token_centroid_positions": "token_centroid_positions.u32",
            "residual_norms": "residual_norms.f32",
            "pq_codes": "pq_codes.u8",
            "pq_codebooks": "pq_codebooks.f32",
        }
    query_payload_files = [
        output_root / "centroids.f32",
        output_root / "centroid_token_ids.i64",
        output_root / "centroid_doc_postings.u32",
        output_root / "centroid_doc_posting_offsets.u64.npy",
        output_root / "doc_ids.txt",
        output_root / "doc_offsets.u64.npy",
    ]
    if config.pq is not None:
        query_payload_files.extend(
            [
                output_root / "token_centroid_positions.u32",
                output_root / "residual_norms.f32",
                output_root / "pq_codes.u8",
                output_root / "pq_codebooks.f32",
            ]
        )
    index_payload_bytes = sum(path.stat().st_size for path in query_payload_files)
    build_payload_bytes = sum(
        path.stat().st_size
        for path in output_root.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    summary = StreamingTachiomBuildSummary(
        snapshot_root=str(snapshot_root),
        output_root=str(output_root),
        dataset_id=str(source_manifest["dataset_id"]),
        model_name=str(source_manifest["model_name"]),
        document_count=document_count,
        document_vector_count=vector_count,
        vector_dim=int(source_manifest["vector_dim"]),
        centroid_count=centroid_count,
        posting_count=posting_count,
        token_type_count=token_type_count,
        sample_vector_count=sample_vector_count,
        pq_enabled=config.pq is not None,
        index_payload_bytes=index_payload_bytes,
        build_payload_bytes=build_payload_bytes,
        created_at_utc=datetime.now(UTC).isoformat(),
        config=config.to_json_ready(),
        files=files,
    )
    (output_root / "manifest.json").write_text(
        json.dumps(summary.to_json_ready(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _token_counts_from_snapshot(
    snapshot_root: Path,
    token_filter: Sequence[int],
) -> dict[int, int]:
    wanted = {int(token) for token in token_filter}
    counts = {token: 0 for token in wanted}
    for _shard, _vectors, token_ids, _offsets, _doc_ids in iter_snapshot_document_shards(
        snapshot_root
    ):
        shard_ids = np.asarray(token_ids)
        unique_ids, unique_counts = np.unique(shard_ids, return_counts=True)
        for token, count in zip(unique_ids, unique_counts, strict=True):
            token_int = int(token)
            if token_int in wanted:
                counts[token_int] += int(count)
    return counts


def _select_occurrence(
    *,
    occurrence_index: int,
    total_count: int,
    sample_count: int,
) -> bool:
    before = (occurrence_index * sample_count) // total_count
    after = ((occurrence_index + 1) * sample_count) // total_count
    return after > before


def _token_uses_kmeans(
    centroid_count: int,
    sample_count: int,
    config: StreamingTachiomBuildConfig,
) -> bool:
    return (
        sample_count > centroid_count
        and centroid_count <= config.kmeans_max_centroids_per_token
        and sample_count <= config.kmeans_max_samples_per_token
    )


def _training_sample_positions(
    *,
    total_count: int,
    sample_count: int | None,
) -> np.ndarray:
    if sample_count is None or sample_count >= total_count:
        return np.arange(total_count, dtype=np.int64)
    return np.linspace(0, total_count - 1, sample_count, dtype=np.int64)


def _token_for_sample_ranges(
    sample_offsets: Mapping[int, tuple[int, int]],
) -> list[tuple[int, int, int]]:
    return [
        (start, stop, token)
        for token, (start, stop) in sorted(sample_offsets.items(), key=lambda item: item[1])
    ]


def _token_for_sample_position(
    ranges: Sequence[tuple[int, int, int]],
    position: int,
) -> int:
    for start, stop, token in ranges:
        if start <= position < stop:
            return token
    raise ValueError("sample position is outside token ranges")


def _validate_payload_lengths(
    *,
    output_root: Path,
    pq_enabled: bool,
    vector_count: int,
    subspace_count: int,
) -> None:
    if not pq_enabled:
        return
    expected_centroid_bytes = vector_count * np.dtype(np.uint32).itemsize
    expected_norm_bytes = vector_count * np.dtype(VECTOR_DTYPE).itemsize
    expected_code_bytes = vector_count * subspace_count * np.dtype(np.uint8).itemsize
    if (output_root / "token_centroid_positions.u32").stat().st_size != expected_centroid_bytes:
        raise RuntimeError("token centroid payload length mismatch")
    if (output_root / "residual_norms.f32").stat().st_size != expected_norm_bytes:
        raise RuntimeError("residual norm payload length mismatch")
    if (output_root / "pq_codes.u8").stat().st_size != expected_code_bytes:
        raise RuntimeError("PQ code payload length mismatch")


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"streaming index output is not empty: {output_root}"
            )
        for path in output_root.iterdir():
            if path.is_dir():
                for child in path.iterdir():
                    child.unlink()
                path.rmdir()
            else:
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def _reconcile_allocations(
    allocation: dict[int, int],
    *,
    target: int,
    weights: dict[int, float],
    caps: dict[int, int],
    floors: dict[int, int],
) -> dict[int, int]:
    current = sum(allocation.values())
    while current < target:
        candidates = [
            token for token, count in allocation.items() if count < caps.get(token, count)
        ]
        if not candidates:
            break
        token = max(candidates, key=lambda item: (weights.get(item, 0.0), -item))
        allocation[token] += 1
        current += 1
    while current > target:
        candidates = [
            token for token, count in allocation.items() if count > floors.get(token, 1)
        ]
        if not candidates:
            break
        token = min(candidates, key=lambda item: (weights.get(item, 0.0), item))
        allocation[token] -= 1
        current -= 1
    return {token: count for token, count in allocation.items() if count > 0}


def _trim_allocation_to_budget(
    allocation: dict[int, int],
    *,
    target: int,
    token_counts: dict[int, int],
) -> dict[int, int]:
    ordered = sorted(
        allocation,
        key=lambda token: (token_counts[token], -token),
        reverse=True,
    )
    trimmed: dict[int, int] = {}
    remaining = target
    for token in ordered:
        if remaining <= 0:
            break
        count = min(allocation[token], remaining)
        trimmed[token] = count
        remaining -= count
    return trimmed
