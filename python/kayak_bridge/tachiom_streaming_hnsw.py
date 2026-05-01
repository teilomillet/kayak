"""HNSW graph sidecar for streaming TAC/PQ artifacts.

This module materializes the existing Python HNSW centroid graph into flat
files next to a streaming TAC/PQ index. It owns graph persistence and loading;
it does not claim the graph builder is paper-scale optimized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE
from .tachiom_hnsw import TachiomCentroidHnswGraph, TachiomHnswConfig


DEFAULT_STREAMING_HNSW_GRAPH_DIR = "hnsw_graph"


@dataclass(frozen=True, slots=True)
class StreamingTachiomHnswGraphBuildSummary:
    index_root: str
    graph_root: str
    centroid_count: int
    level_count: int
    entry_point: int
    edge_count: int
    graph_bytes: int
    build_seconds: float
    created_at_utc: str
    config: dict[str, object]
    files: dict[str, str]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StreamingTachiomHnswGraph:
    graph_root: Path
    centroid_count: int
    level_count: int
    entry_point: int
    ef_search: int
    layer_node_offset_offsets: np.ndarray
    graph_node_offsets: np.ndarray
    graph_neighbor_indices: np.memmap
    graph_bytes: int
    manifest: dict[str, Any]

    def neighbors(self, *, layer_index: int, node_index: int) -> np.ndarray:
        offset_position = int(self.layer_node_offset_offsets[layer_index]) + node_index
        start = int(self.graph_node_offsets[offset_position])
        stop = int(self.graph_node_offsets[offset_position + 1])
        return self.graph_neighbor_indices[start:stop]


def build_streaming_tachiom_hnsw_graph(
    *,
    index_root: Path,
    graph_root: Path | None = None,
    config: TachiomHnswConfig,
    overwrite: bool = False,
) -> StreamingTachiomHnswGraphBuildSummary:
    """Build and persist a centroid HNSW graph for a streaming TAC/PQ index."""

    from .tachiom_streaming_search import load_streaming_tachiom_pq_index

    base_index = load_streaming_tachiom_pq_index(index_root)
    graph_root = _graph_root(index_root, graph_root)
    _prepare_graph_root(graph_root, overwrite=overwrite)
    graph = TachiomCentroidHnswGraph.build(base_index.centroids, config=config)
    layer_offsets, node_offsets, neighbor_indices = _flatten_graph_layers(graph.layers)

    np.save(
        graph_root / "layer_node_offset_offsets.u64.npy",
        np.ascontiguousarray(layer_offsets, dtype=np.uint64),
    )
    np.save(
        graph_root / "graph_node_offsets.u64.npy",
        np.ascontiguousarray(node_offsets, dtype=np.uint64),
    )
    np.ascontiguousarray(neighbor_indices, dtype=np.uint32).tofile(
        graph_root / "graph_neighbor_indices.u32"
    )
    np.ascontiguousarray(graph.levels, dtype=INDEX_OFFSET_DTYPE).tofile(
        graph_root / "levels.i64"
    )

    files = {
        "layer_node_offset_offsets": "layer_node_offset_offsets.u64.npy",
        "graph_node_offsets": "graph_node_offsets.u64.npy",
        "graph_neighbor_indices": "graph_neighbor_indices.u32",
        "levels": "levels.i64",
    }
    graph_bytes = sum((graph_root / name).stat().st_size for name in files.values())
    summary = StreamingTachiomHnswGraphBuildSummary(
        index_root=str(index_root),
        graph_root=str(graph_root),
        centroid_count=base_index.centroid_count,
        level_count=len(graph.layers),
        entry_point=graph.entry_point,
        edge_count=int(neighbor_indices.shape[0]),
        graph_bytes=graph_bytes,
        build_seconds=graph.build_seconds,
        created_at_utc=datetime.now(UTC).isoformat(),
        config=asdict(config),
        files=files,
    )
    (graph_root / "manifest.json").write_text(
        json.dumps(summary.to_json_ready(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_streaming_tachiom_hnsw_graph(
    *,
    index_root: Path,
    graph_root: Path | None = None,
) -> StreamingTachiomHnswGraph:
    graph_root = _graph_root(index_root, graph_root)
    manifest_path = graph_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"streaming HNSW graph manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    centroid_count = int(manifest["centroid_count"])
    edge_count = int(manifest["edge_count"])
    layer_offsets = np.load(graph_root / "layer_node_offset_offsets.u64.npy").astype(
        INDEX_OFFSET_DTYPE,
        copy=False,
    )
    node_offsets = np.load(graph_root / "graph_node_offsets.u64.npy").astype(
        INDEX_OFFSET_DTYPE,
        copy=False,
    )
    level_count = int(manifest["level_count"])
    if layer_offsets.shape != (level_count,):
        raise ValueError("HNSW layer offsets do not match graph manifest")
    if node_offsets.shape != (level_count * (centroid_count + 1),):
        raise ValueError("HNSW node offsets do not match graph manifest")
    if int(node_offsets[-1]) != edge_count:
        raise ValueError("HNSW final node offset does not match edge count")
    neighbor_indices = np.memmap(
        graph_root / "graph_neighbor_indices.u32",
        dtype=np.uint32,
        mode="r",
        shape=(edge_count,),
    )
    return StreamingTachiomHnswGraph(
        graph_root=graph_root,
        centroid_count=centroid_count,
        level_count=level_count,
        entry_point=int(manifest["entry_point"]),
        ef_search=int(manifest["config"]["ef_search"]),
        layer_node_offset_offsets=layer_offsets,
        graph_node_offsets=node_offsets,
        graph_neighbor_indices=neighbor_indices,
        graph_bytes=int(manifest["graph_bytes"]),
        manifest=manifest,
    )


def _flatten_graph_layers(
    layers: Sequence[Sequence[Sequence[int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer_offsets: list[int] = []
    node_offsets: list[int] = []
    neighbor_indices: list[int] = []
    for layer in layers:
        layer_offsets.append(len(node_offsets))
        node_offsets.append(len(neighbor_indices))
        for neighbors in layer:
            neighbor_indices.extend(int(neighbor) for neighbor in neighbors)
            node_offsets.append(len(neighbor_indices))
    return (
        np.asarray(layer_offsets, dtype=np.uint64),
        np.asarray(node_offsets, dtype=np.uint64),
        np.asarray(neighbor_indices, dtype=np.uint32),
    )


def _graph_root(index_root: Path, graph_root: Path | None) -> Path:
    if graph_root is not None:
        return graph_root.resolve()
    return (index_root / DEFAULT_STREAMING_HNSW_GRAPH_DIR).resolve()


def _prepare_graph_root(graph_root: Path, *, overwrite: bool) -> None:
    if graph_root.exists() and any(graph_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"streaming HNSW graph output is not empty: {graph_root}")
        for path in graph_root.iterdir():
            if path.is_dir():
                raise ValueError(f"unexpected directory inside graph root: {path}")
            path.unlink()
    graph_root.mkdir(parents=True, exist_ok=True)
