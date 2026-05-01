from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.tachiom_hnsw import TachiomHnswConfig  # noqa: E402
from kayak_bridge.tachiom_streaming_hnsw import (  # noqa: E402
    build_streaming_tachiom_hnsw_graph,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a persisted HNSW centroid graph for a streaming TAC/PQ index."
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hnsw-max-neighbors", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--hnsw-level-probability", type=float, default=0.0625)
    parser.add_argument("--hnsw-seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_streaming_tachiom_hnsw_graph(
        index_root=args.index,
        graph_root=args.graph,
        config=TachiomHnswConfig(
            max_neighbors=args.hnsw_max_neighbors,
            ef_construction=args.hnsw_ef_construction,
            ef_search=args.hnsw_ef_search,
            level_probability=args.hnsw_level_probability,
            seed=args.hnsw_seed,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.to_json_ready(), indent=2, sort_keys=True))
    print(f"wrote {Path(summary.graph_root) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
