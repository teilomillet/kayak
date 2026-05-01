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

from kayak_bridge.colbert_encoder import DEFAULT_MODEL_NAME  # noqa: E402
from kayak_bridge.msmarco_colbert_snapshot import (  # noqa: E402
    DEFAULT_DATASET_ID,
    build_msmarco_colbert_snapshot,
    estimate_paper_msmarco_snapshot_bytes,
)
from kayak_bridge.msmarco_passage_task import MsmarcoPassagePaths  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize MS MARCO passage ColBERT document vectors and aligned "
            "document token ids into a sharded binary snapshot. This avoids "
            "full-corpus JSON and is the intended paper-scale artifact path."
        )
    )
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument(
        "--vector-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Document vector storage dtype. float16 is the paper-scale default.",
    )
    parser.add_argument(
        "--token-id-dtype",
        choices=("uint32", "int64"),
        default="uint32",
        help="Document token-id storage dtype. uint32 is enough for ColBERT/BERT ids.",
    )
    parser.add_argument("--document-batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument(
        "--shard-max-vectors",
        type=int,
        default=4_000_000,
        help=(
            "Maximum document token vectors per shard. With float16 dim128, "
            "4M vectors is about 1GiB of vector payload."
        ),
    )
    parser.add_argument(
        "--include-query-positives",
        action="store_true",
        help=(
            "For bounded judged slices, include all positive documents for the "
            "selected queries even when they are outside the document prefix."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue after completed shard directories already present in output.",
    )
    parser.add_argument(
        "--estimate-paper-scale",
        action="store_true",
        help="Print the MS MARCO paper-scale byte estimate and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.estimate_paper_scale:
        estimate = estimate_paper_msmarco_snapshot_bytes(
            vector_dim=args.vector_dim,
            vector_dtype=args.vector_dtype,
            token_id_dtype=args.token_id_dtype,
        )
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0

    manifest = build_msmarco_colbert_snapshot(
        paths=MsmarcoPassagePaths(
            collection=args.collection,
            queries=args.queries,
            qrels=args.qrels,
        ),
        output=args.output,
        document_limit=args.document_limit,
        query_limit=args.query_limit,
        dataset_id=args.dataset_id,
        model_name=args.model_name,
        vector_dim=args.vector_dim,
        vector_dtype=args.vector_dtype,
        token_id_dtype=args.token_id_dtype,
        document_batch_size=args.document_batch_size,
        query_batch_size=args.query_batch_size,
        shard_max_vectors=args.shard_max_vectors,
        include_query_positives=args.include_query_positives,
        resume=args.resume,
    )
    print(json.dumps(manifest.to_json_ready(), indent=2, sort_keys=True))
    print(f"wrote {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
