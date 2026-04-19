from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

import kayak

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.openai_dense_baseline import OpenAIEmbeddingTextBatcher
from kayak_bridge.openai_dense_baseline import TiktokenSourceTokenizer
from kayak_bridge.single_vector_chunk_baseline import (
    build_single_vector_chunk_sweep_bundle,
)
from kayak_bridge.source_text_chunking import RawTextChunkSpec


def _parse_chunk_spec(value: str) -> RawTextChunkSpec:
    try:
        size_part, overlap_part = value.split(":", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "chunk specs must look like SIZE:OVERLAP, for example 800:400"
        ) from exc
    return RawTextChunkSpec(
        max_chunk_tokens=int(size_part),
        overlap_tokens=int(overlap_part),
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark an OpenAI-like dense one-vector chunk baseline against "
            "a judged task encoded for Kayak exact late interaction."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--chunk-spec",
        type=_parse_chunk_spec,
        action="append",
        dest="chunk_specs",
        help=(
            "Repeat to add source token chunk sweeps in SIZE:OVERLAP form. "
            "Defaults to 800:400 and 1200:200 to mirror current OpenAI retrieval "
            "guide examples."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model used for chunk and query vectors.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional embedding dimension override passed to the OpenAI API.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size used for OpenAI embedding requests.",
    )
    parser.add_argument(
        "--tiktoken-encoding",
        type=str,
        default=None,
        help="Optional explicit tiktoken encoding name instead of model lookup.",
    )
    parser.add_argument(
        "--chunk-token-limit",
        type=int,
        default=4096,
        help=(
            "Tokenizer-side chunk-size ceiling used for validation. "
            "Defaults to 4096, matching the current OpenAI retrieval guide's "
            "max_chunk_size_tokens limit."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for embedded texts.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument(
        "--normalize-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "L2-normalize query and chunk vectors so Kayak dot-product search "
            "matches cosine-style dense retrieval."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=kayak.NUMPY_REFERENCE_BACKEND,
        help="Kayak backend used for query-time dense chunk scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_specs = args.chunk_specs or [
        RawTextChunkSpec(800, 400),
        RawTextChunkSpec(1200, 200),
    ]
    task = load_task_json(str(args.task))
    tokenizer = TiktokenSourceTokenizer(
        args.embedding_model,
        encoding_name=args.tiktoken_encoding,
        model_max_length=args.chunk_token_limit,
    )
    embedder = OpenAIEmbeddingTextBatcher(
        dimensions=args.dimensions,
        cache_dir=args.cache_dir,
    )
    bundle = build_single_vector_chunk_sweep_bundle(
        task,
        specs=chunk_specs,
        tokenizer=tokenizer,
        embed_texts_fn=embedder.embed_texts,
        embedding_model_name=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        normalize_embeddings=args.normalize_embeddings,
        backend=args.backend,
    )
    _write_json(args.output, bundle.to_json_ready())


if __name__ == "__main__":
    main()
