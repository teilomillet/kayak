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
from kayak_bridge.raw_text_chunk_sweep import build_raw_text_chunk_sweep_bundle
from kayak_bridge.raw_text_chunk_sweep import RawTextChunkSpec


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
            "Benchmark raw-text chunked one-vector baselines on an encoded judged task."
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
            "Defaults to 256:64, 384:128, and 512:128. "
            "These defaults stay within the current ColBERT/BERT tokenizer limit, "
            "but the output still reports how often encoded chunks hit "
            "ColBERT's tighter document-vector cap."
        ),
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument(
        "--backend",
        type=str,
        default=kayak.NUMPY_REFERENCE_BACKEND,
        help="Kayak backend used for query-time scoring on the pooled chunk vectors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_specs = args.chunk_specs or [
        RawTextChunkSpec(256, 64),
        RawTextChunkSpec(384, 128),
        RawTextChunkSpec(512, 128),
    ]
    task = load_task_json(str(args.task))
    bundle = build_raw_text_chunk_sweep_bundle(
        task,
        specs=chunk_specs,
        backend=args.backend,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        encode_batch_size=args.encode_batch_size,
    )
    _write_json(args.output, bundle.to_json_ready())


if __name__ == "__main__":
    main()
