from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.colbert_encoder import DEFAULT_MODEL_NAME  # noqa: E402
from kayak_bridge.msmarco_passage_task import (  # noqa: E402
    DEFAULT_DATASET_ID,
    MsmarcoPassagePaths,
    build_msmarco_passage_task,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an encoded Kayak task JSON from local official MS MARCO "
            "passage ranking files. This command does not download MS MARCO."
        )
    )
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/msmarco_passage/python_task.json"),
    )
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument(
        "--allow-full-corpus-json",
        action="store_true",
        help=(
            "Allow building an encoded JSON task with no document limit. This "
            "can be very large; corpus-scale production runs should use a "
            "streaming snapshot path instead."
        ),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--document-batch-size",
        type=int,
        default=8,
        help=(
            "ColBERT document encoding batch size. Keep explicit because "
            "artifact build time and peak memory depend on it."
        ),
    )
    parser.add_argument(
        "--no-include-relevant-documents",
        action="store_true",
        help=(
            "Do not force selected query positives into limited document "
            "subsets. Full-corpus runs do not need this flag."
        ),
    )
    parser.add_argument(
        "--include-document-token-ids",
        action="store_true",
        help=(
            "Store ColBERT document tokenizer input ids aligned to document "
            "vectors. This is required for token-aware clustering benchmarks."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.document_limit is None and not args.allow_full_corpus_json:
        raise ValueError(
            "building full MS MARCO into one encoded JSON task is intentionally "
            "guarded; pass --allow-full-corpus-json only for deliberate runs"
        )
    task, selection = build_msmarco_passage_task(
        MsmarcoPassagePaths(
            collection=args.collection,
            queries=args.queries,
            qrels=args.qrels,
        ),
        document_limit=args.document_limit,
        query_limit=args.query_limit,
        include_relevant_documents=not args.no_include_relevant_documents,
        include_document_token_ids=args.include_document_token_ids,
        document_batch_size=args.document_batch_size,
        model_name=args.model_name,
        dataset_id=args.dataset_id,
    )
    task["source_selection"] = selection.to_json_ready()
    task["created_at_utc"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(task, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}")
    print(json.dumps(selection.to_json_ready(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
