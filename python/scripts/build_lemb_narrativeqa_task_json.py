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
from kayak_bridge.lemb_narrativeqa_subset import (  # noqa: E402
    DEFAULT_DATASET_ID,
    build_lemb_narrativeqa_colbert_subset,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an encoded Kayak task JSON for the LEMB NarrativeQA "
            "long-document retrieval slice."
        )
    )
    parser.add_argument("--query-limit", type=int, default=8)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/lemb_narrativeqa/python_task.json"),
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
    if args.query_limit <= 0:
        raise ValueError("query_limit must be positive")
    task = build_lemb_narrativeqa_colbert_subset(
        query_limit=args.query_limit,
        model_name=args.model_name,
        dataset_id=args.dataset_id,
        include_document_token_ids=args.include_document_token_ids,
    )
    task["created_at_utc"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(task, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}")
    print(
        json.dumps(
            {
                "document_count": len(task["documents"]),
                "query_count": len(task["queries"]),
                "nominal_document_vector_count": task[
                    "nominal_document_vector_count"
                ],
                "nominal_query_vector_count": task["nominal_query_vector_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
