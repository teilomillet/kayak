from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

from bench_store_support import build_documents, open_store, timed_mean_seconds


def _load_summary(
    store,
    *,
    subset_doc_ids: tuple[str, ...],
    layout: str,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    return {
        "store_stats": asdict(store.stats()),
        "mean_full_load_seconds": timed_mean_seconds(
            lambda: store.load_index(layout=layout),
            warmup=warmup,
            repeats=repeats,
        ),
        "mean_subset_load_seconds": timed_mean_seconds(
            lambda: store.load_index(doc_ids=subset_doc_ids, layout=layout),
            warmup=warmup,
            repeats=repeats,
        ),
        "mean_filtered_load_seconds": timed_mean_seconds(
            lambda: store.load_index(where={"bucket": 0}, layout=layout),
            warmup=warmup,
            repeats=repeats,
        ),
    }


def _safe_ratio(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return candidate / baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bench_store_compare.py",
        description="Compare Kayak store load paths on the same synthetic corpus.",
    )
    parser.add_argument(
        "--stores",
        nargs="+",
        choices=("memory", "directory", "lancedb"),
        default=("directory", "memory"),
    )
    parser.add_argument("--baseline-store", default="directory")
    parser.add_argument("--document-count", type=int, default=1000)
    parser.add_argument("--vectors-per-document", type=int, default=16)
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--layout", default="packed")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if len(args.stores) < 2:
        raise ValueError("stores must contain at least two store kinds")
    if args.baseline_store not in args.stores:
        raise ValueError("baseline-store must be present in stores")
    if args.document_count <= 0:
        raise ValueError("document-count must be positive")
    if args.vectors_per_document <= 0:
        raise ValueError("vectors-per-document must be positive")
    if args.vector_dim <= 0:
        raise ValueError("vector-dim must be positive")
    if args.subset_size <= 0:
        raise ValueError("subset-size must be positive")
    if args.subset_size > args.document_count:
        raise ValueError("subset-size must not exceed document-count")

    documents, metadata = build_documents(
        document_count=args.document_count,
        vectors_per_document=args.vectors_per_document,
        vector_dim=args.vector_dim,
    )
    subset_doc_ids = documents.doc_ids[: args.subset_size]

    summaries: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="kayak-store-compare-bench-") as tmpdir:
        tmp_root = Path(tmpdir)
        for store_kind in args.stores:
            store = open_store(
                store_kind,
                path=(
                    tmp_root / store_kind
                    if store_kind in {"directory", "lancedb"}
                    else None
                ),
                table_name=f"{store_kind}_docs",
            )
            store.upsert(documents, metadata=metadata)
            summaries[store_kind] = _load_summary(
                store,
                subset_doc_ids=subset_doc_ids,
                layout=args.layout,
                warmup=args.warmup,
                repeats=args.repeats,
            )

    baseline = summaries[args.baseline_store]
    ratios = {
        store_kind: {
            "full_load_ratio_vs_baseline": _safe_ratio(
                float(summary["mean_full_load_seconds"]),
                float(baseline["mean_full_load_seconds"]),
            ),
            "subset_load_ratio_vs_baseline": _safe_ratio(
                float(summary["mean_subset_load_seconds"]),
                float(baseline["mean_subset_load_seconds"]),
            ),
            "filtered_load_ratio_vs_baseline": _safe_ratio(
                float(summary["mean_filtered_load_seconds"]),
                float(baseline["mean_filtered_load_seconds"]),
            ),
        }
        for store_kind, summary in summaries.items()
    }

    payload = {
        "stores": list(args.stores),
        "baseline_store": args.baseline_store,
        "document_count": args.document_count,
        "vectors_per_document": args.vectors_per_document,
        "vector_dim": args.vector_dim,
        "subset_size": args.subset_size,
        "layout": args.layout,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "summaries": summaries,
        "ratios": ratios,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
