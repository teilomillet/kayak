from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import kayak

from bench_store_support import build_documents, build_query, open_store, timed_mean_seconds


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bench_store_search.py",
        description="Benchmark exact search on indexes loaded from Kayak stores.",
    )
    parser.add_argument(
        "--store",
        choices=("memory", "directory", "lancedb"),
        default="directory",
    )
    parser.add_argument("--document-count", type=int, default=1000)
    parser.add_argument("--vectors-per-document", type=int, default=16)
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--candidate-window-size", type=int, default=50)
    parser.add_argument("--layout", default="packed")
    parser.add_argument("--backend", default=kayak.NUMPY_REFERENCE_BACKEND)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.document_count <= 0:
        raise ValueError("document-count must be positive")
    if args.vectors_per_document <= 0:
        raise ValueError("vectors-per-document must be positive")
    if args.vector_dim <= 0:
        raise ValueError("vector-dim must be positive")
    if args.candidate_window_size <= 0:
        raise ValueError("candidate-window-size must be positive")
    if args.candidate_window_size > args.document_count:
        raise ValueError("candidate-window-size must not exceed document-count")

    documents, metadata = build_documents(
        document_count=args.document_count,
        vectors_per_document=args.vectors_per_document,
        vector_dim=args.vector_dim,
    )
    query = build_query(
        vectors_per_document=args.vectors_per_document,
        vector_dim=args.vector_dim,
    )
    candidate_doc_ids = documents.doc_ids[: args.candidate_window_size]

    with tempfile.TemporaryDirectory(prefix="kayak-store-search-bench-") as tmpdir:
        store_path = Path(tmpdir) / "store"
        store = open_store(
            args.store,
            path=store_path if args.store in {"directory", "lancedb"} else None,
        )
        store.upsert(documents, metadata=metadata)

        full_index = store.load_index(layout=args.layout)
        candidate_index = store.load_index(
            doc_ids=candidate_doc_ids,
            layout=args.layout,
        )

        full_exact_search_seconds = timed_mean_seconds(
            lambda: full_index.maxsim(query, backend=args.backend),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        candidate_exact_search_seconds = timed_mean_seconds(
            lambda: candidate_index.maxsim(query, backend=args.backend),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        payload = {
            "store_kind": args.store,
            "document_count": args.document_count,
            "vectors_per_document": args.vectors_per_document,
            "vector_dim": args.vector_dim,
            "candidate_window_size": args.candidate_window_size,
            "layout": args.layout,
            "backend": args.backend,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "store_stats": asdict(store.stats()),
            "full_index_document_count": full_index.document_count,
            "candidate_index_document_count": candidate_index.document_count,
            "query_vector_count": query.vector_count,
            "mean_full_exact_search_seconds": full_exact_search_seconds,
            "mean_candidate_exact_search_seconds": candidate_exact_search_seconds,
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
