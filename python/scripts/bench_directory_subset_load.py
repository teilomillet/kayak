from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile

import numpy as np

import kayak

from bench_store_support import build_documents, timed_mean_seconds


def _legacy_reference_subset_load(
    store: kayak.DirectoryLateStore,
    *,
    doc_ids: tuple[str, ...],
    include_text: bool,
) -> kayak.LateIndex:
    token_vectors = np.load(store._token_vectors_path, mmap_mode="r")
    positions = tuple(store._doc_positions[doc_id] for doc_id in doc_ids)
    selected_texts = (
        None
        if not include_text or store._doc_texts is None
        else tuple(store._doc_texts[position] for position in positions)
    )
    selected_matrices = tuple(
        token_vectors[
            int(store._doc_offsets[position]) : int(store._doc_offsets[position + 1])
        ]
        for position in positions
    )
    return kayak.documents(
        doc_ids,
        selected_matrices,
        texts=selected_texts,
    ).pack()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bench_directory_subset_load.py",
        description="Compare the current directory subset load path with a legacy reference.",
    )
    parser.add_argument("--document-count", type=int, default=5000)
    parser.add_argument("--vectors-per-document", type=int, default=32)
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--subset-size", type=int, default=2000)
    parser.add_argument("--shuffle-seed", type=int, default=0)
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
    if args.subset_size <= 0:
        raise ValueError("subset-size must be positive")
    if args.subset_size > args.document_count:
        raise ValueError("subset-size must not exceed document-count")

    documents, metadata = build_documents(
        document_count=args.document_count,
        vectors_per_document=args.vectors_per_document,
        vector_dim=args.vector_dim,
    )

    with tempfile.TemporaryDirectory(prefix="kayak-directory-subset-bench-") as tmpdir:
        store = kayak.DirectoryLateStore(Path(tmpdir) / "store")
        store.upsert(documents, metadata=metadata)
        reopened = kayak.DirectoryLateStore(Path(tmpdir) / "store")

        selected_doc_ids = list(reopened._doc_ids[: args.subset_size])
        random.Random(args.shuffle_seed).shuffle(selected_doc_ids)
        selected_doc_ids_tuple = tuple(selected_doc_ids)

        optimized = reopened.load_index(
            doc_ids=selected_doc_ids_tuple,
            include_text=True,
        )
        reference = _legacy_reference_subset_load(
            reopened,
            doc_ids=selected_doc_ids_tuple,
            include_text=True,
        )
        if optimized.doc_ids != reference.doc_ids:
            raise AssertionError("optimized subset load changed the requested order")
        if optimized.doc_texts != reference.doc_texts:
            raise AssertionError("optimized subset load changed document texts")
        if not np.array_equal(optimized.doc_offsets, reference.doc_offsets):
            raise AssertionError("optimized subset load changed packed offsets")
        if not np.allclose(
            optimized.as_packed_token_matrix(),
            reference.as_packed_token_matrix(),
        ):
            raise AssertionError("optimized subset load changed token vectors")

        optimized_seconds = timed_mean_seconds(
            lambda: reopened.load_index(
                doc_ids=selected_doc_ids_tuple,
                include_text=True,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        reference_seconds = timed_mean_seconds(
            lambda: _legacy_reference_subset_load(
                reopened,
                doc_ids=selected_doc_ids_tuple,
                include_text=True,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        payload = {
            "document_count": args.document_count,
            "vectors_per_document": args.vectors_per_document,
            "vector_dim": args.vector_dim,
            "subset_size": args.subset_size,
            "shuffle_seed": args.shuffle_seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "mean_subset_load_seconds": optimized_seconds,
            "mean_legacy_reference_seconds": reference_seconds,
            "latency_ratio_vs_reference": (
                None if reference_seconds == 0.0 else optimized_seconds / reference_seconds
            ),
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
