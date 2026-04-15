"""Owns narrow TREC run-file export for external retrieval comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import kayak


def write_trec_run(
    path: Path,
    *,
    query_ids: Sequence[str],
    hits_by_query: Sequence[Sequence[kayak.SearchHit]],
    run_name: str,
) -> None:
    if len(query_ids) != len(hits_by_query):
        raise ValueError("query ids and hit lists must have matching lengths")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id, hits in zip(query_ids, hits_by_query, strict=True):
            for rank, hit in enumerate(hits, start=1):
                handle.write(
                    f"{query_id} Q0 {hit.doc_id} {rank} {hit.score:.8f} {run_name}\n"
                )
