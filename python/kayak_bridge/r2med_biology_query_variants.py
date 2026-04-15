"""Owns R2MED/Biology public query-variant loading and caching.

This module owns:
- public query-variant specs for the `R2MED/Biology` benchmark
- fallback from empty rewrite text back to the original benchmark query
- encoded query caches for late-interaction benchmarking

This module does not own:
- corpus snapshots
- score fusion
- leaderboard comparison
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

from .colbert_encoder import DEFAULT_MODEL_NAME, encode_query_text
from .r2med_biology_full import (
    DEFAULT_DATASET_ID,
    EncodedJudgedQuery,
    EncodedJudgedQueryCache,
    _load_positive_doc_ids_by_query,
)

from .cache_paths import configure_local_caches

configure_local_caches()

from datasets import load_dataset


@dataclass(frozen=True, slots=True)
class R2MEDQueryVariantSpec:
    name: str
    config_name: str
    split_name: str
    text_field: str


R2MED_QUERY_VARIANTS = (
    R2MEDQueryVariantSpec("query", "query", "query", "text"),
    R2MEDQueryVariantSpec("hyde_gpt4", "hyde", "gpt4", "hy_doc"),
    R2MEDQueryVariantSpec("hyde_o3_mini", "hyde", "o3_mini", "hy_doc"),
    R2MEDQueryVariantSpec("query2doc_gpt4", "query2doc", "gpt4", "hy_doc"),
    R2MEDQueryVariantSpec("lamer_gpt4", "lamer", "gpt4", "hy_doc"),
    R2MEDQueryVariantSpec(
        "search_r1_qwen3b_ins",
        "search-r1",
        "qwen_3b_ins",
        "hy_doc",
    ),
    R2MEDQueryVariantSpec(
        "search_r1_qwen7b_ins",
        "search-r1",
        "qwen_7b_ins",
        "hy_doc",
    ),
    R2MEDQueryVariantSpec("search_o1_qwq32b", "search-o1", "qwq_32b", "hy_doc"),
    R2MEDQueryVariantSpec(
        "search_o1_qwen3_32b",
        "search-o1",
        "qwen3_32b",
        "hy_doc",
    ),
)


def default_r2med_query_variant(spec_name: str) -> R2MEDQueryVariantSpec:
    for spec in R2MED_QUERY_VARIANTS:
        if spec.name == spec_name:
            return spec
    raise ValueError(f"unknown R2MED query variant: {spec_name}")


def _base_query_texts(*, dataset_id: str) -> dict[str, str]:
    return {
        str(row["id"]): str(row["text"])
        for row in load_dataset(dataset_id, "query", split="query")
    }


def _load_variant_texts(
    *,
    dataset_id: str,
    spec: R2MEDQueryVariantSpec,
) -> dict[str, str]:
    base_query_texts = _base_query_texts(dataset_id=dataset_id)
    if spec.name == "query":
        return base_query_texts

    texts_by_query_id: dict[str, str] = {}
    for row in load_dataset(dataset_id, spec.config_name, split=spec.split_name):
        query_id = str(row["id"])
        text = str(row[spec.text_field]).strip()
        if not text:
            text = base_query_texts[query_id]
        texts_by_query_id[query_id] = text
    return texts_by_query_id


def _mean_query_vector_count(queries: tuple[EncodedJudgedQuery, ...]) -> int:
    if not queries:
        return 0

    total = 0
    for query in queries:
        total += query.vector_count
    return round(total / len(queries))


def build_r2med_query_variant_cache(
    *,
    spec: R2MEDQueryVariantSpec,
    dataset_id: str = DEFAULT_DATASET_ID,
    model_name: str = DEFAULT_MODEL_NAME,
) -> EncodedJudgedQueryCache:
    positives_by_query = _load_positive_doc_ids_by_query(dataset_id=dataset_id)
    texts_by_query_id = _load_variant_texts(dataset_id=dataset_id, spec=spec)

    encoded_queries: list[EncodedJudgedQuery] = []
    vector_dim = 0
    for query_id in sorted(texts_by_query_id, key=int):
        relevant_doc_ids = positives_by_query.get(query_id)
        if not relevant_doc_ids:
            continue

        text = texts_by_query_id[query_id]
        vectors = encode_query_text(text, model_name)
        if vectors:
            vector_dim = len(vectors[0])
        encoded_queries.append(
            EncodedJudgedQuery(
                query_id=query_id,
                text=text,
                relevant_doc_ids=relevant_doc_ids,
                vector_count=len(vectors),
                vectors=vectors,
            )
        )

    return EncodedJudgedQueryCache(
        dataset_id=dataset_id,
        model_name=model_name,
        vector_dim=vector_dim,
        query_count=len(encoded_queries),
        nominal_query_vector_count=_mean_query_vector_count(tuple(encoded_queries)),
        queries=tuple(encoded_queries),
    )


def ensure_r2med_query_variant_cache(
    *,
    path: Path,
    spec: R2MEDQueryVariantSpec,
    dataset_id: str = DEFAULT_DATASET_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    force_rebuild: bool = False,
) -> tuple[EncodedJudgedQueryCache, float, bool]:
    if path.exists() and not force_rebuild:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        queries = tuple(
            EncodedJudgedQuery(
                query_id=str(row["query_id"]),
                text=str(row["text"]),
                relevant_doc_ids=tuple(
                    str(doc_id) for doc_id in row["relevant_doc_ids"]
                ),
                vector_count=int(row["vector_count"]),
                vectors=row["vectors"],
            )
            for row in payload["queries"]
        )
        return (
            EncodedJudgedQueryCache(
                dataset_id=str(payload["dataset_id"]),
                model_name=str(payload["model_name"]),
                vector_dim=int(payload["vector_dim"]),
                query_count=int(payload["query_count"]),
                nominal_query_vector_count=int(payload["nominal_query_vector_count"]),
                queries=queries,
            ),
            0.0,
            True,
        )

    start = time.perf_counter()
    cache = build_r2med_query_variant_cache(
        spec=spec,
        dataset_id=dataset_id,
        model_name=model_name,
    )
    elapsed_seconds = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return cache, elapsed_seconds, False
