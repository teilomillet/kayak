"""Owns task benchmarks for exported latent-proxy artifacts.

This module keeps the artifact question explicit:
- load one exported latent-proxy sidecar
- score stage-1 candidates through the artifact reference evaluator
- exact-rerank the shortlist with Kayak MaxSim on the same task JSON

It does not claim native collection serving. It is the Python-side benchmark
bridge that lets us compare exported artifacts against trained checkpoints
under the same judged task contract.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import kayak

from .latent_proxy_reference import load_latent_proxy_reference_model
from .lemur_task_benchmark import (
    LemurTaskBenchmarkSummary,
    benchmark_queries_with_reference_lemur_model,
    rank_queries_with_reference_lemur_model,
)


def _build_index(task: Mapping[str, Any]) -> kayak.LateIndex:
    return kayak.documents(
        [document["doc_id"] for document in task["documents"]],
        [document["vectors"] for document in task["documents"]],
        texts=[document["text"] for document in task["documents"]],
    ).pack()


def _build_queries(task: Mapping[str, Any]) -> tuple[kayak.LateQuery, ...]:
    return tuple(
        kayak.query(query["vectors"], text=query["text"])
        for query in task["queries"]
    )


def benchmark_queries_with_latent_proxy_artifact_model(
    *,
    task: Mapping[str, Any],
    index: kayak.LateIndex,
    queries: Sequence[kayak.LateQuery],
    model: Any,
    candidate_k: int,
    final_k: int | None = None,
    warmup_iterations: int = 2,
    measurement_iterations: int = 25,
    rerank_backend: str = kayak.NUMPY_REFERENCE_BACKEND,
) -> LemurTaskBenchmarkSummary:
    summary = benchmark_queries_with_reference_lemur_model(
        task=task,
        index=index,
        queries=queries,
        model=model,
        candidate_k=candidate_k,
        final_k=final_k,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        rerank_backend=rerank_backend,
    )
    return replace(
        summary,
        activation=str(model.activation),
        query_divisor=float(model.query_divisor),
        apply_layer_norm=bool(model.apply_layer_norm),
        latent_dim=int(model.latent_dim),
        landmark_count=int(model.landmark_count),
        engine="latent_proxy_artifact_reference",
        index_kind="latent_proxy_artifact",
        stage1_backend="latent_proxy_artifact_reference",
    )


def rank_task_with_latent_proxy_artifact(
    task: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    candidate_k: int,
    device: str = "cpu",
    final_k: int | None = None,
    rerank_backend: str = kayak.NUMPY_REFERENCE_BACKEND,
) -> tuple[tuple[str, ...], ...]:
    index = _build_index(task)
    queries = _build_queries(task)
    model = load_latent_proxy_reference_model(artifact_root, device=device)
    return rank_queries_with_reference_lemur_model(
        task=task,
        index=index,
        queries=queries,
        model=model,
        candidate_k=candidate_k,
        final_k=final_k,
        rerank_backend=rerank_backend,
    )


def benchmark_task_with_latent_proxy_artifact(
    task: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    candidate_k: int,
    device: str = "cpu",
    final_k: int | None = None,
    warmup_iterations: int = 2,
    measurement_iterations: int = 25,
    rerank_backend: str = kayak.NUMPY_REFERENCE_BACKEND,
) -> LemurTaskBenchmarkSummary:
    index = _build_index(task)
    queries = _build_queries(task)
    model = load_latent_proxy_reference_model(artifact_root, device=device)
    return benchmark_queries_with_latent_proxy_artifact_model(
        task=task,
        index=index,
        queries=queries,
        model=model,
        candidate_k=candidate_k,
        final_k=final_k,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        rerank_backend=rerank_backend,
    )
