from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean, median
import time

import numpy as np

import kayak


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    mode: str
    index_layout: str
    batch_size: int
    query_vector_count: int
    document_count: int
    document_vector_count: int
    repeats: int
    warmup_runs: int
    seed: int


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark Python-side exact MaxSim batch dispatch."
    )
    parser.add_argument(
        "--mode",
        choices=("naive_loop", "shared_batch"),
        required=True,
    )
    parser.add_argument(
        "--index-layout",
        choices=("packed", "hybrid_flat_dim128"),
        default="hybrid_flat_dim128",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--query-vector-count", type=int, default=6)
    parser.add_argument("--document-count", type=int, default=192)
    parser.add_argument("--document-vector-count", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return BenchmarkConfig(
        mode=args.mode,
        index_layout=args.index_layout,
        batch_size=args.batch_size,
        query_vector_count=args.query_vector_count,
        document_count=args.document_count,
        document_vector_count=args.document_vector_count,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
    )


def random_tensor(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def build_inputs(config: BenchmarkConfig) -> tuple[kayak.LateQueryBatch, kayak.LateIndex]:
    rng = np.random.default_rng(config.seed)
    query_tensor = random_tensor(
        rng,
        (config.batch_size, config.query_vector_count, 128),
    )
    document_tensor = random_tensor(
        rng,
        (config.document_count, config.document_vector_count, 128),
    )
    query_batch = kayak.query_batch(query_tensor)
    index = kayak.documents(
        [f"doc-{index:04d}" for index in range(config.document_count)],
        document_tensor,
    ).pack()

    if config.index_layout == "hybrid_flat_dim128":
        return (
            query_batch.to_layout("flat_dim128"),
            index.to_layout("hybrid_flat_dim128"),
        )

    return query_batch, index


def assert_supported_environment() -> None:
    info = kayak.backend_info(kayak.MOJO_EXACT_CPU_BACKEND)
    if not info.available:
        raise SystemExit(info.availability_reason)


def assert_matching_scores(
    query_batch: kayak.LateQueryBatch, index: kayak.LateIndex
) -> None:
    numpy_scores = kayak.maxsim_batch(
        query_batch,
        index,
        backend=kayak.NUMPY_REFERENCE_BACKEND,
    )
    mojo_batch_scores = kayak.maxsim_batch(
        query_batch,
        index,
        backend=kayak.MOJO_EXACT_CPU_BACKEND,
    )
    mojo_loop_scores = tuple(
        kayak.maxsim(query, index, backend=kayak.MOJO_EXACT_CPU_BACKEND)
        for query in query_batch.queries
    )

    for actual_scores, expected_scores in zip(
        mojo_batch_scores, numpy_scores, strict=True
    ):
        np.testing.assert_allclose(
            actual_scores.numpy(),
            expected_scores.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    for actual_scores, expected_scores in zip(
        mojo_loop_scores, numpy_scores, strict=True
    ):
        np.testing.assert_allclose(
            actual_scores.numpy(),
            expected_scores.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


def run_once(
    config: BenchmarkConfig,
    query_batch: kayak.LateQueryBatch,
    index: kayak.LateIndex,
) -> None:
    if config.mode == "shared_batch":
        kayak.maxsim_batch(
            query_batch,
            index,
            backend=kayak.MOJO_EXACT_CPU_BACKEND,
        )
        return

    for query in query_batch.queries:
        kayak.maxsim(query, index, backend=kayak.MOJO_EXACT_CPU_BACKEND)


def main() -> None:
    config = parse_args()
    assert_supported_environment()
    query_batch, index = build_inputs(config)
    assert_matching_scores(query_batch, index)

    for _ in range(config.warmup_runs):
        run_once(config, query_batch, index)

    durations = []
    for _ in range(config.repeats):
        started_at = time.perf_counter()
        run_once(config, query_batch, index)
        durations.append(time.perf_counter() - started_at)

    print("benchmark: python_batch_maxsim")
    print("backend: mojo_exact_cpu")
    print(f"mode: {config.mode}")
    print(f"query_layout: {query_batch.layouts[0]}")
    print(f"index_layout: {index.layout}")
    print(f"batch_size: {query_batch.batch_size}")
    print(f"query_vector_count: {config.query_vector_count}")
    print(f"document_count: {index.document_count}")
    print(f"document_vector_count: {config.document_vector_count}")
    print(f"total_index_vectors: {index.total_vector_count}")
    print(f"warmup_runs: {config.warmup_runs}")
    print(f"repeats: {config.repeats}")
    print(f"Min: {min(durations)}")
    print(f"Median: {median(durations)}")
    print(f"Mean: {mean(durations)}")
    print(f"Max: {max(durations)}")


if __name__ == "__main__":
    main()
