from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.task_json_catalog import default_task_json_output_path
from kayak_engine import (
    ExactScoringOptions,
    PreparedExactSearchRuntimeConfig,
    PreparedExactSearchRuntimeStats,
    prepare_exact_search_runtime,
    prepare_exact_search_session,
)
from kayak_engine.mojo_service import load_module
from kayak_engine.payloads import document_payload_parts


COLLECTION_ID = "prepared-runtime-bench"
TENANT_ID = "public"
NAMESPACE_ID = "benchmark"
SNAPSHOT_ID = "snapshot-0001"


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkRow:
    scoring_mode: str
    concurrency_lane_count: int
    worker_count: int
    request_pool_count: int
    load_text_corpus: bool
    prepare_ready_seconds: float
    mean_batch_seconds: float
    throughput_queries_per_second: float
    executed_batch_count: int
    average_batch_size: float
    average_queue_wait_ms: float
    average_batch_execution_ms: float
    ready_parent_rss_kib: int
    ready_worker_rss_kib: int
    ready_total_rss_kib: int
    ready_worker_pid_count: int
    enable_parallel_scoring: bool
    enable_parallel_work_item_oversubscription: bool
    parallel_work_item_count_override: int

    def to_json_ready(self) -> dict[str, object]:
        return {
            "scoring_mode": self.scoring_mode,
            "concurrency_lane_count": self.concurrency_lane_count,
            "worker_count": self.worker_count,
            "request_pool_count": self.request_pool_count,
            "load_text_corpus": self.load_text_corpus,
            "prepare_ready_seconds": self.prepare_ready_seconds,
            "mean_batch_seconds": self.mean_batch_seconds,
            "throughput_queries_per_second": self.throughput_queries_per_second,
            "executed_batch_count": self.executed_batch_count,
            "average_batch_size": self.average_batch_size,
            "average_queue_wait_ms": self.average_queue_wait_ms,
            "average_batch_execution_ms": self.average_batch_execution_ms,
            "ready_parent_rss_kib": self.ready_parent_rss_kib,
            "ready_worker_rss_kib": self.ready_worker_rss_kib,
            "ready_total_rss_kib": self.ready_total_rss_kib,
            "ready_worker_pid_count": self.ready_worker_pid_count,
            "enable_parallel_scoring": self.enable_parallel_scoring,
            "enable_parallel_work_item_oversubscription": (
                self.enable_parallel_work_item_oversubscription
            ),
            "parallel_work_item_count_override": (
                self.parallel_work_item_count_override
            ),
        }


def _parse_csv_ints(value: str) -> list[int]:
    counts: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped == "":
            continue
        parsed = int(stripped)
        if parsed < 1:
            raise ValueError("counts must be positive")
        counts.append(parsed)
    if len(counts) == 0:
        raise ValueError("at least one positive count is required")
    return counts


def _parse_csv_strings(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip() != ""]
    if len(values) == 0:
        raise ValueError("at least one mode is required")
    return values


def _parse_csv_bools(value: str) -> list[bool]:
    values: list[bool] = []
    for part in value.split(","):
        normalized = part.strip().lower()
        if normalized == "":
            continue
        if normalized == "true":
            values.append(True)
            continue
        if normalized == "false":
            values.append(False)
            continue
        raise ValueError(
            "boolean CSV values must be 'true' or 'false'; "
            f"got {part!r}"
        )
    if len(values) == 0:
        raise ValueError("at least one boolean value is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the Python prepared exact-search runtime over explicit "
            "lane and worker configurations."
        )
    )
    parser.add_argument("--task", type=Path, default=None)
    parser.add_argument("--task-key", type=str, default="browsecomp_plus_gold")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measurement-iterations", type=int, default=10)
    parser.add_argument("--request-pool-count", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-wait-ms", type=int, default=1)
    parser.add_argument(
        "--load-text-corpus-values",
        type=str,
        default="false",
        help="Comma-separated boolean values such as false,true",
    )
    parser.add_argument("--lane-counts", type=str, default="1,2,4")
    parser.add_argument("--worker-counts", type=str, default="1,2,4")
    parser.add_argument(
        "--scoring-modes",
        type=str,
        default="default_auto,shared_host_budget,serial_inner",
    )
    return parser.parse_args()


def _task_path(args: argparse.Namespace) -> Path:
    if args.task is not None:
        return args.task
    return default_task_json_output_path(args.task_key)


def _load_task(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"task json does not exist: {path}")
    return load_task_json(str(path))


def _create_collection(
    *,
    module: object,
    service_root: Path,
    task: dict[str, Any],
) -> None:
    json.loads(
        module.create_collection_json(
            str(service_root),
            {
                "collection_id": COLLECTION_ID,
                "tenant_id": TENANT_ID,
                "namespace_id": NAMESPACE_ID,
                "collection_layout_family": "",
                "model_name": str(task["model_name"]),
                "vector_scalar_name": "",
                "vector_dim": int(task["vector_dim"]),
                "default_keep_latest_inactive_count": 1,
            },
        )
    )


def _upsert_documents(
    *,
    module: object,
    service_root: Path,
    task: dict[str, Any],
) -> None:
    documents = [
        {
            "doc_id": str(document["doc_id"]),
            "vectors": document["vectors"],
            "text": str(document["text"]),
            "metadata": {},
        }
        for document in task["documents"]
    ]
    (
        doc_ids,
        document_vectors,
        texts,
        metadata_keys,
        metadata_values,
    ) = document_payload_parts({"documents": documents})
    json.loads(
        module.upsert_documents_json(
            str(service_root),
            {
                "collection_id": COLLECTION_ID,
                "tenant_id": TENANT_ID,
                "namespace_id": NAMESPACE_ID,
                "doc_ids": doc_ids,
                "document_vectors": document_vectors,
                "texts": texts,
                "metadata_keys": metadata_keys,
                "metadata_values": metadata_values,
            },
        )
    )


def _create_snapshot(*, module: object, service_root: Path) -> None:
    json.loads(
        module.create_snapshot_json(
            str(service_root),
            {
                "collection_id": COLLECTION_ID,
                "tenant_id": TENANT_ID,
                "namespace_id": NAMESPACE_ID,
                "snapshot_id": SNAPSHOT_ID,
                "reason": "prepared runtime benchmark",
            },
        )
    )


def _build_request_pool(
    task: dict[str, Any],
    *,
    request_pool_count: int,
) -> list[dict[str, Any]]:
    base_requests = [
        {
            "query_model_name": str(task["model_name"]),
            "query_text": str(query["text"]),
            "query": query["vectors"],
            "final_k": int(task["k"]),
        }
        for query in task["queries"]
    ]
    if len(base_requests) == 0:
        raise ValueError("task must contain at least one query")

    return [
        dict(base_requests[index % len(base_requests)])
        for index in range(request_pool_count)
    ]


def _scoring_options(
    *,
    scoring_mode: str,
    concurrency_lane_count: int,
    worker_count: int,
) -> ExactScoringOptions:
    if scoring_mode == "default_auto":
        return ExactScoringOptions()
    if scoring_mode == "serial_inner":
        return ExactScoringOptions(enable_parallel_scoring=False)
    if scoring_mode == "shared_host_budget":
        host_parallelism = os.cpu_count() or 1
        budget = host_parallelism // max(1, concurrency_lane_count * worker_count)
        if budget < 1:
            budget = 1
        return ExactScoringOptions(
            enable_parallel_work_item_oversubscription=False,
            parallel_work_item_count_override=budget,
        )
    raise ValueError(f"unknown scoring mode: {scoring_mode}")


def _stats_delta(
    before: PreparedExactSearchRuntimeStats,
    after: PreparedExactSearchRuntimeStats,
) -> tuple[int, float, float, float]:
    processed_request_count = (
        after.processed_request_count - before.processed_request_count
    )
    executed_batch_count = after.executed_batch_count - before.executed_batch_count
    total_queue_wait_seconds = (
        after.total_queue_wait_seconds - before.total_queue_wait_seconds
    )
    total_batch_execution_seconds = (
        after.total_batch_execution_seconds - before.total_batch_execution_seconds
    )
    average_batch_size = 0.0
    average_queue_wait_ms = 0.0
    average_batch_execution_ms = 0.0
    if executed_batch_count > 0:
        average_batch_size = processed_request_count / float(executed_batch_count)
        average_batch_execution_ms = (
            total_batch_execution_seconds * 1000.0
        ) / float(executed_batch_count)
    if processed_request_count > 0:
        average_queue_wait_ms = (
            total_queue_wait_seconds * 1000.0
        ) / float(processed_request_count)
    return (
        executed_batch_count,
        average_batch_size,
        average_queue_wait_ms,
        average_batch_execution_ms,
    )


def _rss_kib_for_pids(pids: list[int]) -> int:
    live_pids = [pid for pid in pids if pid > 0]
    if len(live_pids) == 0:
        return 0

    result = subprocess.run(
        [
            "ps",
            "-o",
            "rss=",
            "-p",
            ",".join(str(pid) for pid in live_pids),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total_rss_kib = 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        total_rss_kib += int(stripped)
    return total_rss_kib


def _assert_runtime_matches_session(
    *,
    session: object,
    runtime: object,
    requests: list[dict[str, Any]],
    scoring: ExactScoringOptions,
) -> None:
    expected = [session.search(request, scoring=scoring) for request in requests]
    futures = [runtime.submit(request) for request in requests]
    observed = [future.result(timeout=30.0) for future in futures]
    if expected != observed:
        raise AssertionError("prepared exact runtime results did not match session")


def _measure_runtime(
    *,
    runtime: object,
    requests: list[dict[str, Any]],
    warmup_iterations: int,
    measurement_iterations: int,
) -> tuple[float, int, float, float, float]:
    for _ in range(warmup_iterations):
        futures = [runtime.submit(request) for request in requests]
        _ = [future.result(timeout=30.0) for future in futures]

    stats_before = runtime.stats()
    elapsed_seconds: list[float] = []
    for _ in range(measurement_iterations):
        started_at = time.perf_counter()
        futures = [runtime.submit(request) for request in requests]
        _ = [future.result(timeout=30.0) for future in futures]
        elapsed_seconds.append(time.perf_counter() - started_at)
    stats_after = runtime.stats()
    (
        executed_batch_count,
        average_batch_size,
        average_queue_wait_ms,
        average_batch_execution_ms,
    ) = _stats_delta(stats_before, stats_after)
    return (
        sum(elapsed_seconds) / float(len(elapsed_seconds)),
        executed_batch_count,
        average_batch_size,
        average_queue_wait_ms,
        average_batch_execution_ms,
    )


def main() -> None:
    args = parse_args()
    task_path = _task_path(args)
    task = _load_task(task_path)
    lane_counts = _parse_csv_ints(args.lane_counts)
    worker_counts = _parse_csv_ints(args.worker_counts)
    scoring_modes = _parse_csv_strings(args.scoring_modes)
    load_text_corpus_values = _parse_csv_bools(args.load_text_corpus_values)

    if args.warmup_iterations < 0:
        raise ValueError("warmup-iterations must be non-negative")
    if args.measurement_iterations <= 0:
        raise ValueError("measurement-iterations must be positive")
    if args.request_pool_count <= 0:
        raise ValueError("request-pool-count must be positive")
    if args.max_batch_size <= 0:
        raise ValueError("max-batch-size must be positive")
    if args.max_batch_wait_ms < 0:
        raise ValueError("max-batch-wait-ms must be non-negative")

    request_pool = _build_request_pool(
        task,
        request_pool_count=args.request_pool_count,
    )
    rows: list[RuntimeBenchmarkRow] = []

    with tempfile.TemporaryDirectory(prefix="kayak-prepared-runtime-bench-") as tmpdir:
        service_root = Path(tmpdir) / "service-root"
        module = load_module()
        _create_collection(module=module, service_root=service_root, task=task)
        _upsert_documents(module=module, service_root=service_root, task=task)
        _create_snapshot(module=module, service_root=service_root)

        session_prepare_started_at = time.perf_counter()
        session = prepare_exact_search_session(
            service_root=service_root,
            collection_id=COLLECTION_ID,
            tenant_id=TENANT_ID,
            namespace_id=NAMESPACE_ID,
            snapshot_id=SNAPSHOT_ID,
            load_text_corpus=False,
        )
        session_prepare_seconds = time.perf_counter() - session_prepare_started_at

        print(f"task={task_path}")
        print(f"dataset_id={task['dataset_id']}")
        print(f"slice_name={task['slice_name']}")
        print(f"request_pool_count={len(request_pool)}")
        print(f"max_batch_size={args.max_batch_size}")
        print(f"max_batch_wait_ms={args.max_batch_wait_ms}")
        print(f"document_count={len(task['documents'])}")
        print(f"query_count={len(task['queries'])}")
        print(f"vector_dim={task['vector_dim']}")
        print(f"service_root={service_root}")
        print(f"session_prepare_seconds={session_prepare_seconds:.12f}")

        for load_text_corpus in load_text_corpus_values:
            for concurrency_lane_count in lane_counts:
                for worker_count in worker_counts:
                    for scoring_mode in scoring_modes:
                        scoring = _scoring_options(
                            scoring_mode=scoring_mode,
                            concurrency_lane_count=concurrency_lane_count,
                            worker_count=worker_count,
                        )
                        prepare_started_at = time.perf_counter()
                        runtime = prepare_exact_search_runtime(
                            service_root=service_root,
                            collection_id=COLLECTION_ID,
                            tenant_id=TENANT_ID,
                            namespace_id=NAMESPACE_ID,
                            snapshot_id=SNAPSHOT_ID,
                            load_text_corpus=load_text_corpus,
                            config=PreparedExactSearchRuntimeConfig(
                                concurrency_lane_count=concurrency_lane_count,
                                worker_count=worker_count,
                                max_batch_size=args.max_batch_size,
                                max_batch_wait_ms=args.max_batch_wait_ms,
                                scoring=scoring,
                            ),
                        )
                        try:
                            runtime.wait_until_ready(timeout=60.0)
                            prepare_ready_seconds = (
                                time.perf_counter() - prepare_started_at
                            )
                            ready_parent_rss_kib = _rss_kib_for_pids([os.getpid()])
                            ready_worker_pids = list(runtime.worker_pids())
                            ready_worker_rss_kib = _rss_kib_for_pids(ready_worker_pids)
                            _assert_runtime_matches_session(
                                session=session,
                                runtime=runtime,
                                requests=request_pool,
                                scoring=scoring,
                            )
                            (
                                mean_batch_seconds,
                                executed_batch_count,
                                average_batch_size,
                                average_queue_wait_ms,
                                average_batch_execution_ms,
                            ) = _measure_runtime(
                                runtime=runtime,
                                requests=request_pool,
                                warmup_iterations=args.warmup_iterations,
                                measurement_iterations=args.measurement_iterations,
                            )
                        finally:
                            runtime.close()

                        row = RuntimeBenchmarkRow(
                            scoring_mode=scoring_mode,
                            concurrency_lane_count=concurrency_lane_count,
                            worker_count=worker_count,
                            request_pool_count=len(request_pool),
                            load_text_corpus=load_text_corpus,
                            prepare_ready_seconds=prepare_ready_seconds,
                            mean_batch_seconds=mean_batch_seconds,
                            throughput_queries_per_second=(
                                len(request_pool) / mean_batch_seconds
                            ),
                            executed_batch_count=executed_batch_count,
                            average_batch_size=average_batch_size,
                            average_queue_wait_ms=average_queue_wait_ms,
                            average_batch_execution_ms=average_batch_execution_ms,
                            ready_parent_rss_kib=ready_parent_rss_kib,
                            ready_worker_rss_kib=ready_worker_rss_kib,
                            ready_total_rss_kib=(
                                ready_parent_rss_kib + ready_worker_rss_kib
                            ),
                            ready_worker_pid_count=len(ready_worker_pids),
                            enable_parallel_scoring=scoring.enable_parallel_scoring,
                            enable_parallel_work_item_oversubscription=(
                                scoring.enable_parallel_work_item_oversubscription
                            ),
                            parallel_work_item_count_override=(
                                scoring.parallel_work_item_count_override
                            ),
                        )
                        rows.append(row)
                        print(
                            "RESULT"
                            f"\tload_text_corpus={str(row.load_text_corpus).lower()}"
                            f"\tlanes={row.concurrency_lane_count}"
                            f"\tworkers={row.worker_count}"
                            f"\tscoring_mode={row.scoring_mode}"
                            f"\tprepare_ready_seconds={row.prepare_ready_seconds:.12f}"
                            f"\tmean_batch_seconds={row.mean_batch_seconds:.12f}"
                            f"\tthroughput_qps={row.throughput_queries_per_second:.3f}"
                            f"\texecuted_batch_count={row.executed_batch_count}"
                            f"\taverage_batch_size={row.average_batch_size:.3f}"
                            f"\taverage_queue_wait_ms={row.average_queue_wait_ms:.3f}"
                            f"\taverage_batch_execution_ms={row.average_batch_execution_ms:.3f}"
                            f"\tready_parent_rss_kib={row.ready_parent_rss_kib}"
                            f"\tready_worker_rss_kib={row.ready_worker_rss_kib}"
                            f"\tready_total_rss_kib={row.ready_total_rss_kib}"
                            f"\tready_worker_pid_count={row.ready_worker_pid_count}"
                            f"\tparallel_override={row.parallel_work_item_count_override}"
                        )

    payload = {
        "task_path": str(task_path),
        "dataset_id": str(task["dataset_id"]),
        "slice_name": str(task["slice_name"]),
        "request_pool_count": len(request_pool),
        "warmup_iterations": args.warmup_iterations,
        "measurement_iterations": args.measurement_iterations,
        "host_cpu_count": os.cpu_count() or 1,
        "load_text_corpus_values": load_text_corpus_values,
        "session_prepare_seconds": session_prepare_seconds,
        "results": [row.to_json_ready() for row in rows],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    print(f"Mean: {min(row.mean_batch_seconds for row in rows):.12f}")


if __name__ == "__main__":
    main()
