"""Benchmark hosted prepared-runtime burst admission over the real HTTP seam.

This script owns:
- staging one encoded task into a temporary hosted service root
- preparing hosted exact runtimes with explicit admission caps
- measuring burst outcomes, latency, and runtime queue metrics

It does not own:
- tuning the runtime default by itself
- generic in-process runtime throughput or RSS sweeps
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.task_json_catalog import default_task_json_output_path
from kayak_engine.mojo_service import load_module
from kayak_engine.payloads import document_payload_parts


COLLECTION_ID = "prepared-runtime-burst-bench"
TENANT_ID = "public"
NAMESPACE_ID = "benchmark"
SNAPSHOT_ID = "snapshot-0001"
SERVER_MODULE = "kayak_engine.server"


@dataclass(frozen=True, slots=True)
class BurstBenchmarkRow:
    repeat_index: int
    burst_size: int
    concurrency_lane_count: int
    worker_count: int
    requested_max_outstanding_request_count: int
    resolved_max_outstanding_request_count: int
    prepare_seconds: float
    total_wall_seconds: float
    offered_queries_per_second: float
    accepted_queries_per_second: float
    accepted_request_count: int
    rejected_request_count: int
    accepted_latency_p50_ms: float
    accepted_latency_p95_ms: float
    rejected_latency_p50_ms: float
    rejected_latency_p95_ms: float
    submitted_request_count: int
    completed_request_count: int
    runtime_rejected_request_count: int
    failed_request_count: int
    executed_batch_count: int
    max_observed_batch_size: int
    max_observed_queue_depth: int
    average_batch_size: float
    average_queue_wait_ms: float
    average_batch_execution_ms: float

    def to_json_ready(self) -> dict[str, object]:
        return {
            "repeat_index": self.repeat_index,
            "burst_size": self.burst_size,
            "concurrency_lane_count": self.concurrency_lane_count,
            "worker_count": self.worker_count,
            "requested_max_outstanding_request_count": (
                self.requested_max_outstanding_request_count
            ),
            "resolved_max_outstanding_request_count": (
                self.resolved_max_outstanding_request_count
            ),
            "prepare_seconds": self.prepare_seconds,
            "total_wall_seconds": self.total_wall_seconds,
            "offered_queries_per_second": self.offered_queries_per_second,
            "accepted_queries_per_second": self.accepted_queries_per_second,
            "accepted_request_count": self.accepted_request_count,
            "rejected_request_count": self.rejected_request_count,
            "accepted_latency_p50_ms": self.accepted_latency_p50_ms,
            "accepted_latency_p95_ms": self.accepted_latency_p95_ms,
            "rejected_latency_p50_ms": self.rejected_latency_p50_ms,
            "rejected_latency_p95_ms": self.rejected_latency_p95_ms,
            "submitted_request_count": self.submitted_request_count,
            "completed_request_count": self.completed_request_count,
            "runtime_rejected_request_count": self.runtime_rejected_request_count,
            "failed_request_count": self.failed_request_count,
            "executed_batch_count": self.executed_batch_count,
            "max_observed_batch_size": self.max_observed_batch_size,
            "max_observed_queue_depth": self.max_observed_queue_depth,
            "average_batch_size": self.average_batch_size,
            "average_queue_wait_ms": self.average_queue_wait_ms,
            "average_batch_execution_ms": self.average_batch_execution_ms,
        }


class HostedEngineServer:
    def __init__(self, root: Path):
        env = os.environ.copy()
        python_root = str(REPO_ROOT / "python")
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = (
            python_root
            if existing_pythonpath in (None, "")
            else f"{python_root}{os.pathsep}{existing_pythonpath}"
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                SERVER_MODULE,
                "--root",
                str(root),
                "--port",
                "0",
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.base_url = self._wait_until_ready()

    def _wait_until_ready(self) -> str:
        assert self.process.stdout is not None
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stdout = self.process.stdout.read() if self.process.stdout else ""
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(
                    "hosted engine server exited before becoming ready\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                )
            ready, _, _ = select.select([self.process.stdout], [], [], 0.5)
            if not ready:
                continue
            line = self.process.stdout.readline().strip()
            if "listening on http://" not in line:
                continue
            return line.rsplit(" ", 1)[-1]
        self.close()
        raise RuntimeError("timed out waiting for hosted engine server startup")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10.0)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()

    def __enter__(self) -> "HostedEngineServer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()


def _http_json(
    method: str,
    url: str,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _parse_csv_ints(value: str) -> list[int]:
    counts: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped == "":
            continue
        parsed = int(stripped)
        if parsed < 0:
            raise ValueError("counts must be non-negative")
        counts.append(parsed)
    if len(counts) == 0:
        raise ValueError("at least one integer value is required")
    return counts


def _task_path(args: argparse.Namespace) -> Path:
    if args.task is not None:
        return args.task
    return default_task_json_output_path(args.task_key)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_request_pool(
    task: dict[str, Any],
    *,
    request_pool_count: int,
) -> list[dict[str, Any]]:
    base_requests = [
        {
            "query_model_name": str(task["model_name"]),
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


def _expand_requests(
    request_pool: list[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    return [
        dict(request_pool[index % len(request_pool)])
        for index in range(count)
    ]


def _create_collection(*, module: object, service_root: Path, task: dict[str, Any]) -> None:
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


def _upsert_documents(*, module: object, service_root: Path, task: dict[str, Any]) -> None:
    documents = [
        {
            "doc_id": str(document["doc_id"]),
            "vectors": document["vectors"],
            "text": str(document["text"]),
            "metadata": {},
        }
        for document in task["documents"]
    ]
    doc_ids, document_vectors, texts, metadata_keys, metadata_values = (
        document_payload_parts({"documents": documents})
    )
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
                "reason": "hosted prepared runtime backpressure benchmark",
            },
        )
    )


def _percentile_ms(latencies_seconds: list[float], percentile: float) -> float:
    if len(latencies_seconds) == 0:
        return 0.0
    ordered = sorted(latencies_seconds)
    if len(ordered) == 1:
        return ordered[0] * 1000.0
    rank = (len(ordered) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - float(low)
    seconds = ordered[low] + ((ordered[high] - ordered[low]) * fraction)
    return seconds * 1000.0


def _runtime_stats(server: HostedEngineServer, runtime_id: str) -> dict[str, Any]:
    status, payload = _http_json(
        "POST",
        f"{server.base_url}/v1/prepared-exact-runtimes:stats",
        {"runtime_id": runtime_id},
    )
    if status != 200:
        raise RuntimeError(f"failed to fetch runtime stats: status={status} payload={payload}")
    return payload["runtime"]["stats"]


def _close_runtime(server: HostedEngineServer, runtime_id: str) -> None:
    status, payload = _http_json(
        "POST",
        f"{server.base_url}/v1/prepared-exact-runtimes:close",
        {"runtime_id": runtime_id},
    )
    if status != 200:
        raise RuntimeError(f"failed to close runtime: status={status} payload={payload}")


def _prepare_runtime(
    server: HostedEngineServer,
    *,
    concurrency_lane_count: int,
    worker_count: int,
    max_batch_size: int,
    max_batch_wait_ms: int,
    max_outstanding_request_count: int,
) -> tuple[str, int, float]:
    prepare_started_at = time.perf_counter()
    status, payload = _http_json(
        "POST",
        f"{server.base_url}/v1/prepared-exact-runtimes",
        {
            "collection_id": COLLECTION_ID,
            "tenant_id": TENANT_ID,
            "namespace_id": NAMESPACE_ID,
            "snapshot_id": SNAPSHOT_ID,
            "config": {
                "execution_backend": "process",
                "concurrency_lane_count": concurrency_lane_count,
                "worker_count": worker_count,
                "max_batch_size": max_batch_size,
                "max_batch_wait_ms": max_batch_wait_ms,
                "max_outstanding_request_count": max_outstanding_request_count,
            },
        },
    )
    prepare_seconds = time.perf_counter() - prepare_started_at
    if status != 200:
        raise RuntimeError(f"failed to prepare runtime: status={status} payload={payload}")
    runtime = payload["runtime"]
    return (
        str(runtime["runtime_id"]),
        int(runtime["config"]["max_outstanding_request_count"]),
        prepare_seconds,
    )


def _reference_searches(
    server: HostedEngineServer,
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for request in requests:
        status, payload = _http_json(
            "POST",
            f"{server.base_url}/v1/search",
            {
                "collection_id": COLLECTION_ID,
                "tenant_id": TENANT_ID,
                "namespace_id": NAMESPACE_ID,
                "snapshot_id": SNAPSHOT_ID,
                **request,
            },
        )
        if status != 200:
            raise RuntimeError(
                f"failed to compute stateless reference: status={status} payload={payload}"
            )
        expected.append(payload)
    return expected


def _measure_burst(
    server: HostedEngineServer,
    *,
    runtime_id: str,
    requests: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> BurstBenchmarkRow:
    barrier = threading.Barrier(len(requests) + 1)
    accepted_latencies: list[float] = []
    rejected_latencies: list[float] = []
    accepted_count = 0
    rejected_count = 0
    failures: list[str] = []
    started_at = time.perf_counter()
    lock = threading.Lock()

    def run(index: int) -> None:
        nonlocal accepted_count, rejected_count
        request = requests[index]
        try:
            barrier.wait()
            request_started_at = time.perf_counter()
            status, payload = _http_json(
                "POST",
                f"{server.base_url}/v1/prepared-exact-search",
                {"runtime_id": runtime_id, "request": request},
            )
            latency_seconds = time.perf_counter() - request_started_at
            with lock:
                if status == 200:
                    if payload["search"] != expected[index]:
                        failures.append(f"mismatched search payload at request index {index}")
                        return
                    accepted_count += 1
                    accepted_latencies.append(latency_seconds)
                    return
                if status == 429:
                    rejected_count += 1
                    rejected_latencies.append(latency_seconds)
                    return
                failures.append(f"unexpected status {status} at request index {index}: {payload}")
        except BaseException as exc:  # pragma: no cover - benchmark failure path
            with lock:
                failures.append(f"{type(exc).__name__}: {exc}")

    threads = [
        threading.Thread(target=run, args=(index,), daemon=True)
        for index in range(len(requests))
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=30.0)
    for index, thread in enumerate(threads):
        if thread.is_alive():
            failures.append(f"thread {index} did not finish before timeout")
    total_wall_seconds = time.perf_counter() - started_at
    if failures:
        raise AssertionError("; ".join(failures))

    stats = _runtime_stats(server, runtime_id)
    return BurstBenchmarkRow(
        repeat_index=-1,
        burst_size=len(requests),
        concurrency_lane_count=0,
        worker_count=0,
        requested_max_outstanding_request_count=0,
        resolved_max_outstanding_request_count=0,
        prepare_seconds=0.0,
        total_wall_seconds=total_wall_seconds,
        offered_queries_per_second=len(requests) / total_wall_seconds,
        accepted_queries_per_second=accepted_count / total_wall_seconds,
        accepted_request_count=accepted_count,
        rejected_request_count=rejected_count,
        accepted_latency_p50_ms=_percentile_ms(accepted_latencies, 0.50),
        accepted_latency_p95_ms=_percentile_ms(accepted_latencies, 0.95),
        rejected_latency_p50_ms=_percentile_ms(rejected_latencies, 0.50),
        rejected_latency_p95_ms=_percentile_ms(rejected_latencies, 0.95),
        submitted_request_count=int(stats["submitted_request_count"]),
        completed_request_count=int(stats["completed_request_count"]),
        runtime_rejected_request_count=int(stats["rejected_request_count"]),
        failed_request_count=int(stats["failed_request_count"]),
        executed_batch_count=int(stats["executed_batch_count"]),
        max_observed_batch_size=int(stats["max_observed_batch_size"]),
        max_observed_queue_depth=int(stats["max_observed_queue_depth"]),
        average_batch_size=float(stats["average_batch_size"]),
        average_queue_wait_ms=float(stats["average_queue_wait_ms"]),
        average_batch_execution_ms=float(stats["average_batch_execution_ms"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark hosted prepared exact-runtime burst admission and backpressure "
            "tradeoffs over the real HTTP surface."
        )
    )
    parser.add_argument("--task", type=Path, default=None)
    parser.add_argument("--task-key", type=str, default="browsecomp_plus_gold")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--request-pool-count", type=int, default=32)
    parser.add_argument("--burst-sizes", type=str, default="32,128")
    parser.add_argument("--lane-counts", type=str, default="1")
    parser.add_argument("--worker-counts", type=str, default="2")
    parser.add_argument("--outstanding-counts", type=str, default="32,64,128")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-wait-ms", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.request_pool_count <= 0:
        raise ValueError("request-pool-count must be positive")
    if args.max_batch_size <= 0:
        raise ValueError("max-batch-size must be positive")
    if args.max_batch_wait_ms < 0:
        raise ValueError("max-batch-wait-ms must be non-negative")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    task_path = _task_path(args)
    task = load_task_json(str(task_path))
    burst_sizes = [count for count in _parse_csv_ints(args.burst_sizes) if count > 0]
    lane_counts = [count for count in _parse_csv_ints(args.lane_counts) if count > 0]
    worker_counts = [count for count in _parse_csv_ints(args.worker_counts) if count > 0]
    outstanding_counts = _parse_csv_ints(args.outstanding_counts)
    if len(burst_sizes) == 0 or len(lane_counts) == 0 or len(worker_counts) == 0:
        raise ValueError("burst, lane, and worker counts must contain positive values")

    rows: list[BurstBenchmarkRow] = []
    request_pool = _build_request_pool(task, request_pool_count=args.request_pool_count)
    reference_requests = _expand_requests(request_pool, count=max(burst_sizes))

    with tempfile.TemporaryDirectory(prefix="kayak-hosted-burst-bench-") as tmpdir:
        service_root = Path(tmpdir) / "service-root"
        module = load_module()
        _create_collection(module=module, service_root=service_root, task=task)
        _upsert_documents(module=module, service_root=service_root, task=task)
        _create_snapshot(module=module, service_root=service_root)

        with HostedEngineServer(service_root) as server:
            reference_payloads = _reference_searches(server, reference_requests)

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

            for concurrency_lane_count in lane_counts:
                for worker_count in worker_counts:
                    for requested_outstanding_count in outstanding_counts:
                        for burst_size in burst_sizes:
                            for repeat_index in range(args.repeats):
                                runtime_id, resolved_outstanding_count, prepare_seconds = _prepare_runtime(
                                    server,
                                    concurrency_lane_count=concurrency_lane_count,
                                    worker_count=worker_count,
                                    max_batch_size=args.max_batch_size,
                                    max_batch_wait_ms=args.max_batch_wait_ms,
                                    max_outstanding_request_count=requested_outstanding_count,
                                )
                                try:
                                    measured = _measure_burst(
                                        server,
                                        runtime_id=runtime_id,
                                        requests=reference_requests[:burst_size],
                                        expected=reference_payloads[:burst_size],
                                    )
                                finally:
                                    _close_runtime(server, runtime_id)

                                row = BurstBenchmarkRow(
                                    repeat_index=repeat_index,
                                    burst_size=burst_size,
                                    concurrency_lane_count=concurrency_lane_count,
                                    worker_count=worker_count,
                                    requested_max_outstanding_request_count=(
                                        requested_outstanding_count
                                    ),
                                    resolved_max_outstanding_request_count=(
                                        resolved_outstanding_count
                                    ),
                                    prepare_seconds=prepare_seconds,
                                    total_wall_seconds=measured.total_wall_seconds,
                                    offered_queries_per_second=measured.offered_queries_per_second,
                                    accepted_queries_per_second=measured.accepted_queries_per_second,
                                    accepted_request_count=measured.accepted_request_count,
                                    rejected_request_count=measured.rejected_request_count,
                                    accepted_latency_p50_ms=measured.accepted_latency_p50_ms,
                                    accepted_latency_p95_ms=measured.accepted_latency_p95_ms,
                                    rejected_latency_p50_ms=measured.rejected_latency_p50_ms,
                                    rejected_latency_p95_ms=measured.rejected_latency_p95_ms,
                                    submitted_request_count=measured.submitted_request_count,
                                    completed_request_count=measured.completed_request_count,
                                    runtime_rejected_request_count=measured.runtime_rejected_request_count,
                                    failed_request_count=measured.failed_request_count,
                                    executed_batch_count=measured.executed_batch_count,
                                    max_observed_batch_size=measured.max_observed_batch_size,
                                    max_observed_queue_depth=measured.max_observed_queue_depth,
                                    average_batch_size=measured.average_batch_size,
                                    average_queue_wait_ms=measured.average_queue_wait_ms,
                                    average_batch_execution_ms=measured.average_batch_execution_ms,
                                )
                                rows.append(row)
                                print(
                                    "RESULT"
                                    f"\trepeat={row.repeat_index}"
                                    f"\tburst={row.burst_size}"
                                    f"\tlanes={row.concurrency_lane_count}"
                                    f"\tworkers={row.worker_count}"
                                    f"\trequested_outstanding={row.requested_max_outstanding_request_count}"
                                    f"\tresolved_outstanding={row.resolved_max_outstanding_request_count}"
                                    f"\tprepare_seconds={row.prepare_seconds:.12f}"
                                    f"\ttotal_wall_seconds={row.total_wall_seconds:.12f}"
                                    f"\toffered_qps={row.offered_queries_per_second:.3f}"
                                    f"\taccepted_qps={row.accepted_queries_per_second:.3f}"
                                    f"\taccepted={row.accepted_request_count}"
                                    f"\trejected={row.rejected_request_count}"
                                    f"\taccepted_p50_ms={row.accepted_latency_p50_ms:.3f}"
                                    f"\taccepted_p95_ms={row.accepted_latency_p95_ms:.3f}"
                                    f"\trejected_p50_ms={row.rejected_latency_p50_ms:.3f}"
                                    f"\trejected_p95_ms={row.rejected_latency_p95_ms:.3f}"
                                    f"\tavg_queue_wait_ms={row.average_queue_wait_ms:.3f}"
                                    f"\tmax_queue_depth={row.max_observed_queue_depth}"
                                )

    payload = {
        "task_path": str(task_path),
        "dataset_id": str(task["dataset_id"]),
        "slice_name": str(task["slice_name"]),
        "request_pool_count": len(request_pool),
        "burst_sizes": burst_sizes,
        "lane_counts": lane_counts,
        "worker_counts": worker_counts,
        "outstanding_counts": outstanding_counts,
        "max_batch_size": args.max_batch_size,
        "max_batch_wait_ms": args.max_batch_wait_ms,
        "repeats": args.repeats,
        "results": [row.to_json_ready() for row in rows],
    }
    if args.output is not None:
        _write_json(args.output, payload)
        print(f"wrote {args.output}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Mean: {min(row.total_wall_seconds for row in rows):.12f}")


if __name__ == "__main__":
    main()
