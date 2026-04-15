from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest

from hosted_engine_test_support import HostedEngineServer, http_json


class HostedPreparedExactRuntimeHttpTest(unittest.TestCase):
    def test_network_prepared_exact_runtime_prepare_search_batch_stats_and_close(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="kayak-http-prepared-runtime-") as temp_dir:
            with HostedEngineServer(Path(temp_dir) / "service-root") as server:
                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/collections",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "model_name": "colbertv2",
                        "vector_dim": 2,
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/documents:upsert",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "documents": [
                            {
                                "doc_id": "doc-a",
                                "vectors": [[1.0, 0.0], [0.0, 1.0]],
                                "text": "alpha evidence document",
                            },
                            {
                                "doc_id": "doc-b",
                                "vectors": [[0.0, 1.0], [1.0, 0.0]],
                                "text": "beta evidence document",
                            },
                        ],
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/snapshots",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        "reason": "publish prepared runtime snapshot",
                    },
                )
                self.assertEqual(status, 200)

                runtime_prepare = {
                    "collection_id": "news",
                    "tenant_id": "tenant-a",
                    "namespace_id": "search",
                    "snapshot_id": "snapshot-0001",
                    "config": {
                        "execution_backend": "process",
                        "concurrency_lane_count": 1,
                        "worker_count": 2,
                        "max_batch_size": 8,
                        "max_batch_wait_ms": 25,
                        "scoring": {
                            "enable_parallel_scoring": True,
                            "enable_dim128_fast_path": True,
                            "enable_parallel_work_item_oversubscription": False,
                            "parallel_work_item_count_override": 2,
                        },
                    },
                }
                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes",
                    runtime_prepare,
                )
                self.assertEqual(status, 200)
                self.assertFalse(payload["reused"])
                self.assertEqual(payload["active_runtime_count"], 1)
                runtime = payload["runtime"]
                runtime_id = runtime["runtime_id"]
                self.assertEqual(runtime["snapshot_id"], "snapshot-0001")
                self.assertFalse(runtime["load_text_corpus"])
                self.assertEqual(runtime["config"]["worker_count"], 2)
                self.assertGreater(
                    runtime["config"]["max_outstanding_request_count"],
                    0,
                )
                self.assertEqual(
                    runtime["config"]["scoring"]["parallel_work_item_count_override"],
                    2,
                )
                self.assertEqual(runtime["stats"]["submitted_request_count"], 0)
                self.assertEqual(runtime["stats"]["rejected_request_count"], 0)

                status, payload = http_json(
                    "GET",
                    f"{server.base_url}/v1/prepared-exact-runtimes",
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["active_runtime_count"], 1)
                self.assertEqual(payload["runtimes"][0]["runtime_id"], runtime_id)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes",
                    runtime_prepare,
                )
                self.assertEqual(status, 200)
                self.assertTrue(payload["reused"])
                self.assertEqual(payload["runtime"]["runtime_id"], runtime_id)
                self.assertEqual(payload["active_runtime_count"], 1)

                query_alpha = {
                    "query_model_name": "colbertv2",
                    "query": [[1.0, 0.0], [0.0, 1.0]],
                    "final_k": 2,
                }
                query_beta = {
                    "query_model_name": "colbertv2",
                    "query": [[0.0, 1.0], [1.0, 0.0]],
                    "final_k": 2,
                }
                status, stateless_alpha = http_json(
                    "POST",
                    f"{server.base_url}/v1/search",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        **query_alpha,
                    },
                )
                self.assertEqual(status, 200)
                status, stateless_beta = http_json(
                    "POST",
                    f"{server.base_url}/v1/search",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        **query_beta,
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-search",
                    {
                        "runtime_id": runtime_id,
                        "request": query_alpha,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["runtime_id"], runtime_id)
                self.assertEqual(payload["search"], stateless_alpha)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-search-batch",
                    {
                        "runtime_id": runtime_id,
                        "requests": [query_alpha, query_beta],
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["runtime_id"], runtime_id)
                self.assertEqual(payload["responses"], [stateless_alpha, stateless_beta])

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes:stats",
                    {
                        "runtime_id": runtime_id,
                    },
                )
                self.assertEqual(status, 200)
                runtime = payload["runtime"]
                self.assertEqual(runtime["runtime_id"], runtime_id)
                self.assertEqual(runtime["stats"]["submitted_request_count"], 3)
                self.assertEqual(runtime["stats"]["completed_request_count"], 3)
                self.assertEqual(runtime["stats"]["failed_request_count"], 0)
                self.assertEqual(runtime["stats"]["rejected_request_count"], 0)
                self.assertEqual(runtime["stats"]["processed_request_count"], 3)
                self.assertGreaterEqual(
                    runtime["stats"]["executed_batch_count"],
                    2,
                )
                self.assertGreaterEqual(
                    runtime["stats"]["max_observed_batch_size"],
                    2,
                )

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes:close",
                    {
                        "runtime_id": runtime_id,
                    },
                )
                self.assertEqual(status, 200)
                self.assertTrue(payload["closed"])
                self.assertEqual(payload["active_runtime_count"], 0)
                self.assertEqual(payload["runtime"]["runtime_id"], runtime_id)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes:stats",
                    {
                        "runtime_id": runtime_id,
                    },
                )
                self.assertEqual(status, 404)
                self.assertIn("does not exist", payload["error"])

    def test_network_prepared_exact_runtime_coalesces_concurrent_search_requests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="kayak-http-prepared-runtime-batch-") as temp_dir:
            with HostedEngineServer(Path(temp_dir) / "service-root") as server:
                for path, payload in (
                    (
                        "/v1/collections",
                        {
                            "collection_id": "news",
                            "tenant_id": "tenant-a",
                            "namespace_id": "search",
                            "model_name": "colbertv2",
                            "vector_dim": 2,
                        },
                    ),
                    (
                        "/v1/documents:upsert",
                        {
                            "collection_id": "news",
                            "tenant_id": "tenant-a",
                            "namespace_id": "search",
                            "documents": [
                                {
                                    "doc_id": "doc-a",
                                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                                    "text": "alpha evidence document",
                                },
                                {
                                    "doc_id": "doc-b",
                                    "vectors": [[0.0, 1.0], [1.0, 0.0]],
                                    "text": "beta evidence document",
                                },
                            ],
                        },
                    ),
                    (
                        "/v1/snapshots",
                        {
                            "collection_id": "news",
                            "tenant_id": "tenant-a",
                            "namespace_id": "search",
                            "snapshot_id": "snapshot-0001",
                            "reason": "publish prepared runtime snapshot",
                        },
                    ),
                ):
                    status, _payload = http_json(
                        "POST",
                        f"{server.base_url}{path}",
                        payload,
                    )
                    self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        "config": {
                            "execution_backend": "process",
                            "concurrency_lane_count": 1,
                            "worker_count": 2,
                            "max_batch_size": 8,
                            "max_batch_wait_ms": 25,
                        },
                    },
                )
                self.assertEqual(status, 200)
                runtime_id = payload["runtime"]["runtime_id"]

                requests = [
                    {
                        "query_model_name": "colbertv2",
                        "query": [[1.0, 0.0], [0.0, 1.0]],
                        "final_k": 2,
                    },
                    {
                        "query_model_name": "colbertv2",
                        "query": [[0.0, 1.0], [1.0, 0.0]],
                        "final_k": 2,
                    },
                    {
                        "query_model_name": "colbertv2",
                        "query": [[1.0, 0.0], [1.0, 0.0]],
                        "final_k": 2,
                    },
                    {
                        "query_model_name": "colbertv2",
                        "query": [[0.0, 1.0], [0.0, 1.0]],
                        "final_k": 2,
                    },
                ]
                expected: list[dict] = []
                for request in requests:
                    status, payload = http_json(
                        "POST",
                        f"{server.base_url}/v1/search",
                        {
                            "collection_id": "news",
                            "tenant_id": "tenant-a",
                            "namespace_id": "search",
                            "snapshot_id": "snapshot-0001",
                            **request,
                        },
                    )
                    self.assertEqual(status, 200)
                    expected.append(payload)

                barrier = threading.Barrier(len(requests) + 1)
                responses: list[dict | None] = [None] * len(requests)
                failures: list[BaseException] = []

                def run(index: int, request: dict) -> None:
                    try:
                        barrier.wait()
                        status, payload = http_json(
                            "POST",
                            f"{server.base_url}/v1/prepared-exact-search",
                            {
                                "runtime_id": runtime_id,
                                "request": request,
                            },
                        )
                        if status != 200:
                            raise AssertionError(f"unexpected status: {status}")
                        responses[index] = payload["search"]
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        failures.append(exc)

                threads = [
                    threading.Thread(target=run, args=(index, request), daemon=True)
                    for index, request in enumerate(requests)
                ]
                for thread in threads:
                    thread.start()
                barrier.wait()
                for thread in threads:
                    thread.join(timeout=30.0)

                self.assertEqual(failures, [])
                self.assertEqual(responses, expected)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes:stats",
                    {"runtime_id": runtime_id},
                )
                self.assertEqual(status, 200)
                stats = payload["runtime"]["stats"]
                self.assertEqual(stats["submitted_request_count"], len(requests))
                self.assertEqual(stats["completed_request_count"], len(requests))
                self.assertEqual(stats["failed_request_count"], 0)
                self.assertEqual(stats["rejected_request_count"], 0)
                self.assertLess(stats["executed_batch_count"], len(requests))
                self.assertGreater(stats["max_observed_batch_size"], 1)

    def test_network_prepared_exact_runtime_returns_429_under_overload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kayak-http-prepared-runtime-overload-") as temp_dir:
            with HostedEngineServer(Path(temp_dir) / "service-root") as server:
                for path, payload in (
                    (
                        "/v1/collections",
                        {
                            "collection_id": "news",
                            "tenant_id": "tenant-a",
                            "namespace_id": "search",
                            "model_name": "colbertv2",
                            "vector_dim": 2,
                        },
                    ),
                    (
                        "/v1/documents:upsert",
                        {
                            "collection_id": "news",
                            "tenant_id": "tenant-a",
                            "namespace_id": "search",
                            "documents": [
                                {
                                    "doc_id": "doc-a",
                                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                                    "text": "alpha evidence document",
                                },
                                {
                                    "doc_id": "doc-b",
                                    "vectors": [[0.0, 1.0], [1.0, 0.0]],
                                    "text": "beta evidence document",
                                },
                            ],
                        },
                    ),
                    (
                        "/v1/snapshots",
                        {
                            "collection_id": "news",
                            "tenant_id": "tenant-a",
                            "namespace_id": "search",
                            "snapshot_id": "snapshot-0001",
                            "reason": "publish prepared runtime snapshot",
                        },
                    ),
                ):
                    status, _payload = http_json(
                        "POST",
                        f"{server.base_url}{path}",
                        payload,
                    )
                    self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        "config": {
                            "execution_backend": "process",
                            "concurrency_lane_count": 1,
                            "worker_count": 1,
                            "max_batch_size": 8,
                            "max_batch_wait_ms": 250,
                            "max_outstanding_request_count": 1,
                        },
                    },
                )
                self.assertEqual(status, 200)
                runtime_id = payload["runtime"]["runtime_id"]

                request = {
                    "query_model_name": "colbertv2",
                    "query": [[1.0, 0.0], [0.0, 1.0]],
                    "final_k": 2,
                }

                barrier = threading.Barrier(3)
                results: list[tuple[int, dict] | None] = [None, None]
                failures: list[BaseException] = []

                def run(index: int) -> None:
                    try:
                        barrier.wait()
                        results[index] = http_json(
                            "POST",
                            f"{server.base_url}/v1/prepared-exact-search",
                            {
                                "runtime_id": runtime_id,
                                "request": request,
                            },
                        )
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        failures.append(exc)

                threads = [
                    threading.Thread(target=run, args=(index,), daemon=True)
                    for index in range(2)
                ]
                for thread in threads:
                    thread.start()
                barrier.wait()
                for thread in threads:
                    thread.join(timeout=30.0)

                self.assertEqual(failures, [])
                self.assertTrue(all(result is not None for result in results))
                populated_results = [result for result in results if result is not None]
                statuses = sorted(status for status, _payload in populated_results)
                self.assertEqual(statuses, [200, 429])
                overloaded_payload = next(
                    payload
                    for status, payload in populated_results
                    if status == 429
                )
                self.assertIn("overloaded", overloaded_payload["error"])

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes:stats",
                    {"runtime_id": runtime_id},
                )
                self.assertEqual(status, 200)
                stats = payload["runtime"]["stats"]
                self.assertEqual(stats["submitted_request_count"], 1)
                self.assertEqual(stats["rejected_request_count"], 1)
                self.assertEqual(stats["completed_request_count"], 1)
                self.assertEqual(stats["failed_request_count"], 0)
                self.assertEqual(stats["current_pending_request_count"], 0)

    def test_network_prepared_exact_runtime_is_invalidated_by_reclaim(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kayak-http-prepared-runtime-reclaim-") as temp_dir:
            with HostedEngineServer(Path(temp_dir) / "service-root") as server:
                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/collections",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "model_name": "colbertv2",
                        "vector_dim": 2,
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/documents:upsert",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "documents": [
                            {
                                "doc_id": "doc-a",
                                "vectors": [[1.0, 0.0]],
                                "text": "first document",
                            }
                        ],
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/snapshots",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        "reason": "publish first snapshot",
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/collections:retention",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "default_keep_latest_inactive_count": 0,
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/documents:upsert",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "documents": [
                            {
                                "doc_id": "doc-b",
                                "vectors": [[0.0, 1.0]],
                                "text": "second document",
                            }
                        ],
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/snapshots",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0002",
                        "reason": "publish second snapshot",
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/collections:reclaim-plan",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["plan"]["reclaimable_snapshot_count"], 1)
                reclaim_plan = payload["plan"]
                reclaimable_snapshot_id = next(
                    decision["snapshot_id"]
                    for decision in reclaim_plan["decisions"]
                    if not decision["retain"]
                )

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": reclaimable_snapshot_id,
                        "config": {
                            "execution_backend": "process",
                            "concurrency_lane_count": 1,
                            "worker_count": 1,
                            "max_batch_size": 4,
                            "max_batch_wait_ms": 10,
                        },
                    },
                )
                self.assertEqual(status, 200)
                runtime_id = payload["runtime"]["runtime_id"]

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes:stats",
                    {
                        "runtime_id": runtime_id,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["runtime"]["snapshot_id"], reclaimable_snapshot_id)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/collections:reclaim-execute",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "dry_run": True,
                        "plan": reclaim_plan,
                    },
                )
                self.assertEqual(status, 200)
                self.assertFalse(payload["result"]["applied"])
                self.assertEqual(payload["invalidated_prepared_exact_runtime_count"], 0)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes:stats",
                    {
                        "runtime_id": runtime_id,
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/collections:reclaim-execute",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "dry_run": False,
                        "plan": reclaim_plan,
                    },
                )
                self.assertEqual(status, 200)
                self.assertTrue(payload["result"]["applied"])
                self.assertEqual(payload["invalidated_prepared_exact_runtime_count"], 1)
                self.assertEqual(
                    payload["invalidated_prepared_exact_runtimes"][0]["runtime_id"],
                    runtime_id,
                )
                self.assertEqual(
                    payload["invalidated_prepared_exact_runtimes"][0]["snapshot_id"],
                    reclaimable_snapshot_id,
                )

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/prepared-exact-runtimes:stats",
                    {
                        "runtime_id": runtime_id,
                    },
                )
                self.assertEqual(status, 404)
                self.assertIn("does not exist", payload["error"])

                status, payload = http_json(
                    "GET",
                    f"{server.base_url}/v1/prepared-exact-runtimes",
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["active_runtime_count"], 0)


if __name__ == "__main__":
    unittest.main()
