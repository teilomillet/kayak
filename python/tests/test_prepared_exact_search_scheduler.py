from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

from kayak_engine import (
    PreparedExactSearchRuntime,
    PreparedExactSearchRuntimeConfig,
    PreparedExactSearchSchedulerConfig,
    PreparedExactSearchScheduler,
    SUPPORTED_PREPARED_EXACT_RUNTIME_BACKENDS,
    prepare_exact_search_runtime,
    prepare_exact_search_scheduler,
    prepare_exact_search_session,
)
from kayak_engine.mojo_service import load_module
from kayak_engine.payloads import document_payload_parts


REPO_ROOT = Path(__file__).resolve().parents[2]


def _mojo_backend_available() -> bool:
    if shutil.which("mojo") is not None:
        return True
    return (REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "mojo").exists()


def _create_collection(module: object, service_root: Path) -> None:
    json.loads(
        module.create_collection_json(
            str(service_root),
            {
                "collection_id": "news",
                "tenant_id": "tenant-a",
                "namespace_id": "search",
                "collection_layout_family": "",
                "model_name": "colbertv2",
                "vector_scalar_name": "",
                "vector_dim": 2,
                "default_keep_latest_inactive_count": 1,
            },
        )
    )


def _upsert_documents(module: object, service_root: Path, documents: list[dict]) -> None:
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
                "collection_id": "news",
                "tenant_id": "tenant-a",
                "namespace_id": "search",
                "doc_ids": doc_ids,
                "document_vectors": document_vectors,
                "texts": texts,
                "metadata_keys": metadata_keys,
                "metadata_values": metadata_values,
            },
        )
    )


def _create_snapshot(module: object, service_root: Path, snapshot_id: str) -> None:
    json.loads(
        module.create_snapshot_json(
            str(service_root),
            {
                "collection_id": "news",
                "tenant_id": "tenant-a",
                "namespace_id": "search",
                "snapshot_id": snapshot_id,
                "reason": f"publish {snapshot_id}",
            },
        )
    )


class PreparedExactSearchRuntimeApiTests(unittest.TestCase):
    def test_scheduler_aliases_runtime_surface(self) -> None:
        self.assertIs(PreparedExactSearchScheduler, PreparedExactSearchRuntime)
        self.assertIs(
            PreparedExactSearchSchedulerConfig,
            PreparedExactSearchRuntimeConfig,
        )
        self.assertEqual(SUPPORTED_PREPARED_EXACT_RUNTIME_BACKENDS, ("process",))
        self.assertEqual(
            PreparedExactSearchRuntimeConfig().concurrency_lane_count,
            1,
        )
        self.assertEqual(
            PreparedExactSearchRuntimeConfig().execution_backend,
            "process",
        )

    def test_runtime_config_rejects_unverified_backend(self) -> None:
        with self.assertRaises(ValueError):
            PreparedExactSearchRuntimeConfig(execution_backend="thread")

    def test_runtime_config_rejects_non_positive_concurrency_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            PreparedExactSearchRuntimeConfig(concurrency_lane_count=0)


@unittest.skipUnless(
    _mojo_backend_available(), "prepared exact search runtime requires Mojo"
)
class PreparedExactSearchRuntimeTests(unittest.TestCase):
    def _build_service_root(self) -> tuple[object, tempfile.TemporaryDirectory[str], Path]:
        temp_dir = tempfile.TemporaryDirectory(prefix="kayak-prepared-scheduler-")
        service_root = Path(temp_dir.name) / "service-root"
        module = load_module()
        _create_collection(module, service_root)
        return module, temp_dir, service_root

    def _seed_snapshot(self) -> tuple[Path, object, object, tempfile.TemporaryDirectory[str]]:
        module, temp_dir, service_root = self._build_service_root()
        _upsert_documents(
            module,
            service_root,
            [
                {
                    "doc_id": "doc-a",
                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                    "text": "alpha evidence document",
                    "metadata": {"topic": "alpha"},
                },
                {
                    "doc_id": "doc-b",
                    "vectors": [[0.0, 1.0], [1.0, 0.0]],
                    "text": "beta evidence document",
                    "metadata": {"topic": "beta"},
                },
            ],
        )
        _create_snapshot(module, service_root, "snapshot-0001")
        session = prepare_exact_search_session(
            service_root=service_root,
            collection_id="news",
            tenant_id="tenant-a",
            namespace_id="search",
            snapshot_id="snapshot-0001",
        )
        return service_root, module, session, temp_dir

    def test_runtime_search_matches_direct_prepared_search(self) -> None:
        service_root, _module, session, temp_dir = self._seed_snapshot()
        self.addCleanup(temp_dir.cleanup)

        runtime = session.runtime(
            config=PreparedExactSearchRuntimeConfig(
                concurrency_lane_count=2,
                worker_count=1,
                max_batch_size=32,
                max_batch_wait_ms=1,
            )
        )
        self.addCleanup(runtime.close)
        runtime.wait_until_ready(timeout=30.0)

        request = {
            "query_model_name": "colbertv2",
            "query": [[1.0, 0.0], [0.0, 1.0]],
            "final_k": 2,
        }

        self.assertEqual(runtime.search(request), session.search(request))

    def test_runtime_wait_until_ready_exposes_worker_pids(self) -> None:
        service_root, _module, _session, temp_dir = self._seed_snapshot()
        self.addCleanup(temp_dir.cleanup)

        runtime = prepare_exact_search_runtime(
            service_root=service_root,
            collection_id="news",
            tenant_id="tenant-a",
            namespace_id="search",
            snapshot_id="snapshot-0001",
            config=PreparedExactSearchRuntimeConfig(
                concurrency_lane_count=2,
                worker_count=1,
                max_batch_size=8,
                max_batch_wait_ms=1,
            ),
        )
        self.addCleanup(runtime.close)

        runtime.wait_until_ready(timeout=30.0)
        worker_pids = runtime.worker_pids()

        self.assertEqual(len(worker_pids), 2)
        self.assertTrue(all(isinstance(pid, int) and pid > 0 for pid in worker_pids))

    def test_runtime_coalesces_concurrent_submitters(self) -> None:
        service_root, _module, session, temp_dir = self._seed_snapshot()
        self.addCleanup(temp_dir.cleanup)

        runtime = prepare_exact_search_runtime(
            service_root=service_root,
            collection_id="news",
            tenant_id="tenant-a",
            namespace_id="search",
            snapshot_id="snapshot-0001",
            config=PreparedExactSearchRuntimeConfig(
                concurrency_lane_count=2,
                worker_count=2,
                max_batch_size=8,
                max_batch_wait_ms=25,
            ),
        )
        self.addCleanup(runtime.close)
        self.assertEqual(runtime.config.concurrency_lane_count, 2)

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

        barrier = threading.Barrier(len(requests) + 1)
        responses: list[dict | None] = [None] * len(requests)
        failures: list[BaseException] = []

        def run(index: int, request: dict) -> None:
            try:
                barrier.wait()
                responses[index] = runtime.search(request, timeout=10.0)
            except BaseException as exc:  # pragma: no cover - surfaced via assertion
                failures.append(exc)

        threads = [
            threading.Thread(target=run, args=(index, request), daemon=True)
            for index, request in enumerate(requests)
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=10.0)

        self.assertEqual(failures, [])
        expected = [session.search(request) for request in requests]
        self.assertEqual(responses, expected)

        stats = runtime.stats()
        self.assertEqual(stats.submitted_request_count, len(requests))
        self.assertEqual(stats.completed_request_count, len(requests))
        self.assertEqual(stats.failed_request_count, 0)
        self.assertLess(stats.executed_batch_count, len(requests))
        self.assertGreater(stats.max_observed_batch_size, 1)
        self.assertGreaterEqual(stats.max_observed_queue_depth, 1)

    def test_runtime_rejects_submit_after_close(self) -> None:
        service_root, _module, _session, temp_dir = self._seed_snapshot()
        self.addCleanup(temp_dir.cleanup)

        runtime = prepare_exact_search_scheduler(
            service_root=service_root,
            collection_id="news",
            tenant_id="tenant-a",
            namespace_id="search",
            snapshot_id="snapshot-0001",
        )
        runtime.close()

        with self.assertRaises(RuntimeError):
            runtime.submit(
                {
                    "query_model_name": "colbertv2",
                    "query": [[1.0, 0.0]],
                    "final_k": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
