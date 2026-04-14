from __future__ import annotations

import json
from pathlib import Path
import os
import select
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_MODULE = "kayak_engine.server"


def http_json(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class HostedEngineServer:
    def __init__(self, root: Path):
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        python_root = str(REPO_ROOT / "python")
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
        deadline = time.monotonic() + 180
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
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()

    def __enter__(self) -> "HostedEngineServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = exc_type
        _ = exc
        _ = tb
        self.close()


class HostedEngineHttpTest(unittest.TestCase):
    def test_network_happy_path_for_collection_snapshot_search_and_explain(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kayak-http-service-") as temp_dir:
            with HostedEngineServer(Path(temp_dir) / "service-root") as server:
                status, payload = http_json("GET", f"{server.base_url}/health")
                self.assertEqual(status, 200)
                self.assertEqual(payload["status"], "ok")
                self.assertEqual(payload["collection_count"], 0)

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
                self.assertEqual(payload["collection_id"], "news")
                self.assertEqual(payload["model_name"], "colbertv2")

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
                                "metadata": {"topic": "alpha"},
                            },
                            {
                                "doc_id": "doc-b",
                                "vectors": [[0.0, 1.0], [1.0, 0.0]],
                                "text": "beta evidence document",
                                "metadata": {"topic": "beta"},
                            },
                        ],
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["upserted_count"], 2)
                self.assertEqual(payload["draft_document_count"], 2)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/snapshots",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        "reason": "initial publish",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["snapshot_id"], "snapshot-0001")
                self.assertEqual(payload["document_count"], 2)

                status, payload = http_json("GET", f"{server.base_url}/metrics")
                self.assertEqual(status, 200)
                self.assertEqual(payload["collection_count"], 1)
                self.assertEqual(payload["document_count"], 2)
                self.assertGreaterEqual(payload["vector_count"], 4)

                search_request = {
                    "collection_id": "news",
                    "tenant_id": "tenant-a",
                    "namespace_id": "search",
                    "snapshot_id": "snapshot-0001",
                    "query_model_name": "colbertv2",
                    "query": [[1.0, 0.0], [0.0, 1.0]],
                    "final_k": 2,
                }
                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/search",
                    search_request,
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["collection_id"], "news")
                self.assertEqual(payload["snapshot_id"], "snapshot-0001")
                self.assertEqual(payload["hits"][0]["doc_id"], "doc-a")

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/explain",
                    {
                        **search_request,
                        "query_text": "alpha evidence",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["explain"]["collection_id"], "news")
                self.assertEqual(
                    payload["explain"]["plan"]["candidate_generator_kind"],
                    "exact_full_scan",
                )

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/planned-search",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        "query_model_name": "colbertv2",
                        "query": [[1.0, 0.0], [0.0, 1.0]],
                        "query_text": "alpha evidence",
                        "final_k": 2,
                        "candidate_k": 4,
                        "goal": "balanced",
                        "preferred_candidate_generator_kinds": [
                            "document_proxy",
                            "exact_full_scan",
                        ],
                    },
                )
                self.assertEqual(status, 200)
                self.assertIn("selection", payload)
                self.assertIn("search", payload)
                self.assertEqual(payload["search"]["hits"][0]["doc_id"], "doc-a")

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/search",
                    {
                        **search_request,
                        "query_model_name": "wrong-model",
                    },
                )
                self.assertEqual(status, 400)
                self.assertIn("query_model_name", payload["error"])


if __name__ == "__main__":
    unittest.main()
