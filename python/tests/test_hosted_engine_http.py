from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hosted_engine_test_support import HostedEngineServer, http_json


class HostedEngineHttpTest(unittest.TestCase):
    def test_network_happy_path_for_collection_snapshot_search_explain_and_debug(
        self,
    ) -> None:
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
                    f"{server.base_url}/v1/debug-search",
                    {
                        **search_request,
                        "query_text": "alpha evidence",
                    },
                )
                self.assertEqual(status, 200)
                self.assertIn("search", payload)
                self.assertIn("debug", payload)
                self.assertEqual(payload["search"]["hits"][0]["doc_id"], "doc-a")

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/planned-debug-search",
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
                self.assertIn("debug", payload)
                self.assertEqual(
                    payload["debug"]["search"]["hits"][0]["doc_id"],
                    "doc-a",
                )

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/documents:upsert",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "documents": [
                            {
                                "doc_id": "doc-c",
                                "vectors": [[1.0, 0.0], [0.0, 1.0]],
                                "text": "draft document to delete",
                            }
                        ],
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["draft_document_count"], 3)

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/documents:delete",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "doc_ids": ["doc-c"],
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["deleted_count"], 1)
                self.assertEqual(payload["remaining_draft_document_count"], 2)

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

    def test_network_collection_creation_accepts_document_encoder_compression(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="kayak-http-service-") as temp_dir:
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
                        "document_encoder_compression": {
                            "kind": "memory_tokens",
                            "config": [
                                {
                                    "key": "document_vector_budget",
                                    "value": "8",
                                }
                            ],
                        },
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["collection_id"], "news")

                status, payload = http_json(
                    "POST",
                    f"{server.base_url}/v1/collections:lifecycle",
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    payload["document_encoder_compression"]["kind"],
                    "memory_tokens",
                )
                self.assertEqual(
                    payload["document_encoder_compression"]["config"],
                    [{"key": "document_vector_budget", "value": "8"}],
                )

    def test_network_lifecycle_reclaim_and_snapshot_transfer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kayak-http-ops-") as temp_dir:
            service_root = Path(temp_dir) / "source-root"
            import_root = Path(temp_dir) / "import-root"
            bundle_root = Path(temp_dir) / "bundle-root"

            with HostedEngineServer(service_root) as source_server:
                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/collections",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "model_name": "colbertv2",
                        "vector_dim": 2,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["collection_id"], "ops")

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/documents:upsert",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "documents": [
                            {
                                "doc_id": "doc-a",
                                "vectors": [[1.0, 0.0], [1.0, 0.0]],
                                "text": "first retained document",
                            }
                        ],
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/snapshots",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        "reason": "publish first snapshot",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["snapshot_id"], "snapshot-0001")

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/collections:retention",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "default_keep_latest_inactive_count": 0,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["default_keep_latest_inactive_count"], 0)

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/documents:upsert",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "documents": [
                            {
                                "doc_id": "doc-b",
                                "vectors": [[0.0, 1.0], [0.0, 1.0]],
                                "text": "second active document",
                            }
                        ],
                    },
                )
                self.assertEqual(status, 200)

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/snapshots",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0002",
                        "reason": "publish second snapshot",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["snapshot_id"], "snapshot-0002")

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/collections:lifecycle",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "policy_override": {
                            "keep_latest_inactive_count": 0,
                            "pinned_snapshot_ids": ["snapshot-0001"],
                        },
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["active_snapshot_id"], "snapshot-0002")
                self.assertEqual(payload["effective_keep_latest_inactive_count"], 0)
                self.assertEqual(
                    payload["effective_pinned_snapshot_ids"],
                    ["snapshot-0001"],
                )
                self.assertEqual(
                    payload["reclaim_plan"]["reclaimable_snapshot_count"],
                    0,
                )

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/collections:reclaim-plan",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    payload["plan"]["reclaimable_snapshot_count"],
                    1,
                )
                self.assertEqual(
                    payload["plan"]["decisions"][0]["snapshot_id"],
                    "snapshot-0002",
                )
                plan = payload["plan"]

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/collections:reclaim-execute",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "dry_run": True,
                        "plan": plan,
                    },
                )
                self.assertEqual(status, 200)
                self.assertFalse(payload["result"]["applied"])
                self.assertEqual(payload["result"]["snapshot_count"], 1)

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/collections:reclaim-execute",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "dry_run": False,
                        "plan": plan,
                    },
                )
                self.assertEqual(status, 200)
                self.assertTrue(payload["result"]["applied"])
                self.assertEqual(payload["result"]["snapshot_ids"], ["snapshot-0001"])

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/collections:lifecycle",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["reclaim_plan"]["total_snapshot_count"], 1)

                status, payload = http_json(
                    "POST",
                    f"{source_server.base_url}/v1/snapshots:export",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0002",
                        "bundle_uri": f"file://{bundle_root}",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["snapshot_id"], "snapshot-0002")
                self.assertEqual(payload["segment_count"], 1)

            with HostedEngineServer(import_root) as import_server:
                status, payload = http_json(
                    "POST",
                    f"{import_server.base_url}/v1/snapshots:import",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0002",
                        "source_uri": f"file://{bundle_root}",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["snapshot_id"], "snapshot-0002")

                status, payload = http_json(
                    "POST",
                    f"{import_server.base_url}/v1/search",
                    {
                        "collection_id": "ops",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0002",
                        "query_model_name": "colbertv2",
                        "query": [[0.0, 1.0], [0.0, 1.0]],
                        "final_k": 1,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["hits"][0]["doc_id"], "doc-b")


if __name__ == "__main__":
    unittest.main()
