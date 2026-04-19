from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from kayak_engine import ExactScoringOptions, prepare_exact_search_session
from kayak_engine.mojo_service import load_module
from kayak_engine.payloads import (
    PayloadError,
    document_payload_parts,
    exact_search_request_payload,
    lifecycle_request_payload,
)


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


@unittest.skipUnless(
    _mojo_backend_available(), "prepared exact search session requires Mojo"
)
class PreparedExactSearchSessionTests(unittest.TestCase):
    def _build_service_root(self) -> tuple[object, tempfile.TemporaryDirectory[str], Path]:
        temp_dir = tempfile.TemporaryDirectory(prefix="kayak-prepared-exact-")
        service_root = Path(temp_dir.name) / "service-root"
        module = load_module()
        _create_collection(module, service_root)
        return module, temp_dir, service_root

    def test_prepared_search_matches_stateless_search(self) -> None:
        module, temp_dir, service_root = self._build_service_root()
        self.addCleanup(temp_dir.cleanup)

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
        request = {
            "query_model_name": "colbertv2",
            "query": [[1.0, 0.0], [0.0, 1.0]],
            "final_k": 2,
        }

        actual = session.search(request)
        expected = json.loads(
            module.exact_search_json(
                str(service_root),
                exact_search_request_payload(
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                        "snapshot_id": "snapshot-0001",
                        **request,
                    }
                ),
            )
        )

        self.assertEqual(actual, expected)
        self.assertFalse(session.load_text_corpus)

    def test_create_collection_persists_document_encoder_compression(self) -> None:
        temp_dir = tempfile.TemporaryDirectory(prefix="kayak-prepared-exact-")
        self.addCleanup(temp_dir.cleanup)
        service_root = Path(temp_dir.name) / "service-root"
        module = load_module()

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
        )

        lifecycle = json.loads(
            module.collection_lifecycle_json(
                str(service_root),
                lifecycle_request_payload(
                    {
                        "collection_id": "news",
                        "tenant_id": "tenant-a",
                        "namespace_id": "search",
                    }
                ),
            )
        )
        self.assertEqual(
            lifecycle["document_encoder_compression"]["kind"],
            "memory_tokens",
        )
        self.assertEqual(
            lifecycle["document_encoder_compression"]["config"],
            [{"key": "document_vector_budget", "value": "8"}],
        )

    def test_prepared_batch_matches_repeated_prepared_search(self) -> None:
        module, temp_dir, service_root = self._build_service_root()
        self.addCleanup(temp_dir.cleanup)

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
        ]
        scoring = ExactScoringOptions(
            enable_parallel_work_item_oversubscription=False,
            parallel_work_item_count_override=2,
        )

        expected = [session.search(request, scoring=scoring) for request in requests]
        actual = session.search_batch(
            requests,
            worker_count=2,
            scoring=scoring,
        )

        self.assertEqual(actual, expected)

    def test_prepared_session_stays_pinned_to_original_snapshot(self) -> None:
        module, temp_dir, service_root = self._build_service_root()
        self.addCleanup(temp_dir.cleanup)

        _upsert_documents(
            module,
            service_root,
            [
                {
                    "doc_id": "doc-a",
                    "vectors": [[1.0, 0.0]],
                    "text": "first document",
                    "metadata": {"topic": "alpha"},
                },
                {
                    "doc_id": "doc-b",
                    "vectors": [[0.0, 1.0]],
                    "text": "second document",
                    "metadata": {"topic": "beta"},
                },
            ],
        )
        _create_snapshot(module, service_root, "snapshot-0001")

        pinned_session = prepare_exact_search_session(
            service_root=service_root,
            collection_id="news",
            tenant_id="tenant-a",
            namespace_id="search",
            snapshot_id="snapshot-0001",
        )

        _upsert_documents(
            module,
            service_root,
            [
                {
                    "doc_id": "doc-c",
                    "vectors": [[2.0, 0.0]],
                    "text": "new stronger document",
                    "metadata": {"topic": "gamma"},
                }
            ],
        )
        _create_snapshot(module, service_root, "snapshot-0002")

        refreshed_session = prepare_exact_search_session(
            service_root=service_root,
            collection_id="news",
            tenant_id="tenant-a",
            namespace_id="search",
            snapshot_id="snapshot-0002",
        )
        request = {
            "query_model_name": "colbertv2",
            "query": [[1.0, 0.0]],
            "final_k": 1,
        }

        pinned_response = pinned_session.search(request)
        refreshed_response = refreshed_session.search(request)

        self.assertEqual(pinned_response["snapshot_id"], "snapshot-0001")
        self.assertEqual(refreshed_response["snapshot_id"], "snapshot-0002")
        self.assertEqual(pinned_response["hits"][0]["doc_id"], "doc-a")
        self.assertEqual(refreshed_response["hits"][0]["doc_id"], "doc-c")

    def test_prepared_session_rejects_request_identity_drift(self) -> None:
        module, temp_dir, service_root = self._build_service_root()
        self.addCleanup(temp_dir.cleanup)

        _upsert_documents(
            module,
            service_root,
            [
                {
                    "doc_id": "doc-a",
                    "vectors": [[1.0, 0.0]],
                    "text": "first document",
                    "metadata": {"topic": "alpha"},
                }
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

        with self.assertRaises(PayloadError):
            session.search(
                {
                    "snapshot_id": "snapshot-0002",
                    "query_model_name": "colbertv2",
                    "query": [[1.0, 0.0]],
                    "final_k": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
