"""Small stdlib HTTP server for the hosted-engine transport."""

from __future__ import annotations

import argparse
from concurrent.futures import Future
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Queue
import sys
import threading
from typing import Any, Callable

from .hosted_prepared_exact_runtime_registry import (
    HostedPreparedExactRuntimeRegistry,
)
from .mojo_service import load_module
from .payloads import (
    PayloadError,
    delete_documents_request_payload,
    document_payload_parts,
    exact_search_request_payload,
    execute_reclaim_request_payload,
    export_snapshot_request_payload,
    filter_payload_parts,
    import_snapshot_request_payload,
    lifecycle_request_payload,
    prepared_exact_runtime_config_payload,
    retention_update_request_payload,
    require_bool,
    require_int,
    require_list,
    require_object,
    require_query_vectors,
    require_string,
)


PREPARED_EXACT_ROUTE_PATHS = frozenset(
    {
        "/v1/prepared-exact-runtimes",
        "/v1/prepared-exact-runtimes:close",
        "/v1/prepared-exact-runtimes:stats",
        "/v1/prepared-exact-search",
        "/v1/prepared-exact-search-batch",
    }
)


class HostedEngineExecutor:
    """Runs all direct Mojo engine calls on one dedicated thread."""

    __slots__ = ("_queue", "_ready", "_startup_error", "engine_module", "_thread")

    def __init__(self) -> None:
        self._queue: Queue[tuple[Callable[[], str], Future[str]] | None] = Queue()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self.engine_module: Any | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="KayakHostedEngineExecutor",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=180.0)
        if self._startup_error is not None:
            raise RuntimeError("failed to start hosted engine executor") from self._startup_error
        if self.engine_module is None:
            raise RuntimeError("hosted engine executor did not initialize")

    def _run(self) -> None:
        try:
            self.engine_module = load_module()
        except BaseException as exc:  # pragma: no cover - startup guardrail
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        while True:
            item = self._queue.get()
            if item is None:
                return
            callback, future = item
            if future.cancelled():
                continue
            try:
                future.set_result(callback())
            except BaseException as exc:  # pragma: no cover - surfaced to caller
                future.set_exception(exc)

    def call(self, callback: Callable[[], str]) -> str:
        future: Future[str] = Future()
        self._queue.put((callback, future))
        return future.result(timeout=180.0)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise RuntimeError("hosted engine executor did not stop")


class KayakEngineHttpServer(ThreadingHTTPServer):
    """HTTP server carrying service configuration plus explicit runtime state."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        service_root: Path,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.service_root = service_root
        self.engine_executor = HostedEngineExecutor()
        self.engine_module = self.engine_executor.engine_module
        self.prepared_exact_runtime_registry = HostedPreparedExactRuntimeRegistry(
            service_root=service_root
        )

    def server_close(self) -> None:
        self.prepared_exact_runtime_registry.close_all()
        self.engine_executor.close()
        super().server_close()


class KayakEngineHandler(BaseHTTPRequestHandler):
    """Owns the first boring HTTP transport for the hosted engine."""

    server: KayakEngineHttpServer

    def log_message(self, format: str, *args: object) -> None:
        _ = format
        _ = args
        return

    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                self._write_json(
                    HTTPStatus.OK,
                    self.server.engine_executor.call(
                        lambda: self.server.engine_module.service_health_json(
                            str(self.server.service_root)
                        )
                    ),
                )
                return
            if self.path == "/metrics":
                self._write_json(
                    HTTPStatus.OK,
                    self.server.engine_executor.call(
                        lambda: self.server.engine_module.service_metrics_json(
                            str(self.server.service_root)
                        )
                    ),
                )
                return
            if self.path == "/v1/prepared-exact-runtimes":
                self._write_json(
                    HTTPStatus.OK,
                    json.dumps(
                        {
                            "active_runtime_count": (
                                self.server.prepared_exact_runtime_registry.active_runtime_count()
                            ),
                            "runtimes": self.server.prepared_exact_runtime_registry.list_runtime_summaries(),
                        }
                    ),
                )
                return
            self._write_error(HTTPStatus.NOT_FOUND, "route not found")
        except Exception as exc:  # pragma: no cover - transport guardrail
            self._write_engine_error(exc)

    def do_POST(self) -> None:
        routes: dict[str, Callable[[dict[str, Any]], str]] = {
            "/v1/collections": self._create_collection,
            "/v1/collections:lifecycle": self._collection_lifecycle,
            "/v1/collections:reclaim-execute": self._execute_reclaim,
            "/v1/collections:reclaim-plan": self._build_reclaim_plan,
            "/v1/collections:retention": self._update_collection_retention,
            "/v1/debug-search": self._exact_debug_search,
            "/v1/documents:delete": self._delete_documents,
            "/v1/documents:upsert": self._upsert_documents,
            "/v1/prepared-exact-runtimes": self._prepare_exact_runtime,
            "/v1/prepared-exact-runtimes:close": self._close_exact_runtime,
            "/v1/prepared-exact-runtimes:stats": self._prepared_exact_runtime_stats,
            "/v1/prepared-exact-search": self._prepared_exact_search,
            "/v1/prepared-exact-search-batch": self._prepared_exact_search_batch,
            "/v1/snapshots": self._create_snapshot,
            "/v1/snapshots:export": self._export_snapshot,
            "/v1/snapshots:import": self._import_snapshot,
            "/v1/search": self._exact_search,
            "/v1/explain": self._exact_explain,
            "/v1/planned-debug-search": self._planned_debug_search,
            "/v1/planned-search": self._planned_search,
            "/v1/planned-explain": self._planned_explain,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._write_error(HTTPStatus.NOT_FOUND, "route not found")
            return

        try:
            payload = self._read_json_body()
            if self.path in PREPARED_EXACT_ROUTE_PATHS:
                response = handler(payload)
            else:
                response = self.server.engine_executor.call(lambda: handler(payload))
            self._write_json(HTTPStatus.OK, response)
        except PayloadError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - transport guardrail
            self._write_engine_error(exc)

    def _read_json_body(self) -> dict[str, Any]:
        header = self.headers.get("Content-Length")
        if header is None:
            raise PayloadError("missing Content-Length header")
        try:
            content_length = int(header)
        except ValueError as exc:
            raise PayloadError("invalid Content-Length header") from exc
        body = self.rfile.read(content_length)
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PayloadError("request body must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise PayloadError("request body must decode to an object")
        return decoded

    def _write_json(self, status: HTTPStatus, payload: str) -> None:
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_error(self, status: HTTPStatus, message: str) -> None:
        self._write_json(status, json.dumps({"error": message}))

    def _write_engine_error(self, exc: Exception) -> None:
        message = str(exc)
        status = HTTPStatus.BAD_REQUEST
        lowered = message.lower()
        if (
            "does not exist" in lowered
            or "not resolve" in lowered
            or "no snapshots directory" in lowered
        ):
            status = HTTPStatus.NOT_FOUND
        self._write_error(status, message)

    def _create_collection(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.create_collection_json(
            str(self.server.service_root),
            {
                "collection_id": require_string(payload, "collection_id"),
                "tenant_id": require_string(payload, "tenant_id"),
                "namespace_id": require_string(payload, "namespace_id"),
                "collection_layout_family": require_string(
                    payload, "collection_layout_family", default=""
                ),
                "model_name": require_string(payload, "model_name"),
                "vector_scalar_name": require_string(
                    payload, "vector_scalar_name", default=""
                ),
                "vector_dim": require_int(payload, "vector_dim"),
                "default_keep_latest_inactive_count": require_int(
                    payload, "default_keep_latest_inactive_count", default=1
                ),
            },
        )

    def _upsert_documents(self, payload: dict[str, Any]) -> str:
        (
            doc_ids,
            document_vectors,
            texts,
            metadata_keys,
            metadata_values,
        ) = document_payload_parts(payload)
        return self.server.engine_module.upsert_documents_json(
            str(self.server.service_root),
            {
                "collection_id": require_string(payload, "collection_id"),
                "tenant_id": require_string(payload, "tenant_id"),
                "namespace_id": require_string(payload, "namespace_id"),
                "doc_ids": doc_ids,
                "document_vectors": document_vectors,
                "texts": texts,
                "metadata_keys": metadata_keys,
                "metadata_values": metadata_values,
            },
        )

    def _delete_documents(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.delete_documents_json(
            str(self.server.service_root),
            delete_documents_request_payload(payload),
        )

    def _create_snapshot(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.create_snapshot_json(
            str(self.server.service_root),
            {
                "collection_id": require_string(payload, "collection_id"),
                "tenant_id": require_string(payload, "tenant_id"),
                "namespace_id": require_string(payload, "namespace_id"),
                "snapshot_id": require_string(payload, "snapshot_id"),
                "reason": require_string(payload, "reason"),
            },
        )

    def _export_snapshot(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.export_snapshot_json(
            str(self.server.service_root),
            export_snapshot_request_payload(payload),
        )

    def _import_snapshot(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.import_snapshot_json(
            str(self.server.service_root),
            import_snapshot_request_payload(payload),
        )

    def _collection_lifecycle(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.collection_lifecycle_json(
            str(self.server.service_root),
            lifecycle_request_payload(payload),
        )

    def _build_reclaim_plan(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.build_reclaim_plan_json(
            str(self.server.service_root),
            lifecycle_request_payload(payload),
        )

    def _execute_reclaim(self, payload: dict[str, Any]) -> str:
        request = execute_reclaim_request_payload(payload)
        response = json.loads(
            self.server.engine_module.execute_reclaim_json(
                str(self.server.service_root),
                request,
            )
        )
        invalidated: list[dict[str, object]] = []
        if response["result"]["applied"]:
            reclaimed_snapshot_ids = [
                decision["snapshot_id"]
                for decision in request["plan"]["decisions"]
                if not decision["retain"]
            ]
            invalidated = (
                self.server.prepared_exact_runtime_registry.invalidate_reclaimed_snapshots(
                    collection_id=request["collection_id"],
                    tenant_id=request["tenant_id"],
                    namespace_id=request["namespace_id"],
                    reclaimed_snapshot_ids=reclaimed_snapshot_ids,
                )
            )
        response["invalidated_prepared_exact_runtime_count"] = len(invalidated)
        response["invalidated_prepared_exact_runtimes"] = invalidated
        return json.dumps(response)

    def _update_collection_retention(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.update_collection_retention_policy_json(
            str(self.server.service_root),
            retention_update_request_payload(payload),
        )

    def _exact_search(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.exact_search_json(
            str(self.server.service_root),
            exact_search_request_payload(payload),
        )

    def _exact_explain(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.exact_explain_json(
            str(self.server.service_root),
            exact_search_request_payload(payload),
        )

    def _exact_debug_search(self, payload: dict[str, Any]) -> str:
        return self.server.engine_module.debug_search_json(
            str(self.server.service_root),
            exact_search_request_payload(payload, debug_mode_default=True),
        )

    def _prepare_exact_runtime(self, payload: dict[str, Any]) -> str:
        runtime, reused = self.server.prepared_exact_runtime_registry.prepare_runtime(
            collection_id=require_string(payload, "collection_id"),
            tenant_id=require_string(payload, "tenant_id"),
            namespace_id=require_string(payload, "namespace_id"),
            snapshot_id=require_string(payload, "snapshot_id"),
            load_text_corpus=require_bool(
                payload,
                "load_text_corpus",
                default=False,
            ),
            config=prepared_exact_runtime_config_payload(payload),
        )
        return json.dumps(
            {
                "reused": reused,
                "active_runtime_count": self.server.prepared_exact_runtime_registry.active_runtime_count(),
                "runtime": runtime,
            }
        )

    def _prepared_exact_runtime_stats(self, payload: dict[str, Any]) -> str:
        runtime_id = require_string(payload, "runtime_id")
        return json.dumps(
            {
                "active_runtime_count": self.server.prepared_exact_runtime_registry.active_runtime_count(),
                "runtime": self.server.prepared_exact_runtime_registry.runtime_summary(
                    runtime_id
                ),
            }
        )

    def _close_exact_runtime(self, payload: dict[str, Any]) -> str:
        runtime_id = require_string(payload, "runtime_id")
        closed_runtime = self.server.prepared_exact_runtime_registry.close_runtime(
            runtime_id
        )
        return json.dumps(
            {
                "closed": True,
                "active_runtime_count": self.server.prepared_exact_runtime_registry.active_runtime_count(),
                "runtime": closed_runtime,
            }
        )

    def _prepared_exact_search(self, payload: dict[str, Any]) -> str:
        runtime_id = require_string(payload, "runtime_id")
        request = require_object(payload, "request")
        return json.dumps(
            {
                "runtime_id": runtime_id,
                "search": self.server.prepared_exact_runtime_registry.search(
                    runtime_id,
                    request,
                ),
            }
        )

    def _prepared_exact_search_batch(self, payload: dict[str, Any]) -> str:
        runtime_id = require_string(payload, "runtime_id")
        requests = require_list(payload, "requests")
        return json.dumps(
            {
                "runtime_id": runtime_id,
                "responses": self.server.prepared_exact_runtime_registry.search_batch(
                    runtime_id,
                    requests,
                ),
            }
        )

    def _planned_search(self, payload: dict[str, Any]) -> str:
        clause_fields, clause_operators, clause_values = filter_payload_parts(payload)
        return self.server.engine_module.planned_search_json(
            str(self.server.service_root),
            {
                "collection_id": require_string(payload, "collection_id"),
                "tenant_id": require_string(payload, "tenant_id"),
                "namespace_id": require_string(payload, "namespace_id"),
                "snapshot_id": require_string(payload, "snapshot_id"),
                "query": require_query_vectors(payload),
                "query_model_name": require_string(payload, "query_model_name"),
                "query_text": require_string(payload, "query_text", default=""),
                "final_k": require_int(payload, "final_k"),
                "candidate_k": require_int(payload, "candidate_k"),
                "goal": require_string(payload, "goal", default=""),
                "preferred_candidate_generator_kinds": require_list(
                    payload,
                    "preferred_candidate_generator_kinds",
                    default=[],
                ),
                "debug_mode": require_bool(payload, "debug_mode", default=False),
                "faithfulness_policy_kind": require_string(
                    payload,
                    "faithfulness_policy_kind",
                    default="best_effort",
                ),
                "stage2_reference_kind": require_string(
                    payload,
                    "stage2_reference_kind",
                    default="",
                ),
                "stage3_verifier_kind": require_string(
                    payload,
                    "stage3_verifier_kind",
                    default="",
                ),
                "clause_fields": clause_fields,
                "clause_operators": clause_operators,
                "clause_values": clause_values,
                "gem_graph_cluster_top_k_per_query_token": require_int(
                    payload,
                    "gem_graph_cluster_top_k_per_query_token",
                    default=3,
                ),
                "gem_graph_beam_width": require_int(
                    payload,
                    "gem_graph_beam_width",
                    default=9,
                ),
            },
        )

    def _planned_explain(self, payload: dict[str, Any]) -> str:
        clause_fields, clause_operators, clause_values = filter_payload_parts(payload)
        return self.server.engine_module.planned_explain_json(
            str(self.server.service_root),
            {
                "collection_id": require_string(payload, "collection_id"),
                "tenant_id": require_string(payload, "tenant_id"),
                "namespace_id": require_string(payload, "namespace_id"),
                "snapshot_id": require_string(payload, "snapshot_id"),
                "query": require_query_vectors(payload),
                "query_model_name": require_string(payload, "query_model_name"),
                "query_text": require_string(payload, "query_text", default=""),
                "final_k": require_int(payload, "final_k"),
                "candidate_k": require_int(payload, "candidate_k"),
                "goal": require_string(payload, "goal", default=""),
                "preferred_candidate_generator_kinds": require_list(
                    payload,
                    "preferred_candidate_generator_kinds",
                    default=[],
                ),
                "debug_mode": require_bool(payload, "debug_mode", default=False),
                "faithfulness_policy_kind": require_string(
                    payload,
                    "faithfulness_policy_kind",
                    default="best_effort",
                ),
                "stage2_reference_kind": require_string(
                    payload,
                    "stage2_reference_kind",
                    default="",
                ),
                "stage3_verifier_kind": require_string(
                    payload,
                    "stage3_verifier_kind",
                    default="",
                ),
                "clause_fields": clause_fields,
                "clause_operators": clause_operators,
                "clause_values": clause_values,
                "gem_graph_cluster_top_k_per_query_token": require_int(
                    payload,
                    "gem_graph_cluster_top_k_per_query_token",
                    default=3,
                ),
                "gem_graph_beam_width": require_int(
                    payload,
                    "gem_graph_beam_width",
                    default=9,
                ),
            },
        )

    def _planned_debug_search(self, payload: dict[str, Any]) -> str:
        clause_fields, clause_operators, clause_values = filter_payload_parts(payload)
        return self.server.engine_module.planned_debug_search_json(
            str(self.server.service_root),
            {
                "collection_id": require_string(payload, "collection_id"),
                "tenant_id": require_string(payload, "tenant_id"),
                "namespace_id": require_string(payload, "namespace_id"),
                "snapshot_id": require_string(payload, "snapshot_id"),
                "query": require_query_vectors(payload),
                "query_model_name": require_string(payload, "query_model_name"),
                "query_text": require_string(payload, "query_text", default=""),
                "final_k": require_int(payload, "final_k"),
                "candidate_k": require_int(payload, "candidate_k"),
                "goal": require_string(payload, "goal", default=""),
                "preferred_candidate_generator_kinds": require_list(
                    payload,
                    "preferred_candidate_generator_kinds",
                    default=[],
                ),
                "debug_mode": require_bool(payload, "debug_mode", default=True),
                "faithfulness_policy_kind": require_string(
                    payload,
                    "faithfulness_policy_kind",
                    default="best_effort",
                ),
                "stage2_reference_kind": require_string(
                    payload,
                    "stage2_reference_kind",
                    default="",
                ),
                "stage3_verifier_kind": require_string(
                    payload,
                    "stage3_verifier_kind",
                    default="",
                ),
                "clause_fields": clause_fields,
                "clause_operators": clause_operators,
                "clause_values": clause_values,
                "gem_graph_cluster_top_k_per_query_token": require_int(
                    payload,
                    "gem_graph_cluster_top_k_per_query_token",
                    default=3,
                ),
                "gem_graph_beam_width": require_int(
                    payload,
                    "gem_graph_beam_width",
                    default=9,
                ),
            },
        )


def serve(*, root: Path, host: str, port: int) -> int:
    root.mkdir(parents=True, exist_ok=True)
    server = KayakEngineHttpServer(
        (host, port),
        KayakEngineHandler,
        service_root=root,
    )
    actual_host, actual_port = server.server_address[:2]
    print(
        f"kayak-engine-serve listening on http://{actual_host}:{actual_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kayak-engine-serve",
        description="Run the Kayak hosted-engine HTTP service.",
    )
    parser.add_argument("--root", required=True, help="Service state root.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    return serve(root=Path(args.root), host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
