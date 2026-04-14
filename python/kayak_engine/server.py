"""Small stdlib HTTP server for the hosted-engine transport."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
from typing import Any, Callable

from .mojo_service import load_module
from .payloads import (
    PayloadError,
    document_payload_parts,
    filter_payload_parts,
    require_bool,
    require_int,
    require_list,
    require_query_vectors,
    require_string,
)


class KayakEngineHttpServer(HTTPServer):
    """HTTP server carrying immutable service configuration."""

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        service_root: Path,
        engine_module: Any,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.service_root = service_root
        self.engine_module = engine_module


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
                    self.server.engine_module.service_health_json(
                        str(self.server.service_root)
                    ),
                )
                return
            if self.path == "/metrics":
                self._write_json(
                    HTTPStatus.OK,
                    self.server.engine_module.service_metrics_json(
                        str(self.server.service_root)
                    ),
                )
                return
            self._write_error(HTTPStatus.NOT_FOUND, "route not found")
        except Exception as exc:  # pragma: no cover - transport guardrail
            self._write_engine_error(exc)

    def do_POST(self) -> None:
        routes: dict[str, Callable[[dict[str, Any]], str]] = {
            "/v1/collections": self._create_collection,
            "/v1/documents:upsert": self._upsert_documents,
            "/v1/snapshots": self._create_snapshot,
            "/v1/search": self._exact_search,
            "/v1/explain": self._exact_explain,
            "/v1/planned-search": self._planned_search,
            "/v1/planned-explain": self._planned_explain,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._write_error(HTTPStatus.NOT_FOUND, "route not found")
            return

        try:
            payload = self._read_json_body()
            self._write_json(HTTPStatus.OK, handler(payload))
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

    def _exact_search(self, payload: dict[str, Any]) -> str:
        clause_fields, clause_operators, clause_values = filter_payload_parts(payload)
        return self.server.engine_module.exact_search_json(
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
                "debug_mode": require_bool(payload, "debug_mode", default=False),
                "clause_fields": clause_fields,
                "clause_operators": clause_operators,
                "clause_values": clause_values,
            },
        )

    def _exact_explain(self, payload: dict[str, Any]) -> str:
        clause_fields, clause_operators, clause_values = filter_payload_parts(payload)
        return self.server.engine_module.exact_explain_json(
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
                "debug_mode": require_bool(payload, "debug_mode", default=False),
                "clause_fields": clause_fields,
                "clause_operators": clause_operators,
                "clause_values": clause_values,
            },
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


def serve(*, root: Path, host: str, port: int) -> int:
    root.mkdir(parents=True, exist_ok=True)
    engine_module = load_module()
    server = KayakEngineHttpServer(
        (host, port),
        KayakEngineHandler,
        service_root=root,
        engine_module=engine_module,
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
