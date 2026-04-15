"""Owns one pinned exact-search session for one hosted-engine snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .mojo_service import load_module
from .payloads import PayloadError, exact_search_request_payload
from .prepared_exact_types import (
    ExactScoringOptions,
    PreparedExactSearchRuntimeConfig,
    _normalized_scoring,
    _require_bool,
    _require_non_bool_int,
    _scoring_payload,
)

if TYPE_CHECKING:
    from .prepared_exact_runtime import PreparedExactSearchRuntime


def _normalized_request_for_identity(
    payload: dict[str, Any],
    *,
    collection_id: str,
    tenant_id: str,
    namespace_id: str,
    snapshot_id: str,
    debug_mode_default: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PayloadError("request payload must be an object")

    identity = {
        "collection_id": collection_id,
        "tenant_id": tenant_id,
        "namespace_id": namespace_id,
        "snapshot_id": snapshot_id,
    }
    for key, expected in identity.items():
        if key in payload and payload[key] != expected:
            raise PayloadError(
                f"prepared exact search session {key} does not match request"
            )

    return exact_search_request_payload(
        {**payload, **identity},
        debug_mode_default=debug_mode_default,
    )


class PreparedExactSearchSession:
    """Pins one hosted snapshot in Mojo for repeated exact search from Python."""

    __slots__ = (
        "service_root",
        "collection_id",
        "tenant_id",
        "namespace_id",
        "snapshot_id",
        "load_text_corpus",
        "_engine_module",
        "_prepared_session",
    )

    def __init__(
        self,
        *,
        service_root: Path,
        collection_id: str,
        tenant_id: str,
        namespace_id: str,
        snapshot_id: str,
        load_text_corpus: bool,
        engine_module: Any,
        prepared_session: object,
    ) -> None:
        self.service_root = service_root
        self.collection_id = collection_id
        self.tenant_id = tenant_id
        self.namespace_id = namespace_id
        self.snapshot_id = snapshot_id
        self.load_text_corpus = load_text_corpus
        self._engine_module = engine_module
        self._prepared_session = prepared_session

    def _normalized_request(
        self,
        payload: dict[str, Any],
        *,
        debug_mode_default: bool = False,
    ) -> dict[str, Any]:
        return _normalized_request_for_identity(
            payload,
            collection_id=self.collection_id,
            tenant_id=self.tenant_id,
            namespace_id=self.namespace_id,
            snapshot_id=self.snapshot_id,
            debug_mode_default=debug_mode_default,
        )

    def _search_normalized(
        self,
        request: dict[str, Any],
        *,
        scoring: ExactScoringOptions,
    ) -> dict[str, Any]:
        return json.loads(
            self._engine_module.prepared_exact_search_json(
                self._prepared_session,
                request,
                _scoring_payload(scoring),
            )
        )

    def _search_batch_normalized(
        self,
        requests: list[dict[str, Any]],
        *,
        worker_count: int,
        scoring: ExactScoringOptions,
    ) -> list[dict[str, Any]]:
        return [
            json.loads(response_json)
            for response_json in self._engine_module.prepared_exact_search_batch_json(
                self._prepared_session,
                requests,
                {
                    "worker_count": worker_count,
                    **_scoring_payload(scoring),
                },
            )
        ]

    def search(
        self,
        payload: dict[str, Any],
        *,
        scoring: ExactScoringOptions | None = None,
    ) -> dict[str, Any]:
        resolved_scoring = _normalized_scoring(scoring)
        request = self._normalized_request(payload)
        return self._search_normalized(request, scoring=resolved_scoring)

    def search_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        worker_count: int = 1,
        scoring: ExactScoringOptions | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(requests, list):
            raise TypeError("requests must be a list of exact-search payloads")
        if len(requests) == 0:
            return []

        resolved_worker_count = _require_non_bool_int("worker_count", worker_count)
        if resolved_worker_count < 1:
            raise ValueError("worker_count must be positive")

        resolved_scoring = _normalized_scoring(scoring)
        normalized_requests = [self._normalized_request(payload) for payload in requests]
        return self._search_batch_normalized(
            normalized_requests,
            worker_count=resolved_worker_count,
            scoring=resolved_scoring,
        )

    def runtime(
        self,
        *,
        config: PreparedExactSearchRuntimeConfig | None = None,
    ) -> PreparedExactSearchRuntime:
        from .prepared_exact_runtime import PreparedExactSearchRuntime

        return PreparedExactSearchRuntime(
            service_root=self.service_root,
            collection_id=self.collection_id,
            tenant_id=self.tenant_id,
            namespace_id=self.namespace_id,
            snapshot_id=self.snapshot_id,
            load_text_corpus=self.load_text_corpus,
            config=config,
        )

    def scheduler(
        self,
        *,
        config: PreparedExactSearchRuntimeConfig | None = None,
    ) -> PreparedExactSearchRuntime:
        """Compatibility alias for callers still using scheduler terminology."""

        return self.runtime(config=config)


def prepare_exact_search_session(
    *,
    service_root: str | Path,
    collection_id: str,
    tenant_id: str,
    namespace_id: str,
    snapshot_id: str,
    load_text_corpus: bool = True,
) -> PreparedExactSearchSession:
    """Prepare one local exact-search session pinned to one hosted snapshot."""

    resolved_service_root = Path(service_root)
    resolved_load_text_corpus = _require_bool("load_text_corpus", load_text_corpus)
    engine_module = load_module()
    prepared_session = engine_module.prepare_exact_search_session(
        str(resolved_service_root),
        collection_id,
        tenant_id,
        namespace_id,
        snapshot_id,
        resolved_load_text_corpus,
    )
    return PreparedExactSearchSession(
        service_root=resolved_service_root,
        collection_id=collection_id,
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        snapshot_id=snapshot_id,
        load_text_corpus=resolved_load_text_corpus,
        engine_module=engine_module,
        prepared_session=prepared_session,
    )
