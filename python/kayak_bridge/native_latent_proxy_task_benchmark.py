"""Owns the native collection-backed latent-proxy task benchmark bridge.

This module keeps the runtime boundary explicit:
- materialize one native collection mirror from one task JSON and one exported
  latent-proxy artifact
- benchmark that materialized collection through the Mojo collection runtime

It does not own:
- training checkpoints
- latent-proxy artifact export
- reinterpretation of the native JSON summary schema in Python
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kayak_engine.mojo_service import load_module

from .json_task_loader import load_task_json


DEFAULT_COLLECTION_ID = "latent-proxy-bench"
DEFAULT_TENANT_ID = "public"
DEFAULT_NAMESPACE_ID = "benchmark"
DEFAULT_SNAPSHOT_ID = "snapshot-0001"


def _task_identity(task_path: str | Path) -> tuple[str, str]:
    task = load_task_json(str(task_path))
    return str(task["dataset_id"]), str(task["model_name"])


def materialize_native_latent_proxy_collection(
    *,
    task_path: str | Path,
    artifact_root: str | Path,
    collection_root: str | Path,
    collection_id: str = DEFAULT_COLLECTION_ID,
    tenant_id: str = DEFAULT_TENANT_ID,
    namespace_id: str = DEFAULT_NAMESPACE_ID,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    load_text_corpus: bool = True,
) -> dict[str, Any]:
    dataset_id, model_name = _task_identity(task_path)
    module = load_module()
    return json.loads(
        module.materialize_latent_proxy_collection_json(
            {
                "dataset_id": dataset_id,
                "model_name": model_name,
                "task_path": str(task_path),
                "artifact_root": str(artifact_root),
                "collection_root": str(collection_root),
                "collection_id": collection_id,
                "tenant_id": tenant_id,
                "namespace_id": namespace_id,
                "snapshot_id": snapshot_id,
                "load_text_corpus": load_text_corpus,
            }
        )
    )


def benchmark_materialized_collection_search(
    *,
    task_path: str | Path,
    collection_root: str | Path,
    candidate_k: int,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    candidate_generator_kind: str = "latent_proxy",
) -> dict[str, Any]:
    dataset_id, model_name = _task_identity(task_path)
    module = load_module()
    return json.loads(
        module.benchmark_materialized_collection_search_json(
            {
                "dataset_id": dataset_id,
                "model_name": model_name,
                "task_path": str(task_path),
                "collection_root": str(collection_root),
                "snapshot_id": snapshot_id,
                "candidate_generator_kind": candidate_generator_kind,
                "candidate_k": candidate_k,
            }
        )
    )


def benchmark_native_latent_proxy_task(
    *,
    task_path: str | Path,
    artifact_root: str | Path,
    collection_root: str | Path,
    candidate_k: int,
    collection_id: str = DEFAULT_COLLECTION_ID,
    tenant_id: str = DEFAULT_TENANT_ID,
    namespace_id: str = DEFAULT_NAMESPACE_ID,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    load_text_corpus: bool = True,
    candidate_generator_kind: str = "latent_proxy",
) -> dict[str, Any]:
    _ = materialize_native_latent_proxy_collection(
        task_path=task_path,
        artifact_root=artifact_root,
        collection_root=collection_root,
        collection_id=collection_id,
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        snapshot_id=snapshot_id,
        load_text_corpus=load_text_corpus,
    )
    return benchmark_materialized_collection_search(
        task_path=task_path,
        collection_root=collection_root,
        candidate_k=candidate_k,
        snapshot_id=snapshot_id,
        candidate_generator_kind=candidate_generator_kind,
    )
