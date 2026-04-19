"""Owns Python access to the native latent-proxy projection microbenchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kayak_engine.mojo_service import load_module

from .json_task_loader import load_task_json


def benchmark_latent_proxy_projection_profile(
    *,
    task_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    task = load_task_json(str(task_path))
    module = load_module()
    return json.loads(
        module.benchmark_latent_proxy_projection_profile_json(
            {
                "dataset_id": str(task["dataset_id"]),
                "model_name": str(task["model_name"]),
                "task_path": str(task_path),
                "artifact_root": str(artifact_root),
            }
        )
    )
