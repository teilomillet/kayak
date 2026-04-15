"""Owns named encoded-task JSON builders for cross-engine comparisons.

This module owns:
- the mapping from a stable task key to a Python task builder
- the default cache path for each built task JSON

This module does not own:
- benchmark execution
- LanceDB or Kayak scorecards

Assumption:
- every registered builder returns the standard encoded-task JSON schema used
  by the Python comparison scripts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / ".cache" / "kayak"


def _task_output_path(dataset_key: str) -> Path:
    if dataset_key == "browsecomp_plus_evidence":
        return CACHE_ROOT / "browsecomp_plus_real_subset" / "python_task_evidence.json"
    if dataset_key == "browsecomp_plus_gold":
        return CACHE_ROOT / "browsecomp_plus_real_subset" / "python_task_gold.json"
    if dataset_key == "limit_small":
        return CACHE_ROOT / "limit_small_real_subset" / "python_task.json"
    return CACHE_ROOT / dataset_key / "python_task.json"


def named_task_json_keys() -> tuple[str, ...]:
    return (
        "bright_stackoverflow_real_subset",
        "browsecomp_plus_evidence",
        "browsecomp_plus_gold",
        "fiqa_real_subset",
        "legal_rag_bench_real_subset",
        "lemb_narrativeqa_real_subset",
        "limit_small",
        "r2med_biology_real_subset",
        "scifact_real_subset",
    )


def default_task_json_output_path(dataset_key: str) -> Path:
    if dataset_key not in named_task_json_keys():
        raise ValueError(f"unknown task json key: {dataset_key}")
    return _task_output_path(dataset_key)


def load_task_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_task_json(path: Path, task: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(task, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_named_task_json(dataset_key: str) -> dict[str, Any]:
    if dataset_key == "bright_stackoverflow_real_subset":
        from .bright_subset import build_bright_colbert_subset

        return build_bright_colbert_subset()

    if dataset_key == "browsecomp_plus_evidence":
        from .browsecomp_plus_subset import build_browsecomp_plus_colbert_subset

        return build_browsecomp_plus_colbert_subset()

    if dataset_key == "browsecomp_plus_gold":
        from .browsecomp_plus_subset import build_browsecomp_plus_gold_colbert_subset

        return build_browsecomp_plus_gold_colbert_subset()

    if dataset_key == "fiqa_real_subset":
        from .fiqa_subset import build_fiqa_colbert_subset

        return build_fiqa_colbert_subset()

    if dataset_key == "legal_rag_bench_real_subset":
        from .legal_rag_bench_subset import build_legal_rag_bench_colbert_subset

        return build_legal_rag_bench_colbert_subset()

    if dataset_key == "lemb_narrativeqa_real_subset":
        from .lemb_narrativeqa_subset import build_lemb_narrativeqa_colbert_subset

        return build_lemb_narrativeqa_colbert_subset()

    if dataset_key == "limit_small":
        from .limit_subset import build_limit_small_colbert_subset

        return build_limit_small_colbert_subset()

    if dataset_key == "r2med_biology_real_subset":
        from .r2med_biology_subset import build_r2med_biology_colbert_subset

        return build_r2med_biology_colbert_subset()

    if dataset_key == "scifact_real_subset":
        from .scifact_subset import build_scifact_colbert_subset

        return build_scifact_colbert_subset()

    raise ValueError(f"unknown task json key: {dataset_key}")
