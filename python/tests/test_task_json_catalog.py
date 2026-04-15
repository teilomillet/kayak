from __future__ import annotations

from pathlib import Path

from kayak_bridge.task_json_catalog import (
    REPO_ROOT,
    default_task_json_output_path,
    named_task_json_keys,
)


def test_named_task_json_keys_include_new_public_slices() -> None:
    keys = set(named_task_json_keys())
    assert "bright_stackoverflow_real_subset" in keys
    assert "lemb_narrativeqa_real_subset" in keys
    assert "r2med_biology_real_subset" in keys


def test_default_task_json_output_path_uses_stable_cache_locations() -> None:
    assert default_task_json_output_path(
        "bright_stackoverflow_real_subset"
    ) == REPO_ROOT / ".cache/kayak/bright_stackoverflow_real_subset/python_task.json"
    assert default_task_json_output_path(
        "browsecomp_plus_evidence"
    ) == REPO_ROOT / ".cache/kayak/browsecomp_plus_real_subset/python_task_evidence.json"
    assert default_task_json_output_path(
        "browsecomp_plus_gold"
    ) == REPO_ROOT / ".cache/kayak/browsecomp_plus_real_subset/python_task_gold.json"
