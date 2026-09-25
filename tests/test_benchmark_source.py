"""A benchmark worker must measure its snapshot, even with another editable install."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_benchmark_workers_import_the_measured_checkout(tmp_path: Path) -> None:
    pytest.importorskip("pyperf")
    root = Path(__file__).resolve().parents[1]
    for directory in ("kayak", "benchmarks"):
        shutil.copytree(
            root / directory,
            tmp_path / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    result = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.bench_overhead",
            "--workload",
            "small.validate_python",
            "--debug-single-value",
            "--loops",
            "1",
            "-o",
            str(result),
        ],
        cwd=tmp_path,
        check=True,
        timeout=45,
        capture_output=True,
        text=True,
    )
    payload: object = json.loads(result.read_text())
    assert isinstance(payload, dict)
    metadata: object = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["kayak_directory"] == str((tmp_path / "kayak").resolve())
    assert (
        metadata["sha256_decisions.py"]
        == hashlib.sha256((tmp_path / "kayak/decisions.py").read_bytes()).hexdigest()
    )


def test_benchmark_rejects_an_unrelated_installation(tmp_path: Path) -> None:
    pytest.importorskip("pyperf")
    root = Path(__file__).resolve().parents[1]
    # The harness is copied, but Kayak can only come from the existing installation.
    shutil.copytree(root / "benchmarks", tmp_path / "benchmarks")
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.bench_overhead", "--debug-single-value"],
        cwd=tmp_path,
        check=False,
        timeout=15,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "run the benchmark from its source root" in completed.stderr
