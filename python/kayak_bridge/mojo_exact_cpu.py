"""Builds and loads the explicit Mojo-backed exact CPU scoring module."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

from .cache_paths import PYTHON_MOJO_CACHE, REPO_ROOT, configure_local_caches


configure_local_caches()

MODULE_SHORT_NAME = "_mojo_exact_cpu_bindings"
MODULE_FULL_NAME = f"kayak_bridge.{MODULE_SHORT_NAME}"
BINDING_SOURCE = Path(__file__).with_name(f"{MODULE_SHORT_NAME}.mojo")
ARTIFACTS_DIR = Path(__file__).with_name("_artifacts")


def _hash_inputs(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()[:16]


def _repo_mojo_sources() -> list[Path]:
    source_root = REPO_ROOT / "kayak"
    if not source_root.exists():
        return []
    return sorted(source_root.rglob("*.mojo"))


def _compiled_extension_suffix() -> str:
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    if not suffixes:
        raise RuntimeError("Python runtime does not expose extension suffixes")
    return suffixes[0]


def _detect_mojo_command() -> list[str]:
    configured = os.environ.get("KAYAK_MOJO_CLI")
    if configured:
        return [configured]

    mojo_path = shutil.which("mojo")
    if mojo_path is not None:
        return [mojo_path]

    pixi_mojo = REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "mojo"
    if pixi_mojo.exists():
        return [str(pixi_mojo)]

    pixi_path = shutil.which("pixi")
    if pixi_path is not None:
        return [pixi_path, "run", "mojo"]

    raise RuntimeError(
        "Kayak could not find a Mojo CLI. Set KAYAK_MOJO_CLI or install Mojo."
    )


def _bundled_mojopkg() -> Path | None:
    artifact = ARTIFACTS_DIR / "kayak.mojopkg"
    if artifact.exists():
        return artifact
    return None


def _build_mojopkg(cache_key: str) -> Path:
    bundled = _bundled_mojopkg()
    if bundled is not None:
        return bundled

    source_root = REPO_ROOT / "kayak"
    if not source_root.exists():
        raise RuntimeError(
            "Kayak could not find Mojo sources or a bundled kayak.mojopkg artifact."
        )

    artifact_dir = PYTHON_MOJO_CACHE / cache_key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / "kayak.mojopkg"
    if artifact.exists():
        return artifact

    command = [
        *_detect_mojo_command(),
        "package",
        str(source_root),
        "-o",
        str(artifact),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to build kayak.mojopkg for mojo_exact_cpu backend\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return artifact


def _build_extension(cache_key: str, mojopkg_path: Path) -> Path:
    extension_path = (
        PYTHON_MOJO_CACHE / cache_key
        / f"{MODULE_SHORT_NAME}-{cache_key}{_compiled_extension_suffix()}"
    )
    extension_path.parent.mkdir(parents=True, exist_ok=True)
    if extension_path.exists():
        return extension_path

    command = [
        *_detect_mojo_command(),
        "build",
        str(BINDING_SOURCE),
        "--emit",
        "shared-lib",
        "-o",
        str(extension_path),
        "-I",
        str(mojopkg_path.parent),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to build Mojo extension module for mojo_exact_cpu backend\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return extension_path


def _load_extension(extension_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        MODULE_FULL_NAME, extension_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"failed to create Python import spec for {extension_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_FULL_NAME] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_module() -> ModuleType:
    mojo_sources = _repo_mojo_sources()
    cache_key = _hash_inputs([BINDING_SOURCE, *mojo_sources])
    mojopkg_path = _build_mojopkg(cache_key)
    extension_path = _build_extension(cache_key, mojopkg_path)
    return _load_extension(extension_path)
