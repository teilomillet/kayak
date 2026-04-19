"""Builds and loads the Mojo-backed hosted-engine service bindings."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

from .payloads import create_collection_request_payload

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / ".cache"
HF_HOME = CACHE_ROOT / "huggingface"
HF_DATASETS_CACHE = HF_HOME / "datasets"
TORCH_EXTENSIONS_DIR = CACHE_ROOT / "torch_extensions"
IR_DATASETS_HOME = CACHE_ROOT / "ir_datasets"
PYTHON_MOJO_CACHE = CACHE_ROOT / "python_mojo"
BRIDGE_ROOT = Path(__file__).resolve().parents[1] / "kayak_bridge"
ARTIFACTS_DIR = BRIDGE_ROOT / "_artifacts"
BUNDLED_ENGINE_ROOT = BRIDGE_ROOT / "_engine"

MODULE_SHORT_NAME = "_mojo_service_bindings"
MODULE_FULL_NAME = f"kayak_engine.{MODULE_SHORT_NAME}"
BINDING_SOURCE = Path(__file__).with_name(f"{MODULE_SHORT_NAME}.mojo")


def configure_local_caches() -> None:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    (HF_HOME / "hub").mkdir(parents=True, exist_ok=True)
    HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
    TORCH_EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
    IR_DATASETS_HOME.mkdir(parents=True, exist_ok=True)
    PYTHON_MOJO_CACHE.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(TORCH_EXTENSIONS_DIR))
    os.environ.setdefault("IR_DATASETS_HOME", str(IR_DATASETS_HOME))


configure_local_caches()


def _hash_inputs(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()[:16]


def _mojo_sources() -> list[Path]:
    source_root = _mojo_source_root()
    if source_root is not None:
        return sorted(source_root.rglob("*.mojo"))
    return []


def _compiled_extension_suffix() -> str:
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    if not suffixes:
        raise RuntimeError("Python runtime does not expose extension suffixes")
    return suffixes[0]


def _mojo_binary_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("mojo.exe", "mojo")
    return ("mojo",)


def _candidate_mojo_directories() -> tuple[Path, ...]:
    candidates = [
        Path(sys.executable).resolve().parent,
        Path(sys.prefix) / "bin",
        Path(sys.exec_prefix) / "bin",
        Path(sys.prefix) / "Scripts",
        Path(sys.exec_prefix) / "Scripts",
    ]

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidates.extend([Path(virtual_env) / "bin", Path(virtual_env) / "Scripts"])

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend([Path(conda_prefix) / "bin", Path(conda_prefix) / "Scripts"])

    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return tuple(ordered)


def _command_is_usable(command: list[str]) -> bool:
    result = subprocess.run(
        [*command, "--version"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _interpreter_local_mojo_command() -> list[str] | None:
    for directory in _candidate_mojo_directories():
        for binary_name in _mojo_binary_names():
            candidate = directory / binary_name
            if not candidate.exists() or not candidate.is_file():
                continue
            if not os.access(candidate, os.X_OK):
                continue

            command = [str(candidate)]
            if _command_is_usable(command):
                return command
    return None


def _detect_mojo_command() -> list[str]:
    configured = os.environ.get("KAYAK_MOJO_CLI")
    if configured:
        return [configured]

    interpreter_local = _interpreter_local_mojo_command()
    if interpreter_local is not None:
        return interpreter_local

    mojo_path = shutil.which("mojo")
    if mojo_path is not None and _command_is_usable([mojo_path]):
        return [mojo_path]

    pixi_path = shutil.which("pixi")
    if pixi_path is not None:
        pixi_command = [pixi_path, "run", "mojo"]
        if _command_is_usable(pixi_command):
            return pixi_command

    raise RuntimeError(
        "Kayak Engine could not find a usable Mojo CLI. Set KAYAK_MOJO_CLI, "
        "install Mojo into the active environment, or make `mojo` available "
        "on PATH."
    )


def _mojo_source_root() -> Path | None:
    repo_source_root = REPO_ROOT / "kayak"
    if repo_source_root.exists():
        return repo_source_root

    bundled_source_root = BUNDLED_ENGINE_ROOT / "kayak"
    if bundled_source_root.exists():
        return bundled_source_root

    return None


def _bundled_mojopkg() -> Path | None:
    artifact = ARTIFACTS_DIR / "kayak.mojopkg"
    if artifact.exists():
        return artifact
    return None


def _build_mojopkg(cache_key: str) -> Path:
    bundled = _bundled_mojopkg()
    if bundled is not None:
        return bundled

    source_root = _mojo_source_root()
    if source_root is None:
        raise RuntimeError(
            "Kayak Engine could not find Mojo sources in the repo or the "
            "installed package, and no bundled kayak.mojopkg artifact was present."
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
            "failed to build kayak.mojopkg for hosted engine service\n"
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
            "failed to build Mojo extension module for hosted engine service\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return extension_path


def _load_extension(extension_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(MODULE_FULL_NAME, extension_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"failed to create Python import spec for {extension_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_FULL_NAME] = module
    spec.loader.exec_module(module)
    return module


def _wrap_module(module: ModuleType) -> ModuleType:
    raw_create_collection_json = module.create_collection_json

    def create_collection_json(service_root: str, request: dict[str, object]) -> str:
        return raw_create_collection_json(
            service_root,
            create_collection_request_payload(request),
        )

    module.create_collection_json = create_collection_json
    return module


@lru_cache(maxsize=1)
def load_module() -> ModuleType:
    mojo_sources = _mojo_sources()
    cache_key = _hash_inputs([BINDING_SOURCE, *mojo_sources])
    mojopkg_path = _build_mojopkg(cache_key)
    extension_path = _build_extension(cache_key, mojopkg_path)
    return _wrap_module(_load_extension(extension_path))
