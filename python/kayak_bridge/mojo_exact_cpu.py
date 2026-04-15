"""Builds and loads the explicit Mojo-backed exact CPU scoring module."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
from types import ModuleType

from .bundled_mojopkg_metadata import (
    BundledMojopkgMetadata,
    read_bundled_mojopkg_metadata,
)
from .cache_paths import PYTHON_MOJO_CACHE, REPO_ROOT, configure_local_caches


configure_local_caches()

MODULE_SHORT_NAME = "_mojo_exact_cpu_bindings"
MODULE_FULL_NAME = f"kayak_bridge.{MODULE_SHORT_NAME}"
BINDING_SOURCE = Path(__file__).with_name(f"{MODULE_SHORT_NAME}.mojo")
ARTIFACTS_DIR = Path(__file__).with_name("_artifacts")
REPO_MOJO_WRAPPER = REPO_ROOT / "scripts" / "run_mojo_with_pixi_python.sh"


def _hash_inputs(
    paths: list[Path],
    *,
    extra_tokens: tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    for token in extra_tokens:
        digest.update(token.encode("utf-8"))
    for path in paths:
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()[:16]


def _mojo_source_root() -> Path | None:
    repo_source_root = REPO_ROOT / "kayak"
    if repo_source_root.exists():
        return repo_source_root

    return None


def _mojo_sources() -> list[Path]:
    source_root = _mojo_source_root()
    if source_root is None:
        return []
    return sorted(source_root.rglob("*.mojo"))


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
        candidates.extend(
            [Path(virtual_env) / "bin", Path(virtual_env) / "Scripts"]
        )

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(
            [Path(conda_prefix) / "bin", Path(conda_prefix) / "Scripts"]
        )

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
        env=_mojo_subprocess_env(command),
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
        command = shlex.split(configured)
        if command:
            return command

    if REPO_MOJO_WRAPPER.exists() and _command_is_usable(
        ["bash", str(REPO_MOJO_WRAPPER)]
    ):
        return ["bash", str(REPO_MOJO_WRAPPER)]

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
        "Kayak could not find a usable Mojo CLI. Set KAYAK_MOJO_CLI, install "
        "Mojo into the active Python environment, or make `mojo` available "
        "on PATH."
    )


def _bundled_mojopkg() -> Path | None:
    artifact = ARTIFACTS_DIR / "kayak.mojopkg"
    if artifact.exists():
        return artifact
    return None


def _bundled_mojopkg_metadata() -> BundledMojopkgMetadata | None:
    return read_bundled_mojopkg_metadata(ARTIFACTS_DIR)


def _cache_key_inputs() -> tuple[list[Path], tuple[str, ...]]:
    mojo_sources = _mojo_sources()
    if mojo_sources:
        return [BINDING_SOURCE, *mojo_sources], ()

    bundled_artifact = _bundled_mojopkg()
    if bundled_artifact is None:
        return [BINDING_SOURCE], ()

    metadata = _bundled_mojopkg_metadata()
    if metadata is None:
        return [BINDING_SOURCE, bundled_artifact], ()

    # Installed wheels do not ship Mojo sources, so the bundled artifact and
    # its recorded build identity must participate in the extension cache key.
    return (
        [BINDING_SOURCE, bundled_artifact],
        (
            f"project_version:{metadata.project_version}",
            f"mojo_version:{metadata.mojo_version}",
            f"artifact_sha256:{metadata.artifact_sha256}",
        ),
    )


def _build_mojopkg(cache_key: str) -> Path:
    source_root = _mojo_source_root()
    bundled = _bundled_mojopkg()
    if bundled is not None:
        return bundled

    if source_root is None:
        raise RuntimeError(
            "Kayak could not find Mojo sources in the repo or the installed "
            "package, and no bundled kayak.mojopkg artifact was present."
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
        env=_mojo_subprocess_env(command),
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
        env=_mojo_subprocess_env(command),
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
    _ensure_python_runtime_env()
    cache_paths, cache_tokens = _cache_key_inputs()
    cache_key = _hash_inputs(cache_paths, extra_tokens=cache_tokens)
    mojopkg_path = _build_mojopkg(cache_key)
    try:
        extension_path = _build_extension(cache_key, mojopkg_path)
    except RuntimeError as exc:
        if not _is_bundled_mojopkg_version_mismatch(
            exc,
            mojopkg_path=mojopkg_path,
        ):
            raise
        raise _bundled_mojopkg_version_error(
            exc,
            mojopkg_path=mojopkg_path,
        ) from exc
    return _load_extension(extension_path)


def _mojo_subprocess_env(command: list[str]) -> dict[str, str]:
    """Sanitize subprocess state before invoking Mojo from mixed Python envs.

    Reason:
    - optional adapter demos often run under `uv run --with ...`
    - the repo's supported Mojo toolchain lives in the Pixi environment
    - forwarding UV/PYTHONHOME launcher state into Mojo can break the Python
      runtime that Mojo embeds while building or loading the extension module
    """

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("UV_")
    }
    env.pop("PYTHONHOME", None)
    env.pop("__PYVENV_LAUNCHER__", None)

    pixi_env_root = REPO_ROOT / ".pixi" / "envs" / "default"
    if _command_uses_pixi_runtime(command) and pixi_env_root.exists():
        env["VIRTUAL_ENV"] = str(pixi_env_root)
        env["CONDA_PREFIX"] = str(pixi_env_root)
        env["PATH"] = f"{pixi_env_root / 'bin'}:{env.get('PATH', '')}"
        env["DYLD_LIBRARY_PATH"] = (
            f"{pixi_env_root / 'lib'}:{env.get('DYLD_LIBRARY_PATH', '')}"
        )

    env["MOJO_PYTHON"] = sys.executable
    python_library = _current_python_library()
    if python_library is not None:
        env["MOJO_PYTHON_LIBRARY"] = python_library
    python_home = _current_python_home()
    if python_home is not None:
        env["PYTHONHOME"] = python_home

    return env


def _command_uses_pixi_runtime(command: list[str]) -> bool:
    if len(command) >= 2 and Path(command[0]).name == "pixi" and command[1] == "run":
        return True

    if not command:
        return False

    command_path = Path(command[0]).resolve()
    return ".pixi" in command_path.parts


def _current_python_library() -> str | None:
    libdir = Path(sysconfig.get_config_var("LIBDIR") or "")
    version_prefix = f"libpython{sys.version_info.major}.{sys.version_info.minor}"

    candidates: list[Path] = []
    for pattern in (
        version_prefix + ".dylib",
        version_prefix + ".so",
        version_prefix + ".so.*",
        "libpython*.dylib",
        "libpython*.so",
        "libpython*.so.*",
    ):
        candidates.extend(sorted(libdir.glob(pattern)))

    for candidate in candidates:
        if candidate.suffix == ".a":
            continue
        return str(candidate)

    ld_library = sysconfig.get_config_var("LDLIBRARY")
    if ld_library:
        library_path = libdir / ld_library
        if library_path.exists() and library_path.suffix != ".a":
            return str(library_path)

    return None


def _current_python_home() -> str | None:
    libdir = Path(sysconfig.get_config_var("LIBDIR") or "")
    if libdir.exists():
        return str(libdir.parent)
    return None


def _ensure_python_runtime_env() -> None:
    if "MOJO_PYTHON" not in os.environ:
        os.environ["MOJO_PYTHON"] = sys.executable

    python_library = _current_python_library()
    if python_library is not None and "MOJO_PYTHON_LIBRARY" not in os.environ:
        os.environ["MOJO_PYTHON_LIBRARY"] = python_library

    python_home = _current_python_home()
    if python_home is not None and "PYTHONHOME" not in os.environ:
        os.environ["PYTHONHOME"] = python_home


def _active_mojo_version() -> str | None:
    try:
        command = _detect_mojo_command()
    except RuntimeError:
        return None

    result = subprocess.run(
        [*command, "--version"],
        capture_output=True,
        text=True,
        env=_mojo_subprocess_env(command),
    )
    if result.returncode != 0:
        return None

    version = result.stdout.strip() or result.stderr.strip()
    if not version:
        return None
    return version


def _is_bundled_mojopkg_version_mismatch(
    error: RuntimeError,
    *,
    mojopkg_path: Path,
) -> bool:
    if mojopkg_path != _bundled_mojopkg():
        return False

    message = str(error)
    return (
        "Mojo package is incompatible with the current version of the Mojo compiler"
        in message
    )


def _bundled_mojopkg_version_error(
    error: RuntimeError,
    *,
    mojopkg_path: Path,
) -> RuntimeError:
    metadata = _bundled_mojopkg_metadata()
    active_mojo_version = _active_mojo_version()

    lines = [
        "Kayak could not load the bundled Mojo backend because the installed "
        "wheel and the active Mojo compiler do not match.",
    ]
    if metadata is not None:
        lines.append(
            "Bundled artifact: "
            f"{mojopkg_path.name} built for Kayak {metadata.project_version} "
            f"with {metadata.mojo_version}."
        )
    else:
        lines.append(
            "Bundled artifact metadata was not present, so the wheel does not "
            "record which Mojo version produced the packaged backend."
        )
    if active_mojo_version is not None:
        lines.append(f"Active Mojo CLI: {active_mojo_version}.")
    lines.append(
        "Install a compatible `mojo` release or upgrade `kayak` so the "
        "published wheel is rebuilt against your current Mojo toolchain."
    )
    lines.append(
        "Kayak wheels do not ship engine sources, so runtime source rebuild is "
        "not available."
    )
    lines.append("")
    lines.append("Original build error:")
    lines.append(str(error))
    return RuntimeError("\n".join(lines))
