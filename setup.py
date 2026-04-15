from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


REPO_ROOT = Path(__file__).resolve().parent
MOJO_SOURCE_ROOT = REPO_ROOT / "kayak"
BUNDLED_ARTIFACTS_ROOT = Path("kayak_bridge") / "_artifacts"


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


def _mojo_version(command: list[str]) -> str:
    result = subprocess.run(
        [*command, "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() or result.stderr.strip() or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        self._prune_stale_release_payloads()
        self._build_bundled_mojopkg()

    def _prune_stale_release_payloads(self) -> None:
        stale_paths = (
            Path(self.build_lib) / "kayak_engine",
            Path(self.build_lib) / "kayak_bridge" / "_engine",
        )
        for stale_path in stale_paths:
            if not stale_path.exists():
                continue
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            else:
                stale_path.unlink()
            self.announce(
                f"removed stale packaged payload at {stale_path}",
                level=2,
            )

    def _detect_mojo_command(self) -> list[str] | None:
        configured = os.environ.get("KAYAK_MOJO_CLI")
        if configured:
            command = shlex.split(configured)
            if command:
                return command

        interpreter_local = _interpreter_local_mojo_command()
        if interpreter_local is not None:
            return interpreter_local

        mojo_cli = shutil.which("mojo")
        if mojo_cli is not None and _command_is_usable([mojo_cli]):
            return [mojo_cli]

        pixi_cli = shutil.which("pixi")
        if pixi_cli is not None:
            pixi_command = [pixi_cli, "run", "mojo"]
            if _command_is_usable(pixi_command):
                return pixi_command

        return None

    def _build_bundled_mojopkg(self) -> None:
        mojo_command = self._detect_mojo_command()
        if mojo_command is None:
            raise RuntimeError(
                "Kayak wheel build requires a usable Mojo CLI so the bundled "
                "`kayak.mojopkg` can be produced. Install Mojo in the active "
                "build environment or set KAYAK_MOJO_CLI."
            )

        if not MOJO_SOURCE_ROOT.exists():
            raise RuntimeError(
                "Kayak wheel build requires the repo Mojo sources under "
                "`kayak/`, but they were not found."
            )

        artifact_dir = Path(self.build_lib) / BUNDLED_ARTIFACTS_ROOT
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "kayak.mojopkg"

        command = [
            *mojo_command,
            "package",
            str(MOJO_SOURCE_ROOT),
            "-o",
            str(artifact_path),
        ]
        self.announce(
            f"building bundled Mojo package with: {' '.join(command)}",
            level=2,
        )
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        self._write_mojopkg_metadata(
            artifact_dir=artifact_dir,
            artifact_path=artifact_path,
            mojo_version=_mojo_version(mojo_command),
        )

    def _write_mojopkg_metadata(
        self,
        *,
        artifact_dir: Path,
        artifact_path: Path,
        mojo_version: str,
    ) -> None:
        metadata_path = artifact_dir / "mojopkg_build.json"
        payload = {
            "schema_version": 1,
            "project_version": self.distribution.get_version(),
            "mojo_version": mojo_version,
            "artifact_filename": artifact_path.name,
            "artifact_sha256": _sha256(artifact_path),
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


setup(cmdclass={"build_py": build_py})
