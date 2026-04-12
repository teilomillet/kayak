from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


REPO_ROOT = Path(__file__).resolve().parent
MOJO_SOURCE_ROOT = REPO_ROOT / "kayak"
BUNDLED_ENGINE_ROOT = Path("kayak_bridge") / "_engine" / "kayak"


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


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        self._stage_engine_sources()
        self._build_mojopkg_if_possible()

    def _stage_engine_sources(self) -> None:
        if not MOJO_SOURCE_ROOT.exists():
            self.announce(
                "skipping bundled Mojo source staging because kayak/ Mojo sources are missing",
                level=2,
            )
            return

        bundled_engine_root = Path(self.build_lib) / BUNDLED_ENGINE_ROOT
        bundled_engine_root.parent.mkdir(parents=True, exist_ok=True)
        if bundled_engine_root.exists():
            shutil.rmtree(bundled_engine_root)
        shutil.copytree(MOJO_SOURCE_ROOT, bundled_engine_root)
        self.announce(
            f"staged bundled Mojo sources at {bundled_engine_root}",
            level=2,
        )

    def _detect_mojo_command(self) -> list[str] | None:
        configured = os.environ.get("KAYAK_MOJO_CLI")
        if configured:
            return [configured]

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

    def _build_mojopkg_if_possible(self) -> None:
        mojo_command = self._detect_mojo_command()
        if mojo_command is None:
            self.announce(
                "skipping kayak.mojopkg build because no Mojo CLI was found; "
                "bundled engine sources will still be packaged",
                level=2,
            )
            return

        if not MOJO_SOURCE_ROOT.exists():
            self.announce(
                "skipping kayak.mojopkg build because kayak/ Mojo sources are missing",
                level=2,
            )
            return

        artifact_dir = Path(self.build_lib) / "kayak_bridge" / "_artifacts"
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


setup(cmdclass={"build_py": build_py})
