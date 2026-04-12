from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


REPO_ROOT = Path(__file__).resolve().parent
MOJO_SOURCE_ROOT = REPO_ROOT / "kayak"


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        self._build_mojopkg_if_possible()

    def _build_mojopkg_if_possible(self) -> None:
        mojo_cli = shutil.which("mojo")
        if mojo_cli is None:
            self.announce(
                "skipping kayak.mojopkg build because Mojo is not on PATH",
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
            mojo_cli,
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
