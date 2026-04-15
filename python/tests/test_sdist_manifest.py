from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


SOURCES_PATH = REPO_ROOT / "python" / "kayak.egg-info" / "SOURCES.txt"


def _tracked_paths(prefix: str, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    output = subprocess.check_output(
        ["git", "ls-files", prefix],
        cwd=REPO_ROOT,
        text=True,
    )
    return tuple(
        line
        for line in output.splitlines()
        if line and Path(line).suffix in suffixes
    )


class SdistManifestTests(unittest.TestCase):
    def test_egg_info_sources_cover_required_python_and_mojo_inputs(self) -> None:
        subprocess.run(
            [sys.executable, "setup.py", "egg_info"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        sources = set(SOURCES_PATH.read_text(encoding="utf-8").splitlines())
        required_paths = (
            _tracked_paths("python/kayak", (".py", ".md"))
            + _tracked_paths("python/kayak_bridge", (".py", ".mojo"))
            + _tracked_paths("kayak", (".mojo",))
        )

        missing = tuple(path for path in required_paths if path not in sources)
        self.assertEqual(
            missing,
            (),
            f"SOURCES.txt is missing build inputs: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
