from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import kayak_bridge.mojo_exact_cpu as mojo_exact_cpu


class MojoPackagingTests(unittest.TestCase):
    def test_source_root_falls_back_to_bundled_engine_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            bundled_root = temp_root / "_engine"
            bundled_source_root = bundled_root / "kayak"
            bundled_source_root.mkdir(parents=True)
            source_file = bundled_source_root / "__init__.mojo"
            source_file.write_text("fn main():\n    pass\n", encoding="utf-8")

            with (
                mock.patch.object(
                    mojo_exact_cpu, "REPO_ROOT", temp_root / "missing-repo"
                ),
                mock.patch.object(
                    mojo_exact_cpu, "BUNDLED_ENGINE_ROOT", bundled_root
                ),
            ):
                self.assertEqual(
                    mojo_exact_cpu._mojo_source_root(), bundled_source_root
                )
                self.assertEqual(mojo_exact_cpu._mojo_sources(), [source_file])

    def test_build_mojopkg_prefers_bundled_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifacts_root = temp_root / "_artifacts"
            artifacts_root.mkdir(parents=True)
            bundled_artifact = artifacts_root / "kayak.mojopkg"
            bundled_artifact.write_bytes(b"placeholder mojopkg")

            with mock.patch.object(
                mojo_exact_cpu, "ARTIFACTS_DIR", artifacts_root
            ):
                self.assertEqual(
                    mojo_exact_cpu._build_mojopkg("cache-key"),
                    bundled_artifact,
                )

    def test_detect_mojo_command_finds_virtualenv_local_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            bin_dir = temp_root / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            mojo_binary = bin_dir / "mojo"
            mojo_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            mojo_binary.chmod(0o755)
            resolved_mojo_binary = mojo_binary.resolve()

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                self.assertEqual(command[1:], ["--version"])
                self.assertEqual(Path(command[0]).resolve(), resolved_mojo_binary)
                return subprocess.CompletedProcess(command, 0, "mojo 0.0", "")

            with (
                mock.patch.dict(
                    mojo_exact_cpu.os.environ,
                    {},
                    clear=True,
                ),
                mock.patch.object(
                    mojo_exact_cpu.sys, "executable", str(bin_dir / "python")
                ),
                mock.patch.object(
                    mojo_exact_cpu.sys, "prefix", str(temp_root / ".venv")
                ),
                mock.patch.object(
                    mojo_exact_cpu.sys, "exec_prefix", str(temp_root / ".venv")
                ),
                mock.patch.object(mojo_exact_cpu.shutil, "which", return_value=None),
                mock.patch.object(
                    mojo_exact_cpu.subprocess,
                    "run",
                    side_effect=fake_run,
                ),
            ):
                self.assertEqual(
                    mojo_exact_cpu._detect_mojo_command(),
                    [str(resolved_mojo_binary)],
                )

    def test_detect_mojo_command_skips_unusable_local_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            bin_dir = temp_root / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            mojo_binary = bin_dir / "mojo"
            mojo_binary.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            mojo_binary.chmod(0o755)
            resolved_mojo_binary = mojo_binary.resolve()

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if (
                    command[1:] == ["--version"]
                    and Path(command[0]).resolve() == resolved_mojo_binary
                ):
                    return subprocess.CompletedProcess(command, 1, "", "broken")
                if command == ["/usr/bin/pixi", "run", "mojo", "--version"]:
                    return subprocess.CompletedProcess(command, 0, "mojo 0.0", "")
                self.fail(f"unexpected command probe: {command}")

            def fake_which(command: str) -> str | None:
                if command == "pixi":
                    return "/usr/bin/pixi"
                return None

            with (
                mock.patch.dict(
                    mojo_exact_cpu.os.environ,
                    {},
                    clear=True,
                ),
                mock.patch.object(
                    mojo_exact_cpu.sys, "executable", str(bin_dir / "python")
                ),
                mock.patch.object(
                    mojo_exact_cpu.sys, "prefix", str(temp_root / ".venv")
                ),
                mock.patch.object(
                    mojo_exact_cpu.sys, "exec_prefix", str(temp_root / ".venv")
                ),
                mock.patch.object(
                    mojo_exact_cpu.shutil,
                    "which",
                    side_effect=fake_which,
                ),
                mock.patch.object(
                    mojo_exact_cpu.subprocess,
                    "run",
                    side_effect=fake_run,
                ),
            ):
                self.assertEqual(
                    mojo_exact_cpu._detect_mojo_command(),
                    ["/usr/bin/pixi", "run", "mojo"],
                )


if __name__ == "__main__":
    unittest.main()
