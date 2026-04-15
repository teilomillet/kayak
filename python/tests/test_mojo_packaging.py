from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import kayak_bridge.mojo_exact_cpu as mojo_exact_cpu


class MojoPackagingTests(unittest.TestCase):
    def test_source_root_only_uses_repo_mojo_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            repo_source_root = temp_root / "kayak"
            repo_source_root.mkdir(parents=True)
            source_file = repo_source_root / "__init__.mojo"
            source_file.write_text("fn main():\n    pass\n", encoding="utf-8")

            with mock.patch.object(mojo_exact_cpu, "REPO_ROOT", temp_root):
                self.assertEqual(
                    mojo_exact_cpu._mojo_source_root(), repo_source_root
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

    def test_build_mojopkg_builds_from_repo_sources_when_bundle_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "kayak"
            source_root.mkdir(parents=True)
            (source_root / "__init__.mojo").write_text(
                "fn main():\n    pass\n", encoding="utf-8"
            )
            built_artifact = (
                temp_root
                / ".cache"
                / "python_mojo"
                / "cache-key"
                / "kayak.mojopkg"
            )

            with (
                mock.patch.object(
                    mojo_exact_cpu,
                    "ARTIFACTS_DIR",
                    temp_root / "missing-artifacts",
                ),
                mock.patch.object(
                    mojo_exact_cpu,
                    "PYTHON_MOJO_CACHE",
                    temp_root / ".cache" / "python_mojo",
                ),
                mock.patch.object(mojo_exact_cpu, "REPO_ROOT", temp_root),
                mock.patch.object(mojo_exact_cpu, "_detect_mojo_command", return_value=["mojo"]),
                mock.patch.object(
                    mojo_exact_cpu.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(["mojo"], 0, "", ""),
                ),
            ):
                self.assertEqual(
                    mojo_exact_cpu._build_mojopkg("cache-key"),
                    built_artifact,
                )

    def test_cache_key_inputs_include_bundled_metadata_when_sources_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifacts_root = temp_root / "_artifacts"
            artifacts_root.mkdir(parents=True)
            bundled_artifact = artifacts_root / "kayak.mojopkg"
            bundled_artifact.write_bytes(b"placeholder mojopkg")
            (artifacts_root / "mojopkg_build.json").write_text(
                (
                    "{\n"
                    '  "schema_version": 1,\n'
                    '  "project_version": "0.2.0",\n'
                    '  "mojo_version": "mojo 0.26.3",\n'
                    '  "artifact_filename": "kayak.mojopkg",\n'
                    '  "artifact_sha256": "abc123"\n'
                    "}\n"
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(mojo_exact_cpu, "ARTIFACTS_DIR", artifacts_root),
                mock.patch.object(mojo_exact_cpu, "REPO_ROOT", temp_root / "missing"),
            ):
                self.assertEqual(
                    mojo_exact_cpu._cache_key_inputs(),
                    (
                        [
                            mojo_exact_cpu.BINDING_SOURCE,
                            bundled_artifact,
                        ],
                        (
                            "project_version:0.2.0",
                            "mojo_version:mojo 0.26.3",
                            "artifact_sha256:abc123",
                        ),
                    ),
                )

    def test_detects_bundled_version_mismatch_only_for_bundled_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifacts_root = temp_root / "_artifacts"
            artifacts_root.mkdir(parents=True)
            bundled_artifact = artifacts_root / "kayak.mojopkg"
            bundled_artifact.write_bytes(b"placeholder mojopkg")

            error = RuntimeError(
                "failed to build Mojo extension module for mojo_exact_cpu backend\n"
                "Mojo package is incompatible with the current version of the Mojo compiler"
            )

            with mock.patch.object(mojo_exact_cpu, "ARTIFACTS_DIR", artifacts_root):
                self.assertTrue(
                    mojo_exact_cpu._is_bundled_mojopkg_version_mismatch(
                        error,
                        mojopkg_path=bundled_artifact,
                    )
                )
                self.assertFalse(
                    mojo_exact_cpu._is_bundled_mojopkg_version_mismatch(
                        RuntimeError("different failure"),
                        mojopkg_path=bundled_artifact,
                    )
                )
                self.assertFalse(
                    mojo_exact_cpu._is_bundled_mojopkg_version_mismatch(
                        error,
                        mojopkg_path=temp_root / "other.mojopkg",
                    )
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
                mock.patch.object(
                    mojo_exact_cpu,
                    "REPO_MOJO_WRAPPER",
                    temp_root / "missing-wrapper.sh",
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
                    mojo_exact_cpu,
                    "REPO_MOJO_WRAPPER",
                    temp_root / "missing-wrapper.sh",
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

    def test_detect_mojo_command_prefers_repo_wrapper_when_usable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            wrapper = temp_root / "run_mojo_with_pixi_python.sh"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                self.assertEqual(command, ["bash", str(wrapper), "--version"])
                return subprocess.CompletedProcess(command, 0, "mojo 0.0", "")

            with (
                mock.patch.dict(
                    mojo_exact_cpu.os.environ,
                    {},
                    clear=True,
                ),
                mock.patch.object(
                    mojo_exact_cpu,
                    "REPO_MOJO_WRAPPER",
                    wrapper,
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
                    ["bash", str(wrapper)],
                )

    def test_detect_mojo_command_accepts_configured_command_prefix(self) -> None:
        with mock.patch.dict(
            mojo_exact_cpu.os.environ,
            {
                "KAYAK_MOJO_CLI": "bash /tmp/run_mojo_with_pixi_python.sh",
            },
            clear=True,
        ):
            self.assertEqual(
                mojo_exact_cpu._detect_mojo_command(),
                ["bash", "/tmp/run_mojo_with_pixi_python.sh"],
            )

    def test_bundled_version_error_reports_wheel_and_mojo_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifacts_root = temp_root / "_artifacts"
            artifacts_root.mkdir(parents=True)
            (artifacts_root / "mojopkg_build.json").write_text(
                (
                    "{\n"
                    '  "schema_version": 1,\n'
                    '  "project_version": "0.2.0",\n'
                    '  "mojo_version": "mojo 0.26.3",\n'
                    '  "artifact_filename": "kayak.mojopkg",\n'
                    '  "artifact_sha256": "abc123"\n'
                    "}\n"
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(mojo_exact_cpu, "ARTIFACTS_DIR", artifacts_root),
                mock.patch.object(
                    mojo_exact_cpu,
                    "_active_mojo_version",
                    return_value="mojo 0.26.2",
                ),
            ):
                error = mojo_exact_cpu._bundled_mojopkg_version_error(
                    RuntimeError("original failure"),
                    mojopkg_path=artifacts_root / "kayak.mojopkg",
                )
                message = str(error)
                self.assertIn("Kayak 0.2.0", message)
                self.assertIn("mojo 0.26.3", message)
                self.assertIn("mojo 0.26.2", message)
                self.assertIn("runtime source rebuild is not available", message)

    def test_load_module_raises_explicit_error_for_bundled_version_mismatch(self) -> None:
        mojo_exact_cpu.load_module.cache_clear()
        bundled_artifact = Path("/tmp/bundled/kayak.mojopkg")

        try:
            with (
                mock.patch.object(mojo_exact_cpu, "_ensure_python_runtime_env"),
                mock.patch.object(
                    mojo_exact_cpu,
                    "_cache_key_inputs",
                    return_value=([Path("/tmp/source.mojo")], ()),
                ),
                mock.patch.object(
                    mojo_exact_cpu,
                    "_hash_inputs",
                    return_value="cache-key",
                ),
                mock.patch.object(
                    mojo_exact_cpu,
                    "_build_mojopkg",
                    return_value=bundled_artifact,
                ),
                mock.patch.object(
                    mojo_exact_cpu,
                    "_build_extension",
                    side_effect=RuntimeError(
                        "failed to build Mojo extension module for mojo_exact_cpu backend\n"
                        "Mojo package is incompatible with the current version of the Mojo compiler"
                    ),
                ),
                mock.patch.object(
                    mojo_exact_cpu,
                    "_is_bundled_mojopkg_version_mismatch",
                    return_value=True,
                ),
                mock.patch.object(
                    mojo_exact_cpu,
                    "_bundled_mojopkg_version_error",
                    return_value=RuntimeError("explicit mismatch error"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "explicit mismatch error"):
                    mojo_exact_cpu.load_module()
        finally:
            mojo_exact_cpu.load_module.cache_clear()


if __name__ == "__main__":
    unittest.main()
