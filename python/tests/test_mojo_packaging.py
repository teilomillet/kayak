from __future__ import annotations

from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
