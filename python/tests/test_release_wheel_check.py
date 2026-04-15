from __future__ import annotations

import json
from pathlib import Path
import tempfile
import tomllib
import unittest
import zipfile

from kayak_bridge.release_wheel_check import inspect_wheel


PROJECT_VERSION = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
)["project"]["version"]


class ReleaseWheelCheckTests(unittest.TestCase):
    def test_inspect_wheel_accepts_source_free_release_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel_path = (
                Path(temp_dir) / f"kayak-{PROJECT_VERSION}-py3-none-any.whl"
            )
            with zipfile.ZipFile(wheel_path, "w") as wheel:
                wheel.writestr(
                    "kayak_bridge/_artifacts/kayak.mojopkg",
                    b"placeholder",
                )
                wheel.writestr(
                    "kayak_bridge/_artifacts/mojopkg_build.json",
                    json.dumps(
                        {
                            "schema_version": 1,
                            "project_version": PROJECT_VERSION,
                            "mojo_version": "mojo 0.26.3",
                            "artifact_filename": "kayak.mojopkg",
                            "artifact_sha256": "abc123",
                        }
                    ),
                )

            summary = inspect_wheel(wheel_path)
            self.assertEqual(summary["wheel_version"], PROJECT_VERSION)
            self.assertEqual(summary["project_version"], PROJECT_VERSION)

    def test_inspect_wheel_rejects_engine_source_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel_path = (
                Path(temp_dir) / f"kayak-{PROJECT_VERSION}-py3-none-any.whl"
            )
            with zipfile.ZipFile(wheel_path, "w") as wheel:
                wheel.writestr(
                    "kayak_bridge/_artifacts/kayak.mojopkg",
                    b"placeholder",
                )
                wheel.writestr(
                    "kayak_bridge/_artifacts/mojopkg_build.json",
                    json.dumps(
                        {
                            "schema_version": 1,
                            "project_version": PROJECT_VERSION,
                            "mojo_version": "mojo 0.26.3",
                            "artifact_filename": "kayak.mojopkg",
                            "artifact_sha256": "abc123",
                        }
                    ),
                )
                wheel.writestr("kayak_bridge/_engine/kayak/__init__.mojo", "")

            with self.assertRaisesRegex(
                AssertionError, "forbidden implementation files"
            ):
                inspect_wheel(wheel_path)


if __name__ == "__main__":
    unittest.main()
