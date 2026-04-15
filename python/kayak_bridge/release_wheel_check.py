"""Owns release-wheel layout validation for bundled Mojo artifacts.

This module checks that built wheels contain the packaged `kayak.mojopkg`
artifact and metadata, while excluding engine source payloads.

It does not build wheels or upload releases.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import zipfile


FORBIDDEN_PREFIXES = (
    "kayak_engine/",
    "kayak_bridge/_engine/",
)
REQUIRED_ENTRIES = (
    "kayak_bridge/_artifacts/kayak.mojopkg",
    "kayak_bridge/_artifacts/mojopkg_build.json",
)
WHEEL_VERSION_PATTERN = re.compile(r"^kayak-(?P<version>.+)-py3-none-any\.whl$")


def wheel_version(path: Path) -> str:
    match = WHEEL_VERSION_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"unexpected wheel filename: {path.name}")
    return match.group("version")


def inspect_wheel(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())

        missing = [name for name in REQUIRED_ENTRIES if name not in names]
        if missing:
            raise AssertionError(f"wheel is missing required entries: {missing}")

        forbidden = sorted(
            name
            for name in names
            if any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
        )
        if forbidden:
            raise AssertionError(
                "wheel contains forbidden implementation files: "
                + ", ".join(forbidden[:10])
            )

        metadata = json.loads(
            wheel.read("kayak_bridge/_artifacts/mojopkg_build.json").decode(
                "utf-8"
            )
        )
        if int(metadata["schema_version"]) != 1:
            raise AssertionError("unexpected mojopkg metadata schema version")

        bundled_version = str(metadata["project_version"])
        artifact_version = wheel_version(path)
        if bundled_version != artifact_version:
            raise AssertionError(
                "bundled mojopkg metadata version does not match wheel version: "
                f"{bundled_version} != {artifact_version}"
            )

        artifact_name = str(metadata["artifact_filename"])
        if artifact_name != "kayak.mojopkg":
            raise AssertionError(
                "bundled mojopkg metadata points at an unexpected artifact: "
                f"{artifact_name}"
            )

        return {
            "wheel": str(path),
            "wheel_version": artifact_version,
            "project_version": bundled_version,
            "mojo_version": str(metadata["mojo_version"]),
            "artifact_sha256": str(metadata["artifact_sha256"]),
        }
