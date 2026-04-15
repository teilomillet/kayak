"""Owns bundled Mojo artifact metadata for installed Kayak wheels.

This module reads the small JSON manifest that is written next to the bundled
`kayak.mojopkg` artifact during wheel builds.

It does not own artifact building or runtime loading. Those stay in
`setup.py` and `mojo_exact_cpu.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


METADATA_FILENAME = "mojopkg_build.json"


@dataclass(frozen=True)
class BundledMojopkgMetadata:
    schema_version: int
    project_version: str
    mojo_version: str
    artifact_filename: str
    artifact_sha256: str


def metadata_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / METADATA_FILENAME


def read_bundled_mojopkg_metadata(
    artifacts_dir: Path,
) -> BundledMojopkgMetadata | None:
    path = metadata_path(artifacts_dir)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    return BundledMojopkgMetadata(
        schema_version=int(payload["schema_version"]),
        project_version=str(payload["project_version"]),
        mojo_version=str(payload["mojo_version"]),
        artifact_filename=str(payload["artifact_filename"]),
        artifact_sha256=str(payload["artifact_sha256"]),
    )
