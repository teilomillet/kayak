from __future__ import annotations

import importlib
import unittest
from unittest import mock

import kayak
from kayak_bridge.backend_info import BackendInfo
from kayak_bridge.bundled_mojopkg_metadata import BundledMojopkgMetadata

bridge_info_module = importlib.import_module("kayak_bridge.mojo_bridge_info")


class MojoBridgeInfoTests(unittest.TestCase):
    def test_bridge_info_reports_public_connectivity_shape(self) -> None:
        metadata = BundledMojopkgMetadata(
            schema_version=1,
            project_version="0.2.1",
            mojo_version="mojo 0.26.3",
            artifact_filename="kayak.mojopkg",
            artifact_sha256="abc123",
        )

        with (
            mock.patch.object(
                bridge_info_module,
                "backend_info",
                return_value=BackendInfo(
                    name=kayak.MOJO_EXACT_CPU_BACKEND,
                    available=True,
                    requires_mojo=True,
                    query_layouts=("nested", "flat_dim128"),
                    index_layouts=("packed", "hybrid_flat_dim128"),
                    availability_reason="Kayak can invoke Mojo via: mojo",
                ),
            ),
            mock.patch.object(
                bridge_info_module,
                "_command_or_none",
                return_value=("mojo",),
            ),
            mock.patch.object(
                bridge_info_module,
                "_active_mojo_version",
                return_value="mojo 0.26.4",
            ),
            mock.patch.object(
                bridge_info_module,
                "_bridge_source_kind",
                return_value="bundled_artifact",
            ),
            mock.patch.object(
                bridge_info_module,
                "_metadata_or_none",
                return_value=metadata,
            ),
        ):
            info = kayak.mojo_bridge_info()

        self.assertTrue(info.available)
        self.assertEqual(info.command, ("mojo",))
        self.assertEqual(info.active_mojo_version, "mojo 0.26.4")
        self.assertEqual(info.bridge_source, "bundled_artifact")
        self.assertEqual(info.bundled_project_version, "0.2.1")
        self.assertEqual(info.bundled_mojo_version, "mojo 0.26.3")
        self.assertIsNone(info.module_loaded)
        self.assertIsNone(info.load_error)

    def test_bridge_info_probe_reports_load_failure_without_raising(self) -> None:
        with (
            mock.patch.object(
                bridge_info_module,
                "backend_info",
                return_value=BackendInfo(
                    name=kayak.MOJO_EXACT_CPU_BACKEND,
                    available=True,
                    requires_mojo=True,
                    query_layouts=("nested", "flat_dim128"),
                    index_layouts=("packed", "hybrid_flat_dim128"),
                    availability_reason="Kayak can invoke Mojo via: mojo",
                ),
            ),
            mock.patch.object(
                bridge_info_module,
                "_command_or_none",
                return_value=("mojo",),
            ),
            mock.patch.object(
                bridge_info_module,
                "_active_mojo_version",
                return_value="mojo 0.26.4",
            ),
            mock.patch.object(
                bridge_info_module,
                "_bridge_source_kind",
                return_value="bundled_artifact",
            ),
            mock.patch.object(
                bridge_info_module,
                "_metadata_or_none",
                return_value=None,
            ),
            mock.patch.object(
                bridge_info_module,
                "load_module",
                side_effect=RuntimeError("extension build failed"),
            ),
        ):
            info = kayak.mojo_bridge_info(probe_load=True)

        self.assertFalse(info.module_loaded)
        self.assertEqual(info.load_error, "extension build failed")

    def test_bridge_info_probe_reports_backend_unavailable(self) -> None:
        with (
            mock.patch.object(
                bridge_info_module,
                "backend_info",
                return_value=BackendInfo(
                    name=kayak.MOJO_EXACT_CPU_BACKEND,
                    available=False,
                    requires_mojo=True,
                    query_layouts=("nested", "flat_dim128"),
                    index_layouts=("packed", "hybrid_flat_dim128"),
                    availability_reason="Kayak could not find a usable Mojo CLI.",
                ),
            ),
            mock.patch.object(
                bridge_info_module,
                "_command_or_none",
                return_value=None,
            ),
            mock.patch.object(
                bridge_info_module,
                "_active_mojo_version",
                return_value=None,
            ),
            mock.patch.object(
                bridge_info_module,
                "_bridge_source_kind",
                return_value="bundled_artifact",
            ),
            mock.patch.object(
                bridge_info_module,
                "_metadata_or_none",
                return_value=None,
            ),
        ):
            info = kayak.mojo_bridge_info(probe_load=True)

        self.assertFalse(info.available)
        self.assertFalse(info.module_loaded)
        self.assertEqual(
            info.load_error,
            "Kayak could not find a usable Mojo CLI.",
        )


if __name__ == "__main__":
    unittest.main()
