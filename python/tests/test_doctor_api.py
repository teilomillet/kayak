from __future__ import annotations

import importlib
import unittest
from unittest import mock

import kayak

doctor_module = importlib.import_module("kayak.doctor")


class DoctorApiTests(unittest.TestCase):
    def test_doctor_returns_public_environment_report(self) -> None:
        optional_features = (
            kayak.KayakFeatureStatus(
                key="lancedb_store",
                label="LanceDB store",
                available=True,
                detail="import modules available: lancedb, pyarrow",
            ),
            kayak.KayakFeatureStatus(
                key="pgvector_store",
                label="PgVector store",
                available=False,
                detail="missing import modules: pgvector",
                install_hint='`uv add "psycopg[binary]" pgvector` or `pixi add --pypi "psycopg[binary]" pgvector`',
            ),
        )

        with (
            mock.patch.object(
                doctor_module,
                "available_encoder_kinds",
                return_value=("callable", "colbert"),
            ),
            mock.patch.object(
                doctor_module,
                "available_store_kinds",
                return_value=("memory", "pgvector"),
            ),
            mock.patch.object(
                doctor_module,
                "default_text_retriever_backend",
                return_value=kayak.MOJO_EXACT_CPU_BACKEND,
            ),
            mock.patch.object(
                doctor_module,
                "mojo_bridge_info",
                return_value=kayak.MojoBridgeInfo(
                    available=True,
                    availability_reason="Kayak can invoke Mojo via: mojo",
                    command=("mojo",),
                    active_mojo_version="Mojo 0.26.3",
                    bridge_source="bundled_artifact",
                    bundled_project_version="0.2.1",
                    bundled_mojo_version="Mojo 0.26.3",
                    module_loaded=True,
                    load_error=None,
                ),
            ),
            mock.patch.object(
                doctor_module,
                "_optional_feature_statuses",
                return_value=optional_features,
            ),
        ):
            report = kayak.doctor(probe_mojo_load=True)

        self.assertEqual(report.default_text_backend, kayak.MOJO_EXACT_CPU_BACKEND)
        self.assertEqual(report.encoder_kinds, ("callable", "colbert"))
        self.assertEqual(report.store_kinds, ("memory", "pgvector"))
        self.assertTrue(report.mojo_bridge.available)
        self.assertEqual(report.optional_features, optional_features)
        self.assertEqual(report.to_dict()["default_text_backend"], kayak.MOJO_EXACT_CPU_BACKEND)

    def test_optional_feature_statuses_report_missing_modules_with_install_hints(
        self,
    ) -> None:
        availability = {
            "lancedb": True,
            "pyarrow": True,
            "psycopg": True,
            "pgvector": False,
            "qdrant_client": True,
            "weaviate": False,
            "chromadb": True,
        }

        with mock.patch.object(
            doctor_module,
            "_module_available",
            side_effect=lambda name: availability.get(name, False),
        ):
            statuses = doctor_module._optional_feature_statuses()

        statuses_by_key = {status.key: status for status in statuses}
        self.assertTrue(statuses_by_key["lancedb_store"].available)
        self.assertFalse(statuses_by_key["pgvector_store"].available)
        self.assertIn("pgvector", statuses_by_key["pgvector_store"].detail)
        self.assertIn("uv add", statuses_by_key["pgvector_store"].install_hint)
        self.assertFalse(statuses_by_key["weaviate_store"].available)
        self.assertIn("weaviate", statuses_by_key["weaviate_store"].detail)

    def test_doctor_report_string_is_repl_friendly(self) -> None:
        report = kayak.KayakDoctorReport(
            python_version="3.11.11",
            python_executable="/tmp/python",
            default_text_backend=kayak.NUMPY_REFERENCE_BACKEND,
            encoder_kinds=("callable", "colbert"),
            store_kinds=("memory", "chromadb"),
            mojo_bridge=kayak.MojoBridgeInfo(
                available=False,
                availability_reason="Kayak could not find a usable Mojo CLI.",
                command=None,
                active_mojo_version=None,
                bridge_source="bundled_artifact",
                bundled_project_version="0.2.1",
                bundled_mojo_version="Mojo 0.26.3",
                module_loaded=False,
                load_error="Kayak could not find a usable Mojo CLI.",
            ),
            optional_features=(
                kayak.KayakFeatureStatus(
                    key="chromadb_store",
                    label="Chroma store",
                    available=False,
                    detail="missing import modules: chromadb",
                    install_hint="`uv add chromadb` or `pixi add --pypi chromadb`",
                ),
            ),
        )

        text = str(report)

        self.assertIn("Kayak Doctor", text)
        self.assertIn("default_text_backend: numpy_reference", text)
        self.assertIn("Mojo backend:", text)
        self.assertIn("Optional adapters:", text)
        self.assertIn("Chroma store: missing", text)
        self.assertIn("uv add chromadb", text)


if __name__ == "__main__":
    unittest.main()
