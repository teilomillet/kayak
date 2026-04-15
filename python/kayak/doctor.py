"""Owns environment diagnostics for the public Kayak Python SDK.

This module owns:
- one stable public report for backend and adapter availability
- explicit facts about encoder/store kinds and default text backend selection
- developer-facing install guidance for optional store dependencies

This module does not own:
- import-time dependency enforcement
- backend selection policy
- search or storage behavior
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import platform
import sys

from kayak_bridge import MojoBridgeInfo, mojo_bridge_info

from .encoders import available_encoder_kinds
from .retrievers.backend_policy import default_text_retriever_backend
from .stores import available_store_kinds


@dataclass(frozen=True, slots=True)
class KayakFeatureStatus:
    """One optional Kayak feature or adapter availability check."""

    key: str
    label: str
    available: bool
    detail: str
    install_hint: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready view of this feature status."""
        return {
            "key": self.key,
            "label": self.label,
            "available": self.available,
            "detail": self.detail,
            "install_hint": self.install_hint,
        }


@dataclass(frozen=True, slots=True)
class KayakDoctorReport:
    """Stable public environment report for one Kayak Python runtime."""

    python_version: str
    python_executable: str
    default_text_backend: str
    encoder_kinds: tuple[str, ...]
    store_kinds: tuple[str, ...]
    mojo_bridge: MojoBridgeInfo
    optional_features: tuple[KayakFeatureStatus, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready view of the full doctor report."""
        return {
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "default_text_backend": self.default_text_backend,
            "encoder_kinds": list(self.encoder_kinds),
            "store_kinds": list(self.store_kinds),
            "mojo_bridge": {
                "available": self.mojo_bridge.available,
                "availability_reason": self.mojo_bridge.availability_reason,
                "command": (
                    None
                    if self.mojo_bridge.command is None
                    else list(self.mojo_bridge.command)
                ),
                "active_mojo_version": self.mojo_bridge.active_mojo_version,
                "bridge_source": self.mojo_bridge.bridge_source,
                "bundled_project_version": self.mojo_bridge.bundled_project_version,
                "bundled_mojo_version": self.mojo_bridge.bundled_mojo_version,
                "module_loaded": self.mojo_bridge.module_loaded,
                "load_error": self.mojo_bridge.load_error,
            },
            "optional_features": [
                feature.to_dict() for feature in self.optional_features
            ],
        }

    def __str__(self) -> str:
        lines = [
            "Kayak Doctor",
            "",
            f"python: {self.python_version}",
            f"executable: {self.python_executable}",
            f"default_text_backend: {self.default_text_backend}",
            f"encoder_kinds: {', '.join(self.encoder_kinds)}",
            f"store_kinds: {', '.join(self.store_kinds)}",
            "",
            "Mojo backend:",
            (
                "available"
                if self.mojo_bridge.available
                else "unavailable"
            )
            + f" - {self.mojo_bridge.availability_reason}",
        ]
        if self.mojo_bridge.command is not None:
            lines.append(f"command: {' '.join(self.mojo_bridge.command)}")
        if self.mojo_bridge.active_mojo_version is not None:
            lines.append(
                f"active_mojo_version: {self.mojo_bridge.active_mojo_version}"
            )
        lines.append(f"bridge_source: {self.mojo_bridge.bridge_source}")
        if self.mojo_bridge.module_loaded is not None:
            lines.append(f"module_loaded: {self.mojo_bridge.module_loaded}")
        if self.mojo_bridge.load_error is not None:
            lines.append(f"load_error: {self.mojo_bridge.load_error}")

        lines.extend(["", "Optional adapters:"])
        for feature in self.optional_features:
            status = "available" if feature.available else "missing"
            lines.append(f"- {feature.label}: {status} - {feature.detail}")
            if feature.install_hint is not None and not feature.available:
                lines.append(f"  install: {feature.install_hint}")

        return "\n".join(lines)


_OPTIONAL_FEATURE_SPECS = (
    (
        "lancedb_store",
        "LanceDB store",
        ("lancedb", "pyarrow"),
        "uv add lancedb pyarrow",
        "pixi add --pypi lancedb pyarrow",
    ),
    (
        "pgvector_store",
        "PgVector store",
        ("psycopg", "pgvector"),
        'uv add "psycopg[binary]" pgvector',
        'pixi add --pypi "psycopg[binary]" pgvector',
    ),
    (
        "qdrant_store",
        "Qdrant store",
        ("qdrant_client",),
        "uv add qdrant-client",
        "pixi add --pypi qdrant-client",
    ),
    (
        "weaviate_store",
        "Weaviate store",
        ("weaviate",),
        "uv add weaviate-client",
        "pixi add --pypi weaviate-client",
    ),
    (
        "chromadb_store",
        "Chroma store",
        ("chromadb",),
        "uv add chromadb",
        "pixi add --pypi chromadb",
    ),
)


def _module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def _optional_feature_statuses() -> tuple[KayakFeatureStatus, ...]:
    statuses: list[KayakFeatureStatus] = []
    for key, label, modules, uv_hint, pixi_hint in _OPTIONAL_FEATURE_SPECS:
        missing_modules = tuple(
            module_name
            for module_name in modules
            if not _module_available(module_name)
        )
        if missing_modules:
            detail = f"missing import modules: {', '.join(missing_modules)}"
            install_hint = f"`{uv_hint}` or `{pixi_hint}`"
            statuses.append(
                KayakFeatureStatus(
                    key=key,
                    label=label,
                    available=False,
                    detail=detail,
                    install_hint=install_hint,
                )
            )
            continue

        statuses.append(
            KayakFeatureStatus(
                key=key,
                label=label,
                available=True,
                detail=f"import modules available: {', '.join(modules)}",
            )
        )
    return tuple(statuses)


def doctor(*, probe_mojo_load: bool = False) -> KayakDoctorReport:
    """Return one factual environment report for Kayak development workflows.

    Use this when you want one stable public check that answers:
    - which encoder and store kinds are registered right now
    - which exact backend the high-level text retriever would default to
    - whether the Mojo bridge is available and, optionally, loadable
    - whether optional store adapter dependencies are importable

    ``probe_mojo_load=True`` goes beyond static availability checks and tries
    to load the Mojo extension so the report can include any concrete load
    failure.
    """

    return KayakDoctorReport(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        default_text_backend=default_text_retriever_backend(),
        encoder_kinds=available_encoder_kinds(),
        store_kinds=available_store_kinds(),
        mojo_bridge=mojo_bridge_info(probe_load=probe_mojo_load),
        optional_features=_optional_feature_statuses(),
    )
