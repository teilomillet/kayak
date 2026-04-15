"""Optional-import helper for the public Qdrant store adapter."""

from __future__ import annotations


def require_qdrant() -> tuple[object, object]:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models
    except ImportError as exc:
        raise ImportError(
            "Qdrant store support requires the optional dependency "
            "`qdrant-client`. Install it with `uv add qdrant-client` or "
            "`pixi add --pypi qdrant-client`."
        ) from exc

    return QdrantClient, models
