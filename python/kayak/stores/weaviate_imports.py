"""Optional-import helper for the public Weaviate store adapter."""

from __future__ import annotations


def require_weaviate() -> tuple[object, object]:
    try:
        import weaviate
        import weaviate.classes as wvc
    except ImportError as exc:
        raise ImportError(
            "Weaviate store support requires the optional dependency "
            "`weaviate-client`. Install it with `uv add weaviate-client` or "
            "`pixi add --pypi weaviate-client`."
        ) from exc

    return weaviate, wvc
