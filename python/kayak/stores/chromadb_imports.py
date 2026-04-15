"""Optional-import helper for the public Chroma store adapter."""

from __future__ import annotations


def require_chromadb() -> object:
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "Chroma store support requires the optional dependency "
            "`chromadb`. Install it with `uv add chromadb` or "
            "`pixi add --pypi chromadb`."
        ) from exc

    return chromadb
