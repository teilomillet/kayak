"""Optional-import helper for the public pgvector store adapter."""

from __future__ import annotations


def require_pgvector() -> tuple[object, object]:
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise ImportError(
            "PgVector store support requires the optional dependencies "
            "`pgvector` and `psycopg`. The simplest install path is "
            "`uv add \"psycopg[binary]\" pgvector` or "
            "`pixi add --pypi \"psycopg[binary]\" pgvector`."
        ) from exc

    return psycopg, register_vector
