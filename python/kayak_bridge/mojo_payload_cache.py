"""Caches Mojo bridge payloads for repeated exact searches over one index.

This cache is intentionally object-identity scoped. `LateIndex` instances are
immutable at the Python SDK layer, and repeated-query workloads commonly reuse
the same index object many times.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from .mojo_payloads import MojoIndexPayload, index_payload


MOJO_INDEX_PAYLOAD_CACHE_SIZE = 8


@dataclass(slots=True)
class MojoIndexPayloadEntry:
    index: "LateIndex"
    payload: MojoIndexPayload


_mojo_index_payload_cache: OrderedDict[int, MojoIndexPayloadEntry] = OrderedDict()
_mojo_index_payload_cache_lock = RLock()


def clear_mojo_index_payload_cache() -> None:
    with _mojo_index_payload_cache_lock:
        _mojo_index_payload_cache.clear()


def cached_index_payload(index: "LateIndex") -> MojoIndexPayload:
    key = id(index)
    with _mojo_index_payload_cache_lock:
        cached = _mojo_index_payload_cache.get(key)
        if cached is not None and cached.index is index:
            _mojo_index_payload_cache.move_to_end(key)
            return cached.payload

    payload = index_payload(index)

    with _mojo_index_payload_cache_lock:
        _mojo_index_payload_cache[key] = MojoIndexPayloadEntry(
            index=index,
            payload=payload,
        )
        _mojo_index_payload_cache.move_to_end(key)

        while len(_mojo_index_payload_cache) > MOJO_INDEX_PAYLOAD_CACHE_SIZE:
            _mojo_index_payload_cache.popitem(last=False)

    return payload
