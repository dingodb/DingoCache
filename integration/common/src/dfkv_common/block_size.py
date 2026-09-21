"""Startup validation of logical object sizes against an RDMA client bound."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from numbers import Integral
from typing import Any


def get_max_block_bytes(lib: Any, handle: Any) -> int:
    """Read the effective bound from an opened RDMA client, without fallback.

    Callers must only invoke this for RDMA: unsupported transports return zero.
    The native client owns configuration parsing and payload-cap clamping.
    """
    try:
        getter = lib.dfkv_max_block_bytes
    except AttributeError as exc:
        raise RuntimeError(
            "RDMA block-size preflight requires libdfkv exporting "
            "dfkv_max_block_bytes; upgrade the loaded native library to match "
            "the connector. The effective bound cannot be inferred from "
            "DFKV_RDMA_MAX_BLOCK_BYTES alone."
        ) from exc
    getter.restype = ctypes.c_uint64
    getter.argtypes = [ctypes.c_void_p]
    limit = int(getter(handle))
    if limit == 0:
        raise RuntimeError(
            "dfkv_max_block_bytes returned an effective bound of 0 bytes; "
            "RDMA block-size preflight requires a valid, opened RDMA client. "
            "Check native client initialization and the loaded libdfkv; "
            "DFKV_RDMA_MAX_BLOCK_BYTES alone cannot establish the effective bound."
        )
    return limit


def validate_object_sizes(
    limit: int, sizes: Iterable[int], *, context: str
) -> None:
    """Require nonempty geometry whose largest logical object fits ``limit``.

    Each entry is one object's total bytes: sum scatter/gather segments before
    calling, but keep independently stored pool components as separate entries.
    This checks actual geometry; it never changes client or server limits.
    """
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0:
        raise ValueError(
            f"{context}: effective block bound must be a positive integer "
            f"number of bytes, got {limit!r}"
        )
    required = 0
    for index, size in enumerate(sizes):
        if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
            raise ValueError(
                f"{context}: object size at index {index} must be a positive "
                f"integer number of bytes, got {size!r}"
            )
        required = max(required, int(size))
    if required == 0:
        raise ValueError(
            f"{context}: object geometry is empty; cannot determine required bytes"
        )
    if required > limit:
        raise ValueError(
            f"{context}: required object size {required} bytes exceeds the "
            f"effective RDMA block bound {limit} bytes. Reduce the object "
            "geometry or explicitly configure DFKV_RDMA_MAX_BLOCK_BYTES before "
            "opening the client. The native payload cap and server limits must "
            "also support the desired bound; increasing this setting alone "
            "may not increase the effective limit."
        )
