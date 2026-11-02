"""ctypes representation of the versioned libdfkv client-open ABI."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

DFKV_CLIENT_ABI_VERSION_V2 = 2
DFKV_CLIENT_OPT_REGISTER_WITH_MDS = 1 << 0


class DfkvClientOptionsV2(ctypes.Structure):
    """Byte-for-byte mirror of ``dfkv_client_options_v2`` in dfkv_c_api.h."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("members", ctypes.c_char_p),
        ("key_namespace", ctypes.c_void_p),
        ("key_namespace_len", ctypes.c_uint64),
        ("batch_concurrency", ctypes.c_uint64),
        ("mds_endpoints", ctypes.c_char_p),
        ("mds_group", ctypes.c_char_p),
        ("mds_poll_ms", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("client_id", ctypes.c_char_p),
        ("client_info", ctypes.c_char_p),
        ("client_heartbeat_ms", ctypes.c_int32),
        ("reserved0", ctypes.c_uint32),
    ]


def make_key_buffer(key: bytes) -> tuple[ctypes.c_void_p, Any]:
    """Return an exact binary-key pointer and its caller-owned backing buffer."""
    if not isinstance(key, bytes):
        raise TypeError("dfkv object key must be bytes")
    if not key:
        raise ValueError("dfkv object key must not be empty")
    owner = ctypes.create_string_buffer(key, len(key))
    return ctypes.cast(owner, ctypes.c_void_p), owner


def make_key_array(
    keys: Sequence[bytes],
) -> tuple[Any, Any, list[Any]]:
    """Build aligned ``void*``/``uint64`` key arrays and backing owners."""
    owners: list[Any] = []
    pointers: list[ctypes.c_void_p] = []
    lengths: list[int] = []
    for key in keys:
        pointer, owner = make_key_buffer(key)
        owners.append(owner)
        pointers.append(pointer)
        lengths.append(len(key))
    key_array = (ctypes.c_void_p * len(pointers))(*pointers)
    length_array = (ctypes.c_uint64 * len(lengths))(*lengths)
    return key_array, length_array, owners


def make_client_options_v2(
    key_namespace: bytes,
    *,
    members: str = "",
    batch_concurrency: int = 0,
    mds_endpoints: str = "",
    mds_group: str = "default",
    mds_poll_ms: int = 3000,
    register_client: bool = False,
    client_id: str = "",
    client_info: str = "",
    client_heartbeat_ms: int = 10000,
) -> DfkvClientOptionsV2:
    """Build ABI-v2 options and retain every pointed-to buffer until open.

    ``libdfkv`` copies all pointed-to data synchronously in ``dfkv_open_v2``;
    attaching the buffers to the returned structure makes their lifetime
    unambiguous at the ctypes boundary.
    """

    namespace = bytes(key_namespace)
    if not namespace:
        raise ValueError("dfkv requires a non-empty key namespace")
    if register_client and (not mds_endpoints or not client_id):
        raise ValueError(
            "dfkv MDS registration requires mds_endpoints and client_id"
        )

    namespace_buffer = ctypes.create_string_buffer(namespace, len(namespace) + 1)
    members_bytes = members.encode() if members else None
    endpoints_bytes = mds_endpoints.encode() if mds_endpoints else None
    group_bytes = (mds_group or "default").encode()
    client_id_bytes = client_id.encode() if client_id else None
    client_info_bytes = client_info.encode() if client_info else None

    options = DfkvClientOptionsV2(
        struct_size=ctypes.sizeof(DfkvClientOptionsV2),
        abi_version=DFKV_CLIENT_ABI_VERSION_V2,
        members=members_bytes,
        key_namespace=ctypes.cast(namespace_buffer, ctypes.c_void_p),
        key_namespace_len=len(namespace),
        batch_concurrency=max(0, int(batch_concurrency)),
        mds_endpoints=endpoints_bytes,
        mds_group=group_bytes,
        mds_poll_ms=int(mds_poll_ms),
        flags=(
            DFKV_CLIENT_OPT_REGISTER_WITH_MDS if register_client else 0
        ),
        client_id=client_id_bytes,
        client_info=client_info_bytes,
        client_heartbeat_ms=int(client_heartbeat_ms),
        reserved0=0,
    )
    options._keepalive = (  # type: ignore[attr-defined]
        namespace_buffer,
        members_bytes,
        endpoints_bytes,
        group_bytes,
        client_id_bytes,
        client_info_bytes,
    )
    return options


# ---- Retention (kRetain): mirrors dfkv_retain_options / DFKV_RET_* ----
DFKV_RETAIN_SET = 0
DFKV_RETAIN_REFRESH = 1
DFKV_RETAIN_F_NO_DEBOUNCE = 1
DFKV_RET_MISS = 0
DFKV_RET_HIT_UNPROTECTED = 1
DFKV_RET_PROTECTED_NEW = 2
DFKV_RET_REFRESHED = 3
DFKV_RET_UPGRADED = 4
DFKV_RET_UNCHANGED = 5
DFKV_RET_DENIED = 6
DFKV_RET_UNSUPPORTED = 7
DFKV_RET_SKIPPED_DEBOUNCE = 8
DFKV_RET_ERROR = 9
# A key is present iff its REFRESH/SET code is one of these.
RETAIN_HIT_CODES = frozenset({1, 2, 3, 4, 5, 6, 8})
# The key is protected after the call.
RETAIN_PROTECTED_CODES = frozenset({2, 3, 4, 5, 8})


class DfkvRetainOptions(ctypes.Structure):
    """Byte-for-byte mirror of ``dfkv_retain_options`` in dfkv_c_api.h."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("ttl_s", ctypes.c_uint32),
        ("mode", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint16),
        ("min_refresh_ms", ctypes.c_uint32),
    ]


def make_retain_options(
    ttl_s: int, mode: int, *, flags: int = 0, min_refresh_ms: int = 0
) -> DfkvRetainOptions:
    if mode not in (DFKV_RETAIN_SET, DFKV_RETAIN_REFRESH):
        raise ValueError(f"invalid retain mode {mode}")
    return DfkvRetainOptions(
        struct_size=ctypes.sizeof(DfkvRetainOptions),
        ttl_s=max(0, min(int(ttl_s), 0xFFFFFFFF)),
        mode=mode,
        flags=flags,
        reserved0=0,
        min_refresh_ms=max(0, int(min_refresh_ms)),
    )


def bind_retain(lib: Any) -> bool:
    """Declare the retention symbols when ``lib`` exports them.

    Returns False for an older libdfkv without kRetain support, in which case
    callers keep their existence-check paths.
    """
    try:
        batch_retain = lib.dfkv_batch_retain
        limits = lib.dfkv_retain_limits
        precheck = lib.dfkv_retain_precheck
    except AttributeError:
        return False
    batch_retain.restype = ctypes.c_int
    batch_retain.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
        ctypes.POINTER(DfkvRetainOptions), ctypes.POINTER(ctypes.c_uint8),
    ]
    limits.restype = ctypes.c_int
    limits.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    precheck.restype = ctypes.c_int
    precheck.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_int),
    ]
    return True


def call_batch_retain(lib: Any, handle: Any, keys: Sequence[bytes],
                      ttl_s: int, mode: int, *, flags: int = 0,
                      min_refresh_ms: int = 0) -> list[int]:
    """Run ``dfkv_batch_retain`` and return one result code per key."""
    n = len(keys)
    if n == 0:
        return []
    key_array, length_array, owners = make_key_array(keys)
    out = (ctypes.c_uint8 * n)()
    options = make_retain_options(
        ttl_s, mode, flags=flags, min_refresh_ms=min_refresh_ms)
    rc = lib.dfkv_batch_retain(
        handle, key_array, length_array, n, ctypes.byref(options), out)
    del owners
    if rc != 0:
        raise RuntimeError(f"dfkv_batch_retain rc={rc}")
    return list(out)


def call_retain_limits(lib: Any, handle: Any) -> tuple[int, int]:
    supported = ctypes.c_int(-1)
    max_ttl = ctypes.c_uint32(0)
    rc = lib.dfkv_retain_limits(
        handle, ctypes.byref(supported), ctypes.byref(max_ttl))
    if rc != 0:
        return -1, 0
    return int(supported.value), int(max_ttl.value)


def call_retain_precheck(lib: Any, handle: Any, keys: Sequence[bytes],
                         min_headroom_pct: int = 0) -> bool:
    n = len(keys)
    if n == 0:
        return True
    key_array, length_array, owners = make_key_array(keys)
    ok = ctypes.c_int(0)
    rc = lib.dfkv_retain_precheck(
        handle, key_array, length_array, n, int(min_headroom_pct),
        ctypes.byref(ok))
    del owners
    return rc == 0 and ok.value == 1
