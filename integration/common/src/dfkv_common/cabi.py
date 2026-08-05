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
