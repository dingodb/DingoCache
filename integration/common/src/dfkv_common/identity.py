"""Canonical dfkv namespace/object codec shared by Python connectors.

The bytes are the native protocol contract, not a display key. Keep this file
byte-for-byte identical when vendoring it into connector wheels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import struct
from typing import Union


_MAX_FIELD_BYTES = 1 << 20
_MAX_COMPONENT_BYTES = 256
_NAMESPACE_MAGIC = b"DFKVNS\x00\x02"
_OVERRIDE_NAMESPACE_MAGIC = b"DFKVNS\x00\xff"
_OBJECT_MAGIC = b"DFKVOBJ2"
_LAYOUT_MAGIC = b"DFKVLAYOUT1"
_POOL_MAGIC = b"DFKVPOOL\x02"
_SG_MAGIC = b"DFKVSG\x02"
_TENANT_HASH_MAGIC = b"DFKVTENANT1"

SGLANG_HICACHE_RAW_V1 = "sglang-hicache/raw-v1"
VLLM_RAW_V1 = "vllm/raw-v1"
LMCACHE_RAW_V1 = "lmcache/raw-v1"
BytesLike = Union[str, bytes]


class PoolKind(IntEnum):
    ATTENTION = 1
    MLA = 2
    MAMBA = 3
    SWA = 4
    INDEXER = 5
    DRAFT_ATTENTION = 6
    DRAFT_MLA = 7
    DRAFT_INDEXER = 8
    CUSTOM = 65535


@dataclass(frozen=True)
class RankCoordinate:
    world_size: int = 1
    rank: int = -1

    def validate(self, axis: str) -> None:
        if not 1 <= self.world_size <= 0xFFFFFFFF:
            raise ValueError(f"{axis} world_size must be within uint32 and positive")
        if self.rank < -1 or self.rank >= self.world_size:
            raise ValueError(f"{axis} rank must be -1 or within world_size")

    def serialize(self) -> bytes:
        return struct.pack("<II", self.world_size, self.rank & 0xFFFFFFFF)


@dataclass(frozen=True)
class NamespaceDescriptor:
    tenant_id: str
    model_id: str
    model_revision: str
    pool_kind: PoolKind
    pool_name: str
    dtype: str
    layout_fingerprint: int
    block_tokens: int
    group_id: int
    layer_begin: int
    layer_count: int
    tp: RankCoordinate = RankCoordinate()
    dp: RankCoordinate = RankCoordinate()
    pp: RankCoordinate = RankCoordinate()

    def validate(self) -> None:
        for name in ("tenant_id", "model_id", "model_revision", "pool_name", "dtype"):
            _required_bytes(getattr(self, name), name)
        try:
            PoolKind(self.pool_kind)
        except ValueError as exc:
            raise ValueError("pool_kind is not a defined PoolKind") from exc
        _uint(self.layout_fingerprint, 64, "layout_fingerprint", nonzero=True)
        _uint(self.block_tokens, 32, "block_tokens", nonzero=True)
        _uint(self.group_id, 32, "group_id")
        _uint(self.layer_begin, 32, "layer_begin")
        _uint(self.layer_count, 32, "layer_count", nonzero=True)
        if self.layer_begin + self.layer_count > 0xFFFFFFFF:
            raise ValueError("layer range must not overflow uint32")
        self.tp.validate("tp")
        self.dp.validate("dp")
        self.pp.validate("pp")

    def serialize(self) -> bytes:
        self.validate()
        out = bytearray(_NAMESPACE_MAGIC)
        out += _field(self.tenant_id)
        out += _field(self.model_id)
        out += _field(self.model_revision)
        out += struct.pack("<H", int(self.pool_kind))
        out += _field(self.pool_name)
        out += _field(self.dtype)
        out += struct.pack(
            "<QIIII",
            self.layout_fingerprint,
            self.block_tokens,
            self.group_id,
            self.layer_begin,
            self.layer_count,
        )
        out += self.tp.serialize()
        out += self.dp.serialize()
        out += self.pp.serialize()
        return bytes(out)


@dataclass(frozen=True)
class ObjectDescriptor:
    logical_key: BytesLike
    component: str
    chunk_index: int = 0
    chunk_count: int = 1

    def validate(self) -> None:
        _required_bytes(self.logical_key, "logical_key")
        component = _required_bytes(self.component, "component")
        if len(component) > _MAX_COMPONENT_BYTES:
            raise ValueError("component exceeds 256 bytes")
        _uint(self.chunk_index, 32, "chunk_index")
        _uint(self.chunk_count, 32, "chunk_count", nonzero=True)
        if self.chunk_index >= self.chunk_count:
            raise ValueError("chunk_index must be within chunk_count")

    def serialize(self, namespace: NamespaceDescriptor) -> bytes:
        self.validate()
        namespace_bytes = namespace.serialize()
        return b"".join(
            (
                _OBJECT_MAGIC,
                _field(namespace_bytes),
                _field(self.logical_key),
                _field(self.component),
                struct.pack("<II", self.chunk_index, self.chunk_count),
            )
        )


def _bytes(value: BytesLike, name: str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytes):
        return value
    raise TypeError(f"{name} must be str or bytes")


def _required_bytes(value: BytesLike, name: str) -> bytes:
    encoded = _bytes(value, name)
    if not encoded:
        raise ValueError(f"{name} must not be empty")
    if len(encoded) > _MAX_FIELD_BYTES:
        raise ValueError(f"{name} exceeds 1 MiB")
    return encoded


def _field(value: BytesLike) -> bytes:
    encoded = _bytes(value, "field")
    if len(encoded) > 0xFFFFFFFF:
        raise ValueError("field exceeds uint32 length")
    return struct.pack("<I", len(encoded)) + encoded


def tenant_hash(tenant_identity: BytesLike) -> int:
    """Return the stable 64-bit tenant hash used by native BlockKey.

    SHA-256 input is ``DFKVTENANT1 || uint64le(len(identity)) || identity``;
    the first eight digest bytes are interpreted big-endian.
    """
    identity = _bytes(tenant_identity, "tenant identity")
    framed = _TENANT_HASH_MAGIC + struct.pack("<Q", len(identity)) + identity
    return int.from_bytes(hashlib.sha256(framed).digest()[:8], "big")


def _canonical_namespace_tenant(namespace: bytes) -> bytes | None:
    if not namespace.startswith(_NAMESPACE_MAGIC):
        return None
    pos = len(_NAMESPACE_MAGIC)

    def field() -> bytes | None:
        nonlocal pos
        if len(namespace) - pos < 4:
            return None
        size = struct.unpack_from("<I", namespace, pos)[0]
        pos += 4
        if size == 0 or size > _MAX_FIELD_BYTES or size > len(namespace) - pos:
            return None
        value = namespace[pos:pos + size]
        pos += size
        return value

    first = field()
    if first is None or field() is None or field() is None:
        return None
    if len(namespace) - pos < 2:
        return None
    pool_kind = struct.unpack_from("<H", namespace, pos)[0]
    pos += 2
    if not (1 <= pool_kind <= 8 or pool_kind == 0xFFFF):
        return None
    if field() is None or field() is None:
        return None
    fixed_tail = 8 + 4 * 4 + 3 * 8
    if len(namespace) - pos != fixed_tail:
        return None
    (
        layout_fingerprint,
        block_tokens,
        _group_id,
        layer_begin,
        layer_count,
        tp_world,
        tp_rank,
        dp_world,
        dp_rank,
        pp_world,
        pp_rank,
    ) = struct.unpack_from("<Q10I", namespace, pos)
    if (
        layout_fingerprint == 0
        or block_tokens == 0
        or layer_count == 0
        or layer_begin + layer_count > 0xFFFFFFFF
    ):
        return None
    for world_size, rank in (
        (tp_world, tp_rank),
        (dp_world, dp_rank),
        (pp_world, pp_rank),
    ):
        if world_size == 0 or (rank != 0xFFFFFFFF and rank >= world_size):
            return None
    return first


def namespace_tenant_hash(namespace: bytes) -> int:
    """Hash the canonical tenant field, falling back to all namespace bytes."""
    if not isinstance(namespace, bytes):
        raise TypeError("namespace must be bytes")
    tenant = _canonical_namespace_tenant(namespace)
    return tenant_hash(namespace if tenant is None else tenant)



def _uint(value: int, bits: int, name: str, *, nonzero: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    minimum = 1 if nonzero else 0
    if not minimum <= value < (1 << bits):
        qualifier = "positive " if nonzero else ""
        raise ValueError(f"{name} must be a {qualifier}uint{bits}")



def _layout_value(value: object) -> bytes:
    if isinstance(value, bool):
        return b"bool:" + (b"1" if value else b"0")
    if isinstance(value, int):
        return b"int:" + str(value).encode("ascii")
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"bytes:" + value
    if value is None:
        return b"none:"
    raise TypeError(
        "layout field values must be bool, int, str, bytes, or None")


def layout_fingerprint(
    layout_id: str,
    fields: Mapping[str, object] | None = None,
) -> int:
    """Return the deterministic 64-bit fingerprint stored in a namespace.

    Field names are sorted and both names and typed values are length-framed.
    Python's randomized ``hash()`` and display formatting never enter the
    storage identity.
    """
    _required_bytes(layout_id, "layout_id")
    encoded = bytearray(_LAYOUT_MAGIC)
    encoded += _field(layout_id)
    for name, value in sorted((fields or {}).items()):
        _required_bytes(name, "layout field name")
        encoded += _field(name)
        encoded += _field(_layout_value(value))
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    return value or 1


def canonical_namespace(
    model_identity: str,
    layout_id: str,
    override: str | None = None,
    *,
    tenant_id: str = "default",
    model_revision: str | None = None,
    dtype: str = "opaque",
    block_tokens: int = 1,
    group_id: int = 0,
    layer_begin: int = 0,
    layer_count: int = 1,
    tp_size: int = 1,
    dp_size: int = 1,
    pp_size: int = 1,
    layout_fields: Mapping[str, object] | None = None,
) -> bytes:
    """Build the binary namespace passed to ``dfkv_open_v2``.

    The automatic form is the cross-language ``NamespaceDescriptor`` contract:
    tenant, exact model/revision, runtime raw-layout contract, dtype, token and
    layer geometry, topology sizes, and a deterministic fingerprint of any
    connector-specific layout fields. Rank stays replicated here because the
    per-object pool key carries rank scope.

    An explicit override is deliberately a disjoint namespace form. It is the
    operator's assertion that all producers and consumers using that exact
    override store byte-compatible payloads.
    """
    if override is not None:
        return _OVERRIDE_NAMESPACE_MAGIC + _field(
            _required_bytes(override, "key_namespace override"))
    descriptor = NamespaceDescriptor(
        tenant_id=tenant_id,
        model_id=model_identity,
        model_revision=(
            model_identity if model_revision is None else model_revision),
        pool_kind=PoolKind.CUSTOM,
        pool_name=layout_id,
        dtype=dtype,
        layout_fingerprint=layout_fingerprint(layout_id, layout_fields),
        block_tokens=int(block_tokens),
        group_id=int(group_id),
        layer_begin=int(layer_begin),
        layer_count=int(layer_count),
        tp=RankCoordinate(int(tp_size), -1),
        dp=RankCoordinate(int(dp_size), -1),
        pp=RankCoordinate(int(pp_size), -1),
    )
    return descriptor.serialize()


def pool_key(
    page_hash: BytesLike,
    *,
    pool: BytesLike = "kv",
    dp_size: int = 1,
    dp_rank: int = -1,
    tp_size: int = 1,
    tp_rank: int = -1,
    pcp_size: int = 1,
    pcp_rank: int = 0,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    pp_size: int = 1,
    pp_rank: int = 0,
    group_id: int = 0,
    component: BytesLike = "all",
) -> bytes:
    """Encode one logical cache-object key as self-delimiting bytes."""
    pool_bytes = _required_bytes(pool, "pool")
    page_hash_bytes = _required_bytes(page_hash, "page_hash")
    component_bytes = _required_bytes(component, "component")
    if len(component_bytes) > _MAX_COMPONENT_BYTES:
        raise ValueError("component exceeds 256 bytes")
    axes = (
        ("dp", dp_size, dp_rank),
        ("tp", tp_size, tp_rank),
        ("pcp", pcp_size, pcp_rank),
        ("dcp", dcp_size, dcp_rank),
        ("pp", pp_size, pp_rank),
    )
    encoded_axes = bytearray()
    for name, size, rank in axes:
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError(f"{name} size must be int")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError(f"{name} rank must be int")
        if (
            size <= 0 or size > 0xFFFFFFFF
            or rank < -1 or rank > 0x7FFFFFFF or rank >= size
        ):
            raise ValueError(
                f"{name} coordinate must satisfy uint32 size > 0 and "
                "-1 <= int32 rank < size")
        encoded_axes += struct.pack("<Ii", size, rank)
    _uint(group_id, 32, "group_id")
    return b"".join((
        _POOL_MAGIC,
        _field(pool_bytes),
        _field(page_hash_bytes),
        bytes(encoded_axes),
        struct.pack("<I", group_id),
        _field(component_bytes),
    ))


def sg_key(key: bytes, width: int, group: int) -> bytes:
    """Append self-delimiting scatter geometry to a binary logical key."""
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if not key:
        raise ValueError("key must not be empty")
    _uint(width, 32, "scatter width", nonzero=True)
    _uint(group, 32, "scatter group")
    return key + _SG_MAGIC + struct.pack("<II", width, group)