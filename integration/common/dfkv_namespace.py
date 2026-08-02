"""Canonical dfkv namespace/object codec shared by Python connectors.

The bytes are the native protocol contract, not a display key. Keep this file
byte-for-byte identical when vendoring it into connector wheels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import Union


_MAX_FIELD_BYTES = 1 << 20
_MAX_COMPONENT_BYTES = 256
_NAMESPACE_MAGIC = b"DFKVNS\x00\x02"
_OBJECT_MAGIC = b"DFKVOBJ2"
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


def _uint(value: int, bits: int, name: str, *, nonzero: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    minimum = 1 if nonzero else 0
    if not minimum <= value < (1 << bits):
        qualifier = "positive " if nonzero else ""
        raise ValueError(f"{name} must be a {qualifier}uint{bits}")
