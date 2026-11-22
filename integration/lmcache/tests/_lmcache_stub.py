# SPDX-License-Identifier: Apache-2.0
"""A stand-in for the parts of LMCache the dfkv connector imports.

The connector's unit tests exercise connector logic with fake clients; they
need LMCache only for base classes and a few value types. Where LMCache is
installed it is used as it is. Where it is not (the C++ build's ctest run has
no LMCache), ``install()`` puts this stand-in into ``sys.modules`` so the same
tests run there.

The L2 adapter interface is reproduced with the signatures of the LMCache
releases the connector supports, because they differ in the one place the
adapter has to follow:

  ``submit_lookup_and_lock_task(self, keys)``                      0.4.7
  ``submit_lookup_and_lock_task(self, keys, layout_desc)``         0.5.0-0.5.2
  ``submit_lookup_and_lock_task(self, keys, group_layout_descs)``  0.5.3-0.5.6

(LMCache's controllers pass the second argument positionally), and in where
``Bitmap`` is imported from (``lmcache.native_storage_ops`` up to 0.5.3,
``lmcache.lmcache_native`` from 0.5.4). ``install(api=...)`` selects one of
``"0.4.7"``, ``"0.5.0"``, ``"0.5.3"``, ``"0.5.4"``; the environment variable
``DFKV_TEST_LMCACHE_API`` is the default. The other abstract methods, the
constructor, ``L2StoreResult`` and the base class's byte accounting
(``_notify_keys_*``, ``get_usage``) are the same in all of them; they are
reproduced from ``l2_adapters/base.py`` and ``internal_api.py`` of 0.4.7,
0.5.0, 0.5.2, 0.5.4 and 0.5.6. 0.5.7 adds an abstract ``get_memory_tags``
the adapter does not implement and moves the in-process storage backend to
``lmcache.inprocess.v1.storage_backend``, which is why the dependency stops
before it.
"""

from __future__ import annotations

import abc
import importlib.util
import logging
import os
import sys
import threading
import types
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

APIS = ("0.4.7", "0.5.0", "0.5.3", "0.5.4")


def real_lmcache_available() -> bool:
    try:
        return importlib.util.find_spec("lmcache.v1") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, module)
    return module


class Bitmap:
    def __init__(self, n: int) -> None:
        self._bits = [False] * n

    def set(self, i: int) -> None:
        self._bits[i] = True

    def test(self, i: int) -> bool:
        return self._bits[i]

    def popcount(self) -> int:
        return sum(self._bits)

    def __len__(self) -> int:
        return len(self._bits)


class L2StoreResult(int):
    def __new__(cls, success: bool, bytes_transferred: int) -> "L2StoreResult":
        if bytes_transferred < 0:
            raise ValueError("bytes_transferred must be >= 0")
        return super().__new__(cls, bytes_transferred if success else -1)

    def is_successful(self) -> bool:
        return int(self) >= 0

    def bytes_transferred(self) -> int:
        value = int(self)
        return value if value >= 0 else 0


@dataclass(frozen=True)
class AdapterUsage:
    total_bytes_used: int
    total_capacity_bytes: int
    bytes_by_cache_salt: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def usage_fraction(self) -> float:
        if self.total_capacity_bytes <= 0:
            return -1.0
        return self.total_bytes_used / self.total_capacity_bytes


@dataclass(frozen=True)
class ObjectKey:
    chunk_hash: bytes
    model_name: str
    kv_rank: int
    object_group_id: int = 0
    cache_salt: str = ""

    @staticmethod
    def ComputeKVRank(world_size, global_rank, local_world_size, local_rank):
        return (
            (world_size << 24)
            | (global_rank << 16)
            | (local_world_size << 8)
            | local_rank
        )


class _EventNotifier:
    def __init__(self) -> None:
        self._r, self._w = os.pipe()
        os.set_blocking(self._r, False)

    def fileno(self) -> int:
        return self._r

    def notify(self) -> None:
        os.write(self._w, b"\x01")

    def consume(self) -> None:
        try:
            os.read(self._r, 4096)
        except BlockingIOError:
            pass

    def close(self) -> None:
        for fd in (self._r, self._w):
            try:
                os.close(fd)
            except OSError:
                pass


def _l2_interface(api: str):
    class L2AdapterInterface(abc.ABC):
        def __init__(self, max_capacity_bytes: int = 0) -> None:
            self._listeners = []
            self._max_capacity_bytes = max_capacity_bytes
            self._total_bytes_used = 0
            self._bytes_by_cache_salt = {}
            self._usage_lock = threading.Lock()

        @abc.abstractmethod
        def get_store_event_fd(self) -> int: ...

        @abc.abstractmethod
        def get_lookup_and_lock_event_fd(self) -> int: ...

        @abc.abstractmethod
        def get_load_event_fd(self) -> int: ...

        @abc.abstractmethod
        def submit_store_task(self, keys, objects): ...

        @abc.abstractmethod
        def pop_completed_store_tasks(self): ...

        @abc.abstractmethod
        def query_lookup_and_lock_result(self, task_id): ...

        @abc.abstractmethod
        def submit_unlock(self, keys) -> None: ...

        @abc.abstractmethod
        def submit_load_task(self, keys, objects): ...

        @abc.abstractmethod
        def query_load_result(self, task_id): ...

        @abc.abstractmethod
        def close(self) -> None: ...

        def register_listener(self, listener) -> None:
            self._listeners.append(listener)

        def _notify_keys_stored(self, keys, sizes) -> None:
            with self._usage_lock:
                for key, size in zip(keys, sizes):
                    self._total_bytes_used += size
                    salt = key.cache_salt
                    self._bytes_by_cache_salt[salt] = (
                        self._bytes_by_cache_salt.get(salt, 0) + size
                    )
            for listener in self._listeners:
                listener.on_l2_keys_stored(keys, sizes)

        def _notify_keys_accessed(self, keys) -> None:
            for listener in self._listeners:
                listener.on_l2_keys_accessed(keys)

        def _notify_keys_deleted(self, keys, sizes) -> None:
            with self._usage_lock:
                for key, size in zip(keys, sizes):
                    self._total_bytes_used -= size
                    salt = key.cache_salt
                    left = self._bytes_by_cache_salt.get(salt, 0) - size
                    if left <= 0:
                        self._bytes_by_cache_salt.pop(salt, None)
                    else:
                        self._bytes_by_cache_salt[salt] = left
                if self._total_bytes_used < 0:
                    self._total_bytes_used = 0
            for listener in self._listeners:
                listener.on_l2_keys_deleted(keys)

        @property
        def supports_global_eviction(self) -> bool:
            return self._max_capacity_bytes > 0

        def delete(self, keys) -> None:
            pass

        def get_usage(self) -> AdapterUsage:
            with self._usage_lock:
                return AdapterUsage(
                    total_bytes_used=self._total_bytes_used,
                    total_capacity_bytes=self._max_capacity_bytes,
                    bytes_by_cache_salt=MappingProxyType(
                        {k: v for k, v in self._bytes_by_cache_salt.items() if v > 0}
                    ),
                )

        def report_status(self) -> dict:
            return {}

    if api == "0.5.4":
        def set_backend_identity(self, name, shared) -> None:
            pass

        L2AdapterInterface.set_backend_identity = set_backend_identity

    # The one method whose signature differs between releases.
    if api == "0.4.7":
        def submit_lookup_and_lock_task(self, keys): ...
    elif api == "0.5.0":
        def submit_lookup_and_lock_task(self, keys, layout_desc): ...
    else:
        def submit_lookup_and_lock_task(self, keys, group_layout_descs): ...
    L2AdapterInterface.submit_lookup_and_lock_task = abc.abstractmethod(
        submit_lookup_and_lock_task
    )
    L2AdapterInterface.__abstractmethods__ = frozenset(
        set(L2AdapterInterface.__abstractmethods__)
        | {"submit_lookup_and_lock_task"}
    )
    return L2AdapterInterface


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


class _RemoteConnector:
    """lmcache.v1.storage_backend.connector.base_connector.RemoteConnector,
    reduced to what the dfkv connector calls on it."""

    def __init__(self, config, metadata) -> None:
        self.config = config
        self.metadata = metadata
        # As LMCache's base: the chunk geometry from the metadata.
        get_shapes = getattr(metadata, "get_shapes", None)
        if get_shapes is not None:
            self.meta_shapes = list(get_shapes())
            self.meta_dtypes = list(metadata.get_dtypes())
            self.meta_fmt = "KV_MLA_FMT" if metadata.use_mla else "KV_2LTD"
            self.full_chunk_size_bytes = sum(
                _numel(shape) * dtype.itemsize
                for shape, dtype in zip(self.meta_shapes, self.meta_dtypes)
            )
            self.single_token_size = self.full_chunk_size_bytes // metadata.chunk_size

    def post_init(self) -> None:
        pass

    def reshape_partial_chunk(self, memory_obj, bytes_read):
        full = self.full_chunk_size_bytes
        if bytes_read <= 0 or bytes_read > full or bytes_read % self.single_token_size:
            raise ValueError(f"invalid partial chunk length {bytes_read}")
        memory_obj.valid_bytes = bytes_read
        return memory_obj


def install(api: str | None = None) -> str:
    """Make ``import lmcache...`` work. Returns ``"real"`` if LMCache is
    installed (nothing is touched), else the API version the stand-in has."""
    if real_lmcache_available():
        return "real"
    api = api or os.environ.get("DFKV_TEST_LMCACHE_API", "0.4.7")
    if api not in APIS:
        raise ValueError(f"unknown LMCache API {api!r}; one of {APIS}")
    if getattr(sys.modules.get("lmcache"), "_dfkv_stub_api", api) != api:
        raise RuntimeError("the LMCache stand-in is already installed as another API")

    root = types.ModuleType("lmcache")
    root._dfkv_stub_api = api
    sys.modules["lmcache"] = root

    _module("lmcache.logging").init_logger = logging.getLogger
    utils = _module("lmcache.utils")

    class CacheEngineKey:  # the connector only type-annotates with it
        pass

    utils.CacheEngineKey = CacheEngineKey

    native_name = (
        "lmcache.lmcache_native" if api == "0.5.4" else "lmcache.native_storage_ops"
    )
    _module(native_name).Bitmap = Bitmap

    _module("lmcache.v1")
    mm = _module("lmcache.v1.memory_management")

    class MemoryObj:  # likewise only an annotation
        pass

    mm.MemoryObj = MemoryObj
    _module("lmcache.v1.platform").create_event_notifier = _EventNotifier

    _module("lmcache.v1.storage_backend")
    connector = _module("lmcache.v1.storage_backend.connector")

    class ConnectorAdapter:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class ConnectorContext:
        pass

    connector.ConnectorAdapter = ConnectorAdapter
    connector.ConnectorContext = ConnectorContext
    _module(
        "lmcache.v1.storage_backend.connector.base_connector"
    ).RemoteConnector = _RemoteConnector

    class LocalCPUBackend:
        pass

    _module(
        "lmcache.v1.storage_backend.local_cpu_backend"
    ).LocalCPUBackend = LocalCPUBackend

    _module("lmcache.v1.distributed")
    _module("lmcache.v1.distributed.api").ObjectKey = ObjectKey
    internal = _module("lmcache.v1.distributed.internal_api")
    internal.L2StoreResult = L2StoreResult

    class L1MemoryDesc:
        def __init__(self, ptr: int = 0, size: int = 0) -> None:
            self.ptr = ptr
            self.size = size

    internal.L1MemoryDesc = L1MemoryDesc
    _module("lmcache.v1.distributed.l2_adapters")
    base = _module("lmcache.v1.distributed.l2_adapters.base")
    base.L2AdapterInterface = _l2_interface(api)
    base.L2TaskId = int
    base.AdapterUsage = AdapterUsage

    class L2AdapterConfigBase:
        pass

    _module(
        "lmcache.v1.distributed.l2_adapters.config"
    ).L2AdapterConfigBase = L2AdapterConfigBase
    return api
