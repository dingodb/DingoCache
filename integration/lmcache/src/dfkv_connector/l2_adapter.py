# SPDX-License-Identifier: Apache-2.0
"""dfkv L2 adapter for the LMCache **multiprocess (MP) server**.

LMCache's MP server (``lmcache server`` + ``LMCacheMPConnector`` on the vLLM
side) drives its remote tier through ``L2AdapterInterface`` — a non-blocking,
eventfd-signalled store/lookup/load contract — instead of the in-process
``RemoteConnector`` plugin mechanism used by :mod:`dfkv_connector.remote_connector`.
The two interfaces are not interchangeable: the in-process ``remote_storage_plugins``
config is silently ignored by the MP server, so dfkv could not be an MP L2 tier
until this adapter existed.

This module bridges the gap **without any native (pybind/eventfd) code**: it
reuses the existing ctypes client (:class:`dfkv_connector.native_client.DfkvNativeClient`,
which already does RDMA Put/Get over ``libdfkv.so`` and fans blocking calls out to a
thread pool) and adapts it to the L2 eventfd model with:

* three distinct cross-platform event notifiers (store / lookup / load);
* a background asyncio loop thread that runs the client's ``batch_*`` coroutines;
* a done-callback per submitted task that records the result and ``notify()``-s
  the matching event fd — exactly the completion shape the MP store / prefetch
  controllers poll for.

Design notes:
* **No allocation.** Unlike the in-process connector, the MP controllers own the
  ``MemoryObj`` buffers for both store (read-from) and load (write-into); the
  adapter only borrows their ``byte_array`` memoryviews for the duration of a task.
* **RDMA zero-copy.** When the factory passes an ``l1_memory_desc`` (the server's
  pinned L1 arena), the whole arena is registered once so Put/Get into any slice
  skips per-op MR registration. No-op on TCP transport.
* **Locking is a no-op.** dfkv is a remote KV store with no L1-eviction concept,
  so lookup "locks" / unlocks are tracked only as a client-side refcount (kept for
  interface symmetry; nothing can evict a remote object out from under a loader).
* **Isolation.** ``dfkv_open`` receives an explicit binary model namespace;
  object keys use the same canonical pool-coordinate schema as every connector.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future as _CFuture
from typing import TYPE_CHECKING, List, Optional

from dfkv_common import (
    LMCACHE_RAW_V1,
    canonical_namespace,
    pool_key,
    reject_namespace_override,
)

from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

from .config import parse_dfkv_url
from .native_client import DfkvNativeClient

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

logger = init_logger(__name__)


def _object_key_to_bytes(key: ObjectKey) -> bytes:
    """Encode an ObjectKey using dfkv's canonical binary pool-key schema."""
    global_rank = (int(key.kv_rank) >> 16) & 0xFF
    component = "all"
    if key.cache_salt:
        component = f"salt-{key.cache_salt}"
    return pool_key(
        key.chunk_hash.hex(),
        pool="kv",
        tp_size=max(1, _kv_rank_world(int(key.kv_rank))),
        tp_rank=global_rank,
        group_id=int(key.object_group_id),
        component=component,
    )


def _kv_rank_world(kv_rank: int) -> int:
    """world_size component of an ObjectKey kv_rank (upstream packs 4x8bit)."""
    return (kv_rank >> 24) & 0xFF


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class DfkvL2AdapterConfig(L2AdapterConfigBase):
    """Config for the dfkv L2 adapter (loaded via the ``plugin`` L2 adapter).

    Example ``--l2-adapter`` JSON::

        {"type": "plugin",
         "module_path": "dfkv_connector.l2_adapter",
         "class_name": "DfkvL2Adapter",
         "config_class_name": "DfkvL2AdapterConfig",
         "adapter_params": {
             "url": "dfkv://127.0.0.1:28150/glm",
             "membership": "mds",
             "lib": "/dfkv/lib/libdfkv.so",
             "model_name": "glm-5.2"
         }}

    Fields (all under ``adapter_params``):
      - url (str, required): ``dfkv://<endpoint>/<group>``. For ``mds``
        membership the host part is a comma-separated MDS ip:port list; for
        ``static`` it is a literal member string.
      - membership (str): ``"mds"`` (default) or ``"static"``.
      - lib (str): path to ``libdfkv.so`` (else env ``DFKV_LIB`` /
        ``$DFKV_BUILD/libdfkv.so``).
      - model_name (str): model identity used to derive the binary namespace.
      - mds_poll_ms (int): MDS ring re-discovery interval (default 3000).
      - tenant_id (str): explicit tenant identity (default ``"default"``).
      - model_revision (str): explicit model revision (default ``model_name``).
      - num_workers (int): client I/O parallelism (default 8).
      - max_capacity_gb (float): if > 0, enables aggregate usage reporting /
        global eviction; default 0 (dfkv manages its own capacity).
    """

    def __init__(
        self,
        url: str,
        membership: str = "mds",
        lib: Optional[str] = None,
        model_name: str = "",
        tenant_id: str = "default",
        model_revision: Optional[str] = None,
        mds_poll_ms: int = 3000,
        num_workers: int = 8,
        max_capacity_gb: float = 0.0,
    ) -> None:
        self.url = url
        self.membership = membership
        self.lib = lib
        self.model_name = model_name
        self.tenant_id = tenant_id
        self.model_revision = model_revision or model_name
        self.mds_poll_ms = mds_poll_ms
        self.num_workers = num_workers
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "DfkvL2AdapterConfig":
        url = d.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("dfkv L2 adapter: 'url' must be a non-empty string")

        membership = d.get("membership", "mds")
        if membership not in ("mds", "static"):
            raise ValueError("dfkv L2 adapter: 'membership' must be 'mds' or 'static'")

        lib = d.get("lib")
        if lib is not None and not isinstance(lib, str):
            raise ValueError("dfkv L2 adapter: 'lib' must be a string")

        model_name = d.get("model_name", "")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("dfkv L2 adapter: 'model_name' must be a non-empty string")
        reject_namespace_override(d)
        tenant_id = d.get("tenant_id", "default")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("dfkv L2 adapter: 'tenant_id' must be a non-empty string")
        model_revision = d.get("model_revision", model_name)
        if not isinstance(model_revision, str) or not model_revision:
            raise ValueError(
                "dfkv L2 adapter: 'model_revision' must be a non-empty string"
            )

        mds_poll_ms = d.get("mds_poll_ms", 3000)
        if not isinstance(mds_poll_ms, int) or mds_poll_ms <= 0:
            raise ValueError("dfkv L2 adapter: 'mds_poll_ms' must be a positive int")

        num_workers = d.get("num_workers", 8)
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers <= 0
        ):
            raise ValueError("dfkv L2 adapter: 'num_workers' must be a positive int")

        max_capacity_gb = d.get("max_capacity_gb", 0.0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError(
                "dfkv L2 adapter: 'max_capacity_gb' must be a non-negative number"
            )

        # Reject the retired rank-folding knob instead of ignoring it.
        if "mla_canonical_keys" in d:
            raise ValueError(
                "dfkv L2 adapter: 'mla_canonical_keys' is unsupported because "
                "the MP-server API does not expose enough model/PP metadata to "
                "prove that rank folding is byte-compatible"
            )
        return cls(
            url=url,
            membership=membership,
            lib=lib,
            model_name=model_name,
            tenant_id=tenant_id,
            model_revision=model_revision,
            mds_poll_ms=mds_poll_ms,
            num_workers=num_workers,
            max_capacity_gb=float(max_capacity_gb),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "dfkv L2 adapter config (under adapter_params):\n"
            "- url (str, required): dfkv://<endpoint>/<group>\n"
            "- membership (str): 'mds' (default) or 'static'\n"
            "- lib (str): path to libdfkv.so (else env DFKV_LIB)\n"
            "- model_name (str): model identity for namespace derivation\n"
            "- tenant_id (str): explicit tenant identity (default 'default')\n"
            "- model_revision (str): explicit revision (default model_name)\n"
            "- mds_poll_ms (int): MDS rediscovery interval (default 3000)\n"
            "- num_workers (int): client I/O parallelism (default 8)\n"
            "- max_capacity_gb (float): >0 enables aggregate eviction "
            "(default 0 = dfkv manages capacity)"
        )


# ----------------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------------


class DfkvL2Adapter(L2AdapterInterface):
    """L2 adapter backed by a dfkv ring (ctypes ``libdfkv.so`` + asyncio bridge)."""

    config_class_name = "DfkvL2AdapterConfig"

    def __init__(
        self,
        config: DfkvL2AdapterConfig,
        l1_memory_desc: "Optional[L1MemoryDesc]" = None,
    ) -> None:
        super().__init__(max_capacity_bytes=int(config.max_capacity_gb * (1024**3)))
        self._config = config

        # Preserve LMCache's world-size/global-rank identity. The MP-server
        # adapter receives only ObjectKey and an L1 buffer descriptor, not
        # model/PP metadata; folding rank bits could alias distinct payloads.

        # 3 distinct event notifiers (the contract requires distinct fds).
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Shared task state (touched by submit calls on controller threads and
        # by done-callbacks on the loop thread) — guarded by ``_lock``.
        self._next_task_id: L2TaskId = 0
        self._completed_stores: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookups: dict[L2TaskId, Bitmap] = {}
        self._completed_loads: dict[L2TaskId, Bitmap] = {}
        self._locked_keys: dict[ObjectKey, int] = {}
        # First-store-wins byte accounting (re-store of a known key adds 0).
        self._key_sizes: dict[ObjectKey, int] = {}
        self._warned_no_remove = False
        self._lock = threading.Lock()

        # Background asyncio loop that runs the client's batch_* coroutines.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="dfkv-l2-loop"
        )
        self._loop_thread.start()

        # Register the server's L1 arena for RDMA zero-copy when available.
        rdma_pools = []
        if l1_memory_desc is not None:
            ptr = int(getattr(l1_memory_desc, "ptr", 0) or 0)
            size = int(getattr(l1_memory_desc, "size", 0) or 0)
            if ptr and size:
                rdma_pools = [(ptr, size)]

        namespace = canonical_namespace(
            config.model_name,
            LMCACHE_RAW_V1,
            tenant_id=config.tenant_id,
            model_revision=config.model_revision,
            dtype="opaque-lmcache-l2",
        )

        endpoint = parse_dfkv_url(config.url, membership=config.membership)
        self._client = DfkvNativeClient(
            raw_endpoint=endpoint.raw_endpoint,
            group=endpoint.group,
            membership=endpoint.membership,
            key_namespace=namespace,
            lib_path=config.lib,
            mds_poll_ms=config.mds_poll_ms,
            rdma_pools=rdma_pools,
            loop=self._loop,
            get_parallelism=config.num_workers,
        )
        logger.info(
            "DfkvL2Adapter ready: endpoint=%s group=%s membership=%s "
            "namespace=%s rdma_pools=%d transport=%s",
            endpoint.raw_endpoint,
            endpoint.group,
            endpoint.membership,
            namespace.hex(),
            len(rdma_pools),
            getattr(self._client, "transport_mode", "unknown"),
        )

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def submit_store_task(
        self, keys: List[ObjectKey], objects: List[MemoryObj]
    ) -> L2TaskId:
        binary_keys = [_object_key_to_bytes(k) for k in keys]
        views = [obj.byte_array for obj in objects]
        sizes = [obj.get_size() for obj in objects]
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        fut = asyncio.run_coroutine_threadsafe(
            self._client.batch_set(binary_keys, views), self._loop
        )
        # Hold ``objects`` alive in the closure until the store completes — the
        # memoryviews alias their buffers and the C call must finish reading.
        fut.add_done_callback(
            lambda f: self._on_store_done(f, task_id, list(keys), sizes, objects)
        )
        return task_id

    def _on_store_done(
        self,
        fut: _CFuture,
        task_id: L2TaskId,
        keys: List[ObjectKey],
        sizes: List[int],
        _objects_keepalive: List[MemoryObj],
    ) -> None:
        keys_stored: List[ObjectKey] = []
        sizes_stored: List[int] = []
        task_bytes = 0
        try:
            ok, per_key = fut.result()
        except Exception as e:  # pragma: no cover - network/lib failure path
            logger.warning("dfkv L2 store task %d failed: %s", task_id, e)
            ok, per_key = False, None
        if ok:
            flags = per_key if per_key is not None else [True] * len(keys)
            for key, size, stored in zip(keys, sizes, flags):
                if not stored:
                    continue
                # First-store wins: a re-store of a known key adds 0 bytes but
                # still notifies (so LRU can move_to_end); base counters no-op.
                if key not in self._key_sizes:
                    self._key_sizes[key] = size
                    keys_stored.append(key)
                    sizes_stored.append(size)
                    task_bytes += size
                else:
                    keys_stored.append(key)
                    sizes_stored.append(0)
        with self._lock:
            self._completed_stores[task_id] = L2StoreResult(ok, task_bytes)
        if keys_stored:
            self._notify_keys_stored(keys_stored, sizes_stored)
        self._store_efd.notify()

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            completed = self._completed_stores
            self._completed_stores = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(self, keys: List[ObjectKey]) -> L2TaskId:
        binary_keys = [_object_key_to_bytes(k) for k in keys]
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        fut = asyncio.run_coroutine_threadsafe(
            self._client.batch_exists(binary_keys), self._loop
        )
        fut.add_done_callback(lambda f: self._on_lookup_done(f, task_id, list(keys)))
        return task_id

    def _on_lookup_done(
        self, fut: _CFuture, task_id: L2TaskId, keys: List[ObjectKey]
    ) -> None:
        n = len(keys)
        bitmap = Bitmap(n)
        try:
            per_key = fut.result()
        except Exception as e:  # pragma: no cover
            logger.warning("dfkv L2 lookup task %d failed: %s", task_id, e)
            per_key = None
        with self._lock:
            if per_key is not None:
                for i, found in enumerate(per_key):
                    if found:
                        bitmap.set(i)
                        self._locked_keys[keys[i]] = (
                            self._locked_keys.get(keys[i], 0) + 1
                        )
            self._completed_lookups[task_id] = bitmap
        self._lookup_efd.notify()

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_lookups.pop(task_id, None)

    def submit_unlock(self, keys: List[ObjectKey]) -> None:
        # dfkv objects are remote and never evicted out from under a loader, so
        # this is purely client-side refcount bookkeeping (kept for symmetry).
        with self._lock:
            for key in keys:
                c = self._locked_keys.get(key)
                if c is None:
                    continue
                if c <= 1:
                    del self._locked_keys[key]
                else:
                    self._locked_keys[key] = c - 1

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def submit_load_task(
        self, keys: List[ObjectKey], objects: List[MemoryObj]
    ) -> L2TaskId:
        binary_keys = [_object_key_to_bytes(k) for k in keys]
        views = [obj.byte_array for obj in objects]
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        fut = asyncio.run_coroutine_threadsafe(
            self._client.batch_get(binary_keys, views), self._loop
        )
        # Keep ``objects`` alive until the C call finishes writing into them.
        fut.add_done_callback(
            lambda f: self._on_load_done(f, task_id, list(keys), objects)
        )
        return task_id

    def _on_load_done(
        self,
        fut: _CFuture,
        task_id: L2TaskId,
        keys: List[ObjectKey],
        _objects_keepalive: List[MemoryObj],
    ) -> None:
        n = len(keys)
        bitmap = Bitmap(n)
        accessed: List[ObjectKey] = []
        try:
            _ok, per_key, _lengths = fut.result()
        except Exception as e:  # pragma: no cover
            logger.warning("dfkv L2 load task %d failed: %s", task_id, e)
            per_key = None
        if per_key is not None:
            for i, loaded in enumerate(per_key):
                if loaded:
                    bitmap.set(i)
                    accessed.append(keys[i])
        with self._lock:
            self._completed_loads[task_id] = bitmap
        if accessed:
            self._notify_keys_accessed(accessed)
        self._load_efd.notify()

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_loads.pop(task_id, None)

    # ------------------------------------------------------------------
    # Eviction (delete)
    # ------------------------------------------------------------------

    def delete(self, keys: List[ObjectKey]) -> None:
        """Drop keys from dfkv (L2 eviction). Synchronous per the L2 controller
        contract; fires ``_notify_keys_deleted`` so the eviction policy's byte
        accounting stays in sync. No-op (with a one-time warning) if the loaded
        libdfkv.so predates the remove RPC. Only meaningful when the adapter was
        constructed with ``max_capacity_gb > 0`` (otherwise the controller never
        calls delete — dfkv manages its own capacity)."""
        if not keys:
            return
        if not self._client.supports_remove():
            if not self._warned_no_remove:
                self._warned_no_remove = True
                logger.warning(
                    "dfkv L2 delete requested but libdfkv.so has no remove RPC; "
                    "eviction is a no-op. Rebuild dfkv with the remove RPC."
                )
            return
        binary_keys = [_object_key_to_bytes(k) for k in keys]
        fut = asyncio.run_coroutine_threadsafe(
            self._client.batch_remove(binary_keys), self._loop
        )
        try:
            per_key = fut.result(timeout=30.0)
        except Exception as e:  # pragma: no cover - network/lib failure path
            logger.warning("dfkv L2 delete failed for %d keys: %s", len(keys), e)
            return
        deleted: List[ObjectKey] = []
        sizes: List[int] = []
        with self._lock:
            for key, ok in zip(keys, per_key):
                if not ok:
                    continue
                # Only account keys we tracked a size for (stored via this adapter).
                sz = self._key_sizes.pop(key, None)
                if sz is not None:
                    deleted.append(key)
                    sizes.append(sz)
        if deleted:
            self._notify_keys_deleted(deleted, sizes)

    # ------------------------------------------------------------------
    # Status / Cleanup
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        return {
            "is_healthy": self._loop_thread.is_alive(),
            "type": "DfkvL2Adapter",
            "transport": getattr(self._client, "transport_mode", "unknown"),
            "group": self._config.url,
        }

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        try:
            self._loop.close()
        except Exception:  # pragma: no cover
            pass
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
