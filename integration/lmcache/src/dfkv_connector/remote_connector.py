# SPDX-License-Identifier: Apache-2.0
"""LMCache RemoteConnector backed by a dfkv KV cache cluster.

Ported from the dingofs LMCache connector, with two substantive changes:

  1. Backend swap — talks to dfkv via the C ABI (libdfkv.so, ctypes) instead of
     the dingofs pybind11 ``_dingofs_native`` module. See native_client.py.
  2. Arbitrary block size — the dingofs connector hard-capped a block at 4 MiB
     (its cache node used fixed io_uring buffers). dfkv stores values of any
     size, so that cap is gone; the connector handles whatever
     ``full_chunk_size_bytes`` LMCache computes, and reads back variable-size
     (unfull) chunks at their true length via dfkv_batch_get_auto +
     reshape_partial_chunk.

Hot-path design (unchanged from dingofs):
  - Each batched op runs the blocking dfkv C call in a thread-pool executor and
    awaits it; ctypes releases the GIL so the asyncio loop keeps running.
  - ExistsLRU short-circuits exists / batched_async_contains so the common case
    (just-put key, immediate prefetch check) skips the network entirely.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Any, List, Optional, Tuple

from dfkv_common import LMCACHE_RAW_V1, canonical_namespace
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

from ._telemetry import config as _tcfg
from .access_log import access_log
from .config import parse_dfkv_url
from .exists_cache import ExistsLRU
from .key_mapper import cache_engine_key_to_dfkv_key
from .native_client import DfkvNativeClient

logger = init_logger(__name__)



def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.2f}KiB"
    return f"{n / 1024 / 1024:.2f}MiB"

def _key_label(key: bytes) -> str:
    return f"len={len(key)} sha256={hashlib.sha256(key).hexdigest()[:16]}"




class DfkvConnector(RemoteConnector):
    """RemoteConnector talking to a dfkv cache cluster via the C ABI."""

    _DEFAULT_BATCH_MAX_KEYS = 512
    _ENV_BATCH_MAX_KEYS = "DFKV_CONNECTOR_BATCH_MAX_KEYS"
    _DEFAULT_GET_PARALLELISM = 1
    _ENV_GET_PARALLELISM = "DFKV_CONNECTOR_GET_PARALLELISM"
    _ENV_ASSUME_EXISTS = "DFKV_CONNECTOR_ASSUME_EXISTS"

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        exists_cache_capacity: int = 1_000_000,
        lib_path: Optional[str] = None,
        key_namespace: Optional[str] = None,
        membership: str = "mds",
        mds_poll_ms: int = 3000,
        mla_canonical_keys: Optional[bool] = None,
    ) -> None:
        with access_log("__init__", lambda: f"url={url}") as r:
            super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

            endpoint = parse_dfkv_url(url, membership=membership)
            self._batch_max_keys = self._resolve_positive_int_env(
                self._ENV_BATCH_MAX_KEYS, self._DEFAULT_BATCH_MAX_KEYS
            )
            self._get_parallelism = self._resolve_positive_int_env(
                self._ENV_GET_PARALLELISM, self._DEFAULT_GET_PARALLELISM
            )
            self._assume_exists = self._resolve_bool_env(self._ENV_ASSUME_EXISTS)

            rdma_pools = self._collect_rdma_pools(local_cpu_backend)
            _md = local_cpu_backend.metadata
            self._use_mla = bool(getattr(_md, "use_mla", False))
            self._world_size = max(
                1, int(getattr(_md, "world_size", 1)))
            self._worker_id = int(getattr(_md, "worker_id", 0))
            model_identity = str(getattr(_md, "model_name", ""))
            _dtype_identity = ",".join(
                str(dtype) for dtype in self.meta_dtypes)
            _shape_identity = "|".join(
                str(tuple(shape)) for shape in self.meta_shapes)
            namespace = canonical_namespace(
                model_identity,
                LMCACHE_RAW_V1,
                key_namespace,
                tenant_id=str(getattr(_md, "tenant_id", "default")),
                model_revision=str(
                    getattr(_md, "model_revision", model_identity)),
                dtype=_dtype_identity or "unknown",
                block_tokens=max(
                    1, int(getattr(_md, "chunk_size", 1))),
                layer_count=max(1, len(self.meta_shapes)),
                tp_size=self._world_size,
                layout_fields={
                    "meta_shapes": _shape_identity,
                    "meta_dtypes": _dtype_identity,
                    "meta_format": str(self.meta_fmt),
                    "full_chunk_size_bytes": int(
                        self.full_chunk_size_bytes),
                    "use_mla": self._use_mla,
                },
            )

            # Phase 9: canonical (rank-agnostic) keys for MLA. LMCache embeds
            # worker_id in every key although MLA KV is replicated across TP
            # ranks — 8x storage/writes/reads at TP8, and the same-host
            # rendezvous can never match (different keys). With canonical keys
            # all ranks share one keyspace; writes are striped to a single
            # deterministic owner per chunk (see _stripe_owner). The decision
            # comes from runtime metadata or this explicit constructor option;
            # environment variables may not alter cache identity/key geometry.
            auto_canonical = self._use_mla and self._world_size > 1
            self._canonical_keys = (
                auto_canonical
                if mla_canonical_keys is None
                else auto_canonical and bool(mla_canonical_keys)
            )
            if self._canonical_keys:
                logger.info(
                    "dfkv canonical MLA keys enabled (world=%d): shared "
                    "keyspace + single-writer striping",
                    self._world_size)

            self._local_cpu_backend = local_cpu_backend
            self._exists_lru = ExistsLRU(capacity=exists_cache_capacity)
            self._client = DfkvNativeClient(
                raw_endpoint=endpoint.raw_endpoint,
                group=endpoint.group,
                membership=endpoint.membership,
                key_namespace=namespace,
                tp_size=self._world_size,
                tp_rank=self._worker_id,
                lib_path=lib_path,
                mds_poll_ms=mds_poll_ms,
                rdma_pools=rdma_pools,
                loop=loop,
                get_parallelism=self._get_parallelism,
            )

            total_pool_mib = sum(length for _, length in rdma_pools) / (1024 * 1024)
            logger.info(
                "DfkvConnector ready: membership=%s endpoint=%s group=%s "
                "full_chunk=%d (%.2f MiB) namespace=%s rdma_pools=%d (%.1f MiB) "
                "batch_max_keys=%d get_parallelism=%d assume_exists=%s transport=%s",
                endpoint.membership, endpoint.raw_endpoint, endpoint.group,
                self.full_chunk_size_bytes,
                self.full_chunk_size_bytes / (1024 * 1024),
                namespace.hex(), len(rdma_pools), total_pool_mib,
                self._batch_max_keys, self._get_parallelism, self._assume_exists,
                getattr(self._client, "transport_mode", "unknown"),
            )
            r.result = f"endpoint={endpoint.raw_endpoint} group={endpoint.group}"

    def _key(self, key: CacheEngineKey) -> bytes:
        return cache_engine_key_to_dfkv_key(key, self._canonical_keys)

    def _stripe_owner(self, key: CacheEngineKey) -> bool:
        """With canonical keys every rank would redundantly PUT the same key;
        exactly one deterministic owner writes each chunk instead (identical
        bytes on every rank — MLA — so any single writer is correct)."""
        if not self._canonical_keys:
            return True
        return (key.chunk_hash % self._world_size) == self._worker_id

    @staticmethod
    def _collect_rdma_pools(local_cpu_backend: Any) -> List[Tuple[int, int]]:
        """Extract ``(addr, length)`` pairs covering the pinned CPU arena that
        backs LMCache MemoryObj allocations. Both put-side and get-side buffers
        are slices into this arena, so registering it once with the RDMA NIC
        covers every byte of dfkv Put/Get traffic. Empty list when unreachable
        (scheduler stub, or a paged/P2P allocator we don't yet enumerate)."""
        alloc = getattr(local_cpu_backend, "memory_allocator", None)
        if alloc is None:
            return []
        buf = getattr(alloc, "buffer", None)
        if buf is not None:
            addr = int(buf.data_ptr())
            length = int(buf.numel() * buf.element_size())
            return [(addr, length)]
        logger.warning(
            "DfkvConnector: local_cpu_backend allocator %s has no single "
            "'buffer' attribute; RDMA pool not registered (paged / P2P "
            "allocators not yet supported — per-op MR registration will be used).",
            type(alloc).__name__,
        )
        return []

    @classmethod
    def _resolve_positive_int_env(cls, env_name: str, default: int) -> int:
        raw = os.environ.get(env_name)
        if not raw:
            return default
        try:
            n = int(raw)
        except ValueError as e:
            raise ValueError(f"{env_name} must be an integer, got {raw!r}") from e
        if n <= 0:
            raise ValueError(f"{env_name} must be positive, got {n}")
        return n

    @staticmethod
    def _resolve_bool_env(env_name: str) -> bool:
        return os.environ.get(env_name, "0").lower() in ("1", "true", "yes", "on")

    def _reshape_hit(self, memory_obj: MemoryObj, nbytes: int) -> Optional[MemoryObj]:
        """Trim a fetched chunk to its true stored length. Full chunks pass
        through; an unfull chunk is reshaped by bytes_read. An invalid length
        (shouldn't happen — dfkv stores a whole number of tokens) is discarded
        as a safe miss rather than raised."""
        if nbytes == self.full_chunk_size_bytes:
            return memory_obj
        try:
            return self.reshape_partial_chunk(memory_obj, nbytes)
        except Exception as e:
            logger.warning(
                "discarding hit with invalid length %d (full=%d): %s",
                nbytes, self.full_chunk_size_bytes, e,
            )
            return None

    # ------------------------------------------------------------------
    # exists / batched_async_contains
    # ------------------------------------------------------------------

    async def exists(self, key: CacheEngineKey) -> bool:
        key_bytes = self._key(key)
        with access_log("exists", lambda: _key_label(key_bytes)) as r:
            if self._exists_lru.has(key_bytes):
                r.result = "lru_hit"
                return True
            per_key = await self._client.batch_exists([key_bytes])
            found = bool(per_key) and bool(per_key[0])
            r.result = "found" if found else "not_found"
            if found:
                self._exists_lru.add(key_bytes)
            return found

    def exists_sync(self, key: CacheEngineKey) -> bool:
        key_bytes = self._key(key)
        with access_log("exists_sync", lambda: _key_label(key_bytes)) as r:
            if self._exists_lru.has(key_bytes):
                r.result = "lru_hit"
                return True
            try:
                found = self._client.exists_sync(key_bytes)
            except Exception as e:
                logger.warning(
                    "exists_sync failed for %s: %s", _key_label(key_bytes), e)
                r.result = f"error: {e}"
                return False
            r.result = "found" if found else "not_found"
            if found:
                self._exists_lru.add(key_bytes)
            return found

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        _ = (lookup_id, pin)  # unused by dfkv
        n = len(keys)
        with access_log("batched_async_contains", lambda: f"{n} keys") as r:
            if not keys:
                r.result = "empty"
                return 0
            if self._assume_exists:
                r.result = f"prefix={n} (assume_exists)"
                return n
            binary_keys = [self._key(k) for k in keys]
            i = self._exists_lru.prefix_len(binary_keys)
            if i == n:
                r.result = f"prefix={n} (all lru hit)"
                return n

            remaining = binary_keys[i:]
            for start, end in self._batch_slices(len(remaining)):
                chunk = remaining[start:end]
                per_key = await self._client.batch_exists(chunk)
                if not per_key:
                    r.result = (f"prefix={i + start} (lru={i}, "
                                f"remote={start} +0 no resp)")
                    return i + start
                hit_keys: List[bytes] = []
                for j, found in enumerate(per_key):
                    if not found:
                        if hit_keys:
                            self._exists_lru.add_many(hit_keys)
                        r.result = (f"prefix={i + start + j} "
                                    f"(lru={i}, remote={start + j})")
                        return i + start + j
                    hit_keys.append(chunk[j])
                if hit_keys:
                    self._exists_lru.add_many(hit_keys)
                if len(per_key) < len(chunk):
                    r.result = f"prefix={i + start + len(per_key)} (short_resp)"
                    return i + start + len(per_key)
            r.result = f"prefix={n} (all hit; lru={i})"
            return n

    # ------------------------------------------------------------------
    # get / put
    # ------------------------------------------------------------------

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_bytes = self._key(key)
        with access_log("get", lambda: _key_label(key_bytes)) as r:
            memory_obj = self._allocate_chunk()
            if memory_obj is None:
                r.result = "alloc_failed"
                return None

            handed_off = False
            try:
                _ok, per_key, lengths = await self._client.batch_get(
                    [key_bytes], [memory_obj.byte_array]
                )
                if per_key and per_key[0]:
                    reshaped = self._reshape_hit(memory_obj, lengths[0])
                    if reshaped is not None:
                        self._exists_lru.add(key_bytes)
                        handed_off = True
                        r.result = f"ok {_fmt_bytes(lengths[0])}"
                        return reshaped
                    r.result = "bad_length"
                    return None
                r.result = "not_found"
                return None  # NotFound — caller treats as cache miss
            finally:
                if not handed_off:
                    memory_obj.ref_count_down()

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        # NOTE: we do NOT ref_count_down memory_obj here. The caller hands us a
        # serialized compressed_memory_obj whose lifetime it manages. Mirrors
        # RedisConnector.put / the dingofs connector.
        key_bytes = self._key(key)
        size = len(memory_obj.byte_array)
        with access_log(
            "put",
            lambda: f"{_key_label(key_bytes)}, {_fmt_bytes(size)}",
        ) as r:
            if not self._stripe_owner(key):
                # canonical keys: the owning rank writes this chunk; treating
                # it as stored here prevents this rank from ever re-putting.
                self._exists_lru.add(key_bytes)
                r.result = "striped_skip"
                return
            ok, _ = await self._client.batch_set([key_bytes], [memory_obj.byte_array])
            if ok:
                self._exists_lru.add(key_bytes)
            r.result = "ok" if ok else "partial_fail"

    # ------------------------------------------------------------------
    # batched_put / batched_get
    # ------------------------------------------------------------------

    def support_batched_put(self) -> bool:
        return True

    async def batched_put(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        # NOTE: we do NOT ref_count_down memory_objs here. See put() above.
        n = len(keys)
        size = (len(memory_objs[0].byte_array) * n) if memory_objs else 0
        with access_log("batched_put", lambda: f"{n} keys, {_fmt_bytes(size)}") as r:
            if not keys:
                r.result = "empty"
                return
            if len(keys) != len(memory_objs):
                r.result = "FAIL length_mismatch"
                raise ValueError("keys and memory_objs length mismatch")
            if self._canonical_keys:
                # single-writer striping: this rank only writes the chunks it
                # owns; the rest are (being) written by their owners and are
                # remembered as stored so this rank never re-puts them.
                owned = [i for i, k in enumerate(keys) if self._stripe_owner(k)]
                skipped = [self._key(keys[i]) for i in range(n)
                           if i not in set(owned)]
                if skipped:
                    self._exists_lru.add_many(skipped)
                if not owned:
                    r.result = f"striped_skip {n} keys"
                    return
                keys = [keys[i] for i in owned]
                memory_objs = [memory_objs[i] for i in owned]
                n = len(keys)
            binary_keys = [self._key(k) for k in keys]
            views = [obj.byte_array for obj in memory_objs]
            ok_all = True
            stored_keys: List[bytes] = []
            chunks = 0
            for start, end in self._batch_slices(n):
                chunks += 1
                ok, per_key = await self._client.batch_set(
                    binary_keys[start:end], views[start:end]
                )
                if ok and per_key is None:
                    stored_keys.extend(binary_keys[start:end])
                    continue
                per_key_list = list(per_key or [])
                for i, found in enumerate(per_key_list):
                    if found:
                        stored_keys.append(binary_keys[start + i])
                if not ok or len(per_key_list) < (end - start) or not all(per_key_list):
                    ok_all = False
            if stored_keys:
                self._exists_lru.add_many(stored_keys)
            r.result = (f"ok chunks={chunks}" if ok_all
                        else f"partial_fail stored={len(stored_keys)}/{n} "
                             f"chunks={chunks}")

    def support_batched_get(self) -> bool:
        return True

    async def batched_get(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        n = len(keys)
        with access_log("batched_get", lambda: f"{n} keys") as r:
            if not keys:
                r.result = "empty"
                return []
            binary_keys = [self._key(k) for k in keys]

            objs: List[MemoryObj] = []
            for _ in keys:
                obj = self._allocate_chunk()
                if obj is None:
                    for o in objs:
                        o.ref_count_down()
                    r.result = "alloc_failed"
                    return [None] * n
                objs.append(obj)

            views = [o.byte_array for o in objs]
            out: List[Optional[MemoryObj]] = [None] * n
            hit_keys: List[bytes] = []
            try:
                for start, _end, per_key_list, lengths in await self._batch_get_chunks(
                    binary_keys, views
                ):
                    for local_i, found in enumerate(per_key_list):
                        i = start + local_i
                        if found:
                            reshaped = self._reshape_hit(objs[i], lengths[local_i])
                            if reshaped is not None:
                                out[i] = reshaped
                                hit_keys.append(binary_keys[i])
            except Exception:
                for o in objs:
                    o.ref_count_down()
                raise

            for i, obj in enumerate(objs):
                if out[i] is None:
                    obj.ref_count_down()
            if hit_keys:
                self._exists_lru.add_many(hit_keys)
            r.result = f"hits={len(hit_keys)}/{n}"
            return out

    # ------------------------------------------------------------------
    # ping / list / close
    # ------------------------------------------------------------------

    def support_ping(self) -> bool:
        return True

    async def ping(self) -> int:
        with access_log("ping", lambda: "") as r:
            try:
                self._client.ping_sync()
                r.result = "ok"
                return 0
            except Exception as e:
                logger.warning("ping failed: %s", e)
                r.result = f"FAIL {e}"
                return 1

    async def list(self) -> List[str]:
        with access_log("list", lambda: "") as r:
            # dfkv has no enumeration RPC. LMCache uses list() mostly for
            # diagnostics; returning empty is safe and documented.
            r.result = "unsupported (returning [])"
            return []

    async def close(self) -> None:
        with access_log("close", lambda: ""):
            self._client.close()

    # ------------------------------------------------------------------
    # lifecycle / utility hooks
    # ------------------------------------------------------------------

    def post_init(self) -> None:
        with access_log("post_init", lambda: ""):
            super().post_init()

    def reshape_partial_chunk(self, memory_obj, bytes_read):
        full = self.full_chunk_size_bytes
        with access_log("reshape_partial_chunk",
                        lambda: f"{bytes_read}/{full} bytes"):
            return super().reshape_partial_chunk(memory_obj, bytes_read)

    # ------------------------------------------------------------------
    # batched_get_non_blocking / remove_sync / batched_contains
    # ------------------------------------------------------------------

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        n = len(keys)
        with access_log("batched_get_non_blocking",
                        lambda: f"{n} keys lookup_id={lookup_id}") as r:
            if not keys:
                r.result = "empty"
                return []

            binary_keys = [self._key(k) for k in keys]
            out: List[MemoryObj] = []
            ranges = list(self._batch_slices(n))
            chunks = len(ranges)
            for group_start in range(0, len(ranges), self._get_parallelism):
                group = ranges[group_start:group_start + self._get_parallelism]
                group_objs: List[List[MemoryObj]] = []
                group_views: List[List[memoryview]] = []
                for start, end in group:
                    objs: List[MemoryObj] = []
                    for _ in range(start, end):
                        obj = self._allocate_chunk()
                        if obj is None:
                            for chunk_objs in group_objs:
                                for o in chunk_objs:
                                    o.ref_count_down()
                            for o in objs:
                                o.ref_count_down()
                            r.result = (f"alloc_failed prefix={len(out)}/{n} "
                                        f"chunks={chunks}")
                            return out
                        objs.append(obj)
                    group_objs.append(objs)
                    group_views.append([o.byte_array for o in objs])
                try:
                    results = await asyncio.gather(
                        *(
                            self._client.batch_get(binary_keys[start:end], views)
                            for (start, end), views in zip(group, group_views)
                        )
                    )
                except Exception:
                    for chunk_objs in group_objs:
                        for o in chunk_objs:
                            o.ref_count_down()
                    for o in out:
                        o.ref_count_down()
                    raise

                for idx, (((start, _end), objs), (_ok, per_key, lengths)) in enumerate(
                    zip(zip(group, group_objs), results)
                ):
                    per_key_list = list(per_key or [])
                    local_prefix = 0
                    while (local_prefix < len(objs) and
                           local_prefix < len(per_key_list) and
                           per_key_list[local_prefix]):
                        reshaped = self._reshape_hit(
                            objs[local_prefix], lengths[local_prefix]
                        )
                        if reshaped is None:
                            break  # invalid-length hit => stop the prefix here
                        objs[local_prefix] = reshaped
                        local_prefix += 1

                    if local_prefix:
                        self._exists_lru.add_many(
                            binary_keys[start:start + local_prefix]
                        )
                        out.extend(objs[:local_prefix])

                    if local_prefix < len(objs):
                        for obj in objs[local_prefix:]:
                            obj.ref_count_down()
                        for later_objs in group_objs[idx + 1:]:
                            for obj in later_objs:
                                obj.ref_count_down()
                        r.result = f"prefix={len(out)}/{n} chunks={chunks}"
                        return out

            r.result = f"prefix={len(out)}/{n} chunks={chunks}"
            return out

    def remove_sync(self, key: CacheEngineKey) -> bool:
        key_bytes = self._key(key)
        with access_log("remove_sync", lambda: _key_label(key_bytes)) as r:
            if not self._client.supports_remove():
                r.result = "FAIL no remove RPC"
                raise NotImplementedError(
                    "libdfkv.so has no remove RPC (rebuild dfkv with it)"
                )
            try:
                ok = self._client.remove_sync(key_bytes)
            except Exception as e:
                logger.warning(
                    "remove_sync failed for %s: %s", _key_label(key_bytes), e)
                r.result = f"error: {e}"
                return False
            if ok:
                self._exists_lru.discard(key_bytes)
            r.result = "ok" if ok else "fail"
            return ok

    def support_batched_contains(self) -> bool:
        return False

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        n = len(keys)
        with access_log("batched_contains", lambda: f"{n} keys") as r:
            r.result = "FAIL NotImplementedError"
            raise NotImplementedError(
                "dfkv connector does not support sync batched_contains"
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _allocate_chunk(self) -> Optional[MemoryObj]:
        return self._local_cpu_backend.allocate(
            self.meta_shapes[0],
            self.meta_dtypes[0],
            self.meta_fmt,
        )

    def _batch_slices(self, n: int):
        step = self._batch_max_keys
        for start in range(0, n, step):
            yield start, min(start + step, n)

    async def _batch_get_chunks(self, binary_keys, views):
        ranges = list(self._batch_slices(len(binary_keys)))
        out = []
        for group_start in range(0, len(ranges), self._get_parallelism):
            group = ranges[group_start:group_start + self._get_parallelism]
            results = await asyncio.gather(
                *(
                    self._client.batch_get(binary_keys[start:end], views[start:end])
                    for start, end in group
                )
            )
            for (start, end), (_ok, per_key, lengths) in zip(group, results):
                out.append((start, end, list(per_key or []), list(lengths or [])))
        return out
