# SPDX-License-Identifier: Apache-2.0
"""Asyncio wrapper around the dfkv C ABI (libdfkv.so) loaded via ctypes.

This replaces the dingofs connector's pybind11 ``_dingofs_native`` module +
eventfd completion queue. dfkv's C ABI is synchronous and internally
thread-safe: ``dfkv_batch_*`` block and fan out across owning nodes with their
own thread pool (KVClient::RunParallel), and the membership ring is mutex
guarded — so one ``dfkv_open`` handle is safe to share across many worker
threads. ``ctypes.CDLL`` releases the GIL for the duration of each foreign call,
so dispatching the blocking calls to a ``ThreadPoolExecutor`` gives real
concurrency without a native demux thread or a cross-thread Future bridge.

Reads use ``dfkv_batch_get_auto`` (variable-size): each buffer is full-chunk
sized, and the call reports the true stored payload length per key so the
connector can ``reshape_partial_chunk`` an unfull chunk. Full chunks keep the
RDMA zero-copy datapath.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple
from dfkv_common import (
    DfkvClientOptionsV2,
    make_client_options_v2,
    make_key_array,
    make_key_buffer,
)

from . import access_log as _alog
from .access_log import access_log
from . import _hot_config
from ._telemetry import config as _tcfg
from ._telemetry import metrics as _push_metrics
from ._telemetry import tracing as _push_tracing

__all__ = ["DfkvNativeClient", "load_lib"]

logger = logging.getLogger(__name__)

c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_uint64 = ctypes.c_uint64
c_int = ctypes.c_int
POINTER = ctypes.POINTER


def load_lib(path: Optional[str] = None) -> ctypes.CDLL:
    """Load libdfkv.so and declare the C ABI signatures.

    Path precedence: explicit ``path`` arg → env ``DFKV_LIB`` →
    ``$DFKV_BUILD/libdfkv.so`` → the system dynamic-loader path.
    """
    build = os.environ.get("DFKV_BUILD")
    lib_path = (
        path
        or os.environ.get("DFKV_LIB")
        or (os.path.join(build, "libdfkv.so") if build else "libdfkv.so")
    )
    lib = ctypes.CDLL(lib_path)
    lib.dfkv_open_v2.restype = c_void_p
    lib.dfkv_open_v2.argtypes = [POINTER(DfkvClientOptionsV2)]
    lib.dfkv_put.restype = c_int
    lib.dfkv_put.argtypes = [
        c_void_p, c_void_p, c_uint64, c_void_p, c_uint64]
    lib.dfkv_get.restype = c_int
    lib.dfkv_get.argtypes = lib.dfkv_put.argtypes
    lib.dfkv_get_auto.restype = c_int
    lib.dfkv_get_auto.argtypes = [
        c_void_p, c_void_p, c_uint64, c_void_p, c_uint64,
        POINTER(c_uint64)]
    lib.dfkv_exist.restype = c_int
    lib.dfkv_exist.argtypes = [c_void_p, c_void_p, c_uint64]
    lib.dfkv_register_memory.restype = c_int
    lib.dfkv_register_memory.argtypes = [c_void_p, c_void_p, c_uint64]
    lib.dfkv_batch_put.restype = c_int
    lib.dfkv_batch_put.argtypes = [
        c_void_p, POINTER(c_void_p), POINTER(c_uint64),
        POINTER(c_void_p), POINTER(c_uint64), c_int, POINTER(c_int)]
    lib.dfkv_batch_get.restype = c_int
    lib.dfkv_batch_get.argtypes = lib.dfkv_batch_put.argtypes
    lib.dfkv_batch_get_auto.restype = c_int
    lib.dfkv_batch_get_auto.argtypes = [
        c_void_p, POINTER(c_void_p), POINTER(c_uint64),
        POINTER(c_void_p), POINTER(c_uint64),
        c_int, POINTER(c_int), POINTER(c_uint64)]
    lib.dfkv_batch_exist.restype = c_int
    lib.dfkv_batch_exist.argtypes = [
        c_void_p, POINTER(c_void_p), POINTER(c_uint64),
        c_int, POINTER(c_int)]
    # Remove support remains capability-detected for deployments that omit the
    # optional eviction RPC; when present it uses the binary-key v2 ABI.
    if hasattr(lib, "dfkv_remove"):
        lib.dfkv_remove.restype = c_int
        lib.dfkv_remove.argtypes = [c_void_p, c_void_p, c_uint64]
    if hasattr(lib, "dfkv_batch_remove"):
        lib.dfkv_batch_remove.restype = c_int
        lib.dfkv_batch_remove.argtypes = [
            c_void_p, POINTER(c_void_p), POINTER(c_uint64),
            c_int, POINTER(c_int)]
    lib.dfkv_transport_mode.restype = c_char_p
    lib.dfkv_transport_mode.argtypes = [c_void_p]
    lib.dfkv_version.restype = c_char_p
    lib.dfkv_version.argtypes = []
    lib.dfkv_close.restype = None
    lib.dfkv_close.argtypes = [c_void_p]
    return lib


def _native_version(lib: ctypes.CDLL) -> str:
    """libdfkv.so version via the C dfkv_version(); "" if missing/older lib."""
    try:
        v = lib.dfkv_version()
        return v.decode("utf-8", "replace") if v else ""
    except Exception:
        return ""


def _mv_ptr(mv: memoryview) -> Tuple[int, Any]:
    """Return ``(address, keepalive)`` for a memoryview buffer.

    Zero-copy for a writable, C-contiguous buffer (the common case — every
    MemoryObj byte_array is a slice of LMCache's pinned arena). A read-only
    buffer can't be aliased by ctypes, so we copy into a ctypes array (rare;
    correctness over zero-copy). The keepalive must outlive the C call.
    """
    try:
        arr = (ctypes.c_char * mv.nbytes).from_buffer(mv)
    except (TypeError, ValueError):
        arr = (ctypes.c_char * mv.nbytes).from_buffer_copy(mv)
    return ctypes.addressof(arr), arr


def _total_size(bufs: Sequence[memoryview]) -> int:
    return sum(mv.nbytes for mv in bufs)


def _key_label(key: bytes) -> str:
    """Safe deterministic object-key representation for text/JSON logs."""
    return f"len={len(key)} sha256={hashlib.sha256(key).hexdigest()[:16]}"

class DfkvNativeClient:
    """Awaitable interface over libdfkv.so (ctypes + thread-pool executor)."""

    def __init__(
        self,
        raw_endpoint: str,
        group: str,
        membership: str,
        key_namespace: bytes,
        tp_size: int = 1,
        tp_rank: int = 0,
        lib_path: Optional[str] = None,
        mds_poll_ms: int = 3000,
        rdma_pools: Optional[Sequence[Tuple[int, int]]] = None,
        loop=None,
        get_parallelism: int = 1,
    ) -> None:
        import asyncio

        with access_log(
            "native.__init__",
            lambda: f"membership={membership} endpoint={raw_endpoint} "
                    f"group={group} rdma_pools={len(rdma_pools or [])} "
                    f"parallelism={get_parallelism}",
        ) as r:
            self._lib = load_lib(lib_path)
            self._loop = loop or asyncio.get_running_loop()
            self._closed = False
            if not key_namespace:
                raise ValueError(
                    "DfkvNativeClient requires a non-empty key namespace")
            self._key_namespace = bytes(key_namespace)
            members = raw_endpoint if membership == "static" else ""
            register_client = (
                membership == "mds"
                and _tcfg.truthy(
                    os.environ.get("DFKV_CLIENT_REGISTER", "1"))
            )
            client_id = (
                _tcfg.resolve_connector_id({}, tp_rank=int(tp_rank))
                if register_client
                else ""
            )
            client_info = (
                f"type={_tcfg.TYPE_LMCACHE},"
                f"tp_size={int(tp_size)},"
                f"tp_rank={int(tp_rank)},"
                f"ver={_tcfg.dist_version('dfkv-connector')}"
                if register_client
                else ""
            )
            options = make_client_options_v2(
                self._key_namespace,
                members=members,
                mds_endpoints=(
                    raw_endpoint if membership == "mds" else ""),
                mds_group=group,
                mds_poll_ms=mds_poll_ms,
                register_client=register_client,
                client_id=client_id,
                client_info=client_info,
                client_heartbeat_ms=10000,
            )
            self._h = self._lib.dfkv_open_v2(ctypes.byref(options))
            if not self._h:
                raise RuntimeError("dfkv_open_v2 failed")
            mode_b = self._lib.dfkv_transport_mode(self._h)
            self.transport_mode = (
                mode_b.decode("utf-8", errors="replace") if mode_b else "unknown"
            )
            # Register the host arena(s) once so RDMA Put/Get into any slice of
            # them skips per-op MR registration. No-op on TCP.
            regs = 0
            for base, size in (rdma_pools or []):
                if not base or not size:
                    continue
                self._register_memory(int(base), int(size))
                regs += 1

            self._executor = ThreadPoolExecutor(
                max_workers=max(1, int(get_parallelism)),
                thread_name_prefix="dfkv-io",
            )
            # Unified fleet metrics pushed over OTLP (off by default; zero cost
            # when DFKV_METRICS_ENABLED is unset). LMCache had no dfkv metrics
            # before — this gives it parity with the SGLang/vLLM connectors. Report
            # the connector package + libdfkv.so versions for fleet skew tracking.
            _push_metrics.configure(
                {}, connector_type=_tcfg.TYPE_LMCACHE, tp_rank=int(tp_rank),
                version=_tcfg.dist_version("dfkv-connector"),
                native_version=_native_version(self._lib))
            # Connector-side request tracing (off by default; zero cost when off).
            # Spans for slow / sampled / failed ops pushed over OTLP /v1/traces.
            _push_tracing.configure(
                {}, connector_type=_tcfg.TYPE_LMCACHE, tp_rank=int(tp_rank),
                version=_tcfg.dist_version("dfkv-connector"),
                native_version=_native_version(self._lib))
            # Access log: parse the launch baseline (env-driven), then let a
            # control file toggle it at runtime without restarting LMCache
            # (opt-in via DFKV_HOT_CONFIG). docs/access_log.md -> 运行时热开关.
            _alog.configure({}, tp_rank=int(tp_rank))
            _hot_config.register("access_log", _alog.apply_hot)
            _hot_config.start({}, tp_rank=int(tp_rank))
            r.result = f"ok regs={regs} transport={self.transport_mode}"

    def _register_memory(self, base: int, size: int) -> None:
        """Register one MR or fail startup rather than claiming zero-copy."""
        rc = self._lib.dfkv_register_memory(
            self._h, c_void_p(base), c_uint64(size))
        if rc != 0:
            raise RuntimeError(
                f"dfkv_register_memory(base={base:#x}, size={size}) rc={rc}")

    # ------------------------------------------------------------------
    # blocking ctypes helpers (run in the executor)
    # ------------------------------------------------------------------

    def _batch_set_blocking(
        self, keys: List[bytes], bufs: List[memoryview]
    ) -> Tuple[bool, List[bool]]:
        n = len(keys)
        karr, klens, key_owners = make_key_array(keys)
        ptrs: List[int] = []
        keepalive: List[Any] = []
        for mv in bufs:
            p, ka = _mv_ptr(mv)
            ptrs.append(p)
            keepalive.append(ka)
        parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
        sarr = (c_uint64 * n)(*[mv.nbytes for mv in bufs])
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_put(
            self._h, karr, klens, parr, sarr, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_put rc={rc}")
        per_key = [out[i] == 1 for i in range(n)]
        del keepalive, key_owners  # native call has returned
        return all(per_key), per_key

    def _batch_get_blocking(
        self, keys: List[bytes], bufs: List[memoryview]
    ) -> Tuple[bool, List[bool], List[int]]:
        n = len(keys)
        karr, klens, key_owners = make_key_array(keys)
        ptrs: List[int] = []
        keepalive: List[Any] = []
        for mv in bufs:
            p, ka = _mv_ptr(mv)
            ptrs.append(p)
            keepalive.append(ka)
        parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
        caps = (c_uint64 * n)(*[mv.nbytes for mv in bufs])
        out_hit = (c_int * n)()
        out_len = (c_uint64 * n)()
        rc = self._lib.dfkv_batch_get_auto(
            self._h, karr, klens, parr, caps, n, out_hit, out_len
        )
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_get_auto rc={rc}")
        per_key = [out_hit[i] == 1 for i in range(n)]
        lengths = [int(out_len[i]) for i in range(n)]
        del keepalive, key_owners  # native call has returned
        return all(per_key), per_key, lengths

    def _batch_exists_blocking(self, keys: List[bytes]) -> List[bool]:
        n = len(keys)
        karr, klens, key_owners = make_key_array(keys)
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_exist(
            self._h, karr, klens, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_exist rc={rc}")
        del key_owners
        return [out[i] == 1 for i in range(n)]

    def _batch_remove_blocking(self, keys: List[bytes]) -> List[bool]:
        n = len(keys)
        karr, klens, key_owners = make_key_array(keys)
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_remove(
            self._h, karr, klens, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_remove rc={rc}")
        del key_owners
        return [out[i] == 1 for i in range(n)]

    # ------------------------------------------------------------------
    # public async API (returns the same tuple shapes the connector expects)
    # ------------------------------------------------------------------

    async def batch_set(
        self, keys: Sequence[bytes], bufs: Sequence[memoryview]
    ) -> Tuple[bool, Optional[List[bool]]]:
        n = len(keys)
        with _push_metrics.op("put", num_keys=n) as _m, \
                _push_tracing.span("batch_set", n) as _sp, \
                access_log("native.batch_set",
                           lambda: f"{n} keys, {_total_size(bufs)} bytes") as r:
            if _m or _sp:
                _nb = _total_size(bufs)
                _m.bytes = _nb
                _sp.bytes = _nb
            ok, per_key = await self._loop.run_in_executor(
                self._executor, self._batch_set_blocking, list(keys), list(bufs)
            )
            r.result = "ok" if ok else "partial_fail"
            if _sp:
                _sp.hits = sum(1 for b in per_key if b)
            return ok, per_key

    async def batch_get(
        self, keys: Sequence[bytes], bufs: Sequence[memoryview]
    ) -> Tuple[bool, Optional[List[bool]], List[int]]:
        n = len(keys)
        with _push_metrics.op("get", num_keys=n) as _m, \
                _push_tracing.span("batch_get", n) as _sp, \
                access_log("native.batch_get",
                           lambda: f"{n} keys, {_total_size(bufs)} bytes") as r:
            ok, per_key, lengths = await self._loop.run_in_executor(
                self._executor, self._batch_get_blocking, list(keys), list(bufs)
            )
            hits = sum(1 for b in per_key if b)
            if _m or _sp:
                _nb = sum(lengths)
                _m.bytes = _nb
                _sp.bytes = _nb
                _sp.hits = hits
            r.result = f"hits={hits}/{n}"
            return ok, per_key, lengths

    async def batch_exists(
        self, keys: Sequence[bytes]
    ) -> Optional[List[bool]]:
        n = len(keys)
        with _push_metrics.op("exist", num_keys=n), \
                _push_tracing.span("batch_exists", n) as _sp, \
                access_log("native.batch_exists", lambda: f"{n} keys") as r:
            per_key = await self._loop.run_in_executor(
                self._executor, self._batch_exists_blocking, list(keys)
            )
            hits = sum(1 for b in per_key if b)
            r.result = f"hits={hits}/{n}"
            if _sp:
                _sp.hits = hits
            return per_key

    def supports_remove(self) -> bool:
        """True iff the loaded libdfkv.so exposes the remove RPC (additive)."""
        return hasattr(self._lib, "dfkv_batch_remove")

    async def batch_remove(
        self, keys: Sequence[bytes]
    ) -> List[bool]:
        """Drop keys from the ring. per_key[i] = the owning node confirmed the op
        (removed or already absent). Raises if the lib has no remove RPC."""
        n = len(keys)
        if not self.supports_remove():
            raise RuntimeError(
                "libdfkv.so has no dfkv_batch_remove (rebuild dfkv with the "
                "remove RPC to enable L2 eviction)"
            )
        with _push_metrics.op("remove", num_keys=n), \
                _push_tracing.span("batch_remove", n) as _sp, \
                access_log("native.batch_remove", lambda: f"{n} keys") as r:
            per_key = await self._loop.run_in_executor(
                self._executor, self._batch_remove_blocking, list(keys)
            )
            ok = sum(1 for b in per_key if b)
            r.result = f"ok={ok}/{n}"
            if _sp:
                _sp.hits = ok
            return per_key

    def exists_sync(self, key: bytes) -> bool:
        with access_log("native.exists_sync", lambda: _key_label(key)) as r:
            key_ptr, key_owner = make_key_buffer(key)
            found = self._lib.dfkv_exist(
                self._h, key_ptr, c_uint64(len(key))) == 1
            del key_owner
            r.result = "found" if found else "not_found"
            return found

    def remove_sync(self, key: bytes) -> bool:
        """Drop one key synchronously. True iff the owning node confirmed the op
        (removed or already absent). Raises if the lib has no remove RPC."""
        if not self.supports_remove():
            raise RuntimeError("libdfkv.so has no dfkv_remove")
        with access_log("native.remove_sync", lambda: _key_label(key)) as r:
            key_ptr, key_owner = make_key_buffer(key)
            ok = self._lib.dfkv_remove(
                self._h, key_ptr, c_uint64(len(key))) == 1
            del key_owner
            r.result = "ok" if ok else "fail"
            return ok

    def ping_sync(self) -> None:
        """Cheap liveness check — connectivity is established at construction
        (dfkv_open + MDS discovery). Just verify the handle is live."""
        with access_log("native.ping_sync", lambda: ""):
            if self._closed or not self._h:
                raise RuntimeError("dfkv client is closed")

    def close(self) -> None:
        if self._closed:
            return
        try:
            _hot_config.stop()
        except Exception:  # pragma: no cover
            pass
        with access_log("native.close", lambda: ""):
            self._closed = True
            try:
                self._executor.shutdown(wait=True)
            except Exception:  # pragma: no cover
                pass
            try:
                if self._h:
                    self._lib.dfkv_close(self._h)
                    self._h = None
            except Exception as exc:  # pragma: no cover
                logger.warning("dfkv_close failed: %s", exc)
