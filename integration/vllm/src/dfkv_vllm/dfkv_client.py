"""Device-pointer dfkv client for the vLLM connector.

Every value buffer is a raw integer GPU pointer pre-registered through
``register_memory``. The native ABI stores opaque bytes and returns authoritative
stored lengths separately; it never dereferences device memory on the CPU.
"""
import ctypes
import os
import threading
from typing import Optional, Sequence
from dfkv_common import make_client_options_v2, make_key_array

from ._cabi import load_lib, native_version
from .client_stats import ClientStatsPoller, read_snapshot
from . import access_log as _alog
from .access_log import access_log
from . import _hot_config
from ._telemetry import config as _tcfg
from ._telemetry import metrics as _push_metrics
from ._telemetry import tracing as _push_tracing

c_void_p = ctypes.c_void_p
c_uint64 = ctypes.c_uint64
c_int = ctypes.c_int



def _env_rank() -> int:
    """Best-effort TP/worker rank for the connector_id label (each rank is its own
    process, so host:pid is already unique; this just adds readable detail)."""
    for k in ("DFKV_TP_RANK", "LOCAL_RANK", "RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


class DfkvDeviceClient:
    """Thin device-pointer wrapper over libdfkv.so for the vLLM connector."""

    def __init__(
        self,
        members: str = "",
        key_namespace: bytes = b"",
        lib_path: Optional[str] = None,
        batch_concurrency: int = 0,
        mds_endpoints: str = "",
        mds_group: str = "default",
        mds_poll_ms: int = 3000,
        client_register: bool = True,
        client_id: str = "",
        client_info: str = "",
        client_heartbeat_ms: int = 10000,
    ):
        if not key_namespace:
            raise ValueError("DfkvDeviceClient requires a non-empty key namespace")
        if not members and not mds_endpoints:
            raise ValueError(
                "DfkvDeviceClient needs 'mds_endpoints' (MDS discovery) or "
                "'members' (static list)"
            )
        self._close_lock = threading.Lock()
        self._lib = load_lib(lib_path)
        # ABI v2 constructs one fully configured handle: static membership or
        # MDS discovery, batch fan-out, and optional client registration become
        # visible together instead of mutating a partially initialized client.
        self._key_namespace = bytes(key_namespace)
        options = make_client_options_v2(
            self._key_namespace,
            members=members,
            batch_concurrency=batch_concurrency,
            mds_endpoints=mds_endpoints,
            mds_group=mds_group,
            mds_poll_ms=mds_poll_ms,
            register_client=bool(
                client_register and client_id and mds_endpoints),
            client_id=client_id,
            client_info=client_info,
            client_heartbeat_ms=client_heartbeat_ms,
        )
        self._h = self._lib.dfkv_open_v2(ctypes.byref(options))
        if not self._h:
            raise RuntimeError(
                "dfkv_open_v2 failed "
                f"(members={members!r}, mds={mds_endpoints!r})"
            )
        try:
            mode = self._lib.dfkv_transport_mode(self._h)
        except Exception as exc:
            rejected_handle = self._h
            self._h = None
            self._lib.dfkv_close(rejected_handle)
            raise RuntimeError(
                "DfkvDeviceClient requires GPUDirect RDMA transport, but "
                "dfkv_transport_mode failed; set DFKV_RDMA=1 and configure "
                "a usable RDMA device"
            ) from exc
        if mode != b"rdma":
            rejected_handle = self._h
            self._h = None
            self._lib.dfkv_close(rejected_handle)
            raise RuntimeError(
                "DfkvDeviceClient requires GPUDirect RDMA transport "
                f"(dfkv_transport_mode returned {mode!r}); "
                "set DFKV_RDMA=1 and configure a usable RDMA device"
            )
        # Unified fleet metrics pushed over OTLP to the central Collector (off by
        # default; zero cost when DFKV_METRICS_ENABLED is unset). Env-driven so it
        # needs no connector plumbing: set DFKV_METRICS_ENABLED + OTEL_*. Report
        # both the connector package version and the linked libdfkv.so version so
        # the dashboard can spot version skew across the fleet.
        _push_metrics.configure(
            {}, connector_type=_tcfg.TYPE_VLLM, tp_rank=_env_rank(),
            version=_tcfg.dist_version("dfkv-vllm"),
            native_version=native_version(self._lib))
        # Connector-side request tracing (off by default; zero cost when off).
        # Spans for slow / sampled / failed ops pushed over OTLP /v1/traces.
        _push_tracing.configure(
            {}, connector_type=_tcfg.TYPE_VLLM, tp_rank=_env_rank(),
            version=_tcfg.dist_version("dfkv-vllm"),
            native_version=native_version(self._lib))
        # Access log: parse the launch baseline (env-driven), then let a control
        # file toggle it at runtime without restarting vLLM (opt-in via
        # DFKV_HOT_CONFIG). See docs/access_log.md -> 运行时热开关.
        _alog.configure({}, tp_rank=_env_rank())
        _hot_config.register("access_log", _alog.apply_hot)
        _hot_config.start({}, tp_rank=_env_rank())
        # Surface the C client's ring/MDS health on Prometheus so an empty ring
        # (writes routed nowhere) or an unreachable MDS is visible on the scrape,
        # not just in the client log. Sleeping daemon thread off the request path;
        # DFKV_CLIENT_STATS_POLL_S=0 disables. Mirrors the HiCache stats poller.
        self._stats_poller = None
        try:
            poll_s = float(os.environ.get("DFKV_CLIENT_STATS_POLL_S", 15))
        except (TypeError, ValueError):
            poll_s = 15.0
        if poll_s > 0:
            self._stats_poller = ClientStatsPoller(
                lambda: read_snapshot(self._lib, self._h), _env_rank(),
                transport=self.transport_mode, interval_s=poll_s)
            self._stats_poller.start()

    @property
    def transport_mode(self) -> str:
        return self._lib.dfkv_transport_mode(self._h).decode()

    def max_sg_segs(self) -> int:
        """Return the negotiated RDMA payload-SGE width for one key.

        Older native libraries may lack this export; the caller retains its
        historical width fallback for key-geometry stability."""
        return int(self._lib.dfkv_max_sg_segs(self._h))

    def register_memory(self, base: int, size: int) -> None:
        """Register a (host or GPU device) region as an RDMA MR. One call per
        contiguous KV-cache storage region; later put/get reference offsets."""
        with access_log("register_memory", lambda: f"base={base:#x}, {size} bytes"):
            rc = self._lib.dfkv_register_memory(self._h, c_void_p(base), c_uint64(size))
            if rc != 0:
                raise RuntimeError(
                    f"dfkv_register_memory(base={base:#x}, size={size}) rc={rc}"
                )

    def batch_put(
        self, keys: Sequence[bytes], ptrs: Sequence[int], sizes: Sequence[int]
    ) -> list:
        """Store ``keys[i]`` from buffer ``ptrs[i]`` (device or host) of
        ``sizes[i]`` bytes. Returns per-key status (``0`` = ok, ``1`` = failed).

        NOTE: the dfkv C ABI's ``dfkv_batch_put`` sets ``out[i]=1`` for SUCCESS
        and ``0`` for failure (matching batch_get/batch_exist, but the OPPOSITE
        of the single ``dfkv_put`` which returns 0=ok). We normalize here to the
        connector's "0 = ok" convention so callers don't have to know this."""
        n = len(keys)
        with _push_metrics.op("put", num_keys=n) as _m, \
                _push_tracing.span("batch_put", n) as _sp, \
                access_log("batch_put", lambda: f"{n} keys, {sum(sizes)} bytes") as r:
            if _m or _sp:
                _nb = sum(sizes)
                _m.bytes = _nb
                _sp.bytes = _nb
            karr, klens, key_owners = make_key_array(keys)
            parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
            sarr = (c_uint64 * n)(*sizes)
            out = (c_int * n)()
            rc = self._lib.dfkv_batch_put(
                self._h, karr, klens, parr, sarr, n, out)
            if rc != 0:
                raise RuntimeError(f"dfkv_batch_put rc={rc}")
            res = [0 if ok == 1 else 1 for ok in out]
            ok_n = res.count(0)
            r.result = f"ok={ok_n}/{n}"
            if _sp:
                _sp.hits = ok_n
            del key_owners
            return res

    def batch_get(
        self, keys: Sequence[bytes], ptrs: Sequence[int], caps: Sequence[int]
    ):
        """Load ``keys[i]`` into buffer ``ptrs[i]`` (device or host, capacity
        ``caps[i]``). Returns ``(hits, lengths)``: ``hits[i]`` is 1 on hit, 0 on
        miss; ``lengths[i]`` is the true stored byte length on hit."""
        n = len(keys)
        with _push_metrics.op("get", num_keys=n) as _m, \
                _push_tracing.span("batch_get", n) as _sp, \
                access_log("batch_get_auto", lambda: f"{n} keys") as r:
            karr, klens, key_owners = make_key_array(keys)
            parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
            carr = (c_uint64 * n)(*caps)
            out_hit = (c_int * n)()
            out_len = (c_uint64 * n)()
            rc = self._lib.dfkv_batch_get_auto(
                self._h, karr, klens, parr, carr, n, out_hit, out_len
            )
            if rc != 0:
                raise RuntimeError(f"dfkv_batch_get_auto rc={rc}")
            hits, lens = list(out_hit), list(out_len)
            nhit, nbytes = sum(hits), sum(lens)
            if _m:
                _m.bytes = nbytes
            if _sp:
                _sp.hits = nhit; _sp.bytes = nbytes
            r.result = f"hits={nhit}/{n}, {nbytes} bytes"
            del key_owners
            return hits, lens

    def batch_put_sg(
        self,
        keys: Sequence[bytes],
        seg_ptrs: Sequence[Sequence[int]],
        seg_sizes: Sequence[Sequence[int]],
    ) -> list:
        """Scatter-gather put: ``keys[i]`` stores the in-order concatenation of
        the buffers ``seg_ptrs[i][..]`` (device pointers) of ``seg_sizes[i][..]``
        bytes, as ONE dfkv key + one RDMA multi-SGE op. Each key takes <=29 segs
        (max_sge-1); the C ABI reports a >29 key failed. Returns per-key status
        (``0`` = ok, ``1`` = failed), same "0=ok" normalization as ``batch_put``."""
        n = len(keys)
        with _push_metrics.op("put", num_keys=n) as _m, \
                _push_tracing.span("batch_put_sg", n) as _sp, \
                access_log(
                    "batch_put_sg",
                    lambda: f"{n} keys, {sum(sum(s) for s in seg_sizes)} bytes",
                ) as r:
            if _m or _sp:
                _nb = sum(sum(s) for s in seg_sizes)
                _m.bytes = _nb
                _sp.bytes = _nb
            karr, klens, key_owners = make_key_array(keys)
            # Keep the per-key inner arrays alive for the duration of the call.
            inner_p = [(c_void_p * len(p))(*[c_void_p(x) for x in p]) for p in seg_ptrs]
            inner_s = [(c_uint64 * len(s))(*s) for s in seg_sizes]
            parr = (ctypes.POINTER(c_void_p) * n)(
                *[ctypes.cast(a, ctypes.POINTER(c_void_p)) for a in inner_p]
            )
            sarr = (ctypes.POINTER(c_uint64) * n)(
                *[ctypes.cast(a, ctypes.POINTER(c_uint64)) for a in inner_s]
            )
            narr = (c_int * n)(*[len(p) for p in seg_ptrs])
            out = (c_int * n)()
            rc = self._lib.dfkv_batch_put_sg(
                self._h, karr, klens, parr, sarr, narr, n, out)
            if rc != 0:
                raise RuntimeError(f"dfkv_batch_put_sg rc={rc}")
            res = [0 if ok == 1 else 1 for ok in out]
            ok_n = res.count(0)
            r.result = f"ok={ok_n}/{n}"
            if _sp:
                _sp.hits = ok_n
            del key_owners
            return res

    def batch_get_auto_sg(
        self,
        keys: Sequence[bytes],
        seg_ptrs: Sequence[Sequence[int]],
        seg_caps: Sequence[Sequence[int]],
    ):
        """Scatter-gather get: ``keys[i]``'s stored blob is scattered in order
        across destination buffers ``seg_ptrs[i][..]`` of capacity
        ``seg_caps[i][..]`` (the segment sizes define the split). Returns
        ``(hits, lengths)`` where ``lengths[i]`` is the total stored bytes."""
        n = len(keys)
        with _push_metrics.op("get", num_keys=n) as _m, \
                _push_tracing.span("batch_get_auto_sg", n) as _sp, \
                access_log("batch_get_auto_sg", lambda: f"{n} keys") as r:
            karr, klens, key_owners = make_key_array(keys)
            inner_p = [(c_void_p * len(p))(*[c_void_p(x) for x in p]) for p in seg_ptrs]
            inner_c = [(c_uint64 * len(c))(*c) for c in seg_caps]
            parr = (ctypes.POINTER(c_void_p) * n)(
                *[ctypes.cast(a, ctypes.POINTER(c_void_p)) for a in inner_p]
            )
            carr = (ctypes.POINTER(c_uint64) * n)(
                *[ctypes.cast(a, ctypes.POINTER(c_uint64)) for a in inner_c]
            )
            narr = (c_int * n)(*[len(p) for p in seg_ptrs])
            out_hit = (c_int * n)()
            out_len = (c_uint64 * n)()
            rc = self._lib.dfkv_batch_get_auto_sg(
                self._h, karr, klens, parr, carr, narr, n, out_hit, out_len
            )
            if rc != 0:
                raise RuntimeError(f"dfkv_batch_get_auto_sg rc={rc}")
            hits, lens = list(out_hit), list(out_len)
            nhit, nbytes = sum(hits), sum(lens)
            if _m:
                _m.bytes = nbytes
            if _sp:
                _sp.hits = nhit; _sp.bytes = nbytes
            r.result = f"hits={nhit}/{n}, {nbytes} bytes"
            del key_owners
            return hits, lens

    def batch_exist(self, keys: Sequence[bytes]) -> list:
        """Return ``[1|0]`` per key (1 = present in the cache)."""
        n = len(keys)
        with _push_metrics.op("exist", num_keys=n), \
                _push_tracing.span("batch_exist", n) as _sp, \
                access_log("batch_exist", lambda: f"{n} keys") as r:
            karr, klens, key_owners = make_key_array(keys)
            out = (c_int * n)()
            rc = self._lib.dfkv_batch_exist(
                self._h, karr, klens, n, out)
            if rc != 0:
                raise RuntimeError(f"dfkv_batch_exist rc={rc}")
            res = list(out)
            present = res.count(1)
            r.result = f"present={present}/{n}"
            if _sp:
                _sp.hits = present
            del key_owners
            return res

    def close(self) -> None:
        try:
            if getattr(self, "_stats_poller", None):
                self._stats_poller.stop()
                self._stats_poller = None
        except Exception:
            pass
        try:
            _hot_config.stop()
        except Exception:
            pass
        with self._close_lock:
            handle = getattr(self, "_h", None)
            self._h = None
        if handle:
            with access_log("close", lambda: ""):
                self._lib.dfkv_close(handle)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
