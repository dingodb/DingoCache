"""DingoFS HiCache storage backend for SGLang (loaded via --hicache-storage-backend
dynamic). Zero-copy v1 path: hands raw host-buffer pointers from
mem_pool_host.get_page_buffer_meta() straight to the DingoFS KV client (C ABI).

Object keys use the shared binary, length-framed pool-key schema. MLA stores one
replicated packed-latent object per page (only TP rank 0 writes); MHA stores
separate K/V components per TP rank. Pipeline-parallel rank is always an
explicit coordinate when enabled, so stages holding different layer slices do
not collide. No delimiter rendering participates in storage identity.

This file is the production plugin. On a GPU host it imports the real SGLang
HiCacheStorage; the test harness supplies a no-torch shim with the same surface.
"""
from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import sys
import time
from typing import List, Optional

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig
from dfkv_common import (
    DfkvClientOptionsV2,
    SGLANG_HICACHE_RAW_V1,
    canonical_namespace,
    reject_namespace_override,
    make_client_options_v2,
    make_key_array,
    make_key_buffer,
    pool_key,
    sg_key,
)

from dfkv_access_log import (access_log, configure as _configure_access_log,
                            apply_hot as _access_log_apply_hot,
                            fmt_bytes as _fmt_bytes, fmt_pools as _fmt_pools,
                            fmt_pool_results as _fmt_pool_results)
import dfkv_hot_config as _hot_config
from dfkv_metrics import Metrics as _Metrics, ClientStatsPoller as _ClientStatsPoller
from dfkv_telemetry import metrics as _push_metrics, config as _tcfg
from dfkv_telemetry import tracing as _tracing

# The module carries no logger of its own historically (observability rides the
# access log / tracing spans). One is added here for the read-reject diagnostic,
# which must land in the ENGINE log next to SGLang's "0 pages verified across
# ranks" line to be readable as a pair.
_log = logging.getLogger(__name__)

_FLAG_IS_MLA = 0x1

# Historical/default width: ConnectX-era max_sge=30 and SGE[0] carries the wire
# prefix, leaving 29 payload segments. The live width comes from
# dfkv_max_sg_segs and is encoded in the binary physical key. It is therefore a
# compatibility boundary, not just a local batching knob: clients with
# different HCA limits safely miss instead of interpreting a different layer
# grouping. Mirrors the vLLM connector's default and reasoning.
SG_DEFAULT_WIDTH = 29


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)

def _physical_axis(cfg: dict, name: str) -> tuple[int, int]:
    """Parse one physical sharding axis without accepting lossy coercions."""
    size_key = f"{name}_size"
    rank_key = f"{name}_rank"

    def _integer(value, field):
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text and text.isascii() and text.isdecimal():
                return int(text)
        raise ValueError(f"{field} must be an integer")

    size = _integer(cfg.get(size_key, 1), size_key)
    if size < 1 or size > 0xFFFFFFFF:
        raise ValueError(f"{size_key} must be in [1, 4294967295]")
    if rank_key not in cfg:
        if size > 1:
            raise ValueError(
                f"{rank_key} is required when {size_key} is greater than 1")
        return size, 0
    rank = _integer(cfg[rank_key], rank_key)
    if rank < 0 or rank >= size:
        raise ValueError(f"{rank_key} must be in [0, {size})")
    return size, rank


def resolve_node_dedup(cfg_value, env_value, is_mla: bool, tp_size: int):
    """Decide DFKV_CLIENT_NODE_DEDUP: (value_to_set | None, auto_enabled).

    Precedence: explicit extra-config beats the env, the env beats the
    default. The default is OFF (R2 A/B verdict, 2026-07-17): on the full
    GLM-5.2 100k-token workload the shm rendezvous inverted the L3 value —
    dedup-on hot rounds ran +19-36% TTFT WORSE than cold with 22s p99 batch
    tails, while dedup-off hot rounds beat cold by 43% TTFT / +50-71%
    throughput. The 8x direct-read amplification is cheaper than the
    rendezvous coordination (publish copies through the 512MiB arena +
    wait-timeout fallbacks) at inference batch shapes; the v1.26.0
    conditional auto-on was validated only on microbenchmarks. Rendezvous
    stays available explicitly (node_dedup=1 / DFKV_CLIENT_NODE_DEDUP=1)
    for fabric-constrained multi-node rings until the publish path is
    redesigned (zero-copy publish, per-key streaming). Pure function so the
    policy is unit-testable.
    """
    if cfg_value is not None:
        return ("1" if _truthy(cfg_value) else "0"), False
    if env_value is not None:
        return None, False  # operator's env stands as-is
    return None, False


def _load_lib(path: Optional[str] = None) -> ctypes.CDLL:
    build = os.environ.get("DFKV_BUILD")
    lib_path = (
        path
        or os.environ.get("DFKV_LIB")
        or (os.path.join(build, "libdfkv.so") if build else "libdfkv.so")
    )
    lib = ctypes.CDLL(lib_path)
    lib.dfkv_open_v2.restype = ctypes.c_void_p
    lib.dfkv_open_v2.argtypes = [ctypes.POINTER(DfkvClientOptionsV2)]
    lib.dfkv_put.restype = ctypes.c_int
    lib.dfkv_put.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
        ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_get.restype = ctypes.c_int
    lib.dfkv_get.argtypes = lib.dfkv_put.argtypes
    lib.dfkv_exist.restype = ctypes.c_int
    lib.dfkv_exist.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_register_memory.restype = ctypes.c_int
    lib.dfkv_register_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_batch_put.restype = ctypes.c_int
    lib.dfkv_batch_put.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int)]
    lib.dfkv_batch_get.restype = ctypes.c_int
    lib.dfkv_batch_get.argtypes = lib.dfkv_batch_put.argtypes
    lib.dfkv_batch_exist.restype = ctypes.c_int
    lib.dfkv_batch_exist.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int)]
    # Scatter-gather put + live SG width, for the HiCache L2-bypass (device-direct
    # write) path. Each key gathers num_bufs[i] non-contiguous source buffers
    # (a page's per-layer GPU segments) into one stored blob. Guarded like the
    # vLLM connector: older libdfkv.so lacks the symbols and callers fall back.
    if hasattr(lib, "dfkv_batch_put_sg"):
        lib.dfkv_batch_put_sg.restype = ctypes.c_int
        lib.dfkv_batch_put_sg.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int)]
    # Scatter-gather GET, the read-side mirror of dfkv_batch_put_sg for the
    # HiCache L2-bypass (device-direct read) path. Key i's stored blob is
    # scattered in order across num_dsts[i] destination buffers (a page's
    # per-layer GPU segments); out_hit[i]==1 on hit, out_len[i]=total stored
    # bytes. Same symbol the vLLM connector's load path uses (see
    # integration/vllm/src/dfkv_vllm/_cabi.py). Guarded: older libdfkv.so lacks
    # it and supports_device_transfer() declines the bypass path.
    if hasattr(lib, "dfkv_batch_get_auto_sg"):
        lib.dfkv_batch_get_auto_sg.restype = ctypes.c_int
        lib.dfkv_batch_get_auto_sg.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)]
    if hasattr(lib, "dfkv_max_sg_segs"):
        lib.dfkv_max_sg_segs.restype = ctypes.c_uint32
        lib.dfkv_max_sg_segs.argtypes = [ctypes.c_void_p]
    lib.dfkv_transport_mode.restype = ctypes.c_char_p
    lib.dfkv_transport_mode.argtypes = [ctypes.c_void_p]
    lib.dfkv_stats_snapshot.restype = ctypes.c_uint64
    lib.dfkv_stats_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64]
    lib.dfkv_version.restype = ctypes.c_char_p
    lib.dfkv_version.argtypes = []
    lib.dfkv_close.restype = None
    lib.dfkv_close.argtypes = [ctypes.c_void_p]
    return lib


def _native_version(lib) -> str:
    """libdfkv.so version via the C dfkv_version(); "" if the symbol is missing
    (older lib) or the call fails. Never raises."""
    try:
        v = lib.dfkv_version()
        return v.decode("utf-8", "replace") if v else ""
    except Exception:
        return ""


def _read_snapshot(lib, h) -> str:
    """Read the C client's Prometheus metrics snapshot (size query, then fetch)."""
    need = int(lib.dfkv_stats_snapshot(h, None, 0))
    if need <= 0:
        return ""
    buf = ctypes.create_string_buffer(need + 1)
    lib.dfkv_stats_snapshot(h, buf, need + 1)
    return buf.value.decode("utf-8", "replace")


# One-byte marker for the logical-anchor "kv" pool of V4/DSA models. Native
# dfkv rejects zero-length PUTs because they allocate objects without consuming
# byte quota; charging one byte preserves the existence-marker semantics.
_MARKER_BUF = ctypes.create_string_buffer(1)
_MARKER_PTR = ctypes.addressof(_MARKER_BUF)


def _is_device_transfer(tr) -> bool:
    """A sidecar PoolTransfer is DEVICE-direct (task 4) when it carries device slot
    indices and NO host indices — the indexer RDMAs from/into its GPU buffer. A host
    transfer (host_indices set) keeps the stock host v2 path."""
    return (getattr(tr, "host_indices", None) is None
            and getattr(tr, "device_indices", None) is not None)


def _arrays(subkeys, ptrs, sizes):
    """Build aligned key/length/payload arrays for one batch call."""
    n = len(subkeys)
    karr, klens, key_owners = make_key_array(subkeys)
    parr = (ctypes.c_void_p * n)(*[ctypes.c_void_p(int(p)) for p in ptrs])
    sarr = (ctypes.c_uint64 * n)(*[int(s) for s in sizes])
    out = (ctypes.c_int * n)()
    return karr, klens, parr, sarr, out, key_owners


def _key_label(key: bytes) -> str:
    return f"len={len(key)} sha256={hashlib.sha256(key).hexdigest()[:16]}"


class DfkvHiCache(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: Optional[dict] = None):
        cfg = (kwargs or {}) or (getattr(storage_config, "extra_config", None) or {})
        self.cfg = cfg
        # Enforce the zero-copy deploy contract. SGLang's cache_controller only
        # selects the zero-copy v1 path (batch_set_v1/batch_get_v1) for a
        # 'dynamic' backend when extra_config.interface_v1 is truthy; otherwise it
        # uses the generic set/get copy path. The generic path is implemented and
        # correct, but slower (extra host copies) and for MLA every TP rank writes
        # the page redundantly. Fail fast so a launch-config omission can't quietly
        # degrade to the copy path.
        if not cfg.get("interface_v1"):
            raise ValueError(
                "dfkv requires extra_config interface_v1=1 to select the zero-copy "
                "v1 RDMA path. Omitting it falls back to the generic copy path "
                "(slower; MLA writes redundant per-rank copies)."
            )
        self.model_identity = storage_config.model_name or ""
        self.model = self.model_identity.replace("/", "-")
        self.tp_rank = int(storage_config.tp_rank)
        self.tp_size = int(storage_config.tp_size)
        self.is_mla = bool(storage_config.is_mla_model)
        # PCP and DCP split one logical page into distinct physical shards.
        # A guessed rank would alias those shards, so multi-rank axes require an
        # explicit bounded coordinate before the native client is opened.
        self.pcp_size, self.pcp_rank = _physical_axis(cfg, "pcp")
        self.dcp_size, self.dcp_rank = _physical_axis(cfg, "dcp")
        self._device_direct_requested = _truthy(
            os.environ.get("SGLANG_HICACHE_L2_BYPASS"))
        self._device_registration_failed = False
        # Pipeline-parallel rank/size. PP splits the model across stages by
        # layer, so each pp_rank holds a *different* slice of KV (unlike TP,
        # where MLA latent is replicated). Storage keys MUST therefore carry
        # pp_rank when pp_size > 1, or PP stages overwrite each other's pages.
        # Mirrors SGLang's reference HiCacheFile suffix (hicache_storage.py:
        # `if enable_pp: config_suffix += f"_{pp_size}_{pp_rank}"`, applied
        # unconditionally — including MLA models).
        self.pp_rank = int(getattr(storage_config, "pp_rank", 0))
        self.pp_size = int(getattr(storage_config, "pp_size", 1))
        self.enable_pp = self.pp_size > 1
        # One-shot guard for the logical-anchor notice (V4/DSA models whose
        # primary "kv" pool holds no host buffer — see _note_logical_anchor_once).
        self._anchor_noop_warned = False
        # Access log: idempotent per process (first instance wins). Configured
        # here so tp_rank/model are available for the {rank} path placeholder.
        _configure_access_log(cfg, tp_rank=self.tp_rank, model=self.model)
        # Hot-reload: a background watcher lets ops flip access_log (and future
        # observability knobs) live via a control file — no SGLang restart. Only
        # zero-correctness-impact knobs are hot; structural/native knobs are not.
        _hot_config.register("access_log", _access_log_apply_hot)
        _hot_config.start(cfg, tp_rank=self.tp_rank)
        self._alog_tag = f"r{self.tp_rank}"
        # Client-side read/write counters (Prometheus when available). Lets ops
        # confirm SGLang->dfkv volume from /metrics instead of parsing access logs.
        self._metrics = _Metrics(self.tp_rank)
        # Load the native lib up front so its version can ride on the fleet-metrics
        # resource (dfkv_native_version). The actual client (dfkv_open) is created
        # later, inside the access-log "init" block. The plugin ships with the dfkv
        # repo, so its package version == the native lib version.
        self._lib = _load_lib(cfg.get("lib_path"))
        native_ver = _native_version(self._lib)
        # Unified fleet metrics pushed over OTLP to the central Collector (off by
        # default; zero cost when DFKV_METRICS_ENABLED is unset). Powers the
        # cross-connector global dashboard (per connector_id / type).
        _push_metrics.configure(cfg, connector_type=_tcfg.TYPE_HICACHE,
                                tp_rank=self.tp_rank, model=self.model,
                                version=native_ver, native_version=native_ver)
        # Connector-side request tracing (off by default; zero cost when off).
        # Emits a span per op for slow / sampled / failed requests over OTLP
        # /v1/traces — same identity + endpoint as the fleet metrics above.
        _tracing.configure(cfg, connector_type=_tcfg.TYPE_HICACHE,
                           tp_rank=self.tp_rank, model=self.model,
                           version=native_ver, native_version=native_ver)
        # When metrics are on, enable the C client's active per-peer latency probe
        # (read at dfkv_open below) so every cache node shows avg/max latency even
        # when idle. extra_config 'probe_interval_ms' overrides; 0 disables.
        if _push_metrics.is_enabled():
            probe_ms = int(float(_tcfg.resolve(
                cfg, "probe_interval_ms", _tcfg.ENV_PROBE_INTERVAL_MS, 5000)))
            os.environ["DFKV_PROBE_INTERVAL_MS"] = str(probe_ms)
        # Log the open/discovery setup (the access log is live from here on; the
        # earlier interface_v1 check raises before config is resolved).
        with access_log("init", lambda: (
                f"{self._alog_tag} {self.model} "
                f"tp={self.tp_rank}/{self.tp_size} "
                f"pcp={self.pcp_rank}/{self.pcp_size} "
                f"dcp={self.dcp_rank}/{self.dcp_size} "
                f"mla={int(self.is_mla)}")) as r:
            mds = cfg.get("mds_endpoints", "")
            members = cfg.get("members", "")
            _tcfg.require_ring_endpoint(members, mds)
            # RDMA write pipelining: depth>1 keeps multiple PUTs in flight on one
            # connection, hiding per-op latency (single-rank MLA writes are
            # latency-bound). The C client reads DFKV_RDMA_DEPTH when it builds the
            # transport inside dfkv_open below, so set it from extra_config first.
            # NOTE: the dfkv_server must set the SAME (or larger) DFKV_RDMA_DEPTH in
            # its own env -- client depth must be <= server depth.
            if cfg.get("rdma_depth"):
                os.environ["DFKV_RDMA_DEPTH"] = str(int(cfg["rdma_depth"]))
            if _truthy(cfg.get("require_rdma")):
                if not _truthy(os.environ.get("DFKV_RDMA")):
                    os.environ["DFKV_RDMA"] = "1"
            # rail_affinity (per-tp_rank narrowing) is DEPRECATED and now a no-op:
            # it keyed off tp_rank, which is always 0 under DP-attention (every rank
            # is its own attention TP group of size 1), so it collapsed all ranks to
            # one rail. NUMA-aware rail selection now lives in the C++ client: keep
            # the full multi-rail DFKV_RDMA_DEV and set DFKV_RDMA_NUMA=1, and the
            # client picks a NUMA-local rail per connection (works for TP and DP).
            if cfg.get("rail_affinity"):
                import sys as _sys
                print("[dfkv] WARNING: 'rail_affinity' is deprecated and ignored; "
                      "set DFKV_RDMA_NUMA=1 + multi-rail DFKV_RDMA_DEV for NUMA-aware "
                      "rail selection in the client.", file=_sys.stderr, flush=True)
            if cfg.get("rdma_numa"):
                os.environ.setdefault("DFKV_RDMA_NUMA", "1")
            # Same-host GET rendezvous (phase 5): dedups TP-replicated L3 loads
            # across the rank processes of one node (HiCache destinations are
            # HOST memory — the host flavor applies). Default OFF since the R2
            # A/B (2026-07-17): at inference batch shapes the rendezvous
            # inverted the L3 benefit (see resolve_node_dedup docstring for
            # the measured numbers); direct 8x reads are cheaper until the
            # publish path is redesigned. Explicit settings always win:
            # extra-config `node_dedup` beats the env; "1" opts back in.
            _dedup, _auto = resolve_node_dedup(
                cfg.get("node_dedup"), os.environ.get("DFKV_CLIENT_NODE_DEDUP"),
                self.is_mla, self.tp_size)
            if _dedup is not None:
                os.environ["DFKV_CLIENT_NODE_DEDUP"] = _dedup
            if _auto:
                print(f"[dfkv] node-dedup auto-enabled (mla, tp={self.tp_size}): "
                      "same-host rendezvous collapses replicated L3 loads; "
                      "set node_dedup=0 or DFKV_CLIENT_NODE_DEDUP=0 to disable.",
                      file=sys.stderr, flush=True)
            # Phase 9: exist gate on the backup path. Recomputed pages whose
            # KV is ALREADY in L3 were re-backed wholesale (measured 485 GB
            # per hot round on the canary — write pressure that starved the
            # very reads the cache exists for). One batch_exist (collapsed by
            # the rendezvous) filters them out. backup_exist_gate=0 disables.
            self._backup_exist_gate = _truthy(
                cfg.get("backup_exist_gate",
                        os.environ.get("DFKV_BACKUP_EXIST_GATE", "1")))
            self._put_retry_recovered = 0  # last _put_flat's retry recoveries
            # last _v2_io()'s transferred bytes + I/O seconds, read by
            # batch_set_v2/batch_get_v2 to feed the v2 metrics (mirrors the
            # _put_retry_recovered instance-state handoff pattern).
            self._v2_io_bytes = 0
            self._v2_io_seconds = 0.0
            self._v2_io_pools = ()  # pool names that actually did I/O last _v2_io()
            # self._lib was loaded above (before configure) so the native version
            # could be reported; dfkv_open uses that same handle here.
            # Namespace aliases are not configurable: this binary identity is
            # the payload type boundary and must follow the connector schema.
            reject_namespace_override(cfg)
            _tcfg.require_isolation_name(
                self.model_identity, field="model_name")
            _page_size = int(cfg.get("page_size", 64))
            _layer_num = int(cfg.get("layer_num", 0))
            _dtype = str(cfg.get(
                "kv_cache_dtype", cfg.get("dtype_tag", "unknown")))
            self._key_namespace = canonical_namespace(
                self.model_identity,
                SGLANG_HICACHE_RAW_V1,
                tenant_id=str(cfg.get("tenant_id", "default")),
                model_revision=str(
                    cfg.get("model_revision", self.model_identity)),
                dtype=_dtype,
                block_tokens=max(1, _page_size),
                layer_count=max(1, _layer_num),
                tp_size=self.tp_size,
                dp_size=max(1, int(cfg.get("dp_size", 1))),
                pp_size=self.pp_size,
                layout_fields={
                    "page_size": _page_size,
                    "dtype": _dtype,
                    "dtype_tag": int(cfg.get("dtype_tag", 0)),
                    "is_mla": self.is_mla,
                    "layer_num": _layer_num,
                    "head_num": int(cfg.get("head_num", 0)),
                    "head_dim": int(cfg.get("head_dim", 0)),
                    "pcp_size": self.pcp_size,
                    "dcp_size": self.dcp_size,
                },
            )
            group = cfg.get("mds_group", "default")
            poll_ms = int(cfg.get("mds_poll_ms", 3000))
            register_client = bool(
                mds and _truthy(
                    cfg.get(
                        "client_register",
                        os.environ.get("DFKV_CLIENT_REGISTER", "1"),
                    )
                )
            )
            client_id = (
                _tcfg.resolve_connector_id(cfg, tp_rank=self.tp_rank)
                if register_client
                else ""
            )
            client_info = (
                f"type={_tcfg.TYPE_HICACHE},model={self.model},"
                f"tp_size={self.tp_size},tp_rank={self.tp_rank},"
                f"pcp_size={self.pcp_size},pcp_rank={self.pcp_rank},"
                f"dcp_size={self.dcp_size},dcp_rank={self.dcp_rank},"
                f"ver={native_ver}"
                if register_client
                else ""
            )
            options = make_client_options_v2(
                self._key_namespace,
                members=members,
                batch_concurrency=int(cfg.get("batch_concurrency", 8)),
                mds_endpoints=mds,
                mds_group=group,
                mds_poll_ms=poll_ms,
                register_client=register_client,
                client_id=client_id,
                client_info=client_info,
                client_heartbeat_ms=10000,
            )
            self._h = self._lib.dfkv_open_v2(ctypes.byref(options))
            if not self._h:
                raise RuntimeError("dfkv_open_v2 failed")
            # (base, size) of host regions already handed to dfkv_register_memory,
            # so a buffer shared across multiple hybrid pools (DSA registers the
            # KV anchor plus several sidecar pools) is registered exactly once.
            self._registered_regions = set()
            mode_b = self._lib.dfkv_transport_mode(self._h)
            self.transport_mode = (
                mode_b.decode("utf-8", errors="replace") if mode_b else "unknown"
            )
            if _truthy(cfg.get("require_rdma")):
                if self.transport_mode != "rdma":
                    self._lib.dfkv_close(self._h)
                    self._h = None
                    raise RuntimeError(
                        "dfkv requires RDMA zero-copy transport, "
                        f"got {self.transport_mode}"
                    )
            if (self._device_direct_requested
                    and not self.supports_device_transfer()):
                self._lib.dfkv_close(self._h)
                self._h = None
                raise RuntimeError(
                    "SGLANG_HICACHE_L2_BYPASS=1 requires RDMA, device SG "
                    "put/get support, a positive layer_num, and a valid SG width")
            self.mem_pool_host = None
            mode = f" transport={self.transport_mode}"
            r.result = ("ok mds-discovery" if mds else "ok static") + mode

        # Background mirror of the C client's metrics snapshot onto Prometheus
        # (sleeping daemon thread; off the request path). client_stats_poll_s<=0
        # disables it. extra_config wins, then env, then 10s default.
        self._poller = None
        try:
            poll_s = cfg.get("client_stats_poll_s")
            if poll_s is None:
                poll_s = os.environ.get("DFKV_CLIENT_STATS_POLL_S", 10)
            poll_s = float(poll_s)
        except (TypeError, ValueError):
            poll_s = 10.0
        if poll_s > 0:
            self._poller = _ClientStatsPoller(
                lambda: _read_snapshot(self._lib, self._h), self.tp_rank, poll_s)
            self._poller.start()
        # Per-cache-node latency: parse the same snapshot and push avg/max per peer
        # over OTLP (no-op unless metrics enabled). Off the request path.
        peer_poll_s = float(_tcfg.resolve(
            cfg, "peer_latency_poll_s", _tcfg.ENV_PEER_POLL_S,
            poll_s if poll_s > 0 else 10.0))
        self._peer_lat_poller = _push_metrics.start_peer_latency_poller(
            lambda: _read_snapshot(self._lib, self._h), peer_poll_s)

    def __del__(self):
        try:
            _hot_config.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_poller", None):
                self._poller.stop()
                self._poller = None
        except Exception:
            pass
        try:
            if getattr(self, "_peer_lat_poller", None):
                self._peer_lat_poller.stop()
                self._peer_lat_poller = None
        except Exception:
            pass
        try:
            if getattr(self, "_h", None):
                self._lib.dfkv_close(self._h)
                self._h = None
        except Exception:
            pass

    def register_memory(self, base: int, size: int) -> bool:
        """Register one host/device MR, returning False on native rejection."""
        if not base or not size:
            return False
        rc = self._lib.dfkv_register_memory(
            self._h, ctypes.c_void_p(int(base)), ctypes.c_uint64(int(size)))
        if rc != 0:
            _log.warning(
                "dfkv_register_memory rejected base=%#x size=%d rc=%d",
                int(base), int(size), rc)
            return False
        return True

    def _pool_backing_tensors(self, pool):
        """Yield a host pool's backing tensor(s) for RDMA registration.

        Preference order mirrors SGLang's own Mooncake backend
        (_iter_host_pool_buffers): a hybrid/DSA pool self-reports its physical
        tensors via get_hybrid_pool_buffer(), because they do NOT all live under
        the same attribute name -- DeepSeekV4PagedHostPool/DeepSeekV4StateHostPool
        expose a list under `kv_buffer`, but DSAIndexerPoolHost keeps its buffer in
        `index_k_with_scale_buffer`, and MambaPoolHost in yet others. Probing a
        fixed attribute list (the old behaviour) missed those and logged
        "no backing buffer found". Simple KV pools with no accessor fall back to
        the classic attribute probe."""
        get_buffers = getattr(pool, "get_hybrid_pool_buffer", None)
        if callable(get_buffers):
            for buf in get_buffers() or ():
                if buf is not None:
                    yield buf
            return
        for attr in ("kv_buffer", "host_kv_buffer", "data_buffer", "buffer", "data"):
            buf = getattr(pool, attr, None)
            if buf is None:
                continue
            for t in (buf if isinstance(buf, (list, tuple)) else [buf]):
                if t is not None:
                    yield t
            return  # first attribute that yielded buffers wins

    def _register_pool_buffers(self, pool) -> int:
        """Best-effort: register a host pool's backing buffer(s) so its pages
        transfer zero-copy without per-op MR registration. Registers EVERY tensor
        the pool exposes (a DSA multi-pool model has several), deduping regions
        already registered. Failure is non-fatal (pages just fall back to ad-hoc
        per-buffer registration). Returns #regions newly registered."""
        if not hasattr(self, "_registered_regions"):
            self._registered_regions = set()
        done = 0
        for t in self._pool_backing_tensors(pool):
            if not (hasattr(t, "data_ptr") and hasattr(t, "numel")
                    and hasattr(t, "element_size")):
                continue
            try:
                base = int(t.data_ptr())
                size = int(t.numel()) * int(t.element_size())
                region = (base, size)
                if not base or not size or region in self._registered_regions:
                    continue
                if self.register_memory(base, size):
                    self._registered_regions.add(region)
                    done += 1
            except Exception:
                pass
        return done

    def register_mem_pool_host(self, mem_pool_host):
        self.mem_pool_host = mem_pool_host
        with access_log("register_mem_pool_host",
                        lambda: f"{self._alog_tag}") as r:
            n = 0
            try:
                n = self._register_pool_buffers(mem_pool_host)
            except Exception as e:  # never fail HiCache setup over an optimization
                r.result = f"skip ({type(e).__name__})"
                return
            r.result = f"registered {n} region(s)" if n else "no backing buffer found"

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        name = str(host_pool_name)
        # Per-page component count for this pool. Multi-pool layouts are never
        # guessed: a bad probe would otherwise collapse independent components
        # into one replicated "all" object and serve corrupt/incomplete state.
        if not hasattr(self, "_pool_component_names"):
            self._pool_component_names = {}
        if not hasattr(self, "_pool_replicated"):
            self._pool_replicated = {}
        try:
            page_size = int(getattr(host_pool, "page_size", 1))
            if page_size <= 0:
                raise RuntimeError("pool page_size must be positive")
            try:
                import torch as _t
                indices = _t.arange(page_size, dtype=_t.int64)
            except ImportError:
                indices = list(range(page_size))
            meta = host_pool.get_page_buffer_meta(indices)
            if (not isinstance(meta, (tuple, list)) or len(meta) != 2
                    or meta[0] is None or meta[1] is None):
                raise RuntimeError("pool component probe returned no layout")
            ptrs, sizes = meta
            count = len(ptrs)
            if count == 0 or len(sizes) != count:
                raise RuntimeError("pool component probe returned empty/mismatched layout")
            if hasattr(host_pool, "conv_state_elem_sizes"):
                names = [f"conv{i}" for i in range(
                    len(getattr(host_pool, "conv_state_elem_sizes") or []))]
                if int(getattr(host_pool, "temporal_state_elem_size", 0)) > 0:
                    names.insert(0, "temporal")
                replicated = False
            elif hasattr(host_pool, "v_buffer"):
                if count != 2:
                    raise RuntimeError(
                        f"K/V pool must expose exactly 2 components, got {count}")
                names = ["k", "v"]
                replicated = False
            elif count == 1:
                # The metadata API itself proves the legacy packed single-buffer
                # layout. This is the only safe replicated fallback.
                names = ["all"]
                replicated = True
            else:
                raise RuntimeError(
                    f"ambiguous {count}-component pool without layout metadata")
            if len(names) != count:
                raise RuntimeError(
                    f"pool component metadata mismatch: {len(names)} != {count}")
        except Exception as exc:
            self.registered_pools.pop(name, None)
            self._pool_component_names.pop(name, None)
            self._pool_replicated.pop(name, None)
            raise RuntimeError(
                f"cannot discover physical layout for pool {name!r}") from exc

        self._pool_component_names[name] = tuple(names)
        self._pool_replicated[name] = replicated
        self.registered_pools[name] = host_pool
        with access_log("pool_components",
                        lambda: f"{self._alog_tag} {name}") as result:
            result.result = (
                f"components={self._pool_component_names[name]} "
                f"replicated={self._pool_replicated[name]} (ok)")
        with access_log("register_mem_host_pool_v2",
                        lambda: f"{self._alog_tag} {name}") as result:
            try:
                registered = self._register_pool_buffers(host_pool)
            except Exception as exc:
                result.result = f"skip ({type(exc).__name__})"
                return
            result.result = (
                f"registered {registered} region(s)"
                if registered else "no backing buffer found")

    # --- L2-bypass (device-direct write) -------------------------------------
    def supports_device_transfer(self) -> bool:
        """Return whether required device-direct transfer remains safe to use."""
        if getattr(self, "_device_registration_failed", False):
            return False
        if self.transport_mode != "rdma":
            return False
        if not (hasattr(self._lib, "dfkv_batch_put_sg")
                and hasattr(self._lib, "dfkv_batch_get_auto_sg")
                and hasattr(self._lib, "dfkv_max_sg_segs")):
            return False
        try:
            layer_num = int(self.cfg.get("layer_num", 0))
            max_segs = int(self._lib.dfkv_max_sg_segs(self._h))
        except (TypeError, ValueError, OverflowError):
            return False
        except Exception:
            return False
        if layer_num <= 0:
            return False
        if max_segs < 1:
            return False
        if max_segs < layer_num:
            # A page is split into ceil(layer_num/max_segs) physical scatter
            # groups by _flatten_device. This is the same grouping used by the
            # vLLM connector, so a narrower HCA costs extra keys and operations
            # per page without disabling device-direct transfer.
            print(f"[dfkv] L2-bypass: max_sg_segs={max_segs} < layer_num="
                  f"{layer_num}; chunking each page into "
                  f"{(layer_num + max_segs - 1) // max_segs} scatter groups.",
                  file=sys.stderr, flush=True)
        return True

    def _register_device_regions(self, regions, label: str) -> int:
        """Register GPU regions, failing closed when bypass was requested."""
        required = getattr(self, "_device_direct_requested", False)
        try:
            discovered = list(regions())
            if required and not discovered:
                raise RuntimeError("no device regions discovered")
            registered = 0
            for base, size in discovered:
                region = (int(base), int(size))
                if not region[0] or not region[1]:
                    if required:
                        raise RuntimeError(f"invalid device region {region!r}")
                    continue
                if region in self._registered_regions:
                    continue
                if not self.register_memory(*region):
                    if required:
                        raise RuntimeError(
                            f"native MR registration rejected {region!r}")
                    continue
                self._registered_regions.add(region)
                registered += 1
            return registered
        except Exception as exc:
            if required:
                self._device_registration_failed = True
                raise RuntimeError(
                    f"required {label} GPU MR registration failed") from exc
            _log.warning(
                "optional %s GPU MR registration skipped: %s",
                label, exc)
            return 0

    def register_mem_pool_device(self, mem_pool_device):
        """Register the primary GPU KV pool; bypass makes every MR required."""
        from sglang.srt.mem_cache.device_page_meta import device_pool_regions

        with access_log("register_mem_pool_device",
                        lambda: f"{self._alog_tag}") as result:
            try:
                registered = self._register_device_regions(
                    lambda: device_pool_regions(mem_pool_device), "primary")
            except Exception as exc:
                result.result = f"failed ({type(exc).__name__})"
                raise
            self.mem_pool_device = mem_pool_device
            result.result = (
                f"registered {registered} device region(s)"
                if registered else "no device buffer registered")

    def register_mem_pool_device_sidecar(self, name, device_pool):
        """Register a DSA sidecar GPU pool, strictly in requested bypass mode."""
        from sglang.srt.mem_cache.device_page_meta import sidecar_device_pool_regions

        key = str(name)
        with access_log("register_mem_pool_device_sidecar",
                        lambda: f"{self._alog_tag} {key}") as result:
            try:
                registered = self._register_device_regions(
                    lambda: sidecar_device_pool_regions(device_pool), f"sidecar {key}")
            except Exception as exc:
                result.result = f"failed ({type(exc).__name__})"
                raise
            if not hasattr(self, "_sidecar_device_pools"):
                self._sidecar_device_pools = {}
            self._sidecar_device_pools[key] = device_pool
            # The sidecar metadata API returns one logical object per page; its
            # layers are SG segments, not independent pool components.
            if not hasattr(self, "_pool_component_names"):
                self._pool_component_names = {}
            if not hasattr(self, "_pool_replicated"):
                self._pool_replicated = {}
            self._pool_component_names[key] = ("all",)
            self._pool_replicated[key] = True
            result.result = (
                f"registered {registered} sidecar device region(s)"
                if registered else "no sidecar device buffer registered")

    def register_mem_pool_device_draft(self, mem_pool_device_draft):
        """Register the draft GPU KV pool, strictly in requested bypass mode."""
        from sglang.srt.mem_cache.device_page_meta import device_pool_regions

        with access_log("register_mem_pool_device_draft",
                        lambda: f"{self._alog_tag}") as result:
            try:
                registered = self._register_device_regions(
                    lambda: device_pool_regions(mem_pool_device_draft), "draft")
            except Exception as exc:
                result.result = f"failed ({type(exc).__name__})"
                raise
            self.mem_pool_device_draft = mem_pool_device_draft
            result.result = (
                f"registered {registered} draft device region(s)"
                if registered else "no draft device buffer registered")

    def register_mem_pool_device_draft_sidecar(self, mem_pool_device_draft):
        """Register the draft DSA indexer GPU pool in device-direct mode."""
        from sglang.srt.mem_cache.device_page_meta import sidecar_device_pool_regions

        name = "draft_indexer"
        with access_log("register_mem_pool_device_draft_sidecar",
                        lambda: f"{self._alog_tag}") as result:
            try:
                registered = self._register_device_regions(
                    lambda: sidecar_device_pool_regions(mem_pool_device_draft),
                    "draft indexer")
            except Exception as exc:
                result.result = f"failed ({type(exc).__name__})"
                raise
            if not hasattr(self, "_sidecar_device_pools"):
                self._sidecar_device_pools = {}
            self._draft_sidecar_name = name
            self._sidecar_device_pools[name] = mem_pool_device_draft
            # Same API-proven one-object layout as the target indexer sidecar.
            if not hasattr(self, "_pool_component_names"):
                self._pool_component_names = {}
            if not hasattr(self, "_pool_replicated"):
                self._pool_replicated = {}
            self._pool_component_names[name] = ("all",)
            self._pool_replicated[name] = True
            result.result = (
                f"registered {registered} draft indexer region(s)"
                if registered else "no draft indexer buffer registered")

    def _draft_sidecar_device_set(self, keys, device_indices):
        """Best-effort device-direct SG put of the DSA draft's indexer sidecar (the
        `draft_indexer` namespace), or None when no draft sidecar is registered (a
        dense/non-DSA draft). Failure returns [False]*n so the caller marks those
        draft pages unstored — the read gates coherence, so a page never serves a
        latent without its matching indexer."""
        name = getattr(self, "_draft_sidecar_name", None)
        if not name:
            return None
        try:
            res, _nb, _s = self._sidecar_device_set(name, keys, device_indices)
            return res
        except Exception:
            return [False] * len(keys)

    def _draft_sidecar_device_get(self, keys, device_indices):
        """Read twin of _draft_sidecar_device_set: RDMA the DSA draft indexer straight
        into its GPU buffer, or None when no draft sidecar is registered."""
        name = getattr(self, "_draft_sidecar_name", None)
        if not name:
            return None
        try:
            res, _nb, _s = self._sidecar_device_get(name, keys, device_indices)
            return res
        except Exception:
            return [False] * len(keys)


    # --- canonical pool-key schema -----------------------------------------
    def _keys(self, page_hash: str | bytes) -> List[bytes]:
        tp = -1 if self.is_mla else self.tp_rank
        pp = self.pp_rank if self.enable_pp else 0
        components = ["all"] if self.is_mla else ["k", "v"]
        return [
            pool_key(page_hash, pool="kv",
                     tp_size=self.tp_size, tp_rank=tp,
                     pcp_size=self.pcp_size, pcp_rank=self.pcp_rank,
                     dcp_size=self.dcp_size, dcp_rank=self.dcp_rank,
                     pp_size=self.pp_size if self.enable_pp else 1, pp_rank=pp,
                     component=component)
            for component in components
        ]

    def _sub(self) -> int:
        return 1 if self.is_mla else 2

    def _flatten(self, keys, ptrs, sizes):
        """Expand per-page keys into per-object (sub) flat arrays."""
        sub = self._sub()
        assert len(ptrs) == len(keys) * sub, (len(ptrs), len(keys), sub)
        sks, sp, ss = [], [], []
        for i, k in enumerate(keys):
            for j, sk in enumerate(self._keys(k)):
                sks.append(sk); sp.append(int(ptrs[i * sub + j])); ss.append(int(sizes[i * sub + j]))
        return sub, sks, sp, ss

    def _fold(self, flat_results, npages, sub):
        """A page succeeds iff all its sub-objects succeeded."""
        return [all(flat_results[i * sub + j] for j in range(sub)) for i in range(npages)]

    def _batch_exist_flat(self, subkeys) -> List[bool]:
        if not subkeys:
            return []
        karr, klens, key_owners = make_key_array(subkeys)
        out = (ctypes.c_int * len(subkeys))()
        rc = self._lib.dfkv_batch_exist(
            self._h, karr, klens, len(subkeys), out)
        if rc != 0:
            return [False] * len(subkeys)
        del key_owners
        return [out[i] == 1 for i in range(len(subkeys))]

    def _note_logical_anchor_once(self):
        """One-time notice that the primary KV pool is a logical anchor.

        SGLang builds a *logical anchor* (LogicalHostPool) as the primary "kv"
        pool for V4/DSA multi-pool models such as GLM-5.2 — it carries no KV
        tensor, so get_page_buffer_meta() returns None. The real KV rides the v2
        side-pool path (batch_set_v2/batch_get_v2); the v1 anchor writes a
        one-byte "kv" marker so the v2 existence check can anchor the hit prefix.
        This mirrors the reference backend's logical marker semantics.
        Informational, not an error — the path works, but it is newly enabled,
        confirm the hit rate via metrics."""
        if self._anchor_noop_warned:
            return
        self._anchor_noop_warned = True
        import sys as _sys
        print("[dfkv] NOTE: primary KV pool is a logical anchor "
              "(get_page_buffer_meta -> None) — a V4/DSA multi-pool model "
              f"(model={self.model!r}, e.g. GLM-5.2). The v1 anchor writes a "
              "one-byte 'kv' marker; real KV rides the v2 side-pool path. Verify L3 "
              "hit rate via metrics; if low, the LMCache MP connector "
              "(docs/CONNECTORS.md #4.5) is the alternative path for GLM-5.x DSA.",
              file=_sys.stderr, flush=True)

    def _write_anchor_markers(self, keys) -> List[bool]:
        """Write a one-byte marker object per "kv" sub-key so a later
        batch_exists / batch_exists_v2 can find the primary-pool prefix.

        Used only for the logical-anchor case (V4/DSA, e.g. GLM-5.2): the anchor
        pool holds no KV buffer, so there is nothing to zero-copy. The marker is
        non-empty so native dfkv charges tenant byte quota for every allocated
        object while preserving SGLang's logical-anchor existence semantics.
        Namespace and object-key identity isolate incompatible writers; the
        matching read (batch_get_v1) is a no-op. Returns a per-page success list
        (a page succeeds iff all its sub-object markers were written)."""
        sub = self._sub()
        sks = [sk for k in keys for sk in self._keys(k)]
        if not sks:
            return []
        karr, klens, parr, sarr, out, key_owners = _arrays(
            sks, [_MARKER_PTR] * len(sks), [1] * len(sks))
        self._lib.dfkv_batch_put(
            self._h, karr, klens, parr, sarr, len(sks), out)
        del key_owners
        return self._fold(
            [out[i] == 1 for i in range(len(sks))], len(keys), sub)

    # --- zero-copy v1 batch path (the one the controller calls) ---
    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        n = len(keys)
        with _tracing.span("batch_set_v1", n) as _sp, \
                access_log("batch_set_v1", lambda: f"{self._alog_tag} {n} keys") as r:
            # MLA backup_skip: latent is replicated across TP, only rank 0 writes.
            if self.is_mla and self.tp_rank != 0:
                r.result = "backup_skip"
                if _sp:
                    _sp.attrs = {"dfkv.backup_skip": True}
                return [True] * n
            meta = self.mem_pool_host.get_page_buffer_meta(host_indices)
            if meta is None:
                # Primary "kv" pool is a *logical anchor* holding no KV buffer
                # (V4/DSA-compressed models, e.g. GLM-5.2: SGLang registers a
                # LogicalHostPool whose get_page_buffer_meta() returns None). The
                # real KV lives in compressed side-pools written via batch_set_v2;
                # there is nothing to zero-copy on the anchor, but we still write
                # a one-byte "kv" marker per page so the v2 existence check can
                # anchor the hit prefix. Single-pool models (MLA/MHA) never
                # return None, so this branch is inert for them.
                self._note_logical_anchor_once()
                res = self._write_anchor_markers(keys)
                r.result = f"anchor_marker {sum(res)}/{n}"
                if _sp:
                    _sp.hits = sum(res)
                    _sp.attrs = {"dfkv.anchor_marker": True}
                return res
            ptrs, sizes = meta
            sub, sks, sp, ss = self._flatten(keys, ptrs, sizes)
            t0 = time.perf_counter()
            flat = self._put_flat(sks, sp, ss)
            dur = time.perf_counter() - t0
            res = self._fold(flat, n, sub)
            r.result = f"ok {sum(res)}/{n}"
            if self._put_retry_recovered:
                r.result += f" retry_ok={self._put_retry_recovered}"
            if _sp:
                _sp.hits = sum(res); _sp.bytes = sum(ss)
            self._metrics.on_set(pages=n, ok_pages=sum(res), nbytes=sum(ss), seconds=dur)
            # Fleet op metrics (put/get/exist/...) are accumulated in the C++
            # KVClient (the chokepoint all connectors share) and forwarded over
            # OTLP by the snapshot poller — see dfkv_telemetry.parse_client_ops.
            return res

    def _sg_width(self) -> int:
        """Negotiated SG segments per key (HCA max_sge budget), cached."""
        w = getattr(self, "_sg_width_cache", 0)
        if not w:
            try:
                w = int(self._lib.dfkv_max_sg_segs(self._h))
            except Exception:
                w = 0
            w = w or 29  # ConnectX-era fallback: max_sge=30, one SGE reserved
            self._sg_width_cache = w
        return w

    def _sg_group_key(self, sk: bytes, ci: int) -> bytes:
        return sg_key(sk, self._sg_width(), ci)

    def _flatten_device(self, keys, seg_ptrs, seg_sizes, keys_fn=None, sub=None):
        """Expand per-page keys into per-sub-object (k[/v]) sub-keys, pairing each
        with its per-layer device segment list. Parallel to _flatten, but every
        entry is a segment LIST (one per layer), not a single (ptr, size).

        A page's layer_num segments can exceed the HCA's per-key SG budget
        (max_sge-1, e.g. 29 on ConnectX < 36/61 layers), so each sub-object is
        chunked into binary SG-coordinate keys of at most _sg_width()
        consecutive layers. Write and read derive the identical deterministic
        split from (layer count, width), so layer-major bytes reassemble exactly.
        Chunk count is uniform across pages, so _fold()'s per-page stride is
        sub * nchunks. Width is always part of key identity, so clients that
        negotiate different widths miss (cold) instead of reading mis-split
        bytes.

        keys_fn/sub default to the main-KV key scheme (self._keys / self._sub);
        DSA indexer and EAGLE draft sidecars pass their own key builder and
        object count while reusing the same SG chunking.
        """
        keys_fn = keys_fn or self._keys
        sub = self._sub() if sub is None else sub
        assert len(seg_ptrs) == len(keys) * sub, (len(seg_ptrs), len(keys), sub)
        w = self._sg_width()
        sks, sp, ss = [], [], []
        nchunks = 1
        for i, k in enumerate(keys):
            for j, sk in enumerate(keys_fn(k)):
                p = [int(x) for x in seg_ptrs[i * sub + j]]
                s = [int(x) for x in seg_sizes[i * sub + j]]
                nchunks = max(1, (len(p) + w - 1) // w)
                for ci in range(nchunks):
                    sks.append(self._sg_group_key(sk, ci))
                    sp.append(p[ci * w:(ci + 1) * w])
                    ss.append(s[ci * w:(ci + 1) * w])
        return sub * nchunks, sks, sp, ss

    def _batch_put_sg(self, sks, seg_ptrs, seg_sizes) -> List[bool]:
        """Scatter-gather batch put: sub-key sks[i] stores the in-order
        concatenation of its per-layer segments seg_ptrs[i][..]/seg_sizes[i][..] as
        one dfkv key (one RDMA multi-SGE op). Returns per-key success bools."""
        n = len(sks)
        if n == 0:
            return []
        karr, klens, key_owners = make_key_array(sks)
        # Keep the per-key inner arrays alive for the whole call (mirrors the vLLM
        # connector's batch_put_sg): the outer arrays hold casts of these.
        inner_p = [(ctypes.c_void_p * len(p))(*[ctypes.c_void_p(int(x)) for x in p])
                   for p in seg_ptrs]
        inner_s = [(ctypes.c_uint64 * len(s))(*[int(x) for x in s])
                   for s in seg_sizes]
        parr = (ctypes.POINTER(ctypes.c_void_p) * n)(
            *[ctypes.cast(a, ctypes.POINTER(ctypes.c_void_p)) for a in inner_p])
        sarr = (ctypes.POINTER(ctypes.c_uint64) * n)(
            *[ctypes.cast(a, ctypes.POINTER(ctypes.c_uint64)) for a in inner_s])
        narr = (ctypes.c_int * n)(*[len(p) for p in seg_ptrs])
        out = (ctypes.c_int * n)()
        rc = self._lib.dfkv_batch_put_sg(
            self._h, karr, klens, parr, sarr, narr, n, out)
        if rc != 0:
            return [False] * n
        del key_owners
        return [out[i] == 1 for i in range(n)]

    def _put_sg_flat(self, sks, seg_ptrs, seg_sizes) -> List[bool]:
        """SG put with the same phase-9 exist gate + single transient retry as
        _put_flat (the contiguous host put), so the L2-bypass path inherits the
        identical backup-dedup and burst-failure recovery semantics."""
        self._put_retry_recovered = 0
        if not sks:
            return []
        if self._backup_exist_gate:
            present = list(self._batch_exist_flat(sks))
            todo = [i for i, p in enumerate(present) if not p]
            if not todo:
                return [True] * len(sks)
        else:
            present = [False] * len(sks)
            todo = list(range(len(sks)))
        for attempt in (0, 1):
            if not todo:
                break
            if attempt:
                time.sleep(0.01)  # let the write burst drain first
            out = self._batch_put_sg(
                [sks[i] for i in todo], [seg_ptrs[i] for i in todo],
                [seg_sizes[i] for i in todo])
            failed = []
            for m, i in enumerate(todo):
                present[i] = out[m]
                if not present[i]:
                    failed.append(i)
                elif attempt:
                    self._put_retry_recovered += 1
            todo = failed
        return present

    def batch_set_v1_device(self, keys, device_indices, extra_info=None) -> List[bool]:
        """L2-bypass write-through: RDMA a page straight from its GPU KV slots to L3
        (no D2H staging). Mirrors batch_set_v1 but the page payload is gathered from
        the layer-first device pool as per-layer scatter-gather segments.

        NOTE (ordering): the device segments concatenate LAYER-major, whereas the
        stock host read (batch_get_v1) reconstructs a page-first (TOKEN-major)
        buffer. The two are transposes; a page written here is byte-coherent only
        with a matching device-direct (layer-major) reader — increment 2 — not with
        the unchanged host read path. See the SGLang-side device_page_meta.py."""
        n = len(keys)
        with _tracing.span("batch_set_v1_device", n) as _sp, \
                access_log("batch_set_v1_device",
                           lambda: f"{self._alog_tag} {n} keys") as r:
            # MLA backup_skip: latent is replicated across TP, only rank 0 writes.
            if self.is_mla and self.tp_rank != 0:
                r.result = "backup_skip"
                if _sp:
                    _sp.attrs = {"dfkv.backup_skip": True}
                return [True] * n
            from sglang.srt.mem_cache.device_page_meta import (
                get_device_page_buffer_meta,
            )
            seg_ptrs, seg_sizes = get_device_page_buffer_meta(
                self.mem_pool_device, device_indices)
            sub, sks, sp, ss = self._flatten_device(keys, seg_ptrs, seg_sizes)
            nbytes = sum(sum(s) for s in ss)
            t0 = time.perf_counter()
            flat = self._put_sg_flat(sks, sp, ss)
            dur = time.perf_counter() - t0
            res = self._fold(flat, n, sub)
            r.result = f"ok {sum(res)}/{n} (device-direct)"
            if self._put_retry_recovered:
                r.result += f" retry_ok={self._put_retry_recovered}"
            if _sp:
                _sp.hits = sum(res); _sp.bytes = nbytes
            self._metrics.on_set(pages=n, ok_pages=sum(res), nbytes=nbytes,
                                 seconds=dur)
            return res

    def _batch_get_sg(self, sks, seg_ptrs, seg_caps):
        """Scatter-gather batch get: sub-key sks[i]'s stored blob is scattered in
        order across the destination buffers seg_ptrs[i][..] of capacity
        seg_caps[i][..] (one RDMA multi-SGE op). Returns (hits, lens): hits[i]==1
        on hit, lens[i]=total stored bytes. Mirrors the vLLM connector's
        batch_get_auto_sg (dfkv_client.py) and is the read twin of _batch_put_sg."""
        n = len(sks)
        if n == 0:
            return [], []
        karr, klens, key_owners = make_key_array(sks)
        # Keep the per-key inner arrays alive for the whole call (the outer arrays
        # hold casts of these) — same lifetime discipline as _batch_put_sg.
        inner_p = [(ctypes.c_void_p * len(p))(*[ctypes.c_void_p(int(x)) for x in p])
                   for p in seg_ptrs]
        inner_c = [(ctypes.c_uint64 * len(c))(*[int(x) for x in c])
                   for c in seg_caps]
        parr = (ctypes.POINTER(ctypes.c_void_p) * n)(
            *[ctypes.cast(a, ctypes.POINTER(ctypes.c_void_p)) for a in inner_p])
        carr = (ctypes.POINTER(ctypes.c_uint64) * n)(
            *[ctypes.cast(a, ctypes.POINTER(ctypes.c_uint64)) for a in inner_c])
        narr = (ctypes.c_int * n)(*[len(p) for p in seg_ptrs])
        out_hit = (ctypes.c_int * n)()
        out_len = (ctypes.c_uint64 * n)()
        rc = self._lib.dfkv_batch_get_auto_sg(
            self._h, karr, klens, parr, carr, narr, n, out_hit, out_len)
        if rc != 0:
            return [0] * n, [0] * n
        del key_owners
        return [out_hit[i] for i in range(n)], [int(out_len[i]) for i in range(n)]

    def batch_get_v1_device(self, keys, device_indices, extra_info=None) -> List[bool]:
        """L2-bypass on-demand read: RDMA a page's stored blob straight INTO its
        GPU KV slots (no host staging). The read twin of batch_set_v1_device.

        The device pool is layer-first, so a page's KV scatters across layer_num
        non-contiguous per-layer buffers; get_device_page_buffer_meta yields the
        same per-layer (ptr, size) segment lists the write used, here as the SG GET
        DESTINATIONS (ptr) and capacities (size). Because the stored blob was
        written LAYER-major (batch_set_v1_device), scattering it back across the
        per-layer destination segments reassembles it layer-major into the device
        slots — byte-consistent with the writer. A page succeeds iff every one of
        its sub-objects (k[/v]) hit AND returned its full capacity (a short read is
        a corrupt page -> failure, so the caller recomputes rather than serving it).

        MLA note: the latent is replicated across TP, and only tp_rank 0 wrote it
        (backup_skip). But EVERY rank must READ its own copy into its own device
        slots, so there is NO read-side rank skip (unlike the write)."""
        n = len(keys)
        with _tracing.span("batch_get_v1_device", n) as _sp, \
                access_log("batch_get_v1_device",
                           lambda: f"{self._alog_tag} {n} keys") as r:
            from sglang.srt.mem_cache.device_page_meta import (
                get_device_page_buffer_meta,
            )
            seg_ptrs, seg_caps = get_device_page_buffer_meta(
                self.mem_pool_device, device_indices)
            sub, sks, sp, sc = self._flatten_device(keys, seg_ptrs, seg_caps)
            # Per sub-key expected byte length = sum of its per-layer segment caps.
            want = [sum(c) for c in sc]
            t0 = time.perf_counter()
            hits, lens = self._batch_get_sg(sks, sp, sc)
            dur = time.perf_counter() - t0
            # A sub-object is good only on a full-length hit; fold to per-page.
            flat_ok = [hits[i] == 1 and lens[i] >= want[i] for i in range(len(sks))]
            res = self._fold(flat_ok, n, sub)
            nbytes = sum(lens)
            r.result = f"hits={sum(res)}/{n} (device-direct)"
            short = sum(1 for i in range(len(sks))
                        if hits[i] == 1 and lens[i] < want[i])
            if short:
                r.result += f" short_read={short}"
            if _sp:
                _sp.hits = sum(res); _sp.bytes = nbytes
            self._metrics.on_get(pages=n, hit_pages=sum(res), nbytes=nbytes,
                                 seconds=dur)
            return res

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        n = len(keys)
        with _tracing.span("batch_get_v1", n) as _sp, \
                access_log("batch_get_v1", lambda: f"{self._alog_tag} {n} keys") as r:
            meta = self.mem_pool_host.get_page_buffer_meta(host_indices)
            if meta is None:
                # Logical anchor pool, no buffer to scatter into (V4/DSA models,
                # e.g. GLM-5.2). The "kv" prefix was already confirmed present by
                # batch_exists_v2 (via the one-byte markers written on backup);
                # there is no anchor payload to load. Report all pages present so
                # _page_get_zero_copy counts the anchor prefix complete and the
                # hybrid controller then loads the real KV from side-pools via
                # batch_get_v2. Returning False here would make kv_completed_pages
                # < prefix and skip that side-pool load entirely. Inert for
                # single-pool models (non-None).
                self._note_logical_anchor_once()
                r.result = "anchor_noop"
                if _sp:
                    _sp.attrs = {"dfkv.anchor_noop": True}
                return [True] * n
            ptrs, sizes = meta
            sub, sks, sp, ss = self._flatten(keys, ptrs, sizes)
            karr, klens, parr, sarr, out, key_owners = _arrays(sks, sp, ss)
            t0 = time.perf_counter()
            self._lib.dfkv_batch_get(
                self._h, karr, klens, parr, sarr, len(sks), out)
            dur = time.perf_counter() - t0
            del key_owners
            res = self._fold(
                [out[i] == 1 for i in range(len(sks))], n, sub)
            r.result = f"hits={sum(res)}/{n}"
            if _sp:
                _sp.hits = sum(res); _sp.bytes = sum(ss)
            self._metrics.on_get(pages=n, hit_pages=sum(res), nbytes=sum(ss), seconds=dur)
            # Fleet op metrics now come from the C++ KVClient snapshot (above).
            return res

    def batch_exists(self, keys, extra_info=None) -> int:
        total = len(keys)
        with _tracing.span("batch_exists", total) as _sp, \
                access_log("batch_exists", lambda: f"{self._alog_tag} {total} keys") as r:
            # longest contiguous prefix of pages whose every sub-object exists.
            # Device-direct writes use explicit binary SG coordinates, so
            # existence probes group 0. The bare object key
            # cannot match a chunked store; a different negotiated width is a
            # cold miss rather than a corrupt read. One probe form suffices.
            sub = self._sub()
            if getattr(self, "mem_pool_device", None) is not None:
                sks = [self._sg_group_key(sk, 0) for k in keys for sk in self._keys(k)]
            else:
                sks = [sk for k in keys for sk in self._keys(k)]
            page_ok = self._fold(self._batch_exist_flat(sks), total, sub)
            n = 0
            for ok in page_ok:
                if not ok:
                    break
                n += 1
            r.result = f"prefix={n}/{total}"
            self._log_exist_probe(sks, n, total)
            if _sp:
                _sp.hits = n
            return n

    # --- v2 pool-aware interface (multi-pool models: Mamba/SWA/DeepSeek-V4) ---
    def _pool_components(self, pool_name: str) -> tuple[str, ...]:
        """Return a registered, positively discovered physical pool layout."""
        layouts = getattr(self, "_pool_component_names", {})
        if pool_name not in layouts:
            raise RuntimeError(f"pool {pool_name!r} has no discovered layout")
        return tuple(layouts[pool_name])

    def _pool_is_replicated(self, pool_name: str) -> bool:
        layouts = getattr(self, "_pool_replicated", {})
        if pool_name not in layouts:
            raise RuntimeError(f"pool {pool_name!r} has no discovered layout")
        return bool(layouts[pool_name])

    def _pool_keys(
        self, pool_name: str | bytes, page_hash: str | bytes,
    ) -> List[bytes]:
        if pool_name in ("kv", "__default__"):
            return self._keys(page_hash)
        components = self._pool_components(pool_name)
        pp = self.pp_rank if self.enable_pp else 0
        tp = -1 if self._pool_is_replicated(pool_name) else self.tp_rank
        return [
            pool_key(page_hash, pool=pool_name,
                     tp_size=self.tp_size, tp_rank=tp,
                     pcp_size=self.pcp_size, pcp_rank=self.pcp_rank,
                     dcp_size=self.dcp_size, dcp_rank=self.dcp_rank,
                     pp_size=self.pp_size if self.enable_pp else 1, pp_rank=pp,
                     component=component)
            for component in components
        ]

    def _pool_sub(self, pool_name: str) -> int:
        if pool_name in ("kv", "__default__"):
            return self._sub()
        return len(self._pool_components(pool_name))

    def _put_flat(self, sks, sp, ss) -> List[bool]:
        """batch_put with the phase-9 exist gate (sub-objects already in L3
        are reported stored without rewriting) and a single retry of failed
        keys. Under a cold-round write burst ~0.1% of puts fail transiently,
        and SGLang's backup bookkeeping does not consume per-page failures —
        an unretried miss becomes a phantom "backed" page that the hot round
        then fails to retrieve (R1 finding, 2026-07-17). The burst is
        momentary, so one delayed retry recovers almost all of them; keys
        still failing stay False (honest result, as before)."""
        self._put_retry_recovered = 0
        if not sks:
            return []
        if self._backup_exist_gate:
            present = list(self._batch_exist_flat(sks))
            todo = [i for i, p in enumerate(present) if not p]
            if not todo:
                return [True] * len(sks)
        else:
            present = [False] * len(sks)
            todo = list(range(len(sks)))
        for attempt in (0, 1):
            if not todo:
                break
            if attempt:
                time.sleep(0.01)  # let the write burst drain first
            karr, klens, parr, sarr, out, key_owners = _arrays(
                [sks[i] for i in todo], [sp[i] for i in todo],
                [ss[i] for i in todo])
            self._lib.dfkv_batch_put(
                self._h, karr, klens, parr, sarr, len(todo), out)
            del key_owners
            failed = []
            for m, i in enumerate(todo):
                present[i] = out[m] == 1
                if not present[i]:
                    failed.append(i)
                elif attempt:
                    self._put_retry_recovered += 1
            todo = failed
        return present

    def _v2_io(self, transfers, putting):
        results = {}
        segments = []
        sks, sp, ss = [], [], []
        for tr in transfers:
            name = str(tr.name)
            keys = tr.keys or []
            # Only rank 0 writes physically replicated pools. A follower must
            # still write rank-sharded state: MHA K/V shards and recurrent
            # temporal/conv components hold different bytes on every rank.
            # Skipping those writes loses 15/16 of a TP16 Kimi-K3 recurrent
            # state and can silently change generated output after an L3 hit.
            # Replication is inferred from the registered host layout, not a
            # model-name allowlist, and _pool_keys carries the matching
            # component/rank coordinates. Thus the write gate and collision
            # isolation derive from the same physical contract.
            if (putting and self.is_mla and self.tp_rank != 0
                    and self._pool_is_replicated(name)):
                results[name] = [True] * len(keys)
                continue
            pool = self.registered_pools[name]
            ptrs, sizes = pool.get_page_buffer_meta(tr.host_indices)
            sub = self._pool_sub(name)
            start = len(sks)
            for i, k in enumerate(keys):
                for j, sk in enumerate(self._pool_keys(name, k)):
                    sks.append(sk); sp.append(int(ptrs[i * sub + j])); ss.append(int(sizes[i * sub + j]))
            segments.append((name, len(keys), sub, start, len(sks)))
        t0 = time.perf_counter()
        if sks and putting:
            flat = self._put_flat(sks, sp, ss)
        elif sks:
            karr, klens, parr, sarr, out, key_owners = _arrays(sks, sp, ss)
            self._lib.dfkv_batch_get(
                self._h, karr, klens, parr, sarr, len(sks), out)
            del key_owners
            flat = [out[i] == 1 for i in range(len(sks))]
        else:
            flat = []
        # Expose the actual I/O cost/volume for the v2 metrics (the caller reads
        # these). ss covers only pools that did I/O (MLA backup_skip pools were
        # `continue`d before appending), so bytes are 0 on a pure skip.
        self._v2_io_bytes = sum(ss)
        self._v2_io_seconds = time.perf_counter() - t0
        self._v2_io_pools = tuple(seg[0] for seg in segments)
        for name, nkeys, sub, start, end in segments:
            results[name] = self._fold(flat[start:end], nkeys, sub)
        return results

    def batch_set_v2(self, transfers, extra_info=None) -> dict:
        nkeys = sum(len(tr.keys or []) for tr in (transfers or []))
        with _tracing.span("batch_set_v2", nkeys) as _sp, \
                access_log("batch_set_v2",
                           lambda: f"{self._alog_tag} {_fmt_pools(transfers)}") as r:
            res = self._v2_io(transfers, putting=True)
            r.result = _fmt_pool_results(res)
            if self._put_retry_recovered:
                r.result += f" retry_ok={self._put_retry_recovered}"
            if _sp:
                _sp.hits = sum(sum(rs) for rs in res.values())
            # Pool-granular metrics: report exactly the pools that performed I/O
            # (_v2_io_pools). The MLA rank!=0 replicated latent stays a no-op skip
            # ([True] markers, excluded), but rank-sharded sidecars such as
            # Kimi-K3's mamba/KDA state do real writes on follower ranks — a
            # whole-call tp_rank gate here would hide that write bandwidth
            # entirely from the v2 set metrics.
            io_pages = sum(len(res[name]) for name in self._v2_io_pools)
            if io_pages:
                self._metrics.on_set_v2(
                    pages=io_pages,
                    ok_pages=sum(sum(res[name]) for name in self._v2_io_pools),
                    nbytes=self._v2_io_bytes, seconds=self._v2_io_seconds)
            return res

    def batch_get_v2(self, transfers, extra_info=None) -> dict:
        nkeys = sum(len(tr.keys or []) for tr in (transfers or []))
        with _tracing.span("batch_get_v2", nkeys) as _sp, \
                access_log("batch_get_v2",
                           lambda: f"{self._alog_tag} {_fmt_pools(transfers)}") as r:
            res = self._v2_io(transfers, putting=False)
            r.result = _fmt_pool_results(res)
            if _sp:
                _sp.hits = sum(sum(rs) for rs in res.values())
            if nkeys:
                self._metrics.on_get_v2(
                    pages=sum(len(rs) for rs in res.values()),
                    hit_pages=sum(sum(rs) for rs in res.values()),
                    nbytes=self._v2_io_bytes, seconds=self._v2_io_seconds)
            return res

    # --- DSA L2-bypass: main KV device-direct + sidecar host, one logical op ----
    def _kv_device_set(self, keys, device_indices):
        """Core of batch_set_v1_device (device-direct SG put) without the tracing /
        access-log wrapper, so batch_set_v2_device can reuse it for the anchor KV.
        Returns (per_page_bools, nbytes, seconds). MLA backup_skip on non-zero TP
        rank (replicated latent) short-circuits to all-True, no I/O."""
        n = len(keys)
        if self.is_mla and self.tp_rank != 0:
            return [True] * n, 0, 0.0
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_page_buffer_meta,
        )
        seg_ptrs, seg_sizes = get_device_page_buffer_meta(
            self.mem_pool_device, device_indices)
        sub, sks, sp, ss = self._flatten_device(keys, seg_ptrs, seg_sizes)
        nbytes = sum(sum(s) for s in ss)
        t0 = time.perf_counter()
        flat = self._put_sg_flat(sks, sp, ss)
        dur = time.perf_counter() - t0
        self._log_write_keys(sks, flat, len(sks))
        return self._fold(flat, n, sub), nbytes, dur

    def _log_write_keys(self, sks, flat, nmain):
        """Log deterministic digests for the first and last written keys."""
        if not sks:
            return
        now = time.monotonic()
        if now - getattr(self, "_write_keys_logged", 0.0) < 30.0:
            return
        self._write_keys_logged = now
        ok = sum(1 for i in range(nmain) if flat[i])
        _log.warning(
            "dfkv write keys: ok=%d/%d first_key=%s last_key=%s",
            ok, nmain, _key_label(sks[0]), _key_label(sks[nmain - 1]))

    def _kv_device_get(self, keys, device_indices):
        """Core of batch_get_v1_device (device-direct SG get) without the tracing /
        access-log wrapper, so batch_get_v2_device can reuse it for the anchor KV.
        Returns (per_page_bools, nbytes, seconds). A page is a hit only on a
        full-length read of every sub-object (a short read is a corrupt page)."""
        n = len(keys)
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_page_buffer_meta,
        )
        seg_ptrs, seg_caps = get_device_page_buffer_meta(
            self.mem_pool_device, device_indices)
        sub, sks, sp, sc = self._flatten_device(keys, seg_ptrs, seg_caps)
        want = [sum(c) for c in sc]
        t0 = time.perf_counter()
        hits, lens = self._batch_get_sg(sks, sp, sc)
        dur = time.perf_counter() - t0
        flat_ok = [hits[i] == 1 and lens[i] >= want[i] for i in range(len(sks))]
        if not all(flat_ok):
            self._log_read_reject(sks, hits, lens, want, flat_ok, len(sks))
        return self._fold(flat_ok, n, sub), sum(lens), dur

    def _log_exist_probe(self, sks, n, total):
        """Log deterministic digests for the first and last existence probes."""
        if not sks:
            return
        now = time.monotonic()
        if now - getattr(self, "_exist_probe_logged", 0.0) < 30.0:
            return
        self._exist_probe_logged = now
        _log.warning(
            "dfkv exist probe: prefix=%d/%d first_key=%s last_key=%s",
            n, total, _key_label(sks[0]), _key_label(sks[-1]))

    def _log_read_reject(self, sks, hits, lens, want, flat_ok, nmain):
        """Log why a device read rejected sub-objects, at most once per 30s."""
        now = time.monotonic()
        if now - getattr(self, "_read_reject_logged", 0.0) < 30.0:
            return
        self._read_reject_logged = now
        # On an all-page miss, capture the native client snapshot immediately.
        # Server cache counters cannot distinguish "objects absent" from
        # "requests never reached the server"; served-op, I/O-error and peer
        # health evidence exists only in this per-client snapshot and is not
        # recoverable from the server's Prometheus metrics after the fact.
        if all(h != 1 for h in hits[:nmain]):
            try:
                snap = _read_snapshot(self._lib, self._h)
                keep = [
                    line for line in snap.splitlines()
                    if not line.startswith("#")
                    and any(
                        token in line
                        for token in (
                            "ops_served", "io_error", "unhealthy", "peer_",
                            "ring_", "mds_", "busy_suppressed",
                        )
                    )
                ]
                _log.warning(
                    "dfkv client snapshot @full-miss:\n%s",
                    "\n".join(keep[:12]) or "(empty)",
                )
            except Exception as exc:
                _log.warning(
                    "dfkv client snapshot @full-miss failed: %r", exc)
        miss = sum(1 for i in range(nmain) if hits[i] != 1)
        short = sum(
            1 for i in range(nmain)
            if hits[i] == 1 and lens[i] < want[i]
        )
        first = next(i for i in range(nmain) if not flat_ok[i])
        _log.warning(
            "dfkv device read rejected %d/%d sub-objects "
            "(%d MISS, %d SHORT); first offender: "
            "key=%s hit=%s got=%d want=%d",
            nmain - sum(flat_ok), nmain, miss, short,
            _key_label(sks[first]), hits[first], lens[first], want[first],
        )

    # --- task 4: DSA indexer sidecar device-direct (no host staging) -----------
    def _sidecar_device_set(self, name, keys, device_indices):
        """Device-direct SG put of the DSA indexer sidecar ``name`` using its
        own binary pool/SG identity. The
        indexer is layer-first & PAGE-indexed; get_device_sidecar_page_buffer_meta
        yields its per-layer page-row segments from the registered sidecar device
        pool. Returns (per_page_bools, nbytes, seconds)."""
        n = len(keys)
        if self.is_mla and self.tp_rank != 0:
            return [True] * n, 0, 0.0
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_sidecar_page_buffer_meta,
        )
        pool = self._sidecar_device_pools[str(name)]
        seg_ptrs, seg_sizes = get_device_sidecar_page_buffer_meta(pool, device_indices)
        stride, sks, sp, ss = self._flatten_device(
            keys, seg_ptrs, seg_sizes,
            keys_fn=lambda h: self._pool_keys(name, h), sub=self._pool_sub(name))
        nbytes = sum(sum(s) for s in ss)
        t0 = time.perf_counter()
        flat = self._put_sg_flat(sks, sp, ss)
        dur = time.perf_counter() - t0
        return self._fold(flat, n, stride), nbytes, dur

    def _sidecar_device_get(self, name, keys, device_indices):
        """Device-direct SG get of the DSA indexer sidecar `name`: RDMA its stored
        blob straight INTO its GPU index buffer (no host staging, no H2D). Read twin
        of _sidecar_device_set. A page is a hit only on a full-length read of every
        sub-key. Returns (per_page_bools, nbytes, seconds)."""
        n = len(keys)
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_sidecar_page_buffer_meta,
        )
        pool = self._sidecar_device_pools[str(name)]
        seg_ptrs, seg_caps = get_device_sidecar_page_buffer_meta(pool, device_indices)
        stride, sks, sp, sc = self._flatten_device(
            keys, seg_ptrs, seg_caps,
            keys_fn=lambda h: self._pool_keys(name, h), sub=self._pool_sub(name))
        want = [sum(c) for c in sc]
        t0 = time.perf_counter()
        hits, lens = self._batch_get_sg(sks, sp, sc)
        dur = time.perf_counter() - t0
        flat_ok = [hits[i] == 1 and lens[i] >= want[i] for i in range(len(sks))]
        return self._fold(flat_ok, n, stride), sum(lens), dur

    # --- task 6: EAGLE draft KV device-direct (best-effort L3) ------------------
    def _draft_keys(
        self, page_hash: str | bytes, sub: int,
    ) -> List[bytes]:
        pp = self.pp_rank if self.enable_pp else 0
        tp = -1 if sub == 1 else self.tp_rank
        components = ["all"] if sub == 1 else ["k", "v"]
        return [
            pool_key(page_hash, pool="draft",
                     tp_size=self.tp_size, tp_rank=tp,
                     pcp_size=self.pcp_size, pcp_rank=self.pcp_rank,
                     dcp_size=self.dcp_size, dcp_rank=self.dcp_rank,
                     pp_size=self.pp_size if self.enable_pp else 1, pp_rank=pp,
                     component=component)
            for component in components
        ]

    def batch_set_v1_device_draft(self, keys, device_indices, extra_info=None) -> List[bool]:
        """Best-effort device-direct SG put of the EAGLE draft KV pages (task 6):
        RDMA straight from the draft GPU pool's slots (the same slots the target
        rode) to L3 under the `.draft` namespace. sub (1 for MLA / 2 for MHA) is
        derived from the draft pool meta, independent of the target model. An MLA
        draft's latent is TP-replicated, so only tp_rank 0 writes (backup_skip),
        exactly like the target MLA path."""
        n = len(keys)
        with access_log("batch_set_v1_device_draft",
                        lambda: f"{self._alog_tag} {n} keys") as r:
            from sglang.srt.mem_cache.device_page_meta import (
                get_device_page_buffer_meta,
            )
            seg_ptrs, seg_sizes = get_device_page_buffer_meta(
                self.mem_pool_device_draft, device_indices)
            sub = len(seg_ptrs) // n if n else 1
            if sub == 1 and self.tp_rank != 0:
                r.result = "backup_skip"
                return [True] * n
            stride, sks, sp, ss = self._flatten_device(
                keys, seg_ptrs, seg_sizes,
                keys_fn=lambda h: self._draft_keys(h, sub), sub=sub)
            flat = self._put_sg_flat(sks, sp, ss)
            res = self._fold(flat, n, stride)
            r.result = f"ok {sum(res)}/{n} (draft device-direct)"
            # DSA-draft: also store the draft indexer sidecar device-direct so the
            # loaded draft page is coherent (latent + indexer). None => dense draft.
            side = self._draft_sidecar_device_set(keys, device_indices)
            if side is not None:
                res = [res[i] and side[i] for i in range(n)]
                r.result += f" +indexer {sum(side)}/{n}"
            return res

    def batch_get_v1_device_draft(self, keys, device_indices, extra_info=None) -> List[bool]:
        """Best-effort device-direct SG get of the EAGLE draft KV pages (task 6):
        RDMA straight INTO the draft GPU pool's slots. Read twin of
        batch_set_v1_device_draft. Every rank reads its own copy (no rank skip on
        read), even for a TP-replicated MLA draft."""
        n = len(keys)
        with access_log("batch_get_v1_device_draft",
                        lambda: f"{self._alog_tag} {n} keys") as r:
            from sglang.srt.mem_cache.device_page_meta import (
                get_device_page_buffer_meta,
            )
            seg_ptrs, seg_caps = get_device_page_buffer_meta(
                self.mem_pool_device_draft, device_indices)
            sub = len(seg_ptrs) // n if n else 1
            stride, sks, sp, sc = self._flatten_device(
                keys, seg_ptrs, seg_caps,
                keys_fn=lambda h: self._draft_keys(h, sub), sub=sub)
            want = [sum(c) for c in sc]
            hits, lens = self._batch_get_sg(sks, sp, sc)
            flat_ok = [hits[i] == 1 and lens[i] >= want[i] for i in range(len(sks))]
            res = self._fold(flat_ok, n, stride)
            r.result = f"hits={sum(res)}/{n} (draft device-direct)"
            # DSA-draft: a page is a coherent draft hit only if its indexer sidecar
            # also read back in full. None => dense draft (no sidecar).
            side = self._draft_sidecar_device_get(keys, device_indices)
            if side is not None:
                res = [res[i] and side[i] for i in range(n)]
                r.result += f" +indexer {sum(side)}/{n}"
            return res

    def batch_set_v2_device(
        self, kv_keys, kv_device_indices, sidecar_transfers, extra_info=None
    ) -> dict:
        """DSA L2-bypass backup (GLM-5.2): the anchor "kv" pool (the big MLA latent)
        RDMAs straight from its GPU slots to L3 via the device-direct SG put. Task 4:
        a DEVICE sidecar transfer (device_indices set, host_indices None) RDMAs the
        DSA indexer straight from its GPU index buffer too (no host staging); a host
        sidecar transfer still rides the stock host v2 path. One logical op so the
        split value is written atomically-per-page.

        VALUE LAYOUT: the main KV uses the canonical ``pool=kv`` object keys
        shared by batch_set_v1_device and batch_exists_v2. The indexer uses
        ``pool=indexer`` keys. Each physical component has its own coordinate,
        and every key is scoped by the binary model namespace passed at open.

        METRICS: the anchor device SG put reports on_set (a v1-device physical
        transfer, identical attribution to stock DSA whose anchor rides batch_set_v1),
        and the sidecar reports on_set_v2 — preserving the stock DSA metric split."""
        n = len(kv_keys)
        sidecar_transfers = sidecar_transfers or []
        with _tracing.span("batch_set_v2_device", n) as _sp, \
                access_log("batch_set_v2_device",
                           lambda: f"{self._alog_tag} kv={n} "
                                   f"{_fmt_pools(sidecar_transfers)}") as r:
            kv_res, kv_bytes, kv_secs = self._kv_device_set(kv_keys, kv_device_indices)
            results = {"kv": kv_res}
            # Main-KV device write reports on_set (v1-device), matching stock DSA's
            # anchor attribution; skip the metric on the MLA rank!=0 no-op.
            if n and not (self.is_mla and self.tp_rank != 0):
                self._metrics.on_set(pages=n, ok_pages=sum(kv_res),
                                     nbytes=kv_bytes, seconds=kv_secs)
            # Sidecar. Task 4: a DEVICE sidecar transfer (device_indices set,
            # host_indices None) RDMAs straight from its GPU index buffer; a host
            # sidecar transfer keeps the stock host v2 path. In bypass every sidecar
            # is device now, but keep the host branch for generality/safety.
            dev_side = [tr for tr in sidecar_transfers if _is_device_transfer(tr)]
            host_side = [tr for tr in sidecar_transfers if not _is_device_transfer(tr)]
            side_pages = side_ok = side_bytes = 0
            side_secs = 0.0
            for tr in dev_side:
                name = str(tr.name)
                res, nb, secs = self._sidecar_device_set(name, tr.keys, tr.device_indices)
                results[name] = res
                side_pages += len(res); side_ok += sum(res)
                side_bytes += nb; side_secs += secs
            if host_side:
                side = self._v2_io(host_side, putting=True)
                results.update(side)
                side_pages += sum(len(rs) for rs in side.values())
                side_ok += sum(sum(rs) for rs in side.values())
                side_bytes += self._v2_io_bytes; side_secs += self._v2_io_seconds
            if side_pages and not (self.is_mla and self.tp_rank != 0):
                self._metrics.on_set_v2(
                    pages=side_pages, ok_pages=side_ok,
                    nbytes=side_bytes, seconds=side_secs)
            r.result = f"kv {sum(kv_res)}/{n} (device-direct); " + _fmt_pool_results(
                {k: v for k, v in results.items() if k != "kv"})
            if self._put_retry_recovered:
                r.result += f" retry_ok={self._put_retry_recovered}"
            if _sp:
                _sp.hits = sum(sum(rs) for rs in results.values())
                _sp.bytes = kv_bytes + side_bytes
            return results

    def batch_get_v2_device(
        self, kv_keys, kv_device_indices, sidecar_transfers, extra_info=None
    ) -> dict:
        """DSA L2-bypass on-demand read (GLM-5.2): the read twin of
        batch_set_v2_device. The anchor "kv" pool RDMAs straight INTO its GPU slots
        via the device-direct SG get; a DEVICE sidecar transfer (task 4) RDMAs the
        indexer straight INTO its GPU index buffer too (no host staging / H2D),
        while a host sidecar transfer keeps the stock host v2 path.

        The kv keys/indices and sidecar keys mirror batch_set_v2_device exactly, so a
        page written there reads back byte-identical for BOTH components. Anchor read
        reports on_get (v1-device), sidecar on_get_v2 — the stock DSA read split."""
        n = len(kv_keys)
        sidecar_transfers = sidecar_transfers or []
        with _tracing.span("batch_get_v2_device", n) as _sp, \
                access_log("batch_get_v2_device",
                           lambda: f"{self._alog_tag} kv={n} "
                                   f"{_fmt_pools(sidecar_transfers)}") as r:
            kv_res, kv_bytes, kv_secs = self._kv_device_get(kv_keys, kv_device_indices)
            results = {"kv": kv_res}
            if n:
                self._metrics.on_get(pages=n, hit_pages=sum(kv_res),
                                     nbytes=kv_bytes, seconds=kv_secs)
            dev_side = [tr for tr in sidecar_transfers if _is_device_transfer(tr)]
            host_side = [tr for tr in sidecar_transfers if not _is_device_transfer(tr)]
            side_pages = side_hit = side_bytes = 0
            side_secs = 0.0
            for tr in dev_side:
                name = str(tr.name)
                res, nb, secs = self._sidecar_device_get(name, tr.keys, tr.device_indices)
                results[name] = res
                side_pages += len(res); side_hit += sum(res)
                side_bytes += nb; side_secs += secs
            if host_side:
                side = self._v2_io(host_side, putting=False)
                results.update(side)
                side_pages += sum(len(rs) for rs in side.values())
                side_hit += sum(sum(rs) for rs in side.values())
                side_bytes += self._v2_io_bytes; side_secs += self._v2_io_seconds
            if side_pages:
                self._metrics.on_get_v2(
                    pages=side_pages, hit_pages=side_hit,
                    nbytes=side_bytes, seconds=side_secs)
            r.result = f"kv {sum(kv_res)}/{n} (device-direct); " + _fmt_pool_results(
                {k: v for k, v in results.items() if k != "kv"})
            if _sp:
                _sp.hits = sum(sum(rs) for rs in results.values())
                _sp.bytes = kv_bytes + side_bytes
            return results

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        total = len(keys)
        with access_log("batch_exists_v2",
                        lambda: f"{self._alog_tag} {total} keys, "
                                f"{_fmt_pools(pool_transfers)}") as r:
            from sglang.srt.mem_cache.hicache_storage import PoolTransferResult, PoolHitPolicy
            # primary KV prefix
            kv_pages = self.batch_exists(keys)
            hit = {"kv": kv_pages} if kv_pages else {}
            final = kv_pages
            for tr in (pool_transfers or []):
                if final == 0:
                    break
                name = str(tr.name)
                sub = self._pool_sub(name)
                # Device-direct sidecars store explicit scatter-group keys, so
                # probe group 0 just as batch_exists does for the main KV. A
                # host sidecar stores the bare pool key.
                dev_side = name in getattr(self, "_sidecar_device_pools", {})
                sks = [self._sg_group_key(sk, 0) if dev_side else sk
                       for k in keys[:kv_pages]
                       for sk in self._pool_keys(name, k)]
                present = self._fold(self._batch_exist_flat(sks), kv_pages, sub)
                if tr.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                    # Only the *last* `trailing` pages of a candidate prefix need
                    # this pool (e.g. SWA sliding window / Mamba state) — matches
                    # SGLang's reference batch_exists_v2. The previous `all(present)`
                    # wrongly required every prefix page, collapsing SWA hits to 0
                    # once early window pages had been evicted.
                    trailing = max(1, len(tr.keys) if tr.keys else 1)
                    boundary = 0
                    for prefix_len in range(kv_pages, 0, -1):
                        if all(present[i] for i in
                               range(max(0, prefix_len - trailing), prefix_len)):
                            boundary = prefix_len
                            break
                else:  # ALL_PAGES
                    boundary = 0
                    for ok in present:
                        if not ok:
                            break
                        boundary += 1
                if boundary:
                    hit[name] = boundary
                final = min(final, boundary)
            result = PoolTransferResult(final, hit)
            r.result = (f"kv={result.kv_hit_pages}/{total} "
                        + ",".join(f"{k}={v}" for k, v in hit.items()
                                   if k != "kv")).strip()
            # The usable hit prefix (kv_hit_pages) over the probed candidate
            # prefix (total keys) is the L3 hit-rate signal for V4/DSA models,
            # where the *_v1 get counters only see empty anchor markers.
            if total:
                self._metrics.on_exists_v2(probe_pages=total,
                                           hit_pages=result.kv_hit_pages)
            return result

    # --- required abstract methods (non zero-copy / introspection) ---
    def exists(self, key) -> bool:
        subkeys = self._keys(key)
        with access_log(
            "exists",
            lambda: f"{self._alog_tag} {_key_label(subkeys[0])}",
        ) as r:
            found = True
            for subkey in subkeys:
                key_ptr, key_owner = make_key_buffer(subkey)
                found = self._lib.dfkv_exist(
                    self._h, key_ptr,
                    ctypes.c_uint64(len(subkey))) == 1
                del key_owner
                if not found:
                    break
            r.result = "found" if found else "not_found"
            return found

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        nbytes = 0
        sk = self._keys(key)[0]
        with access_log(
            "set",
            lambda: (
                f"{self._alog_tag} {_key_label(sk)}, {_fmt_bytes(nbytes)}"),
        ) as r:
            if value is None:
                r.result = "fail none"
                return False
            key_ptr, key_owner = make_key_buffer(sk)
            if hasattr(value, "data_ptr"):
                tensor = value.detach().cpu().contiguous()
                nbytes = tensor.numel() * tensor.element_size()
                ok = self._lib.dfkv_put(
                    self._h, key_ptr, ctypes.c_uint64(len(sk)),
                    ctypes.c_void_p(tensor.data_ptr()),
                    ctypes.c_uint64(nbytes)) == 0
            else:
                view = memoryview(value).cast("B")
                nbytes = len(view)
                buf = (ctypes.c_char * nbytes).from_buffer_copy(view)
                ok = self._lib.dfkv_put(
                    self._h, key_ptr, ctypes.c_uint64(len(sk)),
                    ctypes.cast(buf, ctypes.c_void_p),
                    ctypes.c_uint64(nbytes)) == 0
            del key_owner
            r.result = "ok" if ok else "fail"
            return ok

    def get(self, key, target_location=None, target_sizes=None):
        """Read a generic host page using the exact binary object key."""
        nbytes = 0
        sk = self._keys(key)[0]
        with access_log(
            "get",
            lambda: (
                f"{self._alog_tag} {_key_label(sk)}, {_fmt_bytes(nbytes)}"),
        ) as r:
            if target_location is None:
                r.result = "miss no_target"
                return None
            nbytes = target_location.numel() * target_location.element_size()
            key_ptr, key_owner = make_key_buffer(sk)
            rc = self._lib.dfkv_get(
                self._h, key_ptr, ctypes.c_uint64(len(sk)),
                ctypes.c_void_p(target_location.data_ptr()),
                ctypes.c_uint64(nbytes))
            del key_owner
            r.result = "hit" if rc == 1 else "miss"
            return target_location if rc == 1 else None

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        n = len(keys)
        with access_log("batch_set", lambda: f"{self._alog_tag} {n} keys") as r:
            if values is None:
                r.result = "fail none"
                return False
            # list (not short-circuiting all()) so every key is attempted and the
            # logged count is accurate; controller treats the bool as all-or-nothing.
            oks = [self.set(k, v) for k, v in zip(keys, values)]
            r.result = f"ok {sum(oks)}/{n}"
            return all(oks)

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        n = len(keys)
        with access_log("batch_get", lambda: f"{self._alog_tag} {n} keys") as r:
            if target_locations is None:
                r.result = "miss no_targets"
                return [None] * n
            res = [self.get(k, t) for k, t in zip(keys, target_locations)]
            r.result = f"hits={sum(1 for x in res if x is not None)}/{n}"
            return res
