"""DingoFS HiCache storage backend for SGLang (loaded via --hicache-storage-backend
dynamic). Zero-copy v1 path: hands raw host-buffer pointers from
mem_pool_host.get_page_buffer_meta() straight to the DingoFS KV client (C ABI).

Key scheme:
- MLA (GLM-5.1): one packed-latent object per page. The latent is replicated
  across *TP*, so the key has NO tp_rank suffix and only tp_rank 0 writes
  (backup_skip). PP splits the model by *layer*, so the latent is NOT replicated
  across PP stages — when pp_size > 1 every key carries _pp{pp_rank} (including
  MLA) so stages holding different layer-slices do not collide.
- MHA: two objects (_k/_v) per page, suffixed by tp_size/tp_rank (+ _pp{pp_rank}
  when PP is on).

This mirrors SGLang's reference HiCacheFile suffix (hicache_storage.py), where
`enable_pp` appends `_{pp_size}_{pp_rank}` unconditionally — including MLA.

This file is the production plugin. On a GPU host it imports the real SGLang
HiCacheStorage; the test harness supplies a no-torch shim with the same surface.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from typing import List, Optional

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

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


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


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
    lib_path = (path or os.environ.get("DFKV_LIB")
                or os.path.join(os.environ.get("DFKV_BUILD", "/home/ketor/dfkv-dev/build"),
                                "libdfkv.so"))
    lib = ctypes.CDLL(lib_path)
    lib.dfkv_open.restype = ctypes.c_void_p
    lib.dfkv_open.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                              ctypes.c_uint32]
    lib.dfkv_put.restype = ctypes.c_int
    lib.dfkv_put.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_get.restype = ctypes.c_int
    lib.dfkv_get.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_exist.restype = ctypes.c_int
    lib.dfkv_exist.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.dfkv_register_memory.restype = ctypes.c_int
    lib.dfkv_register_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.dfkv_batch_put.restype = ctypes.c_int
    lib.dfkv_batch_put.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                                   ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int)]
    lib.dfkv_batch_get.restype = ctypes.c_int
    lib.dfkv_batch_get.argtypes = lib.dfkv_batch_put.argtypes
    lib.dfkv_batch_exist.restype = ctypes.c_int
    lib.dfkv_batch_exist.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    # Scatter-gather put + live SG width, for the HiCache L2-bypass (device-direct
    # write) path. Each key gathers num_bufs[i] non-contiguous source buffers
    # (a page's per-layer GPU segments) into one stored blob. Guarded like the
    # vLLM connector: older libdfkv.so lacks the symbols and callers fall back.
    if hasattr(lib, "dfkv_batch_put_sg"):
        lib.dfkv_batch_put_sg.restype = ctypes.c_int
        lib.dfkv_batch_put_sg.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
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
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)]
    if hasattr(lib, "dfkv_max_sg_segs"):
        lib.dfkv_max_sg_segs.restype = ctypes.c_uint32
        lib.dfkv_max_sg_segs.argtypes = [ctypes.c_void_p]
    lib.dfkv_set_members.restype = ctypes.c_int
    lib.dfkv_set_members.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.dfkv_refresh_members.restype = ctypes.c_int
    lib.dfkv_refresh_members.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.dfkv_start_mds_discovery.restype = ctypes.c_int
    lib.dfkv_start_mds_discovery.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    # Client registration (additive >= dfkv with the /clients/<id> lease). Guarded
    # at the call site for older libdfkv.so without the symbol (same pattern as the
    # vLLM/LMCache connectors — see integration/vllm + integration/lmcache).
    if hasattr(lib, "dfkv_start_client_registration"):
        lib.dfkv_start_client_registration.restype = ctypes.c_int
        lib.dfkv_start_client_registration.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                       ctypes.c_char_p, ctypes.c_char_p,
                                                       ctypes.c_char_p, ctypes.c_int]
    lib.dfkv_transport_mode.restype = ctypes.c_char_p
    lib.dfkv_transport_mode.argtypes = [ctypes.c_void_p]
    lib.dfkv_set_batch_concurrency.restype = ctypes.c_int
    lib.dfkv_set_batch_concurrency.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
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


# A valid non-null pointer for zero-length "marker" puts (the logical-anchor
# "kv" pool of V4/DSA models). The payload is 0 bytes, so the pointer is never
# dereferenced, but it must be non-null so the batch slot isn't treated as a
# failed (null-key) entry by the C ABI.
_MARKER_BUF = ctypes.create_string_buffer(1)
_MARKER_PTR = ctypes.addressof(_MARKER_BUF)


def _is_device_transfer(tr) -> bool:
    """A sidecar PoolTransfer is DEVICE-direct (task 4) when it carries device slot
    indices and NO host indices — the indexer RDMAs from/into its GPU buffer. A host
    transfer (host_indices set) keeps the stock host v2 path."""
    return (getattr(tr, "host_indices", None) is None
            and getattr(tr, "device_indices", None) is not None)


def _arrays(subkeys, ptrs, sizes):
    """Build parallel C arrays (keys, ptrs, sizes) for a batch call."""
    n = len(subkeys)
    kbuf = [k.encode() for k in subkeys]
    karr = (ctypes.c_char_p * n)(*kbuf)
    parr = (ctypes.c_void_p * n)(*[ctypes.c_void_p(int(p)) for p in ptrs])
    sarr = (ctypes.c_uint64 * n)(*[int(s) for s in sizes])
    out = (ctypes.c_int * n)()
    return karr, parr, sarr, out, kbuf


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
        self.model = (storage_config.model_name or "").replace("/", "-")
        self.tp_rank = int(storage_config.tp_rank)
        self.tp_size = int(storage_config.tp_size)
        self.is_mla = bool(storage_config.is_mla_model)
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
        with access_log("init", lambda: f"{self._alog_tag} {self.model} "
                        f"tp={self.tp_rank}/{self.tp_size} mla={int(self.is_mla)}") as r:
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
                os.environ["DFKV_REQUIRE_RDMA"] = "1"
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
            # self._lib was loaded above (before configure) so the native version
            # could be reported; dfkv_open uses that same handle here.
            flags = _FLAG_IS_MLA if self.is_mla else 0
            # Refuse to start on a missing/invalid isolation identity (ring
            # endpoint already validated above). Opt out with
            # allow_shared_keyspace=1 for a single-model shared keyspace.
            model_hash = _tcfg.require_model_hash(cfg)
            self._h = self._lib.dfkv_open(
                members.encode(), model_hash,
                int(cfg.get("page_size", 64)), int(cfg.get("dtype_tag", 0)), flags,
                self.tp_size, self.tp_rank,
                int(cfg.get("layer_num", 0)), int(cfg.get("head_num", 0)),
                int(cfg.get("head_dim", 0)))
            if not self._h:
                raise RuntimeError("dfkv_open failed")
            # (base, size) of host regions already handed to dfkv_register_memory,
            # so a buffer shared across multiple hybrid pools (DSA registers the
            # KV anchor plus several sidecar pools) is registered exactly once.
            self._registered_regions = set()
            mode_b = self._lib.dfkv_transport_mode(self._h)
            self.transport_mode = (
                mode_b.decode("utf-8", errors="replace") if mode_b else "unknown"
            )
            if (_truthy(cfg.get("require_rdma")) or
                    _truthy(os.environ.get("DFKV_REQUIRE_RDMA"))):
                if self.transport_mode != "rdma":
                    self._lib.dfkv_close(self._h)
                    self._h = None
                    raise RuntimeError(
                        "dfkv requires RDMA zero-copy transport, "
                        f"got {self.transport_mode}"
                    )
            if cfg.get("batch_concurrency"):
                self._lib.dfkv_set_batch_concurrency(
                    self._h, ctypes.c_uint64(int(cfg["batch_concurrency"])))
            if mds:
                group = cfg.get("mds_group", "default")
                poll_ms = int(cfg.get("mds_poll_ms", 3000))
                rc = self._lib.dfkv_start_mds_discovery(self._h, mds.encode(), group.encode(), poll_ms)
                if rc != 0:
                    raise RuntimeError("dfkv_start_mds_discovery failed")
                # Register this SGLang HiCache connector as a cache consumer so
                # `dfkvctl clients` can answer "who is using dfkv" (parity with the
                # vLLM/LMCache connectors added in v1.15.0). Best-effort: a missing
                # symbol (older libdfkv.so) or a registration failure is logged,
                # never fatal — the data path is already up via discovery above.
                # Default on; opt out with extra_config client_register=0 or
                # DFKV_CLIENT_REGISTER=0. SGLang HiCache is a prefix L3 cache with
                # no producer/consumer split, so no 'role' field (the CLI shows '-'
                # for it, same as LMCache which has no role either).
                if _truthy(cfg.get("client_register",
                                    os.environ.get("DFKV_CLIENT_REGISTER", "1"))):
                    cid = _tcfg.resolve_connector_id(cfg, tp_rank=self.tp_rank)
                    info = (f"type={_tcfg.TYPE_HICACHE},model={self.model},"
                            f"tp_size={self.tp_size},tp_rank={self.tp_rank},"
                            f"ver={native_ver}")
                    try:
                        rc2 = self._lib.dfkv_start_client_registration(
                            self._h, mds.encode(), group.encode(), cid.encode(),
                            info.encode(), 10000)
                        if rc2 != 0:
                            raise RuntimeError(f"rc={rc2}")
                    except AttributeError:
                        pass  # older libdfkv.so without the symbol — skip silently
                    except Exception as e:  # noqa: BLE001 — never block startup
                        import warnings
                        warnings.warn(
                            f"dfkv client registration skipped (mds={mds!r}): {e}",
                            stacklevel=2)
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
        """Register a host memory region for RDMA zero-copy (registered once per
        connection; buffers inside it then transfer with no per-op MR register).
        No-op on TCP. Returns True on success."""
        if not base or not size:
            return False
        return self._lib.dfkv_register_memory(
            self._h, ctypes.c_void_p(int(base)), ctypes.c_uint64(int(size))) == 0

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
        self.registered_pools[name] = host_pool
        with access_log("register_mem_host_pool_v2",
                        lambda: f"{self._alog_tag} {name}") as r:
            n = 0
            try:
                n = self._register_pool_buffers(host_pool)
            except Exception as e:
                r.result = f"skip ({type(e).__name__})"
                return
            r.result = f"registered {n} region(s)" if n else "no backing buffer found"

    # --- L2-bypass (device-direct write) -------------------------------------
    def supports_device_transfer(self) -> bool:
        """True iff this backend can RDMA a page straight from GPU KV slots to L3.

        Requires the scatter-gather put AND get ABIs (dfkv_batch_put_sg /
        dfkv_batch_get_auto_sg) plus a negotiated SG width big enough to hold a
        page's per-layer device segments in ONE key (layer_num segments/key — the
        GPU pool is layer-first, so a page's KV is scattered across layer_num
        non-contiguous buffers). The get symbol is required too: increment 2 reads
        device-direct-written pages back via SG GET, and a page written with no
        matching reader is dead. Older libdfkv.so without the symbols, or an HCA
        whose max_sge is too small, keep the stock host (D2H) path. Checked by the
        SGLang controller's capability gate; never raises."""
        if not (hasattr(self._lib, "dfkv_batch_put_sg")
                and hasattr(self._lib, "dfkv_batch_get_auto_sg")
                and hasattr(self._lib, "dfkv_max_sg_segs")):
            return False
        layer_num = int(self.cfg.get("layer_num", 0))
        if layer_num <= 0:
            return False
        try:
            max_segs = int(self._lib.dfkv_max_sg_segs(self._h))
        except Exception:
            return False
        if max_segs < 1:
            return False
        if max_segs < layer_num:
            # Pages split into ceil(layer_num/max_segs) "@sg{n}" sub-keys
            # (_flatten_device) — same chunking the vLLM connector uses, so a
            # narrow HCA costs extra keys per page, not the bypass itself.
            print(f"[dfkv] L2-bypass: max_sg_segs={max_segs} < layer_num="
                  f"{layer_num}; chunking each page into "
                  f"{(layer_num + max_segs - 1) // max_segs} @sg sub-keys.",
                  file=sys.stderr, flush=True)
        return True

    def register_mem_pool_device(self, mem_pool_device):
        """Register the GPU KV pool's per-layer buffers for RDMA (GPUDirect MR;
        dfkv_register_memory accepts device pointers — same call the vLLM connector
        uses). Deduped against host regions already registered."""
        self.mem_pool_device = mem_pool_device
        from sglang.srt.mem_cache.device_page_meta import device_pool_regions

        with access_log("register_mem_pool_device",
                        lambda: f"{self._alog_tag}") as r:
            n = 0
            try:
                for base, size in device_pool_regions(mem_pool_device):
                    region = (base, size)
                    if not base or not size or region in self._registered_regions:
                        continue
                    if self.register_memory(base, size):
                        self._registered_regions.add(region)
                        n += 1
            except Exception as e:  # never fail setup over an optimization
                r.result = f"skip ({type(e).__name__})"
                return
            r.result = (f"registered {n} device region(s)" if n
                        else "no device buffer found")

    def register_mem_pool_device_sidecar(self, name, device_pool):
        """Task 4: register the DSA indexer sidecar's per-layer GPU buffers for
        RDMA (GPUDirect MR), so the indexer RDMAs straight from/into its device
        buffer instead of a host staging slot. `name` is the sidecar PoolName (e.g.
        'indexer'); `device_pool` is the DeepSeekV4IndexerPool holding
        index_k_with_scale_buffer. Deduped against regions already registered."""
        if not hasattr(self, "_sidecar_device_pools"):
            self._sidecar_device_pools = {}
        key = str(name)
        self._sidecar_device_pools[key] = device_pool
        from sglang.srt.mem_cache.device_page_meta import sidecar_device_pool_regions

        with access_log("register_mem_pool_device_sidecar",
                        lambda: f"{self._alog_tag} {key}") as r:
            n = 0
            try:
                for base, size in sidecar_device_pool_regions(device_pool):
                    region = (base, size)
                    if not base or not size or region in self._registered_regions:
                        continue
                    if self.register_memory(base, size):
                        self._registered_regions.add(region)
                        n += 1
            except Exception as e:  # never fail setup over an optimization
                r.result = f"skip ({type(e).__name__})"
                return
            r.result = (f"registered {n} sidecar device region(s)" if n
                        else "no sidecar device buffer found")

    def register_mem_pool_device_draft(self, mem_pool_device_draft):
        """Task 6: register the EAGLE draft model's GPU KV pool for RDMA so draft KV
        pages RDMA straight from/into their GPU slots (device-direct draft L3),
        mirroring register_mem_pool_device for the target pool. Deduped against
        regions already registered."""
        self.mem_pool_device_draft = mem_pool_device_draft
        from sglang.srt.mem_cache.device_page_meta import device_pool_regions

        with access_log("register_mem_pool_device_draft",
                        lambda: f"{self._alog_tag}") as r:
            n = 0
            try:
                for base, size in device_pool_regions(mem_pool_device_draft):
                    region = (base, size)
                    if not base or not size or region in self._registered_regions:
                        continue
                    if self.register_memory(base, size):
                        self._registered_regions.add(region)
                        n += 1
            except Exception as e:  # never fail setup over an optimization
                r.result = f"skip ({type(e).__name__})"
                return
            r.result = (f"registered {n} draft device region(s)" if n
                        else "no draft device buffer found")

    def register_mem_pool_device_draft_sidecar(self, mem_pool_device_draft):
        """DSA-draft extension of task 6: register the draft model's DSA indexer
        sidecar (index_k_with_scale_buffer) GPU buffers for RDMA so the draft indexer
        RDMAs device-direct alongside the draft main latent, keeping a DSA draft's KV
        coherent (latent + indexer) on an L3 hit. The draft pool object IS the
        DSATokenToKVPool (it carries both kv_buffer and index_k_with_scale_buffer), so
        we register the same object as a sidecar device pool under a distinct name and
        drive it through the shared _sidecar_device_set/get machinery (its own
        `draft_indexer` key namespace, layer-first @sg chunking). Deduped against
        regions already registered."""
        if not hasattr(self, "_sidecar_device_pools"):
            self._sidecar_device_pools = {}
        self._draft_sidecar_name = "draft_indexer"
        self._sidecar_device_pools[self._draft_sidecar_name] = mem_pool_device_draft
        from sglang.srt.mem_cache.device_page_meta import sidecar_device_pool_regions

        with access_log("register_mem_pool_device_draft_sidecar",
                        lambda: f"{self._alog_tag}") as r:
            n = 0
            try:
                for base, size in sidecar_device_pool_regions(mem_pool_device_draft):
                    region = (base, size)
                    if not base or not size or region in self._registered_regions:
                        continue
                    if self.register_memory(base, size):
                        self._registered_regions.add(region)
                        n += 1
            except Exception as e:  # never fail setup over an optimization
                r.result = f"skip ({type(e).__name__})"
                return
            r.result = (f"registered {n} draft indexer region(s)" if n
                        else "no draft indexer buffer found")

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

    def set_members(self, members: str):
        """Hot-swap cluster membership, e.g. 'n1=ip:12000,n2=ip:12000'."""
        self._lib.dfkv_set_members(self._h, members.encode())

    def refresh_members(self, seed: str) -> bool:
        """Discover cluster membership from a seed node ('ip:port') and apply it.
        Lets the cluster grow/shrink without restarting clients. Returns True on
        success (seed reachable and returned a non-empty member list)."""
        return self._lib.dfkv_refresh_members(self._h, seed.encode()) == 0

    def start_mds_discovery(self, mds_endpoints: str, group: str = "default", poll_ms: int = 3000) -> bool:
        """Start background MDS-based discovery. mds_endpoints: comma-separated 'ip:port' list.
        Returns True on success."""
        return self._lib.dfkv_start_mds_discovery(self._h, mds_endpoints.encode(), group.encode(), poll_ms) == 0

    # --- key scheme: MLA single object (no tp_rank suffix); MHA two objects ---
    # PP note: when pp_size > 1, every path appends _pp{pp_rank} so stages
    # holding different layer-slices of KV do not collide on the same key.
    # This applies to MLA too — PP splits by layer, the latent is NOT
    # replicated across PP stages (only across TP). See __init__.
    def _pp_suffix(self) -> str:
        return f"_pp{self.pp_rank}" if self.enable_pp else ""

    def _keys(self, page_hash: str) -> List[str]:
        if self.is_mla:
            return [f"{self.model}/{page_hash}_k{self._pp_suffix()}"]
        base = f"{self.model}/{page_hash}_{self.tp_size}_{self.tp_rank}{self._pp_suffix()}"
        return [base + "_k", base + "_v"]

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
        kbuf = [s.encode() for s in subkeys]
        karr = (ctypes.c_char_p * len(kbuf))(*kbuf)
        out = (ctypes.c_int * len(kbuf))()
        rc = self._lib.dfkv_batch_exist(self._h, karr, len(kbuf), out)
        if rc != 0:
            return [False] * len(kbuf)
        return [out[i] == 1 for i in range(len(kbuf))]

    def _note_logical_anchor_once(self):
        """One-time notice that the primary KV pool is a logical anchor.

        SGLang builds a *logical anchor* (LogicalHostPool) as the primary "kv"
        pool for V4/DSA multi-pool models such as GLM-5.2 — it carries no KV
        tensor, so get_page_buffer_meta() returns None. The real KV rides the v2
        side-pool path (batch_set_v2/batch_get_v2); the v1 anchor only writes an
        empty "kv" marker so the v2 existence check can anchor the hit prefix
        (mirrors SGLang's reference backend). Informational, not an error — the
        path works, but it is newly enabled, so surface it once for ops to
        confirm the hit rate via metrics."""
        if self._anchor_noop_warned:
            return
        self._anchor_noop_warned = True
        import sys as _sys
        print("[dfkv] NOTE: primary KV pool is a logical anchor "
              "(get_page_buffer_meta -> None) — a V4/DSA multi-pool model "
              f"(model={self.model!r}, e.g. GLM-5.2). The v1 anchor writes an "
              "empty 'kv' marker; real KV rides the v2 side-pool path. Verify L3 "
              "hit rate via metrics; if low, the LMCache MP connector "
              "(docs/CONNECTORS.md #4.5) is the alternative path for GLM-5.x DSA.",
              file=_sys.stderr, flush=True)

    def _write_anchor_markers(self, keys) -> List[bool]:
        """Write an empty (0-byte) marker object per "kv" sub-key so a later
        batch_exists / batch_exists_v2 can find the primary-pool prefix.

        Used only for the logical-anchor case (V4/DSA, e.g. GLM-5.2): the anchor
        pool holds no KV buffer, so there is nothing to zero-copy — but SGLang's
        v2 existence check still gates the hit prefix on the primary "kv" keys,
        exactly as its reference backend does by writing an empty get_data_page().
        The marker carries the connector's geometry header (so a cross-geometry
        reader still misses); the matching read (batch_get_v1) is a no-op. Returns
        a per-page success list (a page succeeds iff all its sub-object markers
        were written)."""
        sub = self._sub()
        sks = [sk for k in keys for sk in self._keys(k)]
        if not sks:
            return []
        karr, parr, sarr, out, _ = _arrays(sks, [_MARKER_PTR] * len(sks),
                                           [0] * len(sks))
        self._lib.dfkv_batch_put(self._h, karr, parr, sarr, len(sks), out)
        return self._fold([out[i] == 1 for i in range(len(sks))], len(keys), sub)

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
                # an empty "kv" marker per page so the v2 existence check can
                # anchor the hit prefix (SGLang's reference backend does the same
                # with an empty get_data_page()). Single-pool models (MLA/MHA)
                # never return None, so this branch is inert for them.
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

    def _flatten_device(self, keys, seg_ptrs, seg_sizes, keys_fn=None, sub=None):
        """Expand per-page keys into per-sub-object (k[/v]) sub-keys, pairing each
        with its per-layer device segment list. Parallel to _flatten, but every
        entry is a segment LIST (one per layer), not a single (ptr, size).

        A page's layer_num segments can exceed the HCA's per-key SG budget
        (max_sge-1, e.g. 29 on ConnectX < 36/61 layers), so each sub-object is
        chunked into "@sg{n}" sub-keys of <= _sg_width() consecutive layers —
        the same scheme (and key suffix) the vLLM connector uses. Write and read
        derive the identical deterministic split from (layer count, width), so
        the layer-major bytes reassemble exactly. Chunk count is uniform across
        pages, so _fold()'s per-page stride is sub * nchunks.

        keys_fn/sub default to the main-KV key scheme (self._keys / self._sub); the
        DSA indexer sidecar and the EAGLE draft pool pass their own key builder and
        object count so they reuse the identical @sg chunking under a distinct
        namespace (task 4: device-direct sidecar; task 6: device-direct draft)."""
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
                    sks.append(f"{sk}@sg{ci}")
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
        karr = (ctypes.c_char_p * n)(*[k.encode() for k in sks])
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
        rc = self._lib.dfkv_batch_put_sg(self._h, karr, parr, sarr, narr, n, out)
        if rc != 0:
            return [False] * n
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
        karr = (ctypes.c_char_p * n)(*[k.encode() for k in sks])
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
            self._h, karr, parr, carr, narr, n, out_hit, out_len)
        if rc != 0:
            return [0] * n, [0] * n
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
                # batch_exists_v2 (via the empty markers written on backup); there
                # is no anchor payload to load. Report all pages present so
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
            karr, parr, sarr, out, _kb = _arrays(sks, sp, ss)
            t0 = time.perf_counter()
            self._lib.dfkv_batch_get(self._h, karr, parr, sarr, len(sks), out)
            dur = time.perf_counter() - t0
            res = self._fold([out[i] == 1 for i in range(len(sks))], n, sub)
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
            # Device-direct (L2-bypass) writes store "@sg{n}" chunk sub-keys, so
            # existence is probed on "@sg0" — the vLLM connector's probe scheme;
            # the bare sub-key never matches a chunked store. A bypass instance
            # only ever writes chunked keys (model_hash-isolated), so one probe
            # form suffices per mode.
            sub = self._sub()
            if getattr(self, "mem_pool_device", None) is not None:
                sks = [f"{sk}@sg0" for k in keys for sk in self._keys(k)]
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
    def _pool_keys(self, pool_name: str, page_hash: str) -> List[str]:
        # primary KV pool keeps the MLA/MHA split; auxiliary pools are single-object.
        # Both carry the PP suffix for the same layer-slice reason as _keys().
        pps = self._pp_suffix()
        if pool_name in ("kv", "__default__"):
            return self._keys(page_hash)
        base = f"{self.model}/{page_hash}_{pool_name}{pps}"
        return [base + "_k"] if self.is_mla else [base + "_k", base + "_v"]

    def _pool_sub(self, pool_name: str) -> int:
        if pool_name in ("kv", "__default__"):
            return self._sub()
        return 1 if self.is_mla else 2

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
            karr, parr, sarr, out, _kb = _arrays(
                [sks[i] for i in todo], [sp[i] for i in todo],
                [ss[i] for i in todo])
            self._lib.dfkv_batch_put(self._h, karr, parr, sarr,
                                     len(todo), out)
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
            # MLA backup_skip: only tp_rank 0 writes the replicated latent pools.
            if putting and self.is_mla and self.tp_rank != 0:
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
            karr, parr, sarr, out, _ = _arrays(sks, sp, ss)
            self._lib.dfkv_batch_get(self._h, karr, parr, sarr, len(sks), out)
            flat = [out[i] == 1 for i in range(len(sks))]
        else:
            flat = []
        # Expose the actual I/O cost/volume for the v2 metrics (the caller reads
        # these). ss covers only pools that did I/O (MLA backup_skip pools were
        # `continue`d before appending), so bytes are 0 on a pure skip.
        self._v2_io_bytes = sum(ss)
        self._v2_io_seconds = time.perf_counter() - t0
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
            # MLA rank!=0 replicated latent is a no-op skip ([True] markers, no
            # I/O) — don't inflate the write-ok metric with it (mirrors
            # batch_set_v1, which returns before on_set on backup_skip).
            if nkeys and not (self.is_mla and self.tp_rank != 0):
                self._metrics.on_set_v2(
                    pages=sum(len(rs) for rs in res.values()),
                    ok_pages=sum(sum(rs) for rs in res.values()),
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
        return self._fold(flat, n, sub), nbytes, dur

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
        """Print the FIRST key exist actually probed, throttled to one line/30s.

        Pairs with _log_read_reject to settle a sharp contradiction seen on
        2026-07-25: exist reports prefix=1562/1562 (every candidate page present)
        while the device GET that follows reports 4686/4686 sub-objects MISS —
        and 4686 = 1562 x 3 chunks, so the two are talking about the same NUMBER
        of pages. Same key scheme on both sides (`{model}/{hash}_k@sg0`), so if
        the strings still differ it can only be the page_hash itself: exist hashes
        the candidate token sequence, while the GET uses the marker nodes'
        hash_value. Printing both first-keys makes that comparison direct instead
        of inferential."""
        if not sks:
            return
        now = time.monotonic()
        if now - getattr(self, "_exist_probe_logged", 0.0) < 30.0:
            return
        self._exist_probe_logged = now
        _log.warning(
            "dfkv exist probe: prefix=%d/%d first_key=%s last_key=%s",
            n, total, sks[0], sks[-1])

    def _log_read_reject(self, sks, hits, lens, want, flat_ok, nmain):
        """Say WHY a device read rejected sub-objects, throttled to one line/30s.

        Motivation (2026-07-25): SGLang's promotion drops whole loads with
        `0 pages verified across ranks` 24-56 times per hot-L3 round even on a
        clean baseline — exist reports the pages ARE in L3, yet the read is not
        accepted. Without this line the two candidate causes are indistinguishable
        from the outside:

          * MISS  (hits[i] != 1)        — the sub-key is not there at all, i.e. an
                                          exist/read key-scheme disagreement;
          * SHORT (lens[i] < want[i])   — it is there but returned fewer bytes than
                                          the destination expects, i.e. a layout /
                                          @sg-chunking / capacity disagreement.

        Reports the first offender's numbers so `want` vs `got` can be compared
        directly against the page geometry."""
        now = time.monotonic()
        if now - getattr(self, "_read_reject_logged", 0.0) < 30.0:
            return
        self._read_reject_logged = now
        miss = sum(1 for i in range(nmain) if hits[i] != 1)
        short = sum(1 for i in range(nmain)
                    if hits[i] == 1 and lens[i] < want[i])
        first = next(i for i in range(nmain) if not flat_ok[i])
        _log.warning(
            "dfkv device read rejected %d/%d sub-objects (%d MISS, %d SHORT); "
            "first offender: key=%s hit=%s got=%d want=%d",
            nmain - sum(flat_ok), nmain, miss, short,
            sks[first], hits[first], lens[first], want[first])

    # --- task 4: DSA indexer sidecar device-direct (no host staging) -----------
    def _sidecar_device_set(self, name, keys, device_indices):
        """Device-direct SG put of the DSA indexer sidecar `name` (its own key
        namespace, _pool_keys(name, hash), @sg-chunked like the main KV). The
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
    def _draft_keys(self, page_hash: str, sub: int) -> List[str]:
        """Draft-model KV sub-keys, a distinct namespace from the target pages
        (`.draft` infix, PP-aware). The draft model may be MLA or MHA INDEPENDENT of
        the target, so the key scheme follows the DRAFT pool shape (sub), mirroring
        _keys: MLA draft (sub=1) is TP-replicated -> no tp_rank suffix; MHA draft
        (sub=2) is TP-sharded -> tp_size/tp_rank suffix so ranks do not collide."""
        pps = self._pp_suffix()
        if sub == 1:
            return [f"{self.model}/{page_hash}.draft_k{pps}"]
        base = f"{self.model}/{page_hash}.draft_{self.tp_size}_{self.tp_rank}{pps}"
        return [base + "_k", base + "_v"]

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

        VALUE LAYOUT (explicit): the main KV is stored under the v1-style keys
        (_keys(hash) -> "model/hash_k"), byte-for-byte the same key scheme
        batch_set_v1_device / batch_exists use — so an unchanged batch_exists_v2
        anchors the hit prefix on the SAME "kv" keys with no change. The sidecar
        rides its own keys (_pool_keys('indexer', hash) -> "model/hash_indexer_k",
        @sg-chunked when device-direct). The two components never collide; the
        composite is split honestly across two key namespaces, isolated by model_hash.

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
                # Device-direct sidecars (task 4) store "@sg{n}" chunk sub-keys, so
                # probe "@sg0" — the same scheme batch_exists uses for the main KV.
                # A host sidecar stores the bare sub-key.
                dev_side = name in getattr(self, "_sidecar_device_pools", {})
                sks = [(sk + "@sg0") if dev_side else sk
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
        with access_log("exists", lambda: f"{self._alog_tag} {key}") as r:
            found = all(self._lib.dfkv_exist(self._h, sk.encode()) == 1
                        for sk in self._keys(key))
            r.result = "found" if found else "not_found"
            return found

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        nbytes = 0
        with access_log("set",
                        lambda: f"{self._alog_tag} {key}, {_fmt_bytes(nbytes)}") as r:
            if value is None:
                r.result = "fail none"
                return False
            sk = self._keys(key)[0]
            # SGLang's L3 backup path (_generic_page_set -> batch_set) passes torch
            # Tensors, not bytes. Take the raw tensor bytes via data_ptr (dtype-
            # agnostic, works for fp8 which numpy can't represent). Tensor must stay
            # alive across the call (local `t`).
            if hasattr(value, "data_ptr"):
                t = value.detach().cpu().contiguous()
                nbytes = t.numel() * t.element_size()
                ok = self._lib.dfkv_put(self._h, sk.encode(),
                                        ctypes.c_void_p(t.data_ptr()),
                                        ctypes.c_uint64(nbytes)) == 0
            else:
                mv = memoryview(value).cast("B")
                nbytes = len(mv)
                buf = (ctypes.c_char * nbytes).from_buffer_copy(mv)
                ok = self._lib.dfkv_put(self._h, sk.encode(),
                                        ctypes.cast(buf, ctypes.c_void_p),
                                        ctypes.c_uint64(nbytes)) == 0
            r.result = "ok" if ok else "fail"
            return ok

    def get(self, key, target_location=None, target_sizes=None):
        # Generic (non zero-copy) read: dfkv_get reads the page bytes straight
        # into target_location's buffer (a host flat-page tensor). Symmetric with
        # set() (whole page under _keys[0]). Returns target_location on hit, None
        # on miss. SGLang's prod path uses batch_get_v1; this serves the generic
        # path + direct/test callers.
        nbytes = 0
        with access_log("get",
                        lambda: f"{self._alog_tag} {key}, {_fmt_bytes(nbytes)}") as r:
            if target_location is None:
                r.result = "miss no_target"
                return None
            sk = self._keys(key)[0]
            nbytes = target_location.numel() * target_location.element_size()
            rc = self._lib.dfkv_get(self._h, sk.encode(),
                                    ctypes.c_void_p(target_location.data_ptr()),
                                    ctypes.c_uint64(nbytes))
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
