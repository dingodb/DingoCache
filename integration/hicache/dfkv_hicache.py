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

import collections
import contextlib
import ctypes
import functools
import hashlib
import inspect
import threading
import logging
import os
import sys
import time
from typing import List, Optional

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig
from dfkv_common import (
    DfkvClientOptionsV2,
    SGLANG_HICACHE_RAW_V1,
    apply_rank_local_rail_affinity,
    canonical_namespace,
    reject_namespace_override,
    make_client_options_v2,
    make_key_array,
    make_key_buffer,
    pool_key,
    sg_key,
)
from dfkv_common import cabi as dfkv_cabi
from dfkv_common.client_metrics import read_native_snapshot
from dfkv_common.block_size import get_max_block_bytes, validate_object_sizes

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


def _gpu_load_fence_enabled() -> bool:
    return os.environ.get("DFKV_GPU_LOAD_FENCE", "1") == "1"


def _gpu_load_fence(device=None) -> None:
    """Order GPUDirect RDMA writes before subsequent CUDA kernels.

    The GPUDirect RDMA design guide (Synchronization and Memory Ordering)
    requires that writes landing via the GPU BAR complete before the CPU
    thread returns to launch dependent kernels; a completion reaped from the
    CQ proves transmission, not arrival in device memory. Drivers reject
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS on VMM/cuMemMap pools (vLLM LBHNC), so
    the connector issues a device synchronization as the ordering fence
    after any hit scatter. ``device`` is the GPU the bytes landed on; None
    synchronizes the calling thread's current device. Disable with
    DFKV_GPU_LOAD_FENCE=0.
    """
    if not _gpu_load_fence_enabled():
        return
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _cuda_device_of(pool) -> Optional[int]:
    """CUDA device index of a device pool's buffers, if they say."""
    for attr in ("kv_buffer", "k_buffer", "index_k_with_scale_buffer"):
        tensors = getattr(pool, attr, None)
        if tensors is None or len(tensors) == 0:
            continue
        device = getattr(tensors[0], "device", None)
        if getattr(device, "type", None) != "cuda":
            return None
        index = getattr(device, "index", None)
        if isinstance(index, int) and not isinstance(index, bool):
            return index
    return None


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


def _probe_parallel_context():
    """SGLang's parallel context, and why there is none.

    Returns ``(context, None)``, or ``(None, None)`` when this SGLang build
    has no parallel-context API at all, or ``(None, exc)`` when the API is
    there and did not deliver: the state is not initialized yet, or it has
    changed shape. The two failures are not alike. A build without the API
    describes its only CP axis in ``HiCacheStorageConfig``; a build with it
    may run decode CP, whose coordinates exist nowhere else.
    """
    try:
        try:
            from sglang.srt.runtime_context import get_parallel
        except ImportError:
            from sglang.srt.distributed import get_parallel
    except ImportError:
        return None, None
    try:
        return get_parallel(), None
    except (ImportError, AttributeError, AssertionError, RuntimeError) as exc:
        return None, exc


def _load_parallel_context():
    """Return SGLang's initialized parallel context when available."""
    return _probe_parallel_context()[0]


def _resolve_local_physical_rank(parallel=None) -> int:
    """Resolve the node-local process/device coordinate used for HCA affinity.

    This is deliberately independent of TP/DP/PCP/DCP semantics. SGLang's
    world-group local rank is the cross-platform device-assignment coordinate
    for CUDA/ROCm, NPU, XPU, and MUSA backends.
    """
    if parallel is None:
        parallel = _load_parallel_context()
    group = None
    if parallel is not None:
        try:
            group = parallel.world_group
        except (AttributeError, AssertionError, RuntimeError):
            pass
    if group is None:
        try:
            from sglang.srt.distributed.parallel_state import get_world_group
            group = get_world_group()
        except (ImportError, AttributeError, AssertionError, RuntimeError):
            pass
    rank = getattr(group, "local_rank", None) if group is not None else None
    if isinstance(rank, bool):
        raise ValueError("SGLang world-group local_rank must be an integer")
    try:
        rank = int(rank)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rail_affinity requires SGLang world_group.local_rank; "
            "TP/DP/PCP/DCP ranks are not physical affinity coordinates"
        ) from exc
    if rank < 0:
        raise ValueError("SGLang world-group local_rank must be non-negative")
    return rank


def _runtime_flag(parallel, name: str):
    if parallel is None:
        return None
    try:
        value = getattr(parallel, name)
    except (AttributeError, AssertionError, RuntimeError):
        return None
    return bool(value)


def _resolve_prefill_cp_storage_layout(
    cfg: dict,
    *,
    is_mla: bool,
    pcp_size: int,
    dcp_size: int,
    parallel,
    attn_tp_rank: Optional[int] = None,
) -> Optional[str]:
    """Validate the explicit CP storage identity contract.

    SGLang's ``is_mla_model`` establishes TP replication, not a permanent
    promise that every future CP implementation stores replicated bytes. CP
    deployments therefore declare the storage layout explicitly; ambiguous or
    truly sharded layouts fail closed instead of aliasing distinct payloads.

    Two MLA+CP layouts are supported, distinguished by whether the CP ranks
    hold the same bytes:

    ``replicated``
        Every CP rank holds identical latent bytes. One rank is elected the
        writer and all ranks share its key, so L3 stores one copy.
    ``per_cp_rank``
        Each CP rank holds a *different* payload, so each rank persists under
        its own CP key coordinate and every elected replica writer writes.
        This is the layout required by ``enable_dsa_cache_layer_split``, where
        the DSA cache is split by layer across the CP ranks; it is rejected
        without that flag because duplicating replicated bytes ``pcp_size``
        times would waste L3 capacity rather than enable anything.
    """
    accepted = ("replicated", "sharded", "per_cp_rank")
    raw = cfg.get("prefill_cp_storage_layout")
    if raw is not None:
        if not isinstance(raw, str):
            raise ValueError(
                "prefill_cp_storage_layout must be one of " + ", ".join(accepted)
            )
        raw = raw.strip().lower()
        if raw not in accepted:
            raise ValueError(
                "prefill_cp_storage_layout must be one of " + ", ".join(accepted)
            )
    if pcp_size <= 1:
        return raw
    if not is_mla:
        if raw is not None:
            raise ValueError(
                "prefill_cp_storage_layout is supported only for rank-replicated MLA"
            )
        return None
    layer_split = _runtime_flag(parallel, "enable_dsa_cache_layer_split")
    if raw is None:
        raise ValueError(
            "MLA with attention CP requires an explicit "
            "prefill_cp_storage_layout: 'per_cp_rank' when "
            "enable_dsa_cache_layer_split splits the DSA cache across CP "
            "ranks, otherwise 'replicated'; sharded CP is unsupported"
        )
    if raw == "sharded":
        raise ValueError(
            "prefill_cp_storage_layout='sharded' is unsupported: SGLang elects "
            "one MLA storage writer, so dfkv cannot persist every CP shard. "
            "Use 'per_cp_rank' when every CP rank runs its own backend"
        )
    if dcp_size > 1:
        # Physical object keys carry only the PCP coordinate, so a second CP
        # axis would alias distinct DCP shards onto one key.
        raise ValueError(
            f"{raw} Prefill-CP storage is incompatible with DCP; "
            "keep distinct DCP storage coordinates"
        )
    prefill_enabled = _runtime_flag(parallel, "enable_prefill_cp")
    if prefill_enabled is False:
        raise ValueError(
            f"prefill_cp_storage_layout='{raw}' requires enable_prefill_cp"
        )
    if raw == "replicated":
        if layer_split:
            raise ValueError(
                "replicated Prefill-CP storage is incompatible with "
                "enable_dsa_cache_layer_split: the CP ranks hold different "
                "layers, so use prefill_cp_storage_layout='per_cp_rank'"
            )
        return raw
    if not layer_split:
        raise ValueError(
            "prefill_cp_storage_layout='per_cp_rank' requires "
            "enable_dsa_cache_layer_split; replicated CP ranks hold identical "
            "bytes and must share one key"
        )
    if attn_tp_rank is None:
        # Writer election compares the attention-TP rank, because the latent is
        # replicated across attn-TP but not across CP. Without that coordinate
        # the election degrades to the global tp_rank, electing a single writer
        # for the whole world and silently dropping pcp_size-1 of the shards.
        raise ValueError(
            "prefill_cp_storage_layout='per_cp_rank' requires SGLang's "
            "attn_tp_rank so one writer is elected per CP rank"
        )
    return raw

def _resolve_parallel_coordinates(
    cfg: dict,
    parallel=None,
    *,
    probe=None,
    storage_config=None,
    sources: Optional[dict] = None,
) -> tuple[tuple[int, int], tuple[int, int], Optional[int]]:
    """Resolve SGLang's physical CP coordinates, with explicit overrides.

    Recent SGLang releases do not copy attention-CP/DCP coordinates into the
    dynamic HiCache backend's ``extra_config``. Discover them from the initialized
    parallel state so sharded ranks cannot alias the default ``0/1`` keys.

    Without the parallel state an axis that ``extra_config`` does not give is
    never assumed to be unsharded just because nothing said otherwise: a rank
    of a sharded deployment that stores under ``0/1`` reads another rank's
    bytes as its own, with matching sizes. An axis is then taken from
    ``storage_config`` where it carries both its size and its rank: PCP from
    ``attn_cp_size``/``attn_cp_rank`` (which SGLang always passes), DCP from
    ``dcp_size``/``dcp_rank`` (which builds with decode CP pass). Otherwise
    DCP is ``0/1`` only on a build that has no parallel-context API and so no
    decode CP; when the API is there but failed, the coordinate is unknown
    and resolution raises. A context whose fields raise when read counts as
    a failed API.

    ``probe`` is the result of _probe_parallel_context() if the caller has
    already probed; otherwise it probes when ``parallel`` is not given.
    ``sources``, if given, receives where each axis came from, and under
    ``"context"`` whether the parallel context was ``available``,
    ``missing`` (no API) or ``failed``.
    """
    failure = None
    if parallel is None:
        parallel, failure = probe if probe is not None else _probe_parallel_context()

    def note(name, source):
        if sources is not None:
            sources[name] = source

    def read(attr):
        """A context field; a raising field fails the context as a whole."""
        nonlocal parallel, failure
        if parallel is None:
            return None
        try:
            return getattr(parallel, attr, None)
        except (AssertionError, RuntimeError) as exc:
            parallel, failure = None, exc
            return None

    runtime_fields = {
        "pcp": ("attn_cp_size", "attn_cp_rank"),
        "dcp": ("attn_dcp_size", "attn_dcp_rank"),
    }
    storage_fields = {
        "pcp": ("attn_cp_size", "attn_cp_rank"),
        "dcp": ("dcp_size", "dcp_rank"),
    }
    runtime_values = {}
    for name, (size_attr, rank_attr) in runtime_fields.items():
        runtime_values[name] = (read(size_attr), read(rank_attr))
    attn_tp_rank = read("attn_tp_rank")
    if parallel is None:
        attn_tp_rank = None   # a later field failed the whole context

    resolved = {}
    for name in runtime_fields:
        size_attr, rank_attr = storage_fields[name]
        if f"{name}_size" in cfg or f"{name}_rank" in cfg:
            resolved[name] = _physical_axis(cfg, name)
            note(name, "extra_config")
        elif parallel is not None:
            note(name, "parallel_context")
            size_value, rank_value = runtime_values[name]
            if name == "dcp" and size_value is None and rank_value is None:
                # Legacy SGLang exposes only attn_cp_*: that single axis is
                # already represented by PCP above. It has no independent DCP
                # shard, so duplicating attn_cp_* here would corrupt keys and
                # replica-writer election.
                resolved[name] = (1, 0)
            else:
                resolved[name] = _physical_axis(
                    {
                        f"{name}_size": size_value,
                        f"{name}_rank": rank_value,
                    },
                    name,
                )
        elif (storage_config is not None
              and hasattr(storage_config, size_attr)
              and hasattr(storage_config, rank_attr)):
            resolved[name] = _physical_axis(
                {
                    f"{name}_size": getattr(storage_config, size_attr),
                    f"{name}_rank": getattr(storage_config, rank_attr),
                },
                name,
            )
            note(name, "storage_config")
        elif failure is not None:
            raise RuntimeError(
                f"SGLang's parallel context could not be read "
                f"({type(failure).__name__}: {failure}), so the {name} "
                f"coordinate of this rank is unknown. Storing under "
                f"{name}=0/1 could serve another rank's KV as this rank's. "
                f"Pass {name}_size and {name}_rank in the HiCache "
                f"extra_config, or create the storage backend after SGLang "
                f"has initialized its parallel state"
            ) from failure
        else:
            resolved[name] = (1, 0)
            note(name, "default")

    note("context", "available" if parallel is not None
         else "failed" if failure is not None else "missing")
    return (resolved["pcp"], resolved["dcp"],
            int(attn_tp_rank) if attn_tp_rank is not None else None)


def _is_mla_replica_writer(
    is_mla: bool,
    tp_rank: int,
    attn_tp_rank: Optional[int],
    dcp_size: int,
    dcp_rank: int,
) -> bool:
    """Elect one writer for each physical PCP/DCP shard.

    Under DCP the writer of shard ``d`` is the attention-TP rank ``d``. That
    gives every shard exactly one writer only if the DCP rank is the
    attention-TP rank modulo the DCP width, as SGLang builds it (DCP groups
    are contiguous slices of the TP group, attention-TP ranks change
    fastest). Under another mapping some shards would have no writer and
    never hit, with backup reported done, so a rank whose coordinates do not
    follow it is refused.
    """
    if not is_mla:
        return True
    rank = attn_tp_rank if attn_tp_rank is not None else tp_rank
    if dcp_size > 1 and dcp_rank != rank % dcp_size:
        raise RuntimeError(
            f"MLA writer election under DCP needs dcp_rank == attention-TP "
            f"rank % dcp_size; this rank has dcp_rank={dcp_rank}, rank={rank}, "
            f"dcp_size={dcp_size}. Another rank layout would leave DCP shards "
            f"without a writer; check dcp_size/dcp_rank in the HiCache "
            f"extra_config")
    return rank == (dcp_rank if dcp_size > 1 else 0)



def _is_hicache_storage_writer(
    *,
    is_mla: bool,
    tp_rank: int,
    attn_tp_rank: Optional[int],
    pcp_rank: int,
    dcp_size: int,
    dcp_rank: int,
    prefill_cp_storage_layout: Optional[str],
) -> bool:
    writer = _is_mla_replica_writer(
        is_mla, tp_rank, attn_tp_rank, dcp_size, dcp_rank
    )
    if (
        writer
        and is_mla
        and prefill_cp_storage_layout == "replicated"
        and pcp_rank != 0
    ):
        return False
    return writer

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
    """Read one complete native client metrics snapshot."""
    return read_native_snapshot(lib, h)


# One-byte marker for the logical-anchor "kv" pool of V4/DSA models. Native
# dfkv rejects zero-length PUTs because they allocate objects without consuming
# byte quota; charging one byte preserves the existence-marker semantics.
_LAYOUT_FIELDS = ("layout", "dtype")


def _runtime_layout_facts(pool, fields=_LAYOUT_FIELDS) -> tuple:
    """What one registered pool states about the byte format of its objects.

    Same-size objects can hold different bytes: an MHA host pool in
    ``page_first`` (token-major) and one in ``page_first_direct`` or
    ``page_head`` give pages of the same size in different orders, two
    element types of one width (bf16 and fp16) differ only in value, and
    two pool classes can pack the same layout and dtype differently (an
    MXFP8 MHA pool, a DSA pool). The exact-size read check cannot tell them
    apart; these facts can: the pool's class, ``layout`` and ``dtype``.
    A field the pool does not have is recorded as absent. Device pools have
    no layout choice and are described by class and ``dtype`` only.
    """
    return (("class", type(pool).__name__),) + tuple(
        (field, str(getattr(pool, field, ""))) for field in fields)


def _layout_tag(facts, objects: int) -> str:
    """Key tag of one pool format: its facts and its objects per page."""
    return hashlib.sha256(repr((facts, int(objects))).encode()).hexdigest()[:16]


_MARKER_BUF = ctypes.create_string_buffer(1)
_MARKER_PTR = ctypes.addressof(_MARKER_BUF)


# What one transfer did: bytes moved, seconds in native I/O, the names of the
# pools that did I/O (a skipped replicated pool is not among them), and the
# objects whose put failed once and succeeded on the retry.
_V2Io = collections.namedtuple(
    "_V2Io", ("nbytes", "seconds", "pools", "recovered"))


def _pool_transfer_result(cls, final, hit, restorable):
    """A PoolTransferResult, with the restorable prefix lengths if the
    SGLang build has that field: with a trailing pool they are not every
    length up to ``final``, and the controller intersects them across
    ranks instead of taking each rank's longest."""
    if "restorable_prefix_pages" in getattr(cls, "__dataclass_fields__", {}):
        return cls(final, hit, restorable)
    return cls(final, hit)


def _meta_has_no_layout(meta) -> bool:
    """True when get_page_buffer_meta() explicitly declares "no physical layout".

    Two wire forms exist for the logical-anchor pool of V4/DSA models:
    - None                       (LogicalHostPool, SGLang <= v0.5.13)
    - a 2-tuple containing None  (DeepSeekV4PagedHostPool, SGLang v0.5.17+)
    Both mean "this pool holds no host buffer"; the real KV rides side pools.
    Malformed shapes (wrong arity, empty/mismatched components) are NOT
    treated as logical — they stay hard errors in the callers.
    """
    if meta is None:
        return True
    return (isinstance(meta, (tuple, list)) and len(meta) == 2
            and (meta[0] is None or meta[1] is None))


def _is_device_transfer(tr) -> bool:
    """A sidecar PoolTransfer is DEVICE-direct (task 4) when it carries device slot
    indices and NO host indices — the indexer RDMAs from/into its GPU buffer. A host
    transfer (host_indices set) keeps the stock host v2 path."""
    return (getattr(tr, "host_indices", None) is None
            and getattr(tr, "device_indices", None) is not None)


def _l3_page_span(host_pool) -> int:
    """Host slots one L3 key covers: the pool's page, widened by its DCP size.

    Under DCP SGLang addresses a host pool by logical pages of
    ``page_size * dcp_size`` slots (one physical page per DCP rank) and
    ``get_page_buffer_meta`` asserts whole logical pages. Pools without DCP
    (and KDA state pools, page 1) keep their page size. Computed here instead
    of importing SGLang's ``l3_page_span`` so builds without it still load.
    """
    page_size = int(getattr(host_pool, "page_size", 1) or 1)
    return page_size * int(getattr(host_pool, "dcp_size", 1) or 1)


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


def _with_retain_scope(method):
    """Bind the call's extra_info TTL for the nested write helpers."""
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        with self._retain_scope(bound.arguments.get("extra_info")):
            return method(self, *args, **kwargs)
    return wrapper


class DfkvHiCache(HiCacheStorage):
    # Defaults for instances built without __init__ (tests, subclasses).
    _has_retain = False
    _retain_tls = threading.local()
    _pending_retain = None

    # DCP L3 admission. SGLang builds that gate HiCache L3 under DCP accept a
    # 'dynamic' backend only when its class declares this (read before any
    # instance exists); other SGLang builds ignore it. Keys already carry the
    # DCP coordinate, the MLA writer is elected per DCP shard, and pool
    # registration probes a whole logical page (_l3_page_span), so a
    # DCP-widened page maps to this rank's physical rows.
    supports_dcp_storage = True

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
        probe = _probe_parallel_context()
        # Runtime PCP/DCP coordinates describe SGLang's compute topology. The
        # storage identity may intentionally collapse replicated Prefill-CP
        # ranks onto the elected writer's key; keep the two concepts separate.
        self._coordinate_sources = {}
        (
            (self.pcp_size, self.pcp_rank),
            (self.dcp_size, self.dcp_rank),
            attn_tp_rank,
        ) = _resolve_parallel_coordinates(
            cfg, probe=probe, storage_config=storage_config,
            sources=self._coordinate_sources)
        context = self._coordinate_sources["context"]
        # The context as resolution used it: None if a field failed it.
        parallel = probe[0] if context == "available" else None
        self._parallel_context_missing = parallel is None
        if parallel is None:
            # Once per storage backend. The coordinates decide which rank
            # writes and which bytes a rank reads, so say where they came from.
            _log.warning(
                "dfkv: SGLang's parallel context is %s; storage "
                "coordinates pcp=%d/%d (from %s) dcp=%d/%d (from %s); the "
                "MLA writer is elected by tp_rank=%d because attn_tp_rank "
                "is unknown", context,
                self.pcp_rank, self.pcp_size,
                self._coordinate_sources.get("pcp"),
                self.dcp_rank, self.dcp_size,
                self._coordinate_sources.get("dcp"), self.tp_rank)
        self.prefill_cp_storage_layout = _resolve_prefill_cp_storage_layout(
            cfg,
            is_mla=self.is_mla,
            pcp_size=self.pcp_size,
            dcp_size=self.dcp_size,
            parallel=parallel,
            attn_tp_rank=attn_tp_rank,
        )
        self._storage_pcp_rank = (
            0
            if self.prefill_cp_storage_layout == "replicated"
            else self.pcp_rank
        )
        # Rail affinity is a node-local physical property. Resolve it only when
        # enabled so storage-only/legacy SGLang deployments remain unaffected.
        self.physical_rank = (
            _resolve_local_physical_rank(parallel)
            if _truthy(cfg.get("rail_affinity"))
            else None
        )
        # The connector enforces one physical writer for replicated Prefill-CP
        # even when an SGLang build invokes every CP rank's backend. This keeps
        # shared-key identity from turning into eight duplicate PUT streams.
        self._mla_replica_writer = _is_hicache_storage_writer(
            is_mla=self.is_mla,
            tp_rank=self.tp_rank,
            attn_tp_rank=attn_tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            prefill_cp_storage_layout=self.prefill_cp_storage_layout,
        )
        self._device_direct_requested = _truthy(
            os.environ.get("SGLANG_HICACHE_L2_BYPASS"))
        self._device_registration_failed = False
        # Set by the first primary-KV device transfer; see
        # _device_direct_active().
        self._device_direct_used = False
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
        self._anchor_notice_printed = False
        # Access log: idempotent per process (first instance wins). Configured
        # here so tp_rank/model are available for the {rank} path placeholder.
        _configure_access_log(cfg, tp_rank=self.tp_rank, model=self.model)
        # Hot-reload: a background watcher lets ops flip access_log (and future
        # observability knobs) live via a control file — no SGLang restart. Only
        # zero-correctness-impact knobs are hot; structural/native knobs are not.
        _hot_config.register("access_log", _access_log_apply_hot)
        _hot_config.start(cfg, tp_rank=self.tp_rank)
        self._alog_tag = (
            f"r{self.tp_rank}-p{self.physical_rank}"
            if self.physical_rank is not None
            else f"r{self.tp_rank}"
        )
        # Client-side read/write counters (Prometheus when available). Lets ops
        # confirm SGLang->dfkv volume from /metrics instead of parsing access logs.
        self._metrics = _Metrics(self.tp_rank)
        if self._parallel_context_missing:
            self._metrics.on_parallel_context_fallback()
        # Load the native lib up front so its version can ride on the fleet-metrics
        # resource (dfkv_native_version). The actual client (dfkv_open) is created
        # later, inside the access-log "init" block. The plugin ships with the dfkv
        # repo, so its package version == the native lib version.
        self._lib = _load_lib(cfg.get("lib_path"))
        # Retention (kRetain): optional on older libdfkv builds.
        self._has_retain = dfkv_cabi.bind_retain(self._lib)
        self._retain_tls = threading.local()
        # SGLang touches a request's prefix right after insert; pages still
        # in the backup queue answer MISS and are protected after their write.
        self._pending_retain: "collections.OrderedDict[bytes, int]" = (
            collections.OrderedDict())
        self._pending_lock = threading.Lock()
        native_ver = _native_version(self._lib)
        # Unified fleet metrics pushed over OTLP to the central Collector (off by
        # default; zero cost when DFKV_METRICS_ENABLED is unset). Powers the
        # cross-connector global dashboard (per connector_id / type).
        metrics_acquired = tracing_acquired = False
        try:
            # configure() acquires before validating identity, so mark each
            # acquisition first and roll it back even when configuration raises.
            metrics_acquired = True
            _push_metrics.configure(
                cfg, connector_type=_tcfg.TYPE_HICACHE,
                tp_rank=self.tp_rank, model=self.model,
                version=native_ver, native_version=native_ver)
            tracing_acquired = True
            _tracing.configure(
                cfg, connector_type=_tcfg.TYPE_HICACHE,
                tp_rank=self.tp_rank, model=self.model,
                version=native_ver, native_version=native_ver)
        except Exception:
            if tracing_acquired:
                _tracing.release()
            if metrics_acquired:
                _push_metrics.release()
            raise
        self._telemetry_acquired = True
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
                f"storage_pcp={self._storage_pcp_rank}/{self.pcp_size} "
                f"dcp={self.dcp_rank}/{self.dcp_size} "
                f"physical_rank={self.physical_rank if self.physical_rank is not None else '-'} "
                f"mla={int(self.is_mla)}")) as r:
            mds = cfg.get("mds_endpoints", "")
            members = cfg.get("members", "")
            _tcfg.require_ring_endpoint(members, mds)
            # Multi-SGE device writes use a dedicated connection lane in the
            # native transport, so each QP still has one in-flight CUDA range.
            # Depth fans a batch across independent QPs and is required to keep
            # the SSD/RDMA pipeline full.
            requested_depth = int(
                cfg.get("rdma_depth", os.environ.get("DFKV_RDMA_DEPTH", "1")))
            os.environ["DFKV_RDMA_DEPTH"] = str(max(1, requested_depth))
            if _truthy(cfg.get("require_rdma")):
                if not _truthy(os.environ.get("DFKV_RDMA")):
                    os.environ["DFKV_RDMA"] = "1"
            # Resolve rank-local affinity before dfkv_open_v2 so each process
            # registers pools only on its node-local primary and bounded
            # fallbacks. Never substitute TP/DP/PCP/DCP for physical rank.
            affinity_rank = (
                self.physical_rank
                if self.physical_rank is not None
                else self.tp_rank
            )
            self._rail_affinity = apply_rank_local_rail_affinity(
                cfg, affinity_rank, os.environ)
            self._metrics.set_identity(
                physical_rank=self.physical_rank,
                pcp_rank=self.pcp_rank,
                dcp_rank=self.dcp_rank,
                storage_pcp_rank=self._storage_pcp_rank,
                primary_dev=self._rail_affinity.primary,
            )
            if not self._rail_affinity.enabled and cfg.get("rdma_numa"):
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
            # Off by default because it changes every object key: see
            # _pool_id() and docs/CONNECTORS.md.
            self._bind_runtime_layout = _truthy(
                cfg.get("bind_runtime_layout", False))
            self._backup_exist_gate = _truthy(
                cfg.get("backup_exist_gate",
                        os.environ.get("DFKV_BACKUP_EXIST_GATE", "1")))
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
                batch_concurrency=int(cfg.get("batch_concurrency", 0)),
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
        try:
            if getattr(self, "_telemetry_acquired", False):
                self._telemetry_acquired = False
                _push_metrics.release()
                _tracing.release()
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

    def _validate_pool_object_sizes(self, sizes, *, context):
        if self.transport_mode == "rdma":
            validate_object_sizes(
                get_max_block_bytes(self._lib, self._h), sizes, context=context)

    def _check_pool_dcp(self, pool, label: str) -> None:
        """Refuse a pool whose DCP width contradicts this rank's coordinates.

        Under decode CP SGLang sizes a host pool's logical page by its
        ``dcp_size``. A pool that says DCP while this backend resolved another
        width means the coordinates are wrong (not discovered, or misstated in
        extra_config), and every key of this rank would carry them.
        """
        resolved = getattr(self, "dcp_size", None)
        if resolved is None:
            return
        try:
            pool_dcp = int(getattr(pool, "dcp_size", 1) or 1)
        except (TypeError, ValueError):
            return
        if pool_dcp > 1 and pool_dcp != resolved:
            raise RuntimeError(
                f"{label} is laid out for dcp_size={pool_dcp}, but this "
                f"backend resolved dcp={getattr(self, 'dcp_rank', 0)}/"
                f"{resolved} (from "
                f"{getattr(self, '_coordinate_sources', {}).get('dcp', '?')}"
                f"). Storing with the wrong DCP coordinate would mix the KV "
                f"of different ranks; pass dcp_size and dcp_rank in the "
                f"HiCache extra_config")

    def _note_pool_layout(self, name: str, pool, *, device=False,
                          extra=()) -> None:
        """Record a pool's format facts for its keys (bind_runtime_layout).

        Which pool tags which key form: host-form keys (bare object keys)
        take the host pool's facts (class, layout, dtype); device-form keys
        (the scatter-gather objects of L2-bypass, laid out by the device
        pool) take the device pool's (class, dtype, and for the primary KV
        the per-token strides and layer count of a page). So instances that
        differ only in host layout share device-form objects, and a device
        pool class that packs rows differently does not. This holds for the
        primary KV and for side pools. A pool that has only one of the two
        (a device-only sidecar, the draft, a host-only side pool) is tagged
        by that one in both forms.
        """
        if not getattr(self, "_bind_runtime_layout", False):
            return
        attr = "_device_layout_facts" if device else "_host_layout_facts"
        if not hasattr(self, attr):
            setattr(self, attr, {})
        fields = ("dtype",) if device else _LAYOUT_FIELDS
        facts = _runtime_layout_facts(pool, fields) + tuple(extra)
        known = getattr(self, attr).setdefault(name, facts)
        if known != facts:
            raise RuntimeError(
                f"pool {name!r} registered again with another layout, dtype "
                f"or class ({known} -> {facts}); its object keys would change")
        if known is facts:
            for field in fields:
                if not hasattr(pool, field):
                    _log.warning(
                        "dfkv: bind_runtime_layout: %s pool %r has no %r; its "
                        "keys do not tell that field apart",
                        "device" if device else "host", name, field)

    def _pool_id(self, name, objects: int, device: bool = False):
        """The pool field of an object key.

        With ``bind_runtime_layout`` the format tag of the pool that lays out
        the key's objects is part of it (see _note_pool_layout for which pool
        that is), so a reader whose pool stores another byte format (same
        object size, different bytes) misses instead of loading the bytes as
        its own. Keys are only built for registered pools; one that is not
        yet registered has no known format and is an error, not an untagged
        key.
        """
        if not getattr(self, "_bind_runtime_layout", False):
            return name
        if isinstance(name, bytes):
            name = name.decode()
        host = getattr(self, "_host_layout_facts", {}).get(name)
        dev = getattr(self, "_device_layout_facts", {}).get(name)
        facts = (dev or host) if device else (host or dev)
        if facts is None:
            raise RuntimeError(
                f"bind_runtime_layout: object keys of pool {name!r} asked for "
                f"before the pool was registered")
        return f"{name}@{_layout_tag(facts, objects)}"

    def register_mem_pool_host(self, mem_pool_host):
        self._check_pool_dcp(mem_pool_host, "HiCache primary host pool")
        self._note_pool_layout("kv", mem_pool_host)
        if self.transport_mode == "rdma":
            import torch
            indices = torch.arange(
                _l3_page_span(mem_pool_host), dtype=torch.int64)
            meta = mem_pool_host.get_page_buffer_meta(indices)
            if not _meta_has_no_layout(meta):
                self._validate_pool_object_sizes(
                    meta[1], context="HiCache primary host pool")
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

    def _is_layer_first_kv_pool(self, host_pool, count: int, sizes) -> bool:
        """Recognize the per-layer component layout of a layer-split MLA pool.

        Only reachable under ``prefill_cp_storage_layout='per_cp_rank'``, which
        this connector accepts solely for ``enable_dsa_cache_layer_split``. No
        previously startable configuration can take this branch, so existing
        key schemas stay byte-identical. ``page_first``/``page_first_direct``
        pools pack every layer into one object and keep the single-component
        schema; only ``layer_first`` exposes the layers separately.
        """
        if getattr(self, "prefill_cp_storage_layout", None) != "per_cp_rank":
            return False
        if not hasattr(host_pool, "kv_buffer"):
            return False
        if getattr(host_pool, "layout", None) != "layer_first":
            return False
        if count <= 1 or int(getattr(host_pool, "layer_num", 0) or 0) != count:
            return False
        # layer_first emits one constant element size per layer; anything else
        # is a layout this mapping does not describe.
        return len({int(size) for size in sizes}) == 1

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        name = str(host_pool_name)
        self._check_pool_dcp(host_pool, f"HiCache host pool {name!r}")
        self._note_pool_layout(
            "kv" if name in ("kv", "__default__") else name, host_pool)
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
            span = _l3_page_span(host_pool)
            if span <= 0:
                raise RuntimeError("pool logical page span must be positive")
            try:
                import torch as _t
                arange = getattr(_t, "arange")
                indices = arange(span, dtype=_t.int64)
            except (ImportError, AttributeError):
                indices = list(range(span))
            meta = host_pool.get_page_buffer_meta(indices)
            if _meta_has_no_layout(meta):
                # The DSV4/DSA hybrid stack (SGLang v0.5.17+) registers the
                # *logical anchor* "kv" pool through this v2 entry too. It holds
                # no host buffer, so there is no layout to discover and nothing
                # to physically register — but raising here crashes HiCache
                # setup for the whole scheduler. Record it marker-only: v2
                # writes and reads use one-byte markers; real KV data lives in
                # the side pools. Malformed layouts (wrong
                # arity, empty/mismatched components) still fail below.
                if not hasattr(self, "_pool_logical"):
                    self._pool_logical = {}
                self._pool_logical[name] = True
                self._pool_component_names[name] = ("all",)
                self._pool_replicated[name] = True
                self.registered_pools[name] = host_pool
                self._note_logical_anchor_once()
                with access_log("pool_components",
                                lambda: f"{self._alog_tag} {name}") as result:
                    result.result = "logical anchor (no host layout); marker-only"
                return
            if not isinstance(meta, (tuple, list)) or len(meta) != 2:
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
            elif self._is_layer_first_kv_pool(host_pool, count, sizes):
                # layer-split DSA: a "layer_first" MLA host pool emits one
                # pointer per *owned* layer per page (page-major, layer-minor),
                # which is exactly the order _flatten expects. The layers a CP
                # rank owns are a shard of the model, so the components are
                # distinct payloads under this rank's CP key coordinate.
                names = [f"l{i:03d}" for i in range(count)]
                replicated = False
                if name in ("kv", "__default__"):
                    self._kv_layer_components = tuple(names)
            elif hasattr(host_pool, "v_buffer"):
                if count != 2:
                    raise RuntimeError(
                        f"K/V pool must expose exactly 2 components, got {count}")
                names = ["k", "v"]
                replicated = False
            elif count == 1:
                # One object per page: the packed single-buffer layout. It is
                # the same on every TP rank only for an MLA model, whose
                # attention state is not split by heads. Elsewhere a single
                # buffer can hold this rank's heads (the sparse index-K pool
                # of an MHA model splits its heads over TP): stored under one
                # replicated key, every rank would overwrite the others' and
                # read a different rank's heads back at the same size.
                names = ["all"]
                replicated = self.is_mla
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
            getattr(self, "_pool_logical", {}).pop(name, None)
            raise RuntimeError(
                f"cannot discover physical layout for pool {name!r}") from exc

        self._validate_pool_object_sizes(
            sizes, context=f"HiCache host pool {name!r} components={tuple(names)}")

        self._pool_component_names[name] = tuple(names)
        self._pool_replicated[name] = replicated
        self.registered_pools[name] = host_pool
        getattr(self, "_pool_logical", {}).pop(name, None)
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

    def _note_load_device(self, pool) -> None:
        """Remember the GPU device-direct reads land on.

        CUDA's current device is per thread, and SGLang's storage threads
        do not set it: a thread that never did is on device 0. The ordering
        fence after a device-direct read, and the native GPU dedup's fence
        before it publishes, synchronize the current device; on a rank whose
        GPU is not 0 they would order the wrong GPU. The device is taken
        from the pool's buffers, else from the registering thread (SGLang
        registers from the rank's own thread, which has set its device).
        """
        if getattr(self, "_load_device", None) is not None:
            return
        index = _cuda_device_of(pool)
        if index is None:
            try:
                import torch
            except ImportError:
                return
            if torch.cuda.is_available():
                index = int(torch.cuda.current_device())
        self._load_device = index

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

    def _device_direct_active(self) -> bool:
        """Whether this process stores primary KV in the device-direct form.

        Device-direct objects are scatter-gather groups under their own keys;
        host objects are bare keys. Probes, retention and the host read must
        address the form that is actually written. A registered device pool
        does not say which: SGLang may register it without using the bypass,
        and the host path then writes bare keys. The form is device-direct
        when the bypass was requested and its registration held, or once this
        process has done a primary-KV device transfer.
        """
        if getattr(self, "mem_pool_device", None) is None:
            return False
        if getattr(self, "_device_registration_failed", False):
            return False
        return bool(getattr(self, "_device_direct_requested", False)
                    or getattr(self, "_device_direct_used", False))

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
            self._note_load_device(mem_pool_device)
            self._note_pool_layout(
                "kv", mem_pool_device, device=True,
                extra=self._device_segment_facts(mem_pool_device))
            result.result = (
                f"registered {registered} device region(s)"
                if registered else "no device buffer registered")
            if getattr(self, "_device_direct_requested", False):
                self._record_kv_sg_groups(mem_pool_device)

    def _device_segment_facts(self, mem_pool_device) -> tuple:
        """Per-token strides and layer count of a primary-KV device page.

        Device-form objects concatenate one segment per layer, each
        ``page_size * stride`` bytes, so these say how a page is packed.
        Taken from the metadata function the transfers use, with the token
        slots of page 0 (no I/O); () when that call is not possible with
        this SGLang build.
        """
        if not getattr(self, "_bind_runtime_layout", False):
            return ()
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_page_buffer_meta,
        )
        try:
            page_size = int(mem_pool_device.page_size)
            _ptrs, sizes = get_device_page_buffer_meta(
                mem_pool_device, list(range(page_size)))
            strides = tuple(int(seg[0]) // page_size for seg in sizes)
            return (("strides", strides), ("layers", len(sizes[0])))
        except Exception as exc:
            _log.warning(
                "dfkv: bind_runtime_layout: page geometry of the device pool "
                "unknown (%s: %s); its keys are tagged without it",
                type(exc).__name__, exc)
            return ()

    def _record_kv_sg_groups(self, mem_pool_device) -> None:
        """Record how many SG groups a primary-KV page is stored in.

        Device transfers record it (_flatten_device), but a process probes,
        reads the presence anchor and retains pages before its first one,
        e.g. pages other instances wrote. The count is that of one page's
        per-layer segments, from the same metadata function the transfers
        use, called with the token slots of page 0 (pointer arithmetic, no
        I/O). If that call is not possible with this SGLang build the count
        stays unknown and group 0 alone is addressed until the first
        transfer, as before.
        """
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_page_buffer_meta,
        )
        try:
            page = list(range(int(mem_pool_device.page_size)))
            seg_ptrs, _sizes = get_device_page_buffer_meta(mem_pool_device, page)
            segments = len(seg_ptrs[0])
        except Exception as exc:
            _log.warning(
                "dfkv: SG group count of the device pool unknown until the "
                "first device transfer (%s: %s); until then only group 0 "
                "of a page is probed and retained", type(exc).__name__, exc)
            return
        w = self._sg_width()
        if not hasattr(self, "_sg_chunks"):
            self._sg_chunks = {}
        self._sg_chunks["kv"] = max(1, (segments + w - 1) // w)

    def _set_sidecar_components(self, key: str) -> None:
        """Components of a device sidecar: one object per page.

        A host pool registered under the same name has already discovered
        its layout; the device registration must agree with it rather than
        overwrite it.
        """
        if not hasattr(self, "_pool_component_names"):
            self._pool_component_names = {}
        if not hasattr(self, "_pool_replicated"):
            self._pool_replicated = {}
        layout = (("all",), bool(self.is_mla))
        if key in self._pool_component_names:
            host = (tuple(self._pool_component_names[key]),
                    bool(self._pool_replicated.get(key)))
            if host != layout:
                raise RuntimeError(
                    f"device sidecar {key!r} has one replicated={layout[1]} "
                    f"object per page, but its host pool registered "
                    f"components={host[0]} replicated={host[1]}")
            return
        self._pool_component_names[key] = layout[0]
        self._pool_replicated[key] = layout[1]

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
            self._note_load_device(device_pool)
            self._note_pool_layout(key, device_pool, device=True)
            # The sidecar metadata API returns one logical object per page; its
            # layers are SG segments, not independent pool components. Like a
            # one-object host pool it is the same on every TP rank only for an
            # MLA model.
            self._set_sidecar_components(key)
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
            self._note_load_device(mem_pool_device_draft)
            self._note_pool_layout("draft", mem_pool_device_draft, device=True)
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
            self._note_pool_layout(name, mem_pool_device_draft, device=True)
            # Same API-proven one-object layout as the target indexer sidecar.
            self._set_sidecar_components(name)
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
            res, _io = self._sidecar_device_set(name, keys, device_indices)
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
            res, _io = self._sidecar_device_get(name, keys, device_indices)
            return res
        except Exception:
            return [False] * len(keys)


    # --- canonical pool-key schema -----------------------------------------
    def _storage_pcp(self, replicated: bool) -> tuple[int, int]:
        """Return the CP coordinate encoded in one physical object key."""
        rank = (
            getattr(self, "_storage_pcp_rank", 0)
            if replicated
            and getattr(self, "prefill_cp_storage_layout", None) == "replicated"
            else self.pcp_rank
        )
        return self.pcp_size, rank

    def _kv_components(self) -> tuple[str, ...]:
        """Components of one physical "kv" page object.

        A layer-split MLA pool stores its owned layers as separate objects (see
        _is_layer_first_kv_pool); every other pool keeps the legacy packed
        schema, so their keys are unchanged.
        """
        layers = getattr(self, "_kv_layer_components", None)
        if layers:
            return layers
        return ("all",) if self.is_mla else ("k", "v")

    def _require_single_object_page(self, api: str) -> None:
        """Fail closed where a one-object-per-page API cannot carry a page.

        The v1 set/get entries take a single value and target per key, so they
        can only address one component. A layer-split page is many components;
        addressing just the first would store or serve one layer of ten and
        report success. The zero-copy v2 path carries every component.
        """
        if getattr(self, "_kv_layer_components", None):
            raise RuntimeError(
                f"{api} cannot address a layer-split page "
                f"({len(self._kv_components())} per-layer objects); "
                "this layout requires the zero-copy v2 path"
            )

    def _keys(self, page_hash: str | bytes, device: bool = False) -> List[bytes]:
        tp = -1 if self.is_mla else self.tp_rank
        pp = self.pp_rank if self.enable_pp else 0
        pcp_size, pcp_rank = self._storage_pcp(self.is_mla)
        components = self._kv_components()
        return [
            pool_key(page_hash, pool=self._pool_id("kv", len(components), device),
                     tp_size=self.tp_size, tp_rank=tp,
                     pcp_size=pcp_size, pcp_rank=pcp_rank,
                     dcp_size=self.dcp_size, dcp_rank=self.dcp_rank,
                     pp_size=self.pp_size if self.enable_pp else 1, pp_rank=pp,
                     component=component)
            for component in components
        ]

    def _sub(self) -> int:
        return len(self._kv_components())

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

    # ---- retention ----
    @staticmethod
    def _retain_ttl_of(extra_info) -> int:
        """Per-request TTL carried in HiCacheStorageExtraInfo.extra_info."""
        if extra_info is None:
            return 0
        info = extra_info if isinstance(extra_info, dict) else getattr(
            extra_info, "extra_info", None)
        if not isinstance(info, dict):
            return 0
        value = info.get("retain_ttl_s", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return 0
        return min(value, 0xFFFFFFFF)

    @contextlib.contextmanager
    def _retain_scope(self, extra_info):
        prev = getattr(self._retain_tls, "ttl", 0)
        self._retain_tls.ttl = self._retain_ttl_of(extra_info)
        try:
            yield
        finally:
            self._retain_tls.ttl = prev

    _PENDING_RETAIN_MAX = 1 << 20

    def _retain_set(self, keys, ttl) -> list:
        try:
            codes = dfkv_cabi.call_batch_retain(
                self._lib, self._h, keys, ttl, dfkv_cabi.DFKV_RETAIN_SET)
        except Exception as e:  # retention never fails a backup or touch
            self._warn_retain("retain SET", e)
            return [dfkv_cabi.DFKV_RET_ERROR] * len(keys)
        unprotected = sum(
            1 for c in codes if c not in dfkv_cabi.RETAIN_PROTECTED_CODES)
        if unprotected:
            _log.debug("dfkv retain SET: %d/%d keys unprotected (codes=%s)",
                       unprotected, len(codes), sorted(set(codes)))
        return codes

    def _retain_after_write(self, sks, stored) -> None:
        """SET every stored sub-object of this write that carries a TTL:
        from the write's extra_info, or parked by an earlier batch_touch."""
        if not self._has_retain:
            return
        ctx_ttl = getattr(self._retain_tls, "ttl", 0)
        pending = getattr(self, "_pending_retain", None)
        by_ttl: dict = {}
        if pending:
            with self._pending_lock:
                for sk, ok in zip(sks, stored):
                    if not ok:
                        continue
                    ttl = max(ctx_ttl, pending.pop(sk, 0))
                    if ttl > 0:
                        by_ttl.setdefault(ttl, []).append(sk)
        elif ctx_ttl > 0:
            by_ttl[ctx_ttl] = [sk for sk, ok in zip(sks, stored) if ok]
        for ttl, keys in by_ttl.items():
            if keys:
                self._retain_set(keys, ttl)

    _RETAIN_WARN_INTERVAL_S = 60.0

    def _warn_retain(self, what: str, exc: Exception) -> None:
        """Rate-limited warning for retention calls on the request path."""
        now = time.monotonic()
        if now - getattr(self, "_retain_warn_at", float("-inf")) < self._RETAIN_WARN_INTERVAL_S:
            self._retain_warn_suppressed = getattr(self, "_retain_warn_suppressed", 0) + 1
            return
        suppressed = getattr(self, "_retain_warn_suppressed", 0)
        self._retain_warn_at = now
        self._retain_warn_suppressed = 0
        _log.warning("dfkv %s failed: %s (%d similar suppressed)", what, exc, suppressed)

    # --- retention ownership: exactly the objects this rank writes ---------
    def _rank_writes_kv(self) -> bool:
        """Whether this rank writes main-KV objects.

        Replicated MLA latent has one elected writer per physical CP/DCP
        coordinate; every other layout writes on every rank.
        """
        return not self.is_mla or self._mla_replica_writer

    def _rank_writes_pool(self, name: str) -> bool:
        """Write gate of one v2 pool: followers still write rank-sharded state."""
        return self._rank_writes_kv() or not self._pool_is_replicated(name)

    def _retain_side_pools(self):
        for name in getattr(self, "_pool_component_names", ()):
            if name not in ("kv", "__default__") and self._rank_writes_pool(name):
                yield name

    def _stored_keys(self, pool: str, page_hash) -> List[bytes]:
        """Physical keys one page occupies in ``pool``, in the form written.

        Device-direct (L2-bypass) writes store SG-chunked keys, so retention on
        the bare key would miss them. The chunk count is recorded by
        _flatten_device, and for primary KV also when the device pool is
        registered with the bypass requested (_record_kv_sg_groups). Where
        neither has happened only chunk 0 is known.
        """
        if pool in ("kv", "__default__"):
            device = self._device_direct_active()
            base = self._keys(page_hash, device)
            pool = "kv"
        else:
            device = pool in getattr(self, "_sidecar_device_pools", {})
            base = self._pool_keys(pool, page_hash, device)
        if not device:
            return base
        chunks = getattr(self, "_sg_chunks", {}).get(pool, 1)
        return [self._sg_group_key(sk, ci) for sk in base for ci in range(chunks)]

    def _retain_object_keys(self, keys) -> List[bytes]:
        """Every object of these pages that this rank writes, and nothing else."""
        sks = ([sk for k in keys for sk in self._stored_keys("kv", k)]
               if self._rank_writes_kv() else [])
        for pool in self._retain_side_pools():
            for k in keys:
                sks.extend(self._stored_keys(pool, k))
        return sks

    def retain_owns_objects(self) -> bool:
        """Whether this rank writes any object and so must vote on a TTL.

        A follower may skip replicated MLA KV yet own rank-sharded side-pool
        state (e.g. KDA). Ownership depends on the registered pools, so asking
        before any pool is registered is a caller ordering bug, not a no.
        """
        if (getattr(self, "mem_pool_host", None) is None
                and getattr(self, "mem_pool_device", None) is None
                and not getattr(self, "_pool_component_names", None)):
            raise RuntimeError(
                "retain_owns_objects() called before any HiCache pool was registered")
        return self._rank_writes_kv() or next(self._retain_side_pools(), None) is not None

    # Prefixes batch_touch protected recently: identity -> (ttl, monotonic s).
    _TOUCH_DEBOUNCE_MAX = 4096
    _touch_seen = None
    _touch_lock = threading.Lock()

    def _touch_is_fresh(self, prefix, ttl: int) -> bool:
        with self._touch_lock:
            seen = self._touch_seen
            entry = seen.get(prefix) if seen else None
        if entry is None:
            return False
        seen_ttl, at = entry
        return seen_ttl >= ttl and time.monotonic() - at < ttl / 10.0

    def _touch_note(self, prefix, ttl: int) -> None:
        with self._touch_lock:
            if self._touch_seen is None:
                self._touch_seen = collections.OrderedDict()
            seen = self._touch_seen
            seen[prefix] = (ttl, time.monotonic())
            seen.move_to_end(prefix)
            while len(seen) > self._TOUCH_DEBOUNCE_MAX:
                seen.popitem(last=False)

    def retain_precheck(self, keys, extra_info=None) -> int:
        """TTL this rank admits for the whole prefix ``keys``, without a write.

        Returns min(requested, ring cap) when the ring supports retention and
        every owner of this rank's objects reported enough headroom; otherwise
        0 (do not promise or bill a TTL). Advisory: capacity can change before
        the asynchronous SET, and the caller combines all owning ranks' votes.
        """
        ttl = self._retain_ttl_of(extra_info)
        if not ttl or not keys or not self._has_retain:
            return 0
        try:
            supported, cap = dfkv_cabi.call_retain_limits(self._lib, self._h)
            if supported != 1 or cap <= 0:
                return 0
            sks = self._retain_object_keys(keys)
            if sks and dfkv_cabi.call_retain_precheck(self._lib, self._h, sks):
                return min(ttl, cap)
        except Exception as e:
            self._warn_retain("retention precheck", e)
        return 0

    def batch_touch(self, keys, extra_info=None) -> None:
        """Renew retention of pages served from device/host memory.

        Called by SGLang for prefix hits that never reach storage. Refreshes
        the objects of those pages that this rank writes (main KV and side
        pools).

        With a ``retain_ttl_s`` in ``extra_info`` the prefix is protected
        with a SET, which the native client never debounces, and every call
        rebuilt the keys of the whole prefix for it: on a hot prefix that is
        thousands of keys and two round trips per request. A prefix that this
        process protected for at least that TTL within the last tenth of it
        is therefore skipped before any key is built -- the same renewal
        interval the native client applies to a REFRESH with a TTL hint.
        """
        if not keys or not self._has_retain:
            return
        ttl = self._retain_ttl_of(extra_info)
        prefix = None
        if ttl > 0:
            # Exact identity of the prefix without building its object keys,
            # and of the key form those keys take: a prefix protected while
            # its pages were bare keys, or while only SG group 0 was known,
            # was not protected in the form written later.
            prefix = (len(keys), keys[0], keys[-1], hash(tuple(keys)),
                      self._device_direct_active(),
                      tuple(sorted(getattr(self, "_sg_chunks", {}).items())))
            if self._touch_is_fresh(prefix, ttl):
                return
        sks = self._retain_object_keys(keys)
        if not sks:
            return
        try:
            # Decision 9: protect the whole prefix or none of it.
            if ttl > 0 and dfkv_cabi.call_retain_precheck(
                    self._lib, self._h, sks):
                codes = self._retain_set(sks, ttl)
                # Skip the next touches only if this one left nothing undone:
                # every object protected, or parked below to be protected by
                # its write.
                if all(c in dfkv_cabi.RETAIN_PROTECTED_CODES
                       or c == dfkv_cabi.DFKV_RET_MISS for c in codes):
                    self._touch_note(prefix, ttl)
                missing = [sk for sk, c in zip(sks, codes)
                           if c == dfkv_cabi.DFKV_RET_MISS]
                if missing:
                    with self._pending_lock:
                        for sk in missing:
                            self._pending_retain[sk] = max(
                                ttl, self._pending_retain.get(sk, 0))
                            self._pending_retain.move_to_end(sk)
                        while len(self._pending_retain) > self._PENDING_RETAIN_MAX:
                            self._pending_retain.popitem(last=False)
            else:
                dfkv_cabi.call_batch_retain(
                    self._lib, self._h, sks, ttl,
                    dfkv_cabi.DFKV_RETAIN_REFRESH)
        except Exception as e:
            self._warn_retain("batch_touch", e)

    def _batch_exist_flat(self, subkeys) -> List[bool]:
        if not subkeys:
            return []
        if self._has_retain:
            # Every existence probe is a REFRESH: a hit renews whatever
            # retention the object carries, at no extra round trip.
            try:
                codes = dfkv_cabi.call_batch_retain(
                    self._lib, self._h, subkeys, 0,
                    dfkv_cabi.DFKV_RETAIN_REFRESH)
            except RuntimeError:
                return [False] * len(subkeys)
            return [c in dfkv_cabi.RETAIN_HIT_CODES for c in codes]
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
        if self._anchor_notice_printed:
            return
        self._anchor_notice_printed = True
        import sys as _sys
        print("[dfkv] NOTE: primary KV pool is a logical anchor "
              "(get_page_buffer_meta -> None) — a V4/DSA multi-pool model "
              f"(model={self.model!r}, e.g. GLM-5.2). The v1 anchor writes a "
              "one-byte 'kv' marker; real KV rides the v2 side-pool path. Verify L3 "
              "hit rate via metrics; if low, the LMCache MP connector "
              "(docs/CONNECTORS.md #4.5) is the alternative path for GLM-5.x DSA.",
              file=_sys.stderr, flush=True)

    def _write_anchor_markers(self, keys) -> List[bool]:
        pages = self._write_anchor_markers_impl(keys)
        if getattr(self._retain_tls, "ttl", 0) > 0:
            sub = self._sub()
            self._retain_after_write(
                [sk for k in keys for sk in self._keys(k)],
                [ok for ok in pages for _ in range(sub)])
        return pages

    def _write_anchor_markers_impl(self, keys) -> List[bool]:
        """Write a one-byte marker object per "kv" sub-key so a later
        batch_exists / batch_exists_v2 can find the primary-pool prefix.

        Used only for the logical-anchor case (V4/DSA, e.g. GLM-5.2): the anchor
        pool holds no KV buffer, so there is nothing to zero-copy. The marker is
        non-empty so native dfkv charges tenant byte quota for every allocated
        object while preserving SGLang's logical-anchor existence semantics.
        Namespace and object-key identity isolate incompatible writers; the
        matching read loads that byte to refresh actual consumption recency.
        A page succeeds iff all its sub-object markers were written."""
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
            # MLA latent is replicated only within one attention-TP subgroup.
            # PCP/DCP shards elect an independent writer per physical coordinate.
            if self.is_mla and not self._mla_replica_writer:
                r.result = "backup_skip"
                if _sp:
                    _sp.attrs = {"dfkv.backup_skip": True}
                return [True] * n
            meta = self.mem_pool_host.get_page_buffer_meta(host_indices)
            if _meta_has_no_layout(meta):
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
            flat, recovered = self._put_flat(sks, sp, ss)
            dur = time.perf_counter() - t0
            res = self._fold(flat, n, sub)
            r.result = f"ok {sum(res)}/{n}"
            if recovered:
                r.result += f" retry_ok={recovered}"
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

    def _flatten_device(self, keys, seg_ptrs, seg_sizes, keys_fn=None, sub=None,
                        pool="kv"):
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
        object count while reusing the same SG chunking. ``pool`` names the
        retained pool whose chunk count is recorded for _stored_keys (None for
        the draft, which retention does not cover).
        """
        keys_fn = keys_fn or (lambda h: self._keys(h, device=True))
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
        if pool is not None and keys:
            if not hasattr(self, "_sg_chunks"):
                self._sg_chunks = {}
            self._sg_chunks[pool] = nchunks
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

    def _put_sg_flat(self, sks, seg_ptrs, seg_sizes):
        """(stored per sub-key, retry recoveries); see _put_flat."""
        stored, recovered = self._put_sg_flat_impl(sks, seg_ptrs, seg_sizes)
        self._retain_after_write(sks, stored)
        return stored, recovered

    def _put_sg_flat_impl(self, sks, seg_ptrs, seg_sizes):
        """SG put with the same phase-9 exist gate + single transient retry as
        _put_flat (the contiguous host put), so the L2-bypass path inherits the
        identical backup-dedup and burst-failure recovery semantics."""
        recovered = 0
        if not sks:
            return [], 0
        if self._backup_exist_gate:
            present = list(self._batch_exist_flat(sks))
            todo = [i for i, p in enumerate(present) if not p]
            if not todo:
                return [True] * len(sks), 0
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
                    recovered += 1
            todo = failed
        return present, recovered

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
            # MLA latent is replicated only within one attention-TP subgroup.
            # PCP/DCP shards elect an independent writer per physical coordinate.
            if self.is_mla and not self._mla_replica_writer:
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
            # After _flatten_device: it records the group count probes use.
            self._device_direct_used = True
            nbytes = sum(sum(s) for s in ss)
            t0 = time.perf_counter()
            flat, recovered = self._put_sg_flat(sks, sp, ss)
            dur = time.perf_counter() - t0
            res = self._fold(flat, n, sub)
            r.result = f"ok {sum(res)}/{n} (device-direct)"
            if recovered:
                r.result += f" retry_ok={recovered}"
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
        device = getattr(self, "_load_device", None)
        scope = contextlib.nullcontext()
        if device is not None:
            # The read and both fences (the native GPU dedup's before it
            # publishes, ours after) act on the current device: make it the
            # destination's for the call, whatever this thread was on.
            import torch
            if torch.cuda.is_available():
                scope = torch.cuda.device(device)
        with scope:
            rc = self._lib.dfkv_batch_get_auto_sg(
                self._h, karr, klens, parr, carr, narr, n, out_hit, out_len)
            if rc != 0:
                return [0] * n, [0] * n
            del inner_p; del inner_c; del parr; del carr; del narr
            if any(h == 1 for h in out_hit):
                _gpu_load_fence(device)
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
            # After _flatten_device: it records the group count probes use.
            self._device_direct_used = True
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
            if _meta_has_no_layout(meta):
                # Load the one-byte anchors only on this actual read path.
                # EXISTS remains non-promoting. Each native GET destination is
                # distinct because batch workers may scatter concurrently.
                self._note_logical_anchor_once()
                sub = self._sub()
                sks = [sk for key in keys for sk in self._keys(key)]
                if not sks:
                    return []
                markers = (ctypes.c_ubyte * len(sks))()
                base = ctypes.addressof(markers)
                karr, klens, parr, sarr, out, key_owners = _arrays(
                    sks, range(base, base + len(sks)), [1] * len(sks))
                t0 = time.perf_counter()
                self._lib.dfkv_batch_get(
                    self._h, karr, klens, parr, sarr, len(sks), out)
                dur = time.perf_counter() - t0
                del key_owners
                res = self._fold([out[i] == 1 for i in range(len(sks))], n, sub)
                r.result = f"anchor_marker hits={sum(res)}/{n}"
                if _sp:
                    _sp.hits = sum(res); _sp.bytes = len(sks)
                    _sp.attrs = {"dfkv.anchor_marker": True}
                self._metrics.on_get(
                    pages=n, hit_pages=sum(res), nbytes=len(sks), seconds=dur)
                return res
            # Device-direct PUT writes SG-chunked keys via _flatten_device
            # (_sg_group_key), but this host-buffer GET path uses flat keys
            # via _flatten/_keys. The key namespaces differ, so a host GET
            # would miss every page that device-direct PUT wrote, and the
            # stored bytes are layer-major, not this buffer's layout. So this
            # call loads nothing into the host buffer: it is the presence
            # anchor the prefetch controller counts, and the real KV is
            # loaded through the device path. A page is reported present only
            # if its stored objects exist: every SG group once this process
            # has done a device transfer and so knows the group count, group
            # 0 before that (as batch_exists probes). A page that is not in
            # the store must not be reported as retrieved; the device read
            # checks every group's length.
            if self._device_direct_active():
                sks = [sk for key in keys
                       for sk in self._stored_keys("kv", key)]
                per_page = len(sks) // n if n else 0
                res = self._fold(self._batch_exist_flat(sks), n, per_page)
                r.result = f"device_direct_anchor present={sum(res)}/{n}"
                if _sp:
                    _sp.hits = sum(res)
                    _sp.attrs = {"dfkv.device_direct_anchor": True}
                return res
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
        t0 = time.perf_counter()
        with _tracing.span("batch_exists", total) as _sp, \
                access_log("batch_exists", lambda: f"{self._alog_tag} {total} keys") as r:
            # longest contiguous prefix of pages whose every sub-object exists.
            # Device-direct writes use explicit binary SG coordinates: every
            # known SG group of a page is probed, the same keys as the
            # presence anchor in batch_get_v1 and retention. The bare object
            # key cannot match a chunked store; a different negotiated width
            # is a cold miss rather than a corrupt read.
            sks = [sk for k in keys for sk in self._stored_keys("kv", k)]
            per_page = len(sks) // total if total else 0
            page_ok = self._fold(self._batch_exist_flat(sks), total, per_page)
            n = 0
            for ok in page_ok:
                if not ok:
                    break
                n += 1
            self._metrics.on_exists(
                probe_pages=total,
                present_pages=sum(page_ok),
                contiguous_pages=n,
                seconds=time.perf_counter() - t0,
            )
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
        device: bool = False,
    ) -> List[bytes]:
        if pool_name in ("kv", "__default__"):
            return self._keys(page_hash, device)
        components = self._pool_components(pool_name)
        pp = self.pp_rank if self.enable_pp else 0
        replicated = self._pool_is_replicated(pool_name)
        tp = -1 if replicated else self.tp_rank
        pcp_size, pcp_rank = self._storage_pcp(replicated)
        return [
            pool_key(page_hash,
                     pool=self._pool_id(pool_name, len(components), device),
                     tp_size=self.tp_size, tp_rank=tp,
                     pcp_size=pcp_size, pcp_rank=pcp_rank,
                     dcp_size=self.dcp_size, dcp_rank=self.dcp_rank,
                     pp_size=self.pp_size if self.enable_pp else 1, pp_rank=pp,
                     component=component)
            for component in components
        ]

    def _pool_sub(self, pool_name: str) -> int:
        if pool_name in ("kv", "__default__"):
            return self._sub()
        return len(self._pool_components(pool_name))

    def _put_flat(self, sks, sp, ss):
        """(stored per sub-key, objects recovered by the retry).

        The recovery count is returned with the result, not left on the
        instance, where a concurrent put would overwrite it.
        """
        stored, recovered = self._put_flat_impl(sks, sp, ss)
        self._retain_after_write(sks, stored)
        return stored, recovered

    def _put_flat_impl(self, sks, sp, ss):
        """batch_put with the phase-9 exist gate (sub-objects already in L3
        are reported stored without rewriting) and a single retry of failed
        keys. Under a cold-round write burst ~0.1% of puts fail transiently,
        and SGLang's backup bookkeeping does not consume per-page failures —
        an unretried miss becomes a phantom "backed" page that the hot round
        then fails to retrieve (R1 finding, 2026-07-17). The burst is
        momentary, so one delayed retry recovers almost all of them; keys
        still failing stay False (honest result, as before)."""
        recovered = 0
        if not sks:
            return [], 0
        if self._backup_exist_gate:
            present = list(self._batch_exist_flat(sks))
            todo = [i for i, p in enumerate(present) if not p]
            if not todo:
                return [True] * len(sks), 0
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
                    recovered += 1
            todo = failed
        return present, recovered

    def _v2_io(self, transfers, putting):
        """One host-buffer v2 transfer: (per-pool page results, _V2Io).

        The I/O volume is returned, never left on the instance: SGLang runs
        backup (set) and prefetch (get) on different threads, and a set that
        read the instance after a concurrent get had finished took the get's
        bytes, seconds and pool names for its own.
        """
        results = {}
        segments = []
        sks, sp, ss = [], [], []
        marker_buffers = []  # keep per-pool GET destinations alive through native I/O
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
            if putting and not self._rank_writes_pool(name):
                results[name] = [True] * len(keys)
                continue
            pool = self.registered_pools[name]
            logical = getattr(self, "_pool_logical", {}).get(name, False)
            if not logical:
                meta = pool.get_page_buffer_meta(tr.host_indices)
                if _meta_has_no_layout(meta):
                    # Late no-layout declaration: the pool registered with a
                    # physical-looking probe but now reports "no host buffer".
                    # Same contract as registration time — flip to marker-only.
                    if not hasattr(self, "_pool_logical"):
                        self._pool_logical = {}
                    self._pool_logical[name] = logical = True
                    self._note_logical_anchor_once()
            if logical:
                sub = self._pool_sub(name)
                start = len(sks)
                if putting:
                    marker_ptr = _MARKER_PTR
                else:
                    markers = (ctypes.c_ubyte * (len(keys) * sub))()
                    marker_buffers.append(markers)
                    marker_ptr = ctypes.addressof(markers)
                for k in keys:
                    for sk in self._pool_keys(name, k):
                        sp.append(marker_ptr if putting else marker_ptr + len(sks) - start)
                        sks.append(sk); ss.append(1)
                segments.append((name, len(keys), sub, start, len(sks)))
                continue
            ptrs, sizes = meta
            sub = self._pool_sub(name)
            start = len(sks)
            for i, k in enumerate(keys):
                for j, sk in enumerate(self._pool_keys(name, k)):
                    sks.append(sk); sp.append(int(ptrs[i * sub + j])); ss.append(int(sizes[i * sub + j]))
            segments.append((name, len(keys), sub, start, len(sks)))
        t0 = time.perf_counter()
        recovered = 0
        if sks and putting:
            flat, recovered = self._put_flat(sks, sp, ss)
        elif sks:
            karr, klens, parr, sarr, out, key_owners = _arrays(sks, sp, ss)
            self._lib.dfkv_batch_get(
                self._h, karr, klens, parr, sarr, len(sks), out)
            del key_owners
            flat = [out[i] == 1 for i in range(len(sks))]
        else:
            flat = []
        # The actual I/O cost/volume for the v2 metrics. ss covers only pools
        # that did I/O (MLA backup_skip pools were `continue`d before
        # appending), so bytes are 0 on a pure skip.
        io = _V2Io(nbytes=sum(ss), seconds=time.perf_counter() - t0,
                   pools=tuple(seg[0] for seg in segments),
                   recovered=recovered)
        for name, nkeys, sub, start, end in segments:
            results[name] = self._fold(flat[start:end], nkeys, sub)
        return results, io

    def batch_set_v2(self, transfers, extra_info=None) -> dict:
        nkeys = sum(len(tr.keys or []) for tr in (transfers or []))
        with _tracing.span("batch_set_v2", nkeys) as _sp, \
                access_log("batch_set_v2",
                           lambda: f"{self._alog_tag} {_fmt_pools(transfers)}") as r:
            res, io = self._v2_io(transfers, putting=True)
            r.result = _fmt_pool_results(res)
            if io.recovered:
                r.result += f" retry_ok={io.recovered}"
            if _sp:
                _sp.hits = sum(sum(rs) for rs in res.values())
            # Pool-granular metrics: report exactly the pools that performed I/O
            # (io.pools). The MLA rank!=0 replicated latent stays a no-op skip
            # ([True] markers, excluded), but rank-sharded sidecars such as
            # Kimi-K3's mamba/KDA state do real writes on follower ranks — a
            # whole-call tp_rank gate here would hide that write bandwidth
            # entirely from the v2 set metrics.
            io_pages = sum(len(res[name]) for name in io.pools)
            if io_pages:
                self._metrics.on_set_v2(
                    pages=io_pages,
                    ok_pages=sum(sum(res[name]) for name in io.pools),
                    nbytes=io.nbytes, seconds=io.seconds)
            return res

    def batch_get_v2(self, transfers, extra_info=None) -> dict:
        nkeys = sum(len(tr.keys or []) for tr in (transfers or []))
        with _tracing.span("batch_get_v2", nkeys) as _sp, \
                access_log("batch_get_v2",
                           lambda: f"{self._alog_tag} {_fmt_pools(transfers)}") as r:
            res, io = self._v2_io(transfers, putting=False)
            r.result = _fmt_pool_results(res)
            if _sp:
                _sp.hits = sum(sum(rs) for rs in res.values())
            if nkeys:
                self._metrics.on_get_v2(
                    pages=sum(len(rs) for rs in res.values()),
                    hit_pages=sum(sum(rs) for rs in res.values()),
                    nbytes=io.nbytes, seconds=io.seconds)
            return res

    # --- DSA L2-bypass: main KV device-direct + sidecar host, one logical op ----
    def _kv_device_set(self, keys, device_indices):
        """Core of batch_set_v1_device (device-direct SG put) without the tracing /
        access-log wrapper, so batch_set_v2_device can reuse it for the anchor KV.
        Returns (per_page_bools, _V2Io). MLA backup_skip on non-zero TP
        rank (replicated latent) short-circuits to all-True, no I/O."""
        n = len(keys)
        if self.is_mla and not self._mla_replica_writer:
            return [True] * n, _V2Io(0, 0.0, (), 0)
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_page_buffer_meta,
        )
        seg_ptrs, seg_sizes = get_device_page_buffer_meta(
            self.mem_pool_device, device_indices)
        sub, sks, sp, ss = self._flatten_device(keys, seg_ptrs, seg_sizes)
        # After _flatten_device: it records the group count probes use.
        self._device_direct_used = True
        nbytes = sum(sum(s) for s in ss)
        t0 = time.perf_counter()
        flat, recovered = self._put_sg_flat(sks, sp, ss)
        dur = time.perf_counter() - t0
        self._log_write_keys(sks, flat, len(sks))
        return self._fold(flat, n, sub), _V2Io(nbytes, dur, ("kv",), recovered)

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
        Returns (per_page_bools, _V2Io), like the write twin. A page is a hit
        only on a full-length read of every sub-object (a short read is a
        corrupt page)."""
        n = len(keys)
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_page_buffer_meta,
        )
        seg_ptrs, seg_caps = get_device_page_buffer_meta(
            self.mem_pool_device, device_indices)
        sub, sks, sp, sc = self._flatten_device(keys, seg_ptrs, seg_caps)
        # After _flatten_device: it records the group count probes use.
        self._device_direct_used = True
        want = [sum(c) for c in sc]
        t0 = time.perf_counter()
        hits, lens = self._batch_get_sg(sks, sp, sc)
        dur = time.perf_counter() - t0
        flat_ok = [hits[i] == 1 and lens[i] >= want[i] for i in range(len(sks))]
        if not all(flat_ok):
            self._log_read_reject(sks, hits, lens, want, flat_ok, len(sks))
        return self._fold(flat_ok, n, sub), _V2Io(sum(lens), dur, ("kv",), 0)

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
        pool. Returns (per_page_bools, _V2Io); no I/O on a replica follower."""
        n = len(keys)
        if self.is_mla and not self._mla_replica_writer:
            return [True] * n, _V2Io(0, 0.0, (), 0)
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_sidecar_page_buffer_meta,
        )
        pool = self._sidecar_device_pools[str(name)]
        seg_ptrs, seg_sizes = get_device_sidecar_page_buffer_meta(pool, device_indices)
        stride, sks, sp, ss = self._flatten_device(
            keys, seg_ptrs, seg_sizes,
            keys_fn=lambda h: self._pool_keys(name, h, True), sub=self._pool_sub(name),
            pool=str(name))
        nbytes = sum(sum(s) for s in ss)
        t0 = time.perf_counter()
        flat, recovered = self._put_sg_flat(sks, sp, ss)
        dur = time.perf_counter() - t0
        return self._fold(flat, n, stride), _V2Io(
            nbytes, dur, (str(name),), recovered)

    def _sidecar_device_get(self, name, keys, device_indices):
        """Device-direct SG get of the DSA indexer sidecar `name`: RDMA its stored
        blob straight INTO its GPU index buffer (no host staging, no H2D). Read twin
        of _sidecar_device_set. A page is a hit only on a full-length read of every
        sub-key. Returns (per_page_bools, _V2Io), like the write twin."""
        n = len(keys)
        from sglang.srt.mem_cache.device_page_meta import (
            get_device_sidecar_page_buffer_meta,
        )
        pool = self._sidecar_device_pools[str(name)]
        seg_ptrs, seg_caps = get_device_sidecar_page_buffer_meta(pool, device_indices)
        stride, sks, sp, sc = self._flatten_device(
            keys, seg_ptrs, seg_caps,
            keys_fn=lambda h: self._pool_keys(name, h, True), sub=self._pool_sub(name),
            pool=str(name))
        want = [sum(c) for c in sc]
        t0 = time.perf_counter()
        hits, lens = self._batch_get_sg(sks, sp, sc)
        dur = time.perf_counter() - t0
        flat_ok = [hits[i] == 1 and lens[i] >= want[i] for i in range(len(sks))]
        return self._fold(flat_ok, n, stride), _V2Io(
            sum(lens), dur, (str(name),), 0)

    # --- task 6: EAGLE draft KV device-direct (best-effort L3) ------------------
    def _draft_keys(
        self, page_hash: str | bytes, sub: int,
    ) -> List[bytes]:
        pp = self.pp_rank if self.enable_pp else 0
        replicated = sub == 1
        tp = -1 if replicated else self.tp_rank
        pcp_size, pcp_rank = self._storage_pcp(replicated)
        components = ["all"] if replicated else ["k", "v"]
        return [
            pool_key(page_hash,
                     pool=self._pool_id("draft", len(components), True),
                     tp_size=self.tp_size, tp_rank=tp,
                     pcp_size=pcp_size, pcp_rank=pcp_rank,
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
            if sub == 1 and not self._mla_replica_writer:
                r.result = "backup_skip"
                return [True] * n
            stride, sks, sp, ss = self._flatten_device(
                keys, seg_ptrs, seg_sizes,
                keys_fn=lambda h: self._draft_keys(h, sub), sub=sub, pool=None)
            flat, recovered = self._put_sg_flat(sks, sp, ss)
            res = self._fold(flat, n, stride)
            r.result = f"ok {sum(res)}/{n} (draft device-direct)"
            if recovered:
                r.result += f" retry_ok={recovered}"
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
                keys_fn=lambda h: self._draft_keys(h, sub), sub=sub, pool=None)
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
            kv_res, kv_io = self._kv_device_set(kv_keys, kv_device_indices)
            results = {"kv": kv_res}
            recovered = kv_io.recovered
            # Main-KV device write reports on_set (v1-device), matching stock DSA's
            # anchor attribution; no metric when it did no I/O (replica follower).
            if n and kv_io.pools:
                self._metrics.on_set(pages=n, ok_pages=sum(kv_res),
                                     nbytes=kv_io.nbytes, seconds=kv_io.seconds)
            # Sidecar. Task 4: a DEVICE sidecar transfer (device_indices set,
            # host_indices None) RDMAs straight from its GPU index buffer; a host
            # sidecar transfer keeps the stock host v2 path. In bypass every sidecar
            # is device now, but keep the host branch for generality/safety.
            dev_side = [tr for tr in sidecar_transfers if _is_device_transfer(tr)]
            host_side = [tr for tr in sidecar_transfers if not _is_device_transfer(tr)]
            side_pages = side_ok = side_bytes = 0
            side_secs = 0.0
            # Sidecar metrics count exactly the pools that did I/O: a
            # replicated sidecar is skipped on a follower, but a rank-sharded
            # host sidecar is written by every rank and must be counted there.
            side_ios = []
            for tr in dev_side:
                name = str(tr.name)
                res, io = self._sidecar_device_set(name, tr.keys, tr.device_indices)
                results[name] = res
                side_ios.append(({name: res}, io))
            if host_side:
                side, io = self._v2_io(host_side, putting=True)
                results.update(side)
                side_ios.append((side, io))
            for side, io in side_ios:
                side_pages += sum(len(side[name]) for name in io.pools)
                side_ok += sum(sum(side[name]) for name in io.pools)
                side_bytes += io.nbytes; side_secs += io.seconds
                recovered += io.recovered
            if side_pages:
                self._metrics.on_set_v2(
                    pages=side_pages, ok_pages=side_ok,
                    nbytes=side_bytes, seconds=side_secs)
            r.result = f"kv {sum(kv_res)}/{n} (device-direct); " + _fmt_pool_results(
                {k: v for k, v in results.items() if k != "kv"})
            if recovered:
                r.result += f" retry_ok={recovered}"
            if _sp:
                _sp.hits = sum(sum(rs) for rs in results.values())
                _sp.bytes = kv_io.nbytes + side_bytes
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
            kv_res, kv_io = self._kv_device_get(kv_keys, kv_device_indices)
            results = {"kv": kv_res}
            if n:
                self._metrics.on_get(pages=n, hit_pages=sum(kv_res),
                                     nbytes=kv_io.nbytes, seconds=kv_io.seconds)
            dev_side = [tr for tr in sidecar_transfers if _is_device_transfer(tr)]
            host_side = [tr for tr in sidecar_transfers if not _is_device_transfer(tr)]
            side_pages = side_hit = side_bytes = 0
            side_secs = 0.0
            for tr in dev_side:
                name = str(tr.name)
                res, io = self._sidecar_device_get(name, tr.keys, tr.device_indices)
                results[name] = res
                side_pages += len(res); side_hit += sum(res)
                side_bytes += io.nbytes; side_secs += io.seconds
            if host_side:
                side, io = self._v2_io(host_side, putting=False)
                results.update(side)
                side_pages += sum(len(rs) for rs in side.values())
                side_hit += sum(sum(rs) for rs in side.values())
                side_bytes += io.nbytes; side_secs += io.seconds
            if side_pages:
                self._metrics.on_get_v2(
                    pages=side_pages, hit_pages=side_hit,
                    nbytes=side_bytes, seconds=side_secs)
            r.result = f"kv {sum(kv_res)}/{n} (device-direct); " + _fmt_pool_results(
                {k: v for k, v in results.items() if k != "kv"})
            if _sp:
                _sp.hits = sum(sum(rs) for rs in results.values())
                _sp.bytes = kv_io.nbytes + side_bytes
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
            # Prefix lengths every pool can serve. An ALL_PAGES pool serves a
            # prefix when all its pages are present; a TRAILING_PAGES pool
            # (SWA window, Mamba state) when the window ending at the prefix
            # is. Each pool's own boundary taken separately and their minimum
            # is not enough: the minimum can be a length at which a trailing
            # window is incomplete, so the load of that length would miss.
            restorable = set(range(1, kv_pages + 1))
            for tr in (pool_transfers or []):
                if not restorable:
                    break
                name = str(tr.name)
                sub = self._pool_sub(name)
                # Device-direct sidecars store explicit scatter-group keys, so
                # probe group 0 just as batch_exists does for the main KV. A
                # host sidecar stores the bare pool key.
                dev_side = name in getattr(self, "_sidecar_device_pools", {})
                sks = [self._sg_group_key(sk, 0) if dev_side else sk
                       for k in keys[:kv_pages]
                       for sk in self._pool_keys(name, k, dev_side)]
                present = self._fold(self._batch_exist_flat(sks), kv_pages, sub)
                if tr.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                    # Only the *last* `trailing` pages of a candidate prefix need
                    # this pool, as in SGLang's reference batch_exists_v2.
                    trailing = max(1, len(tr.keys) if tr.keys else 1)
                    serves = {
                        prefix_len for prefix_len in range(1, kv_pages + 1)
                        if all(present[i] for i in
                               range(max(0, prefix_len - trailing), prefix_len))
                    }
                    boundary = max(serves, default=0)
                else:  # ALL_PAGES
                    boundary = 0
                    for ok in present:
                        if not ok:
                            break
                        boundary += 1
                    serves = set(range(1, boundary + 1))
                if boundary:
                    hit[name] = boundary
                restorable &= serves
            final = max(restorable, default=0)
            result = _pool_transfer_result(
                PoolTransferResult, final, hit, sorted(restorable))
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
        self._require_single_object_page("set")
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
        self._require_single_object_page("get")
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


for _name in ("batch_set_v1", "batch_set_v1_device", "batch_set_v2",
              "batch_set_v2_device", "batch_set_v1_device_draft"):
    setattr(DfkvHiCache, _name, _with_retain_scope(getattr(DfkvHiCache, _name)))
del _name
