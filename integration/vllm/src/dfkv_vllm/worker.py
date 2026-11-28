# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# The transfer-thread scaffolding (KVTransferThread, KVCacheStoreSendingThread,
# KVCacheStoreRecvingThread) is adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Worker-side logic for DfkvStoreConnector.

Includes the store worker, transfer threads, lookup server, and
DfkvDeviceClient integration. The DfkvDeviceClient drives libdfkv.so over
GPUDirect RDMA; the storage backend is dfkv rather than a Mooncake store, so
Mooncake-store-only features (replica-tier classification, owner-DirectIO disk
offload staging, ReplicateConfig/preferred_segment) are dropped.
"""

import dataclasses
import functools
import json
import logging
import os
import queue
import socket
import threading
import time
import zlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from fractions import Fraction

from typing import Any, TypeVar

import torch
from dfkv_common import (
    apply_rank_local_rail_affinity,
    canonical_namespace,
    pool_key_hash_field,
    reject_namespace_override,
)
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_world_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MLAAttentionSpec,
)

# dfkv: get_dp_engine_index replaces mooncake_utils.get_mooncake_dp_engine_index;
# dfkv handles its own RDMA bootstrap so the transfer-engine helpers are dropped.
from ._determinism import ensure_deterministic_block_hashing
from .client_ranks import ENV_NAME as CLIENT_RANKS_ENV
from .client_ranks import (
    ELIDE_ENV,
    configure_load_convergence,
    participant,
    resolve_client_ranks,
    should_create_client,
)
from .cache_api import (
    cache_group_view,
    group_block_size,
    is_full_attention,
    is_mamba_group,
    logical_bytes_per_block,
    manager_api,
    prefix_cacheable,
    resolve_dcp_group,
    store_group_ids,
    store_groups,
)
from .coordinator import (
    DfkvStoreCoordinator,
    ExternalCachedBlockPool,
    partial_hash_hits_enabled,
)
from .data import (
    VLLM_RAW_LAYOUT,
    ChunkedTokenDatabase,
    DfkvStoreConnectorMetadata,
    KeyMetadata,
    PoolKey,
    ReqMeta,
    key_affixes,
    key_diagnostic_label,
    requires_request_level_loads,
    split_block_contiguous_runs,
)
from .dfkv_client import DfkvDeviceClient, SgDescriptorBatch, _prebuilt_sg_batch
from .dfkv_utils import get_dp_engine_index
from .metrics import DfkvStoreConnectorStats
from .rail_affinity import physical_affinity_rank
from .transfer_protocol import TransferResults
from ._telemetry import config as _tcfg  # connector identity + client_register switch
from .protocol import (
    LOOKUP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
    _decode_lookup_request,
    _decode_lookup_response,
    _encode_lookup_request,
    _encode_lookup_response,
)
from dfkv_common import cabi as dfkv_cabi

logger = init_logger(__name__)

_T = TypeVar("_T")


def _rotate_list(values: list[_T], offset: int) -> list[_T]:
    return values[offset:] + values[:offset]

def _process_log_suffix(
    *,
    dp_rank: int,
    pp_rank: int,
    tp_rank: int,
    global_rank: int,
    pid: int | None = None,
    prefix: str | None = None,
) -> str:
    """Return a process-unique native-client log suffix.

    DP/PP/TP/global coordinates make the file attributable across distributed
    layouts. PID is the collision fence when ranks are reused by colocated
    engines or the launcher cannot provide a globally unique rank.
    """
    process_id = os.getpid() if pid is None else pid
    identity = (
        f"dp{dp_rank}_pp{pp_rank}_tp{tp_rank}_"
        f"g{global_rank}_p{process_id}"
    )
    if prefix == identity or (prefix and prefix.endswith(f"_{identity}")):
        return prefix
    return f"{prefix}_{identity}" if prefix else identity

def _batch_rotation_offset(
    req_id: str,
    block_hashes: list[BlockHash],
    tp_rank: int,
    batch_size: int,
) -> int:
    """Choose a stable, rank-staggered first object for one request batch."""
    if batch_size <= 1:
        return 0
    seed = zlib.crc32(req_id.encode("utf-8"))
    if block_hashes:
        seed = zlib.crc32(bytes(block_hashes[0]), seed)
    return (seed + tp_rank) % batch_size

@dataclasses.dataclass(frozen=True)
class _KeyStripeIdentity:
    """The stored key coordinate and replica stripe owned by one worker."""

    tp_rank: int
    dcp_rank: int
    stripe_idx: int
    stripe_step: int


def _save_needs_step_fence(group) -> bool:
    """Whether this group's SAVE sources may change in the next model step.

    Full-attention managers retain their blocks until the request finishes.
    On the modern API the manager is the source of that guarantee; on legacy
    engines the exact policy-typed specs carry it. Unknown layouts stay fenced.
    """
    view = cache_group_view(group)
    if view.manager_config is not None:
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

        return (
            view.manager_cls is not FullAttentionManager
            or any(
                type(spec) not in (AttentionSpec, MLAAttentionSpec)
                for _, spec in view.layer_specs
            )
        )

    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return any(
        type(spec) not in (FullAttentionSpec, MLAAttentionSpec)
        or any(
            getattr(spec, field, None) is not None
            for field in ("sliding_window", "attention_chunk_size", "rswa_window")
        )
        for _, spec in view.layer_specs
    )


def _warn_pipeline_lookup_scope(pp_size: int, kv_transfer_config: Any) -> bool:
    """With pipeline parallelism the lookup sees one stage only: say so.

    The scheduler's lookup runs on rank 0 and probes the objects of rank 0's
    pipeline stage. The other stages' objects are in namespaces of their own
    (each stage's cache groups list only its layers) and are not probed, so a
    prefix whose objects another stage has lost is admitted and that stage's
    LOAD fails. With kv_load_failure_policy="fail", vLLM's default, the
    request then fails instead of recomputing. Returns whether it warned.
    """
    policy = getattr(kv_transfer_config, "kv_load_failure_policy", None)
    if pp_size <= 1 or policy != "fail":
        return False
    logger.warning(
        "dfkv: pipeline parallel size %d: the scheduler's lookup checks the "
        "objects of the first pipeline stage only. If another stage has lost "
        "its objects of a prefix, that stage's load fails, and with "
        "kv_load_failure_policy=\"fail\" the request fails. Set "
        "kv_load_failure_policy=\"recompute\" to recompute such prefixes.",
        pp_size,
    )
    return True


def _identity_dp_size(extra_config: Any, dp_size: int) -> int:
    """The data-parallel size the object identity carries.

    Each DP replica computes the same bytes for the same prefix (``dp_rank``
    is already -1 in every key), so the DP size does not change an object;
    binding it only makes deployments of different DP sizes miss each
    other's objects. ``share_across_dp_sizes`` drops it. Off by default,
    because it changes every key: objects stored so far are not found once.
    """
    if _tcfg.truthy(extra_config.get("share_across_dp_sizes", False)):
        return 1
    return dp_size


def _resolve_model_revision(extra_config: Any, model_config: Any) -> str:
    """The weights version the namespace binds.

    Explicit ``model_revision``, else the engine's revision, else the commit
    hash of a hub checkout, else the model identity itself. That last
    fallback is a weights version only for a hub id: for a local directory
    it stays the same when other weights are put there, and objects computed
    with the old weights would be loaded for the new ones. Then a warning,
    or with ``require_model_revision`` a startup error.
    """
    model_identity = str(model_config.model)
    hf_config = getattr(model_config, "hf_config", None)
    revision = (
        extra_config.get("model_revision")
        or getattr(model_config, "revision", None)
        or getattr(hf_config, "_commit_hash", None)
    )
    if revision:
        return str(revision)
    if os.path.exists(model_identity):
        message = (
            f"dfkv: model {model_identity!r} is a local path and no weights "
            "version is known (no model_revision, revision or commit hash): "
            "the cache namespace cannot tell other weights in that path apart. "
            "Set kv_connector_extra_config.model_revision"
        )
        if _tcfg.truthy(extra_config.get("require_model_revision", False)):
            raise ValueError(message + ".")
        logger.warning("%s (require_model_revision=true makes this an error).",
                       message)
    return model_identity


def _cache_group_layout(groups: Sequence) -> str:
    """Bind cached bytes to geometry and the manager's allocation policy."""
    def encode(value):
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, Fraction):
            return {"numerator": value.numerator, "denominator": value.denominator}
        if isinstance(value, Enum):
            return value.value
        raise TypeError(f"Unsupported cache-spec value: {type(value).__qualname__}")

    def qualified(value: type) -> str:
        return f"{value.__module__}.{value.__qualname__}"

    entries = []
    for group in groups:
        if not manager_api(group):
            # Keep the legacy geometry projection byte-for-byte stable. v5
            # carries the generation/layout migration independently.
            entries.append({
                "layers": sorted(group.layer_names),
                "spec_type": type(group.kv_cache_spec).__qualname__,
                "spec": dataclasses.asdict(group.kv_cache_spec),
                "is_eagle_group": group.is_eagle_group,
            })
            continue
        view = cache_group_view(group)
        entries.append({
            "schema": "manager-config-v1",
            "layers": sorted(group.layer_names),
            "block_size": view.block_size,
            "dcp_sharded": group.dcp_sharded,
            "host_resident": group.host_resident,
            "enable_kv_transfer": group.enable_kv_transfer,
            "manager": qualified(view.manager_cls),
            "manager_params": dataclasses.asdict(view.manager_config.params),
            "specs": [
                {
                    "layer": layer,
                    "spec_type": qualified(type(spec)),
                    "spec": dataclasses.asdict(spec),
                }
                for layer, spec in view.layer_specs
            ],
            "logical_bytes_per_block": logical_bytes_per_block(view),
        })
    return json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        default=encode,
    )


def _byte_order_layout_fields(
    parallel_config,
    extra_config: Any,
    *,
    dcp_size: int,
    pcp_size: int,
) -> dict[str, object]:
    """Namespace fields for two settings that reorder an object's bytes
    without changing its length.

    * ``cp_kv_cache_interleave_size``: under DCP/PCP it decides which tokens
      a rank's block holds (``i`` tokens on rank r, the next ``i`` on r+1).
    * ``enable_cross_layers_blocks``: vLLM up to 0.26 then hands one tensor
      in its own layer order; per-layer registration concatenates layers in
      layer-name order (``layers.10`` before ``layers.2``).

    Each appears only when it changes the bytes from what the default
    gives, so a namespace with the defaults is the one it always was.
    """
    fields: dict[str, object] = {}
    # cp_kv_cache_interleave_size replaced dcp_kv_cache_interleave_size;
    # an engine that has only the older name configures it there.
    interleave = getattr(parallel_config, "cp_kv_cache_interleave_size", None)
    if interleave is None:
        interleave = getattr(parallel_config, "dcp_kv_cache_interleave_size", None)
    interleave = int(interleave or 1)
    if (dcp_size > 1 or pcp_size > 1) and interleave != 1:
        fields["cp_kv_cache_interleave_size"] = interleave
    if str(extra_config.get("enable_cross_layers_blocks", "False")).lower() == "true":
        fields["cross_layer_blocks"] = True
    return fields


def _dcp_group_rank(
    global_rank: int, dcp_size: int, group_ranks: tuple[int, ...] | None
) -> int:
    """Resolve rank_in_group from real membership when main exposes it."""
    if group_ranks is not None:
        try:
            return group_ranks.index(global_rank)
        except ValueError as exc:
            raise RuntimeError(
                f"global rank {global_rank} is absent from DCP group {group_ranks}"
            ) from exc
    # Legacy vLLM and small test stand-ins without group metadata retain the
    # original PCP x TP transpose-order formula.
    return global_rank % dcp_size


def _key_stripe_identity(
    *,
    tp_rank: int,
    tp_size: int,
    pcp_rank: int,
    pcp_size: int,
    dcp_rank: int,
    dcp_size: int,
    num_kv_heads: int,
    use_mla: bool,
    dcp_group_ranks: tuple[int, ...] | None = None,
    dcp_global_rank_base: int = 0,
) -> _KeyStripeIdentity:
    """Map a vLLM TP/PCP/DCP rank to its key namespace and store stripe.

    Main sorts global ranks inside each DCP group; group membership, not a
    modulo shortcut, determines rank_in_group and the replica order.
    """
    # Attention-free Mamba models report zero KV heads. There is no head
    # replication step to derive; their state sharding is selected by group.
    replica_step = (
        tp_size // num_kv_heads if 0 < num_kv_heads < tp_size else 1
    )
    key_tp_rank = tp_rank // replica_step if replica_step > 1 else tp_rank
    expected_dcp_rank = _dcp_group_rank(
        dcp_global_rank_base + tp_rank * pcp_size + pcp_rank,
        dcp_size,
        dcp_group_ranks,
    )
    if dcp_rank != expected_dcp_rank:
        raise RuntimeError(
            "vLLM DCP rank does not match the PCP x TP group layout: "
            f"tp={tp_rank}/{tp_size} pcp={pcp_rank}/{pcp_size} "
            f"dcp={dcp_rank}/{dcp_size} expected_dcp_rank={expected_dcp_rank}"
        )

    replica_ranks = [
        candidate
        for candidate in range(tp_size)
        if (
            (
                candidate // replica_step
                if replica_step > 1
                else candidate
            )
            == key_tp_rank
            and _dcp_group_rank(
                dcp_global_rank_base + candidate * pcp_size + pcp_rank,
                dcp_size,
                dcp_group_ranks,
            ) == dcp_rank
        )
    ]
    stripe_idx = replica_ranks.index(tp_rank)
    metadata_tp_rank = (
        -1 if use_mla and pcp_size == 1 and dcp_size == 1 else key_tp_rank
    )
    return _KeyStripeIdentity(
        tp_rank=metadata_tp_rank,
        dcp_rank=dcp_rank,
        stripe_idx=stripe_idx,
        stripe_step=len(replica_ranks),
    )


@functools.lru_cache(maxsize=None)
def _key_namespaces(
    *,
    tp_size: int,
    pcp_size: int,
    dcp_size: int,
    num_kv_heads: int,
    use_mla: bool,
    tp_sharded: bool,
    dcp_group_ranks: tuple[int, ...] | None = None,
    dcp_global_rank_base: int = 0,
) -> tuple[tuple[int, int, int], ...]:
    """Distinct ``(tp_rank, pcp_rank, dcp_rank)`` key coordinates that the
    ``TP x PCP`` workers of one PP stage write for a cache group.

    A prefix is loadable only if every one of them holds the chunk: each
    worker later loads from its own coordinate.
    """
    namespaces: dict[tuple[int, int, int], None] = {}
    for pcp_rank in range(pcp_size):
        for tp_rank in range(tp_size):
            dcp_rank = _dcp_group_rank(
                dcp_global_rank_base + tp_rank * pcp_size + pcp_rank,
                dcp_size,
                dcp_group_ranks,
            )
            if not tp_sharded:
                tp_rank = _key_stripe_identity(
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    pcp_rank=pcp_rank,
                    pcp_size=pcp_size,
                    dcp_rank=dcp_rank,
                    dcp_size=dcp_size,
                    num_kv_heads=num_kv_heads,
                    use_mla=use_mla,
                    dcp_group_ranks=dcp_group_ranks,
                    dcp_global_rank_base=dcp_global_rank_base,
                ).tp_rank
            namespaces[(tp_rank, pcp_rank, dcp_rank)] = None
    return tuple(namespaces)


# dfkv: Removed Mooncake-store-only helpers:
#   * disk-offload staging budget math (_align_up,
#     _estimate_disk_offload_staging_bytes, _get_usable_disk_offload_buffer
#     _budget_bytes, _split_disk_offload_load_batches) -- dfkv has no
#     owner-side DirectIO staging budget, so a GET is issued as one batch.
#   * replica-tier classification (_call_replica_predicate,
#     _classify_replica_tier, _get_replica_tiers_by_key,
#     _log_mooncake_load_tier_summary) -- dfkv has no batch_get_replica_desc /
#     memory-vs-disk replica tiers.
#   * MooncakeStoreConfig / _parse_size / MooncakeMode and the NO_AVAILABLE
#     _HANDLE pressure code -- dfkv is configured via kv_connector_extra_config
#     and has no embedded/standalone-store topology or offload pressure signal.

# dfkv: batch_put returns a per-key rc (0 == ok, non-zero == failure).
# Normalize that result into the failed-index list the send thread expects.
def _put_failed_indices(rcs: list[int]) -> list[int]:
    return [i for i, rc in enumerate(rcs) if rc != 0]

def _logical_block_ids(
    mask: Sequence[bool],
    block_ids: Sequence[int],
) -> list[int]:
    # Full allocator tables may include future, uncomputed blocks. Select by
    # logical position, not from their tail. A mask with no selected blocks
    # needs no table at all (modern Mamba uses boundary-only offloads).
    positions = [index for index, selected in enumerate(mask) if selected]
    if not positions:
        return [-1] * len(mask)
    # Only an actually compact state table maps its entries consecutively onto
    # selected mask positions.
    if len(block_ids) >= len(mask):
        return [
            int(block_ids[index]) if selected else -1
            for index, selected in enumerate(mask)
        ]
    if len(block_ids) != len(positions):
        raise ValueError(
            f"block table has {len(block_ids)} ids for "
            f"{len(positions)} selected state slots"
        )
    logical = [-1] * len(mask)
    for position, block_id in zip(
        positions,
        block_ids,
        strict=True,
    ):
        logical[position] = int(block_id)
    return logical



# ============================================================
# Transfer Threads
# ============================================================

# Each direction owns one queue. ReqMeta objects retain block/hash lists and
# CUDA-event references, so an unbounded queue can grow host memory and keep
# scheduler-owned GPU blocks pinned indefinitely when the native client slows.
# Reject-new is deliberately non-blocking: blocking result collection on queue
# capacity would prevent vLLM from observing completions and freeing blocks.
DEFAULT_TRANSFER_QUEUE_CAPACITY = 256
MAX_TRANSFER_QUEUE_CAPACITY = 65536
DEFAULT_RECV_WORKERS = 1
MAX_RECV_WORKERS = 32
DEFAULT_LOAD_WINDOW_KEYS = 0
MAX_LOAD_WINDOW_KEYS = 65536
DEFAULT_LOAD_WINDOW_MIN_KEYS = 0
MAX_LOAD_WINDOW_MIN_KEYS = 65536
# A waiting request is re-polled every scheduler step; reuse its lookup (and
# the retention REFRESH it carries) for this long instead of re-probing.
DEFAULT_LOOKUP_CACHE_TTL_MS = 1000
MAX_LOOKUP_CACHE_TTL_MS = 60000
# A lookup the rank-0 server does not answer in time is a miss; 0 = wait
# forever (the scheduler then hangs with a hung or dead rank 0).
DEFAULT_LOOKUP_TIMEOUT_MS = 10000
MAX_LOOKUP_TIMEOUT_MS = 600000
_TRANSFER_STOP = object()


def _parse_bounded_int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(f"{name} must be an integer")
    if not low <= parsed <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {parsed}")
    return parsed

def _parse_transfer_queue_capacity(value: Any) -> int:
    return _parse_bounded_int("transfer_queue_capacity", value, 1, MAX_TRANSFER_QUEUE_CAPACITY)

def _parse_recv_workers(value: Any) -> int:
    return _parse_bounded_int("recv_workers", value, 1, MAX_RECV_WORKERS)

def _parse_load_window_keys(value: Any) -> int:
    return _parse_bounded_int("load_window_keys", value, 0, MAX_LOAD_WINDOW_KEYS)

def _parse_load_window_min_keys(value: Any) -> int:
    return _parse_bounded_int("load_window_min_keys", value, 0, MAX_LOAD_WINDOW_MIN_KEYS)

def _parse_lookup_cache_ttl_ms(value: Any) -> int:
    return _parse_bounded_int("lookup_cache_ttl_ms", value, 0, MAX_LOOKUP_CACHE_TTL_MS)

def _parse_lookup_timeout_ms(value: Any) -> int:
    return _parse_bounded_int("lookup_timeout_ms", value, 0, MAX_LOOKUP_TIMEOUT_MS)

def _parse_segment_affinity_pages(value: Any) -> int:
    # Keep the upper range non-restrictive: zero is the legacy off switch;
    # a positive value is the number of absolute per-group pages per segment.
    parsed = _parse_bounded_int(
        "segment_affinity_pages", value, 0, (1 << 63) - 1
    )
    return parsed


def _segment_anchor_key(
    db: ChunkedTokenDatabase,
    start: int,
    key: PoolKey,
    block_hashes: list[BlockHash],
    pages: int,
) -> PoolKey:
    """Route to the actual object at this group's absolute segment start.

    A boundary/partial object at the segment's first page is itself the
    anchor; using its full-block hash would name a different, possibly absent
    object. Later pages use the full object key of their segment's first page,
    independent of batch filtering, ownership, submission order and rotation.
    """
    if pages == 0:
        return key
    if start < 0 or start % db.block_size:
        raise ValueError(
            f"segment affinity requires a page-aligned start, got {start} "
            f"for block_size={db.block_size}"
        )
    page = start // db.block_size
    first_page = page // pages * pages
    if first_page == page:
        return key
    try:
        first_hash = db.chunk_hash(first_page, block_hashes)
    except IndexError as exc:
        raise ValueError(
            "segment affinity lacks the absolute segment-first hash for "
            f"page={first_page} (object page={page})"
        ) from exc
    return PoolKey(key.key_metadata, first_hash.hex())


def _load_windows(
    length: int,
    window_keys: int,
    min_keys: int = 0,
) -> tuple[tuple[int, int], ...]:
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return ()
    if length < min_keys or window_keys <= 0 or window_keys >= length:
        return ((0, length),)
    return tuple(
        (start, min(start + window_keys, length))
        for start in range(0, length, window_keys)
    )

def _batch_get_auto_sg_windowed(
    client: Any,
    keys: list[bytes],
    seg_ptrs: list[list[int]],
    seg_caps: list[list[int]],
    window_keys: int,
    min_keys: int,
    route_keys: list[bytes] | None = None,
) -> tuple[list[bool], list[int]]:
    windows = _load_windows(len(keys), window_keys, min_keys)
    if not windows:
        return [], []
    if len(windows) == 1:
        if route_keys is None:
            return client.batch_get_auto_sg(keys, seg_ptrs, seg_caps)
        return client.batch_get_auto_sg(
            keys, seg_ptrs, seg_caps, route_keys=route_keys
        )

    # A flat descriptor batch is windowed in place (no per-segment
    # re-marshalling of each window); plain nested lists are sliced.
    batch = _prebuilt_sg_batch(seg_ptrs, seg_caps)
    hits: list[bool] = []
    lengths: list[int] = []
    for start, stop in windows:
        if batch is not None:
            window = batch.window(start, stop)
            window_ptrs, window_caps = window.ptrs, window.caps
        else:
            window_ptrs, window_caps = seg_ptrs[start:stop], seg_caps[start:stop]
        window_routes = (
            None if route_keys is None else route_keys[start:stop]
        )
        if window_routes is None:
            window_hits, window_lengths = client.batch_get_auto_sg(
                keys[start:stop], window_ptrs, window_caps,
            )
        else:
            window_hits, window_lengths = client.batch_get_auto_sg(
                keys[start:stop], window_ptrs, window_caps,
                route_keys=window_routes,
            )
        hits.extend(window_hits)
        lengths.extend(window_lengths)
    return hits, lengths


@dataclasses.dataclass
class _ReceiveRequestState:
    request: ReqMeta | None
    phase: str = "queued"
    cancel_requested: bool = False
    fail_closed_on_cancel: bool = False
    failed: bool = False
    completion: threading.Event = dataclasses.field(default_factory=threading.Event)




class KVTransferThread(threading.Thread):
    """Base class for async KV cache transfer threads."""

    def __init__(
        self,
        client: Any,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        name: str,
        record_operation: Callable[..., None] | None = None,
        record_observation: Callable[[str, float], None] | None = None,
        queue_capacity: int = DEFAULT_TRANSFER_QUEUE_CAPACITY,
        record_pool_sample: Callable[[str, int], None] | None = None,
    ):
        super().__init__(daemon=False, name=name)
        self.client = client
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.token_databases = token_databases
        self._record_operation_cb = record_operation
        self._record_observation_cb = record_observation
        self._record_pool_sample_cb = record_pool_sample
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []
        self._stop_lock = threading.Lock()
        self._accepting = True
        self._stop_enqueued = False

    def add_request(self, request: ReqMeta) -> bool:
        """Submit without blocking; reject when closed or saturated.

        Rejection is completed synchronously through ``_cancel_request`` so a
        load becomes a recompute and a save's fence can reach zero. This keeps
        scheduler cleanup progressing instead of pinning blocks behind a full
        queue.
        """
        return self._submit(self.request_queue, request)

    def _submit(self, target: "queue.Queue[Any]", request: ReqMeta) -> bool:
        reason = "closed"
        with self._stop_lock:
            if self._accepting:
                try:
                    target.put_nowait(request)
                    return True
                except queue.Full:
                    reason = "saturated"
        self._cancel_request(request)
        logger.warning(
            "%s rejected request %s: transfer queue %s (capacity=%d)",
            self.name,
            getattr(request, "req_id", "<unknown>"),
            reason,
            target.maxsize,
        )
        return False

    def get_and_clear_finished_requests(self) -> set[str]:
        with self.done_task_lock:
            finished = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished

    def set_finished_request(self, req_id: str):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
        self.ready_event.set()
        self._serve(self.request_queue)

    def _serve(self, source: "queue.Queue[Any]") -> None:
        while True:
            request_data = source.get()
            try:
                if request_data is _TRANSFER_STOP:
                    return
                self._handle_request(request_data)
            except Exception as e:
                logger.error("Error in %s: %s", self.name, e)
            finally:
                source.task_done()

    def _handle_request(self, req_meta: Any):
        raise NotImplementedError

    def _cancel_request(self, req_meta: Any) -> None:
        """Resolve a request that will never run."""
        req_id = getattr(req_meta, "req_id", None)
        if req_id is not None:
            self.set_finished_request(req_id)

    def stop(self, *, cancel_pending: bool = True) -> None:
        """Stop accepting work, then cancel or drain and join deterministically.

        ``cancel_pending=True`` is the worker-shutdown policy: queued loads are
        failed into recompute and queued saves decrement their finish fences.
        The one active native operation is allowed to finish before the client
        is closed. ``False`` drains all accepted work before the sentinel.
        """
        with self._stop_lock:
            first_stop = not self._stop_enqueued
            self._accepting = False
            if first_stop:
                self._stop_enqueued = True
                if cancel_pending or self.ident is None:
                    while True:
                        try:
                            pending = self.request_queue.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            if pending is not _TRANSFER_STOP:
                                try:
                                    self._cancel_request(pending)
                                except Exception:
                                    logger.exception(
                                        "%s failed to resolve cancelled request %s",
                                        self.name,
                                        getattr(pending, "req_id", "<unknown>"),
                                    )
                        finally:
                            self.request_queue.task_done()
                if self.ident is not None:
                    self.request_queue.put(_TRANSFER_STOP)

        if self.ident is None:
            return
        self.request_queue.join()
        if threading.current_thread() is not self:
            self.join()

    close = stop

    def _record_operation(
        self,
        operation: str,
        start_time: float,
        num_keys: int,
        *,
        num_logical_keys: int | None = None,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        if self._record_operation_cb is None:
            return
        self._record_operation_cb(
            operation=operation,
            duration_seconds=time.perf_counter() - start_time,
            num_keys=num_keys,
            num_logical_keys=num_logical_keys,
            num_bytes=num_bytes,
            status=status,
            num_failed_keys=num_failed_keys,
        )

    def _note_retain_codes(
        self, operation: str, codes: list, start_time: float
    ) -> None:
        """Account one retention batch started at ``start_time``; keys left
        unprotected count as failed (DENIED, MISS after a put, unsupported,
        or transport error)."""
        unprotected = sum(
            1 for c in codes if c not in dfkv_cabi.RETAIN_PROTECTED_CODES)
        self._record_operation(
            operation,
            start_time,
            len(codes),
            num_logical_keys=len(codes),
            status="partial_failure" if unprotected else "ok",
            num_failed_keys=unprotected,
        )
        if unprotected:
            logger.debug("dfkv %s: %d/%d keys left unprotected (codes=%s)",
                         operation, unprotected, len(codes), sorted(set(codes)))

    def _record_observation(self, name: str, start_time: float) -> None:
        if self._record_observation_cb is not None:
            self._record_observation_cb(name, time.perf_counter() - start_time)

    def _record_pool_sample(self, name: str, value: int) -> None:
        if self._record_pool_sample_cb is not None:
            self._record_pool_sample_cb(name, value)

    def update_kv_event(self, events: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(events)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events


class KVCacheStoreSendingThread(KVTransferThread):
    """Background thread for storing KV cache blocks to the store."""

    def __init__(
        self,
        client: Any,
        coord: DfkvStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        stripe_idx: int,
        stripe_step: int,
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
        record_operation: Callable[..., None] | None = None,
        queue_capacity: int = DEFAULT_TRANSFER_QUEUE_CAPACITY,
        tp_sharded_groups: frozenset[int] = frozenset(),
        segment_affinity_pages: int = 0,
    ):
        super().__init__(
            client,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreSendingThread",
            record_operation=record_operation,
            queue_capacity=queue_capacity,
        )
        # Workers sharing one key coordinate split its chunks without
        # replicated writes. CLIENT_RANKS convergence may later set
        # stripe_idx=None for a non-participant.
        self.stripe_idx: int | None = stripe_idx
        self.stripe_step = stripe_step
        self.tp_sharded_groups = tp_sharded_groups
        self.coord = coord
        self.segment_affinity_pages = _parse_segment_affinity_pages(
            segment_affinity_pages
        )
        self.kv_role = kv_role
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self._save_generations: dict[str, int] = {}
        # Per request: (save generation, {group: tokens}) -- how far this
        # rank's chunks of each full-attention or window group are known
        # stored (PUT or found). A SAVE starts each such group there instead
        # of at chunk 0: the earlier chunks are neither keyed nor probed again.
        # A chunk whose PUT failed holds the mark back, so a later step of the
        # request still retries it.
        self._saved_upto: dict[str, tuple[int, dict[int, int]]] = {}
        self._next_save_generation = 0
        self.enable_kv_event = enable_kv_event
        # Which request _handle_request is executing right now. Published under
        # done_task_lock together with the dequeue gate below, so a request is
        # either dropped at the gate or visibly in flight to
        # wait_for_inflight_put() -- never invisibly both.
        # Requests _handle_request is executing, by worker thread.
        self._active_req_ids: dict[int, str] = {}
        self._active_cv = threading.Condition(self.done_task_lock)
        # SAVEs whose sources the next model step can change (recurrent
        # state, window blocks) get a worker of their own: the model thread
        # waits for exactly those (fence_queue.join()), never behind a PUT of
        # immutable attention blocks that is already running on this thread.
        self.fence_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
        self._fence_worker = threading.Thread(
            target=self._serve_fenced,
            daemon=False,
            name="KVCacheStoreSendingThread-fence",
        )
        self._fence_stop_enqueued = False

    def run(self):
        self._fence_worker.start()
        super().run()

    def _serve_fenced(self) -> None:
        self._serve(self.fence_queue)

    def add_fenced_request(self, request: ReqMeta) -> bool:
        """Submit a SAVE the caller will wait for with ``fence_queue.join()``."""
        return self._submit(self.fence_queue, request)

    def _submit(self, target: "queue.Queue[Any]", request: ReqMeta) -> bool:
        # A rejected SAVE (queue full, or closing) stores nothing; for a
        # hybrid model the attention part can be rejected while the state
        # part was stored. Counted as the save_rejected operation.
        if super()._submit(target, request):
            return True
        self._record_operation(
            "save_rejected", time.perf_counter(), 0, num_logical_keys=0,
            status="error",
        )
        return False

    def stop(self, *, cancel_pending: bool = True) -> None:
        worker = self._fence_worker
        with self._stop_lock:
            self._accepting = False
            first_stop = not self._fence_stop_enqueued
            self._fence_stop_enqueued = True
            if first_stop:
                if cancel_pending or worker.ident is None:
                    while True:
                        try:
                            pending = self.fence_queue.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            if pending is not _TRANSFER_STOP:
                                self._cancel_request(pending)
                        except Exception:
                            logger.exception(
                                "%s failed to resolve cancelled request %s",
                                self.name,
                                getattr(pending, "req_id", "<unknown>"),
                            )
                        finally:
                            self.fence_queue.task_done()
                if worker.ident is not None:
                    self.fence_queue.put(_TRANSFER_STOP)
        if worker.ident is not None:
            self.fence_queue.join()
            if threading.current_thread() is not worker:
                worker.join()
        super().stop(cancel_pending=cancel_pending)

    close = stop

    def add_stored_request(self, request: ReqMeta):
        with self.done_task_lock:
            req_id = request.req_id
            if req_id not in self.stored_requests:
                self._next_save_generation += 1
                self._save_generations[req_id] = self._next_save_generation
            request.save_generation = self._save_generations[req_id]
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, request: ReqMeta):
        with self.done_task_lock:
            req_id = request.req_id
            if self._save_generations.get(req_id) == request.save_generation:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
            self._save_generations.pop(req_id, None)
            self._saved_upto.pop(req_id, None)

    def _saved_marks(self, req_meta: ReqMeta) -> dict[int, int]:
        with self.done_task_lock:
            entry = self._saved_upto.get(req_meta.req_id)
        if entry is None or entry[0] != req_meta.save_generation:
            return {}
        return dict(entry[1])

    def _advance_saved_marks(self, req_meta: ReqMeta, marks: dict[int, int]) -> None:
        with self.done_task_lock:
            if self._save_generations.get(req_meta.req_id) != req_meta.save_generation:
                return  # preempted or finished meanwhile
            entry = self._saved_upto.get(req_meta.req_id)
            if entry is None or entry[0] != req_meta.save_generation:
                entry = (req_meta.save_generation, {})
                self._saved_upto[req_meta.req_id] = entry
            for g_idx, upto in marks.items():
                if upto > entry[1].get(g_idx, 0):
                    entry[1][g_idx] = upto

    def wait_for_inflight_put(self, req_id: str, timeout_s: float = 30.0) -> bool:
        """Block until no store for ``req_id`` is currently executing.

        Queued-but-not-started entries are handled by
        delete_finished_stored_request() + the generation gate (old entries
        cannot revive when the same request ID resumes); the one entry already
        executing cannot be cancelled -- it holds an RDMA read against the
        request's GPU blocks -- so the preemption path must wait it out
        before those blocks can be handed to another request. Returns False
        when the diagnostic interval expires; callers that protect GPU block
        ownership must keep waiting until this returns True.
        """
        with self._active_cv:
            return self._active_cv.wait_for(
                lambda: req_id not in self._active_req_ids.values(), timeout_s
            )

    def _partial_tail_chunks(
        self, req_meta: ReqMeta, token_len: int
    ) -> list[tuple[int, int, int, PoolKey, int]]:
        """``(group, start, end, key, block_id)`` of the whole blocks a
        partial hash hit needs beyond the positional SAVE up to ``token_len``.

        Each is keyed by the hash of the prefix ending at its boundary: the
        recurrent states vLLM handed off off the scheduler block grid, and,
        on the SAVE that first covers the prompt, every other group's blocks
        from the last scheduler block boundary to the prompt checkpoint (the
        attention block that holds it). Both use the checkpoint vLLM hands
        off, so a hit there finds its attention tail and states together.
        """
        coord = self.coord
        if not coord.enable_partial_hash_hits:
            return []
        chunks: list[tuple[int, int, int, PoolKey, int]] = []
        for g_idx, blocks in (req_meta.state_blocks or {}).items():
            if g_idx not in self.tp_sharded_groups:
                continue
            db = self.token_databases[g_idx]
            for boundary, block_id in blocks.items():
                if boundary <= token_len and boundary % db.block_size == 0:
                    continue  # the positional SAVE stores it
                chunk = db.boundary_chunk(boundary, req_meta.block_hashes)
                if chunk is not None and block_id > 0:
                    chunks.append((g_idx, *chunk, block_id))

        if req_meta.num_prompt_tokens is None:
            return chunks
        boundary = coord.partial_tail_boundary(req_meta.num_prompt_tokens)
        if boundary is None:
            return chunks
        masks = coord.load_mask(req_meta.block_hashes, boundary)
        first = boundary // coord.lcm_block_size * coord.lcm_block_size
        for g_idx, db in enumerate(self.token_databases):
            if g_idx in self.tp_sharded_groups or not db.cacheable:
                continue
            mask = masks[g_idx]
            block_ids = _logical_block_ids(mask, req_meta.block_ids[g_idx])
            for idx in range(first // db.block_size, len(mask)):
                # Payload replicas split the tail like positional chunks.
                if (
                    not mask[idx]
                    or block_ids[idx] <= 0
                    or self.stripe_idx is None
                    or idx % self.stripe_step != self.stripe_idx
                ):
                    continue
                chunk = db.boundary_chunk(
                    min((idx + 1) * db.block_size, boundary),
                    req_meta.block_hashes,
                )
                if chunk is not None:
                    chunks.append((g_idx, *chunk, block_ids[idx]))
        return chunks

    def _handle_request(self, req_meta: ReqMeta):
        # Positional saves stay on ``lcm_block_size`` multiples, which is
        # also ``store_mask``'s precondition; partial hash hits add whole
        # blocks keyed off that grid (``_partial_tail_chunks``).
        lcm_block_size = self.coord.lcm_block_size
        token_len = req_meta.token_len_chunk // lcm_block_size * lcm_block_size
        block_ids_per_group = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        # Publish the active request under the same lock as the generation
        # gate. A preemption either invalidates this generation before any GPU
        # read or observes it in flight and waits for it to complete.
        worker_id = threading.get_ident()
        with self.done_task_lock:
            self._active_req_ids[worker_id] = req_id
            gate_ok = self._save_generations.get(req_id) == req_meta.save_generation
        if not gate_ok:
            with self._active_cv:
                del self._active_req_ids[worker_id]
                self._active_cv.notify_all()
            return

        # Decrement the in-flight counter in ``finally`` (the base run loop
        # always signals ``task_done``) so the scheduler can release the GPU
        # blocks it pinned for this request even when the store path raises.
        # A SAVE split by the worker covers only its own groups; the stripe
        # positions above still count every group's chunks.
        save_groups = req_meta.save_groups
        try:
            tail_chunks = [
                chunk
                for chunk in self._partial_tail_chunks(req_meta, token_len)
                if save_groups is None or chunk[0] in save_groups
            ]
            if token_len == 0 and not tail_chunks:
                return

            # Within each lcm region only per-spec relevant chunks are loaded
            # (e.g., SWA or linear attn), so mask out irrelevant chunks
            store_masks = (
                self.coord.store_mask(token_len)
                if req_meta.prefill_end_tokens is None
                else self.coord.store_mask(
                    token_len,
                    num_prompt_tokens=req_meta.prefill_end_tokens,
                )
            )
            starts: list[int] = []
            ends: list[int] = []
            keys: list[bytes] = []
            route_keys: list[bytes] | None = (
                [] if self.segment_affinity_pages else None
            )
            block_hashes: list[BlockHash] = []
            parent_hashes: list[BlockHash | None] = []
            group_indices: list[int] = []
            descriptor_chunks: list[tuple[object, int, int, Sequence[int]]] = []
            emit_events: list[bool] = []
            logical_block_ids_per_group: list[list[int]] = []
            stripe_position = 0
            skipped_state = False
            saved_marks = self._saved_marks(req_meta)
            # Per marked group: the first chunk start that is not done (not
            # masked in yet, or this rank's and not confirmed stored), and
            # this pass's chunks of it as (start, index into keys).
            blocked: dict[int, int] = {}
            marked_chunks: dict[int, list[tuple[int, int]]] = {}
            for g_idx, db in enumerate(self.token_databases):
                mask = store_masks[g_idx]
                logical_block_ids_per_group.append(
                    _logical_block_ids(mask, block_ids_per_group[g_idx])
                )
                # KV events chain each block to its predecessor in the request,
                # not to whichever chunk this rank happened to store before it:
                # every rank must emit an identical event for aggregation.
                parent_hash: BlockHash | None = None
                state_blocks = (
                    req_meta.state_blocks.get(g_idx, {})
                    if g_idx in self.tp_sharded_groups
                    and req_meta.state_blocks is not None
                    else None
                )
                # Recurrent states are saved at a few boundaries and keep no
                # mark; other groups resume where the request's earlier steps
                # left off. Skipped chunks still count stripe positions and
                # the chain of parent hashes, exactly as if enumerated.
                marked = (
                    g_idx not in self.tp_sharded_groups
                    and (save_groups is None or g_idx in save_groups)
                )
                # With a retention TTL the SAVE's SET re-protects every chunk
                # of the prefix (prefix closure: early chunks must not expire
                # before later ones), so nothing is skipped.
                skip = (
                    min(saved_marks.get(g_idx, 0), token_len)
                    if marked and req_meta.retain_ttl_s <= 0
                    else 0
                )
                if skip:
                    skipped = skip // db.block_size
                    stripe_position += sum(
                        1 for i in range(min(skipped, len(mask))) if mask[i]
                    )
                    parent_hash = BlockHash(
                        bytes(db.chunk_hash(skipped - 1, req_meta.block_hashes))
                    )
                if marked:
                    marked_chunks[g_idx] = []
                for start, end, key in db.process_tokens(
                    token_len, req_meta.block_hashes, skip
                ):
                    chunk_idx = start // db.block_size
                    chunk_hash = BlockHash(bytes.fromhex(key.chunk_hash))
                    chunk_parent, parent_hash = parent_hash, chunk_hash
                    if state_blocks is not None:
                        # Save exactly the boundaries vLLM committed this
                        # step, from the blocks it named; never infer a state
                        # block from its position in the mirrored table.
                        if end not in state_blocks or chunk_idx >= len(mask):
                            continue
                        logical_block_ids_per_group[g_idx][chunk_idx] = (
                            state_blocks[end]
                        )
                    elif chunk_idx >= len(mask) or not mask[chunk_idx]:
                        if marked and g_idx not in blocked:
                            blocked[g_idx] = start
                        continue
                    # State checkpoints contain this physical TP rank's shard;
                    # every rank must save them. Only payload replicas stripe.
                    # Count all selected chunks to retain the attention stripe
                    # positions used before introducing per-group ownership.
                    selected = g_idx in self.tp_sharded_groups or (
                        self.stripe_idx is not None
                        and stripe_position % self.stripe_step == self.stripe_idx
                    )
                    stripe_position += 1
                    if not selected:
                        continue
                    if save_groups is not None and g_idx not in save_groups:
                        continue
                    if (
                        g_idx in self.tp_sharded_groups
                        and state_blocks is None
                        and req_meta.step_end is not None
                        and end != req_meta.step_end
                    ):
                        # No engine hand-off: only a step that ends on the
                        # boundary is known to leave that boundary's state in
                        # its positional slot. Fail closed otherwise; lookup
                        # needs the state, so the prefix is not admitted.
                        skipped_state = True
                        continue
                    if logical_block_ids_per_group[g_idx][chunk_idx] <= 0:
                        # HMA may already have released an earlier window.
                        # Null/padding slots are not valid immutable objects.
                        continue
                    if marked:
                        marked_chunks[g_idx].append((start, len(keys)))
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.to_bytes())
                    if route_keys is not None:
                        route_keys.append(_segment_anchor_key(
                            db, start, key, req_meta.block_hashes,
                            self.segment_affinity_pages,
                        ).to_bytes())
                    block_hashes.append(chunk_hash)
                    parent_hashes.append(chunk_parent)
                    group_indices.append(g_idx)
                    descriptor_chunks.append(
                        (db, start, end, logical_block_ids_per_group[g_idx])
                    )
                    emit_events.append(True)

            def advance_marks(confirmed: list[bool]) -> None:
                """Move each marked group's mark to its first chunk not done."""
                marks = {}
                for g_idx, chunks in marked_chunks.items():
                    upto = min(token_len, blocked.get(g_idx, token_len))
                    for start, index in chunks:
                        if not confirmed[index]:
                            upto = min(upto, start)
                            break
                    marks[g_idx] = upto
                if marks:
                    self._advance_saved_marks(req_meta, marks)

            # Like vLLM's own partial-block cache entries, objects keyed off
            # the block grid emit no KV events.
            for g_idx, start, end, key, block_id in tail_chunks:
                db = self.token_databases[g_idx]
                starts.append(start)
                ends.append(end)
                keys.append(key.to_bytes())
                if route_keys is not None:
                    route_keys.append(_segment_anchor_key(
                        db, start, key, req_meta.block_hashes,
                        self.segment_affinity_pages,
                    ).to_bytes())
                block_hashes.append(BlockHash(bytes.fromhex(key.chunk_hash)))
                parent_hashes.append(None)
                group_indices.append(g_idx)
                descriptor_chunks.append((db, 0, db.block_size, [block_id]))
                emit_events.append(False)

            if skipped_state:
                logger.warning(
                    "Skipped recurrent-state SAVE for request %s: boundary %d "
                    "has no engine hand-off and is not the step end",
                    req_id,
                    token_len,
                )

            if not keys:
                return

            # A logical chunk is exactly one dfkv object. The native client
            # owns any HCA-width windowing, so dedup probes the logical keys
            # directly and a missing object is rewritten as a whole.
            save_exists_start = time.perf_counter()
            retain_ttl_s = req_meta.retain_ttl_s
            route_kwargs = (
                {} if route_keys is None else {"route_keys": route_keys}
            )
            try:
                if retain_ttl_s > 0:
                    # SET doubles as the dedup probe: present keys are
                    # protected in the same round trip (prefix closure keeps
                    # every saved chunk of this request protected).
                    codes = self.client.batch_retain(
                        keys, retain_ttl_s, dfkv_cabi.DFKV_RETAIN_SET,
                        **route_kwargs)
                    exists_states = [
                        1 if c in dfkv_cabi.RETAIN_HIT_CODES else 0
                        for c in codes
                    ]
                    self._note_retain_codes(
                        "save_retain_set", codes, save_exists_start)
                else:
                    exists_states = self.client.batch_exist(
                        keys, **route_kwargs)
                if len(exists_states) != len(keys):
                    raise RuntimeError(
                        "batch_exist returned incomplete per-key results: "
                        f"keys={len(keys)} statuses={len(exists_states)}"
                    )
            except Exception:
                self._record_operation(
                    "save_exists",
                    save_exists_start,
                    len(keys),
                    num_logical_keys=len(keys),
                    status="error",
                    num_failed_keys=len(keys),
                )
                raise
            self._record_operation(
                "save_exists",
                save_exists_start,
                len(keys),
                num_logical_keys=len(keys),
            )
            missing_indices = [
                i for i, exists in enumerate(exists_states) if exists != 1
            ]
            confirmed = [exists == 1 for exists in exists_states]

            if not missing_indices:
                advance_marks(confirmed)
                return

            starts = [starts[i] for i in missing_indices]
            ends = [ends[i] for i in missing_indices]
            keys = [keys[i] for i in missing_indices]
            if route_keys is not None:
                route_keys = [route_keys[i] for i in missing_indices]
                route_kwargs["route_keys"] = route_keys
            block_hashes = [block_hashes[i] for i in missing_indices]
            parent_hashes = [parent_hashes[i] for i in missing_indices]
            group_indices = [group_indices[i] for i in missing_indices]
            descriptor_chunks = [descriptor_chunks[i] for i in missing_indices]
            emit_events = [emit_events[i] for i in missing_indices]

            logger.debug(
                "Storing KV cache for %d blocks (groups=%s) for request %s",
                len(keys),
                set(group_indices),
                req_id,
            )

            descriptor_batch = SgDescriptorBatch.from_chunks(descriptor_chunks)
            # One entry per key when KV events are on; None for keys off the
            # block grid.
            stored_events: list[BlockStored | None] = []
            new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]

            for idx, (s, e, g_idx) in enumerate(
                zip(starts, ends, group_indices, strict=True)
            ):
                db = self.token_databases[g_idx]

                if self.enable_kv_event and not emit_events[idx]:
                    stored_events.append(None)
                elif self.enable_kv_event:
                    token_ids = (
                        req_meta.token_ids[s:e]
                        if req_meta.token_ids is not None
                        else None
                    )
                    lookahead = req_meta.num_hash_lookahead_tokens
                    event_kwargs = {}
                    if lookahead and req_meta.token_ids is not None:
                        # An event for a lookahead-dependent hash must carry
                        # both the block tokens and the following L tokens.
                        if e + lookahead <= len(req_meta.token_ids):
                            event_kwargs["lookahead_token_ids"] = [
                                req_meta.token_ids[e : e + lookahead]
                            ]
                        else:
                            token_ids = []
                    stored_event = BlockStored(
                        block_hashes=[new_block_hashes[idx]],
                        parent_block_hash=(
                            maybe_convert_block_hash(parent_hashes[idx])
                            if parent_hashes[idx] is not None
                            else None
                        ),
                        token_ids=token_ids,
                        block_size=db.block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                        group_idx=db.metadata.group_id,
                        **event_kwargs,
                    )
                    stored_events.append(stored_event)

            if current_event is not None:
                current_event.synchronize()

            # One key and one complete segment vector represent each logical
            # chunk. libdfkv splits vectors wider than the negotiated HCA SGE
            # limit into ordered WR windows under one object completion.
            batch_bytes = descriptor_batch.total_bytes
            num_segments = descriptor_batch.num_segments
            put_start = time.perf_counter()
            successful_events: list[BlockStored] = []
            try:
                res = self.client.batch_put_sg(
                    keys, descriptor_batch.ptrs, descriptor_batch.sizes,
                    **route_kwargs,
                )
                if len(res) != len(keys):
                    raise RuntimeError(
                        "batch_put_sg returned incomplete per-key results: "
                        f"keys={len(keys)} statuses={len(res)}"
                    )
                failed = _put_failed_indices(res)
                for i, rc in zip(missing_indices, res, strict=True):
                    confirmed[i] = rc == 0
                advance_marks(confirmed)
                if stored_events:
                    successful_events = [
                        event
                        for event, rc in zip(stored_events, res, strict=True)
                        if rc == 0 and event is not None
                    ]
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_logical_keys=len(keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                logger.debug(
                    "dfkv save_put: keys=%d segments=%d bytes=%d ms=%.1f "
                    "failed=%d",
                    len(keys), num_segments, batch_bytes,
                    (time.perf_counter() - put_start) * 1000.0, len(failed),
                )
                if failed and logger.isEnabledFor(logging.WARNING):
                    failed_codes = set(res[i] for i in failed)
                    # A native multi-WR failure fails the object atomically.
                    # Dropping this save is non-fatal to inference; no sibling
                    # physical keys exist to probe or scrub.
                    logger.warning(
                        "batch_put_sg failed: %d/%d keys failed "
                        "(codes=%s, batch_bytes=%d), first_key=%s",
                        len(failed),
                        len(keys),
                        failed_codes,
                        batch_bytes,
                        key_diagnostic_label(keys[0]) if keys else "N/A",
                    )
                if retain_ttl_s > 0:
                    stored = [k for k, rc in zip(keys, res, strict=True)
                              if rc == 0]
                    if stored:
                        retain_start = time.perf_counter()
                        stored_kwargs = {}
                        if route_keys is not None:
                            stored_kwargs["route_keys"] = [
                                route for route, rc in zip(
                                    route_keys, res, strict=True
                                ) if rc == 0
                            ]
                        self._note_retain_codes(
                            "save_retain_set",
                            self.client.batch_retain(
                                stored, retain_ttl_s,
                                dfkv_cabi.DFKV_RETAIN_SET, **stored_kwargs),
                            retain_start)
            except Exception as e:
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_logical_keys=len(keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(keys),
                )
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(
                        "Failed to put keys %s, error: %s",
                        [key_diagnostic_label(key) for key in keys[:3]],
                        e,
                    )
            if self.enable_kv_event and successful_events:
                self.update_kv_event(successful_events)
        finally:
            with self._active_cv:
                del self._active_req_ids[worker_id]
                self._active_cv.notify_all()
            self.dec_stored_request(req_meta)

    def _cancel_request(self, req_meta: Any) -> None:
        # Keep a zero-valued entry so _get_and_clear_finished_sending can
        # acknowledge a finished request whose save was rejected/cancelled.
        self.dec_stored_request(req_meta)


class KVCacheStoreRecvingThread(KVTransferThread):
    """Bounded worker pool for loading KV cache blocks from the store.

    The object remains a ``Thread`` for connector/API compatibility; that
    thread is worker zero and ``recv_workers - 1`` bounded peer threads share
    its request queue.  A request has one queue entry and one lifecycle state.
    The state retains its GPU block fence until that request alone completes or
    is cancelled, independent of other requests' completion order.
    """

    def __init__(
        self,
        client: Any,
        coord: DfkvStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        record_operation: Callable[..., None] | None = None,
        record_observation: Callable[[str, float], None] | None = None,
        client_provider: Callable[[], Any] | None = None,
        queue_capacity: int = DEFAULT_TRANSFER_QUEUE_CAPACITY,
        recv_workers: int = DEFAULT_RECV_WORKERS,
        load_window_keys: int = DEFAULT_LOAD_WINDOW_KEYS,
        load_window_min_keys: int = DEFAULT_LOAD_WINDOW_MIN_KEYS,
        record_pool_sample: Callable[[str, int], None] | None = None,
        request_level_loads: bool = False,
        segment_affinity_pages: int = 0,
    ):
        super().__init__(
            client,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
            record_operation=record_operation,
            record_observation=record_observation,
            record_pool_sample=record_pool_sample,
            queue_capacity=queue_capacity,
        )
        # CUDA's current device is thread-local. Capture the model owner's
        # device here; receive-pool threads must not fence their default GPU 0.
        self._cuda_device = (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        )
        self.recv_workers = _parse_recv_workers(recv_workers)
        self.load_window_keys = _parse_load_window_keys(load_window_keys)
        self.load_window_min_keys = _parse_load_window_min_keys(
            load_window_min_keys
        )
        self.client_provider = client_provider
        self.segment_affinity_pages = _parse_segment_affinity_pages(
            segment_affinity_pages
        )
        self.request_level_loads = request_level_loads
        self._failed_requests: set[str] = set()
        self._invalid_block_ids_lock = threading.Lock()
        self._invalid_block_ids: set[int] = set()
        # Failed LOADs whose destination blocks are still to be resolved:
        # outside _request_states_lock, when the errors are collected.
        self._failed_load_requests: list[ReqMeta] = []
        self.coord = coord
        self._request_states_lock = threading.Lock()
        self._request_states: dict[str, _ReceiveRequestState] = {}
        self._active_workers = 0
        self._pool_stop_lock = threading.Lock()
        self._pool_stop_started = False
        self._pool_stop_done = threading.Event()
        # Removes of stale objects (see _remove_stale) run here, off the
        # thread that loads: a load can run on the model thread.
        self._stale_remover: ThreadPoolExecutor | None = None
        self._stale_remover_lock = threading.Lock()
        self._warned_stale_remove = False
        self._peer_threads = [
            threading.Thread(
                target=self._run_receive_worker,
                daemon=False,
                name=f"KVCacheStoreRecvingThread-{worker_idx}",
            )
            for worker_idx in range(1, self.recv_workers)
        ]

    @property
    def worker_threads(self) -> tuple[threading.Thread, ...]:
        return (self, *self._peer_threads)

    def start(self) -> None:
        with self._pool_stop_lock:
            if self._pool_stop_started:
                raise RuntimeError("receive pool is closed")
            super().start()
            for worker in self._peer_threads:
                worker.start()

    def run(self) -> None:
        self.ready_event.set()
        self._run_receive_worker()

    def add_request(self, request: ReqMeta) -> bool:
        request._dfkv_receive_enqueued_at = time.perf_counter()  # type: ignore[attr-defined]
        reason = "closed"
        with self._stop_lock:
            if self._accepting:
                with self._request_states_lock:
                    if request.req_id in self._request_states:
                        logger.warning(
                            "%s ignored duplicate request %s",
                            self.name,
                            request.req_id,
                        )
                        return False
                    state = _ReceiveRequestState(request=request)
                    self._request_states[request.req_id] = state
                    try:
                        self.request_queue.put_nowait(request)
                    except queue.Full:
                        reason = "saturated"
                        self._terminalize_locked(state, failed=True)
                    else:
                        self._record_pool_sample(
                            "receive_queue_depth", self.request_queue.qsize()
                        )
                        return True
            else:
                state = _ReceiveRequestState(request=request)
                with self._request_states_lock:
                    existing = self._request_states.get(request.req_id)
                    if existing is not None:
                        return False
                    self._request_states[request.req_id] = state
                    self._terminalize_locked(state, failed=True)
        logger.warning(
            "%s rejected request %s: transfer queue %s (capacity=%d)",
            self.name,
            request.req_id,
            reason,
            self.request_queue.maxsize,
        )
        return False

    def get_and_clear_receive_results(self) -> tuple[set[str], set[str]]:
        """Drain terminal completions and failures in one atomic snapshot."""
        with self._request_states_lock:
            with self.done_task_lock:
                finished = self.finished_requests
                failed = self._failed_requests
                self.finished_requests = set()
                self._failed_requests = set()
            for req_id in finished:
                state = self._request_states.get(req_id)
                if state is not None and state.phase == "terminal":
                    self._request_states.pop(req_id, None)
        return finished, failed

    def get_and_clear_finished_requests(self) -> set[str]:
        finished, _ = self.get_and_clear_receive_results()
        return finished

    def cancel_requests(
        self,
        req_ids: set[str] | list[str] | tuple[str, ...],
        *,
        wait: bool = True,
        fail_closed: bool = True,
    ) -> None:
        """Cancel loads and fence active native calls through completion."""
        wait_events: list[threading.Event] = []
        with self._request_states_lock:
            for req_id in req_ids:
                state = self._request_states.get(req_id)
                if state is None or state.phase == "terminal":
                    continue
                if state.phase == "queued":
                    self._terminalize_locked(state, failed=fail_closed)
                else:
                    state.cancel_requested = True
                    state.fail_closed_on_cancel = (
                        state.fail_closed_on_cancel or fail_closed
                    )
                    wait_events.append(state.completion)
        if wait:
            for completion in wait_events:
                completion.wait()

    def _terminalize_locked(
        self,
        state: _ReceiveRequestState,
        *,
        failed: bool,
    ) -> bool:
        if state.phase == "terminal":
            return False
        req = state.request
        req_id = req.req_id if req is not None else None
        state.phase = "terminal"
        state.request = None
        failed = failed or state.failed
        if failed and not self.request_level_loads and req is not None:
            # A LOAD reports its own destinations, never the request's local
            # prefix-cache hit; resolved when the errors are collected.
            with self._invalid_block_ids_lock:
                self._failed_load_requests.append(req)
        if req_id is not None:
            with self.done_task_lock:
                self.finished_requests.add(req_id)
                if failed and self.request_level_loads:
                    self._failed_requests.add(req_id)
        state.completion.set()
        return True

    def _run_receive_worker(self) -> None:
        while True:
            request = self.request_queue.get()
            try:
                self._record_pool_sample(
                    "receive_queue_depth", self.request_queue.qsize()
                )
                if request is _TRANSFER_STOP:
                    return
                with self._request_states_lock:
                    state = self._request_states.get(request.req_id)
                    if (
                        state is None
                        or state.phase != "queued"
                        or state.request is not request
                    ):
                        continue
                    state.phase = "active"
                    self._active_workers += 1
                    self._record_pool_sample(
                        "receive_active_workers", self._active_workers
                    )
                failed = False
                try:
                    self._handle_request(request)
                except Exception:
                    failed = True
                    logger.exception(
                        "Error in %s for request %s", self.name, request.req_id
                    )
                finally:
                    with self._request_states_lock:
                        state = self._request_states.get(request.req_id)
                        if state is not None and state.phase == "active":
                            self._active_workers -= 1
                            failed = (
                                failed
                                or (
                                    state.cancel_requested
                                    and state.fail_closed_on_cancel
                                )
                            )
                            self._terminalize_locked(state, failed=failed)
                            self._record_pool_sample(
                                "receive_active_workers", self._active_workers
                            )
            finally:
                self.request_queue.task_done()

    def stop(self, *, cancel_pending: bool = True) -> None:
        with self._pool_stop_lock:
            first_stop = not self._pool_stop_started
            if first_stop:
                self._pool_stop_started = True
        if not first_stop:
            self._pool_stop_done.wait()
            return

        try:
            with self._stop_lock:
                self._accepting = False
            if cancel_pending:
                while True:
                    try:
                        pending = self.request_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if pending is not _TRANSFER_STOP:
                            self.cancel_requests(
                                (pending.req_id,),
                                wait=False,
                                fail_closed=False,
                            )
                    finally:
                        self.request_queue.task_done()
                with self._request_states_lock:
                    for state in self._request_states.values():
                        if state.phase == "active":
                            state.cancel_requested = True
            # Inline model-thread loads share the ownership fence but are not
            # among worker_threads; shutdown must also join those native calls.
            with self._request_states_lock:
                active_completions = [
                    state.completion for state in self._request_states.values()
                    if state.phase == "active"
                ]
            for completion in active_completions:
                completion.wait()
            if self.ident is None:
                while True:
                    try:
                        self.request_queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        self.request_queue.task_done()
                with self._request_states_lock:
                    for state in self._request_states.values():
                        if state.phase != "terminal":
                            self._terminalize_locked(state, failed=False)
                    self._request_states.clear()
                self._record_pool_sample("receive_queue_depth", 0)
                self._record_pool_sample("receive_active_workers", 0)
                return
            for _ in self.worker_threads:
                self.request_queue.put(_TRANSFER_STOP)
            self.request_queue.join()
            for worker in self.worker_threads:
                if threading.current_thread() is not worker:
                    worker.join()
            with self._request_states_lock:
                if any(state.phase != "terminal" for state in self._request_states.values()):
                    raise RuntimeError("receive pool stopped with non-terminal requests")
                self._request_states.clear()
            self._record_pool_sample("receive_queue_depth", 0)
            self._record_pool_sample("receive_active_workers", 0)
        finally:
            self._shutdown_stale_remover()
            self._pool_stop_done.set()

    close = stop


    def _record_load_failure(self, req_id: str, block_ids: list[int]) -> None:
        if self.request_level_loads:
            with self._request_states_lock:
                self._request_states[req_id].failed = True
        else:
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(block_ids)

    def get_and_clear_block_ids_with_load_errors(self) -> set[int]:
        if self.request_level_loads:
            return set()
        with self._invalid_block_ids_lock:
            invalid_block_ids = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
            failed_requests, self._failed_load_requests = (
                self._failed_load_requests, []
            )
        for request in failed_requests:
            invalid_block_ids.update(self._destination_block_ids(request))
        return invalid_block_ids

    def load_request_sync(self, request: ReqMeta) -> None:
        """Load on the model thread; parked requests report outcomes later."""
        if not self.request_level_loads:
            self._handle_request(request)
            return
        with self._stop_lock:
            with self._request_states_lock:
                if request.req_id in self._request_states:
                    return
                state = _ReceiveRequestState(request=request, phase="active")
                self._request_states[request.req_id] = state
                if not self._accepting:
                    self._terminalize_locked(state, failed=True)
                    return
        failed = False
        try:
            if self._cuda_device is not None:
                # A runner may defer this hook until after forward for parked
                # loads. Finish outstanding kernels before modifying the pool.
                torch.cuda.synchronize(self._cuda_device)
            self._handle_request(request)
        except Exception:
            failed = True
            raise
        finally:
            with self._request_states_lock:
                self._terminalize_locked(
                    state,
                    failed=failed or (
                        state.cancel_requested and state.fail_closed_on_cancel
                    ),
                )


    def _load_chunks(self, req_meta: ReqMeta):
        """The chunks this LOAD writes: per group, from the end of the local
        prefix-cache hit to the load length, within the group's load mask.
        Yields ``(group, db, start, end, key, chunk_idx, logical_block_ids)``;
        the destination block is ``logical_block_ids[chunk_idx]``."""
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )
        # Skip chunks the consumer's per-group spec wouldn't populate
        # locally (e.g. SWA pre-window) even if the producer stored them.
        load_mask_per_group = self.coord.load_mask(req_meta.block_hashes, token_len)
        for g_idx, db in enumerate(self.token_databases):
            mask = load_mask_per_group[g_idx]
            logical_block_ids = _logical_block_ids(mask, req_meta.block_ids[g_idx])
            full_len = token_len // db.block_size * db.block_size
            chunks = list(db.process_tokens(full_len, req_meta.block_hashes, mask_num))
            if full_len != token_len:
                # A partial hash hit ends inside this block: it was saved
                # whole under the hash of the hit's own prefix.
                tail = db.boundary_chunk(token_len, req_meta.block_hashes)
                if tail is not None and tail[0] >= mask_num:
                    chunks.append(tail)
            for start, end, key in chunks:
                chunk_idx = start // db.block_size
                if chunk_idx >= len(mask) or not mask[chunk_idx]:
                    continue
                yield g_idx, db, start, end, key, chunk_idx, logical_block_ids

    def _destination_block_ids(self, req_meta: ReqMeta) -> tuple[int, ...]:
        """The blocks a failed or rejected LOAD of ``req_meta`` is reported
        with. Only the LOAD's own destinations: the blocks of the request's
        local prefix-cache hit are shared with other running requests, and
        vLLM would roll every request back that holds one. If the request's
        geometry cannot even be resolved, or it names no destination, its
        whole block table: a failed LOAD must always fail something. Ids
        <= 0 (the null block, padding) are never among them."""
        try:
            destinations = tuple(
                logical_block_ids[chunk_idx]
                for _g, _db, _s, _e, _k, chunk_idx, logical_block_ids in (
                    self._load_chunks(req_meta)
                )
            )
        except Exception:
            logger.exception(
                "dfkv: cannot resolve the LOAD destinations of request %s; "
                "reporting its whole block table", req_meta.req_id,
            )
            destinations = ()
        # Block 0 is vLLM's null block, in every request that pads with it:
        # never reported.
        destinations = tuple(b for b in destinations if b > 0)
        blocks = destinations or tuple(
            b for ids in req_meta.block_ids for b in ids if b > 0
        )
        if not blocks:
            logger.warning(
                "dfkv: failed LOAD of request %s has no allocated block to "
                "report (its only destinations are the null block)",
                req_meta.req_id,
            )
        return blocks

    def _handle_request(self, req_meta: ReqMeta):
        req_id = req_meta.req_id
        enqueued_at = getattr(req_meta, "_dfkv_receive_enqueued_at", None)
        if enqueued_at is not None:
            self._record_observation("receive_queue_wait", enqueued_at)
        geometry_start = time.perf_counter()
        try:
            descriptor_chunks: list[
                tuple[object, int, int, Sequence[int]]
            ] = []
            key_list: list[bytes] = []
            route_key_list: list[bytes] | None = (
                [] if self.segment_affinity_pages else None
            )
            block_id_list: list[int] = []
            for g_idx, db, start, end, key, chunk_idx, logical_block_ids in (
                self._load_chunks(req_meta)
            ):
                block_id = logical_block_ids[chunk_idx]
                if block_id <= 0:
                    raise ValueError(
                        f"LOAD requires an allocated block: group={g_idx} "
                        f"chunk={chunk_idx} block_id={block_id}"
                    )
                key_list.append(key.to_bytes())
                if route_key_list is not None:
                    route_key_list.append(_segment_anchor_key(
                        db, start, key, req_meta.block_hashes,
                        self.segment_affinity_pages,
                    ).to_bytes())
                descriptor_chunks.append((db, start, end, logical_block_ids))
                block_id_list.append(block_id)

            # An empty descriptor batch finishes immediately; in particular,
            # avoid taking a modulo by zero while choosing the first object.
            if not key_list:
                self._record_observation("geometry_preparation", geometry_start)
                return

            # Rotate submission order only. The request and its first chunk
            # choose a stable base shard, while adjacent TP ranks start at
            # adjacent objects instead of repeating one fixed burst pattern.
            # All aligned metadata follows the same permutation, so keys,
            # object-internal segment order, and failure attribution are intact.
            rotation = _batch_rotation_offset(
                req_id, req_meta.block_hashes, self.tp_rank, len(key_list)
            )
            rotated_keys = _rotate_list(key_list, rotation)
            rotated_route_keys = (
                None if route_key_list is None
                else _rotate_list(route_key_list, rotation)
            )
            rotated_chunks = _rotate_list(descriptor_chunks, rotation)
            rotated_block_ids = _rotate_list(block_id_list, rotation)
            descriptor_batch = SgDescriptorBatch.from_chunks(rotated_chunks)

            # One logical chunk remains one batch key with its complete
            # destination segment vector. Optional key windows bound each
            # native dedup publication; libdfkv still owns HCA-width multi-WR
            # windowing within each object.
            client = self.client
            if client is None and self.client_provider is not None:
                client = self.client_provider()  # lazy un-elide
                if client is not None:
                    self.client = client
            self._record_observation("geometry_preparation", geometry_start)
            chunk_totals = descriptor_batch.logical_lengths
            batch_bytes = descriptor_batch.total_bytes
            num_segments = descriptor_batch.num_segments

            load_get_start = time.perf_counter()
            try:
                # Native node dedup publishes a fetched result only when the
                # enclosing batch_get_auto_sg call completes. Bounded key
                # windows let follower ranks consume large replicated loads
                # before their wait deadline while preserving result order.
                # Require the exact stored length; misses and either short or
                # oversized objects fail the chunk closed and force vLLM to
                # recompute it.
                if client is None:  # no client (un-elide failed): miss -> recompute
                    hits, lens = [False] * len(rotated_keys), [0] * len(rotated_keys)
                else:
                    hits, lens = _batch_get_auto_sg_windowed(
                        client,
                        rotated_keys,
                        descriptor_batch.ptrs,
                        descriptor_batch.caps,
                        self.load_window_keys,
                        self.load_window_min_keys,
                        rotated_route_keys,
                    )
                if len(hits) != len(rotated_keys) or len(lens) != len(rotated_keys):
                    raise RuntimeError(
                        "batch_get_auto_sg returned incomplete per-key results: "
                        f"keys={len(rotated_keys)} hits={len(hits)} lens={len(lens)}"
                    )
            except Exception as e:
                self._record_load_failure(req_id, rotated_block_ids)
                self._record_operation(
                    "load_get",
                    load_get_start,
                    len(rotated_keys),
                    num_logical_keys=len(rotated_keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(rotated_keys),
                )
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "Failed to get dfkv batch %s, error: %s",
                        [
                            key_diagnostic_label(key)
                            for key in rotated_keys[:3]
                        ],
                        e,
                    )
                return
            finally:
                # Fence even a failed/partial native batch before publishing
                # its terminal outcome and allowing destination block reuse.
                if (
                    self._cuda_device is not None
                    and (
                        self.request_level_loads
                        or os.environ.get("DFKV_GPU_LOAD_FENCE", "1") == "1"
                    )
                ):
                    torch.cuda.synchronize(self._cuda_device)

            failed_indices = [
                i
                for i, (hit, got_len) in enumerate(zip(hits, lens, strict=True))
                if hit != 1 or got_len != chunk_totals[i]
            ]
            failed_block_ids = [rotated_block_ids[i] for i in failed_indices]
            self._record_operation(
                "load_get",
                load_get_start,
                len(rotated_keys),
                num_logical_keys=len(rotated_keys),
                num_bytes=batch_bytes,
                status="partial_failure" if failed_block_ids else "ok",
                num_failed_keys=len(failed_block_ids),
            )
            logger.debug(
                "dfkv load_get: req=%s keys=%d segments=%d bytes=%d "
                "ms=%.1f failed=%d",
                req_id, len(rotated_keys), num_segments, batch_bytes,
                (time.perf_counter() - load_get_start) * 1000.0,
                len(failed_block_ids),
            )
            if failed_block_ids:
                self._record_load_failure(req_id, failed_block_ids)
                if logger.isEnabledFor(logging.WARNING):
                    failed_detail = [
                        (
                            key_diagnostic_label(rotated_keys[i]),
                            hits[i],
                            lens[i],
                            chunk_totals[i],
                        )
                        for i in failed_indices[:3]
                    ]
                    logger.warning(
                        "Failed to get %d dfkv keys from batch "
                        "(batch_keys=%d, first_failures="
                        "[(key, hit, got, expected)]=%s)",
                        len(failed_block_ids),
                        len(rotated_keys),
                        failed_detail,
                    )
                # A hit is only reported for an object read whole whose
                # stored length (as the server reports it) fits the
                # destination, so a mismatch here is a shorter object.
                stale = [
                    rotated_keys[i]
                    for i in failed_indices
                    if hits[i] == 1 and lens[i] < chunk_totals[i]
                ]
                if stale:
                    stale_routes = None
                    if rotated_route_keys is not None:
                        stale_routes = [
                            rotated_route_keys[i]
                            for i in failed_indices
                            if hits[i] == 1 and lens[i] < chunk_totals[i]
                        ]
                    self._schedule_stale_removal(
                        client, stale, stale_routes
                    )

        except Exception as e:
            # Any unexpected failure in the load path -> recompute this
            # request's blocks (never hang vLLM's WAITING_FOR_REMOTE_KVS).
            logger.error("dfkv recv thread failed for req %s: %s", req_id, e)
            self._record_load_failure(
                req_id, list(self._destination_block_ids(req_meta))
            )
    def _cancel_request(self, req_meta: Any) -> None:
        self.cancel_requests(
            (req_meta.req_id,), wait=False, fail_closed=True
        )

    def _schedule_stale_removal(
        self, client: Any, keys: list[bytes],
        route_keys: list[bytes] | None = None,
    ) -> None:
        """Hand aligned keys/anchors to the remover without blocking the load."""
        with self._stale_remover_lock:
            if self._pool_stop_started:
                return
            if self._stale_remover is None:
                self._stale_remover = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="dfkv-stale-remove"
                )
            self._stale_remover.submit(
                self._remove_stale, client, keys, route_keys
            )

    def _shutdown_stale_remover(self) -> None:
        # Queued removes are dropped; one already in the native client is
        # waited for, so it cannot outlive the client handle.
        with self._stale_remover_lock:
            remover, self._stale_remover = self._stale_remover, None
        if remover is not None:
            remover.shutdown(wait=True, cancel_futures=True)

    def _remove_stale(
        self, client: Any, keys: list[bytes],
        route_keys: list[bytes] | None = None,
    ) -> None:
        """Remove objects found under their key shorter than this layout's.

        The namespace binds the cache geometry, so a shorter object under a
        key is not a variant some other reader can use: it is
        one written by an older connector or engine layout (a recurrent-state
        object of the first state tensor only, from before every tensor was
        saved). Left in place, every SAVE's exists probe would keep finding it
        and never write the right one, and every hit would fail its load
        until the server evicted it. Removed, the next SAVE writes it again.
        """
        start = time.perf_counter()
        # This runs as a background future, whose exceptions nobody reads:
        # every failure is caught here, counted and logged.
        try:
            if not client.supports_remove():
                self._record_operation(
                    "load_remove_stale", start, len(keys),
                    status="unsupported", num_failed_keys=len(keys),
                )
                return
            if route_keys is None:
                removed = client.batch_remove(keys)
            else:
                removed = client.batch_remove(keys, route_keys=route_keys)
            not_removed = sum(1 for rc in removed if rc != 1)
        except Exception as e:
            # Best effort: the object is only left in place.
            self._record_operation(
                "load_remove_stale", start, len(keys),
                status="error", num_failed_keys=len(keys),
            )
            log = logger.debug if self._warned_stale_remove else logger.warning
            self._warned_stale_remove = True
            log("dfkv stale-object remove failed (further failures are only "
                "counted as load_remove_stale errors): %r", e)
            return
        self._record_operation(
            "load_remove_stale", start, len(keys),
            status="partial_failure" if not_removed else "ok",
            num_failed_keys=not_removed,
        )




# ============================================================
# Store Worker
# ============================================================


class DfkvStoreWorker:
    """Worker-side component for DfkvStoreConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.dp_rank = get_dp_engine_index(parallel_config)
        self.dp_size = parallel_config.data_parallel_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.local_rank = int(get_world_group().local_rank)

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.pp_rank = pipeline_rank(
            parallel_config.rank, self.tp_size, self.pcp_size, self.pp_size
        )
        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_rank = dcp_group.rank_in_group if self.dcp_size > 1 else 0
        group_ranks = getattr(dcp_group, "ranks", None)
        self._dcp_group_ranks = (
            tuple(group_ranks) if group_ranks is not None else None
        )
        self._dcp_global_rank_base = (
            parallel_config.rank - self.tp_rank * self.pcp_size - self.pcp_rank
        )
        self.affinity_rank = physical_affinity_rank(
            local_rank=self.local_rank,
            tp_rank=self.tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
        )

        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._rail_affinity = apply_rank_local_rail_affinity(
            extra, self.affinity_rank, os.environ
        )
        if self._rail_affinity.enabled:
            logger.info(
                "dfkv rail affinity: status=%s physical_rank=%d "
                "dp=%d/%d pp=%d/%d tp=%d/%d pcp=%d/%d dcp=%d/%d "
                "available=%s selected=%s primary=%s fallbacks=%d",
                self._rail_affinity.reason,
                self.affinity_rank,
                self.dp_rank,
                self.dp_size,
                self.pp_rank,
                self.pp_size,
                self.tp_rank,
                self.tp_size,
                self.pcp_rank,
                self.pcp_size,
                self.dcp_rank,
                self.dcp_size,
                ",".join(self._rail_affinity.available) or "<none>",
                ",".join(self._rail_affinity.selected) or "<none>",
                self._rail_affinity.primary or "<none>",
                self._rail_affinity.fallback_count,
            )
        self.transfer_queue_capacity = _parse_transfer_queue_capacity(
            extra.get(
                "transfer_queue_capacity",
                DEFAULT_TRANSFER_QUEUE_CAPACITY,
            )
        )
        self.recv_workers = _parse_recv_workers(
            extra.get("recv_workers", DEFAULT_RECV_WORKERS)
        )
        self.load_window_keys = _parse_load_window_keys(
            extra.get("load_window_keys", DEFAULT_LOAD_WINDOW_KEYS)
        )
        self.load_window_min_keys = _parse_load_window_min_keys(
            extra.get(
                "load_window_min_keys",
                DEFAULT_LOAD_WINDOW_MIN_KEYS,
            )
        )
        self.segment_affinity_pages = _parse_segment_affinity_pages(
            extra.get("segment_affinity_pages", 0)
        )
        logger.info(
            "dfkv transfer queues: capacity=%d per direction, recv_workers=%d, "
            "load_window_keys=%d, load_window_min_keys=%d, overload=reject-new, "
            "shutdown=cancel-pending",
            self.transfer_queue_capacity,
            self.recv_workers,
            self.load_window_keys,
            self.load_window_min_keys,
        )
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_done = threading.Event()
        # Store keys embed block_hashes: refuse to start with process-local
        # hashing (silent 0% cross-instance/cross-restart hit rate otherwise).
        ensure_deterministic_block_hashing(vllm_config.cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = extra.get("load_async", True)
        if not isinstance(self.load_async, bool):
            raise ValueError("dfkv connector: load_async must be a boolean")
        self.request_level_loads = requires_request_level_loads(kv_cache_config)
        logger.info(
            "dfkv load mode: %s",
            "async-overlap" if self.load_async else "serialized-model-thread",
        )
        self.cache_config = vllm_config.cache_config
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.num_layers = model_config.get_num_layers(parallel_config)

        self.use_mla = False
        if (
            hasattr(model_config, "use_mla")
            and isinstance(model_config.use_mla, bool)
            and model_config.use_mla
        ):
            self.use_mla = True

        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()

        key_stripe = _key_stripe_identity(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pcp_rank=self.pcp_rank,
            pcp_size=self.pcp_size,
            dcp_rank=self.dcp_rank,
            dcp_size=self.dcp_size,
            num_kv_heads=self.num_kv_head,
            use_mla=self.use_mla,
            dcp_group_ranks=getattr(self, "_dcp_group_ranks", None),
            dcp_global_rank_base=getattr(self, "_dcp_global_rank_base", 0),
        )
        self.head_or_tp_rank = key_stripe.tp_rank
        self.dcp_rank = key_stripe.dcp_rank
        self.stripe_idx = key_stripe.stripe_idx
        self.stripe_step = key_stripe.stripe_step
        self._store_group_ids = store_group_ids(kv_cache_config)
        groups = store_groups(kv_cache_config)
        # Mamba/KDA states are TP-sharded even when attention uses a replicated
        # MLA latent. Packed groups are classified across their distinct specs.
        self._tp_sharded_groups = frozenset(
            g_idx
            for g_idx, group in enumerate(groups)
            if is_mamba_group(group) and prefix_cacheable(group)
        )
        # KV events: how many TP workers share one stored copy of a group's
        # block. Replicated payloads are striped across their replica set, so
        # one worker reports each block; TP-sharded state needs every worker.
        replication_factors = [1] * (max(self._store_group_ids, default=-1) + 1)
        for g_idx, engine_group_id in enumerate(self._store_group_ids):
            replication_factors[engine_group_id] = (
                1 if g_idx in self._tp_sharded_groups else key_stripe.stripe_step
            )
        self.group_tp_replication_factors = tuple(replication_factors)
        self._kv_cache_groups = [
            resolve_dcp_group(group, self.dcp_size) for group in groups
        ]
        self._kv_cache_group_views = [
            cache_group_view(group) for group in self._kv_cache_groups
        ]
        # Groups whose SAVE sources the next model step may change: recurrent
        # state is updated or copied into blocks, windows free blocks that are
        # reused at once, both while the request is still running. Their
        # PUTs finish before the step returns; the others' blocks stay with
        # the request until vLLM is told the save is done.
        self._fenced_store_groups = frozenset(
            g_idx
            for g_idx, view in enumerate(self._kv_cache_group_views)
            if view.prefix_cacheable and _save_needs_step_fence(view)
        )

        # CLIENT_RANKS (issue #111): converge store-side dfkv clients onto a
        # subset of ranks when the KV object is fully TP-replicated.
        # Layout-clamped, never rejects (one fleet-wide env template must be
        # safe everywhere).
        replicated = self.head_or_tp_rank < 0 and not self._tp_sharded_groups
        requested = os.environ.get(CLIENT_RANKS_ENV)
        self.client_ranks, cr_reason = resolve_client_ranks(
            requested, self.tp_size, replicated
        )
        logger.info(
            "client_ranks: requested=%s effective=%d/%d reason=%s%s",
            requested or "<unset>",
            self.client_ranks,
            self.tp_size,
            cr_reason,
            " (store convergence active)" if self.client_ranks < self.tp_size else "",
        )
        self.load_convergence, load_reason = configure_load_convergence(
            os.environ, self.tp_size, self.client_ranks, replicated
        )
        logger.info("client_ranks load convergence: %s", load_reason)

        self.metadata = KeyMetadata(
            model_name=model_config.model.rstrip("/").split("/")[-1],
            dp_size=_identity_dp_size(
                vllm_config.kv_transfer_config.kv_connector_extra_config,
                self.dp_size,
            ),
            dp_rank=-1,
            tp_size=self.tp_size,
            tp_rank=self.head_or_tp_rank,
            pcp_size=self.pcp_size,
            pcp_rank=self.pcp_rank,
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank,
        )

        # dfkv uses a native device client and selects its own RNIC through
        # DFKV_RDMA_DEV; all connector settings come from extra_config.
        # Membership: prefer MDS discovery (production) when mds_endpoints is set;
        # else fall back to a static members list. The client requires one of them.
        mds_endpoints = extra.get("mds_endpoints", "")
        mds_group = extra.get("mds_group", "default")
        # Client registration (so `dfkvctl clients` can list this instance): on by
        # default when MDS is in use; opt out via extra_config["client_register"]
        # = false or DFKV_CLIENT_REGISTER=0. The identity string is parsed by the
        # CLI into type/model/role/tp columns; client_id defaults to host:pid:rank.
        client_register = _tcfg.truthy(
            extra.get("client_register",
                      os.environ.get("DFKV_CLIENT_REGISTER", "1")))
        client_id = ""
        client_info = ""
        if mds_endpoints and client_register:
            client_id = _tcfg.resolve_connector_id(extra, tp_rank=self.tp_rank)
            client_info = (
                f"type={_tcfg.TYPE_VLLM},model={self.metadata.model_name},"
                f"role={self.kv_role},tp_size={self.tp_size},"
                f"tp_rank={self.tp_rank},ver={_tcfg.dist_version('dfkv-vllm')}"
            )
        # Namespace aliases are not configurable: this binary identity is the
        # payload type boundary and must follow the connector schema.
        _tcfg.require_ring_endpoint(extra.get("members", ""), mds_endpoints)
        reject_namespace_override(extra)
        model_identity = str(model_config.model)
        _tcfg.require_isolation_name(
            model_identity, field="model_name")
        _model_revision = _resolve_model_revision(extra, model_config)
        _cache_dtype = str(
            getattr(self.cache_config, "cache_dtype", "unknown"))
        _group_layout = _cache_group_layout(self._kv_cache_groups)
        _logical_bytes_per_block = [
            logical_bytes_per_block(view) for view in self._kv_cache_group_views
        ]
        key_namespace = canonical_namespace(
            model_identity,
            VLLM_RAW_LAYOUT.decode("ascii"),
            tenant_id=str(extra.get("tenant_id", "default")),
            model_revision=str(_model_revision),
            dtype=_cache_dtype,
            block_tokens=max(1, self.block_size),
            layer_count=max(1, self.num_layers),
            tp_size=self.tp_size,
            dp_size=_identity_dp_size(extra, self.dp_size),
            pp_size=self.pp_size,
            layout_fields={
                "storage_layout": VLLM_RAW_LAYOUT.decode("ascii"),
                "cache_dtype": _cache_dtype,
                "model_dtype": str(getattr(model_config, "dtype", "unknown")),
                "block_size": self.block_size,
                "hash_block_size": self.hash_block_size,
                "num_layers": self.num_layers,
                "num_kv_heads": self.num_kv_head,
                "use_mla": self.use_mla,
                "pcp_size": self.pcp_size,
                "dcp_size": self.dcp_size,
                "group_layout": _group_layout,
                "logical_bytes_per_block": json.dumps(
                    _logical_bytes_per_block, separators=(",", ":")
                ),
                "kv_cache_layout": getattr(kv_cache_config, "kv_cache_layout", None),
                **_byte_order_layout_fields(
                    parallel_config,
                    extra,
                    dcp_size=self.dcp_size,
                    pcp_size=self.pcp_size,
                ),
            },
        )
        # Same-host rendezvous defaults on only for replicated MLA topology:
        # it exists for — MLA with REPLICATED KV across tp ranks (dcp/pcp
        # shard the KV, so their per-rank keys never rendezvous). The vLLM SG
        # data path is GPUDirect, so both flavors are defaulted: the CUDA-IPC
        # one for payloads, the host one for exist probes. Explicit env
        # settings ("0" included) always win over the auto default.
        if (self.use_mla and self.tp_size > 1 and self.dcp_size <= 1
                and getattr(self, "pcp_size", 1) <= 1):
            if os.environ.get("DFKV_CLIENT_NODE_DEDUP") is None:
                os.environ["DFKV_CLIENT_NODE_DEDUP"] = "1"
                logger.info(
                    "dfkv node-dedup auto-enabled (mla, tp=%d): same-host "
                    "rendezvous collapses replicated L3 loads; set "
                    "DFKV_CLIENT_NODE_DEDUP=0 to disable", self.tp_size)
            if (os.environ.get("DFKV_CLIENT_NODE_DEDUP") == "1"
                    and os.environ.get("DFKV_CLIENT_NODE_DEDUP_GPU") is None):
                os.environ["DFKV_CLIENT_NODE_DEDUP_GPU"] = "1"

        # Native per-node logs must not collide across colocated DP/PP/TP
        # workers. Preserve an operator prefix, but always append the complete
        # rank identity and PID collision fence instead of trusting TP alone.
        os.environ["DFKV_CLIENT_LOG_SUFFIX"] = _process_log_suffix(
            dp_rank=self.dp_rank,
            pp_rank=self.pp_rank,
            tp_rank=self.tp_rank,
            global_rank=parallel_config.rank,
            prefix=os.environ.get("DFKV_CLIENT_LOG_SUFFIX"),
        )

        client_kwargs = dict(
            members=extra.get("members", ""),
            mds_endpoints=mds_endpoints,
            mds_group=mds_group,
            mds_poll_ms=int(extra.get("mds_poll_ms", 3000)),
            key_namespace=key_namespace,
            lib_path=extra.get("lib"),
            batch_concurrency=int(extra.get("batch_concurrency", 0)),
            client_register=client_register,
            client_id=client_id,
            client_info=client_info,
            model=model_identity,
            cache_role=str(self.kv_role),
            require_rdma=_tcfg.truthy(extra.get("require_rdma", True)),
            tp_rank=self.tp_rank,
            segment_affinity=self.segment_affinity_pages > 0,
        )
        # Converged replicated layouts enable native same-host GPU rendezvous:
        # one TP rank performs each remote GET and CUDA IPC publishes identical
        # MLA bytes to followers. Producer non-participants also skip eager
        # client creation by default; an explicit CLIENT_ELIDE=0 overrides.
        self._lazy_client_kwargs: dict | None = None
        self._lazy_client_lock = threading.Lock()
        self._kv_pool_regions: list[tuple[int, int]] = []
        elide_default = "1" if self.load_convergence else "0"
        create_client, elide_reason = should_create_client(
            self.kv_role,
            self.tp_rank,
            self.tp_size,
            self.client_ranks,
            _tcfg.truthy(os.environ.get(ELIDE_ENV, elide_default)),
        )
        if not create_client:
            logger.info("dfkv client elided: %s", elide_reason)
            self.client = None
            self._lazy_client_kwargs = client_kwargs
        else:
            self.client = DfkvDeviceClient(**client_kwargs)

        # dfkv: no disk-offload staging budget (Mooncake owner-DirectIO only).

        # Start lookup server on rank 0 for scheduler-side prefix queries
        self.lookup_server: LookupKeyServer | None = None
        if vllm_config.parallel_config.rank == 0:
            self.lookup_server = LookupKeyServer(self, vllm_config)
            _warn_pipeline_lookup_scope(self.pp_size, vllm_config.kv_transfer_config)

        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
        self.finished_store_req: set[str] = set()
        self._kv_connector_stats_lock = threading.Lock()
        self.kv_connector_stats = DfkvStoreConnectorStats()

        self._kv_cache_config = kv_cache_config
        manager_groups = manager_api(kv_cache_config)
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        if spec_cfg is not None and callable(
            getattr(spec_cfg, "use_eagle_block_drop", None)
        ):
            use_eagle = bool(spec_cfg.use_eagle_block_drop())
        else:
            use_eagle = not manager_groups and bool(
                spec_cfg.use_eagle()
                if spec_cfg is not None
                and callable(getattr(spec_cfg, "use_eagle", None))
                else False
            ) and not getattr(spec_cfg, "disable_eagle_block_drop", False)
        self.coord = DfkvStoreCoordinator(
            self._kv_cache_groups,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            use_eagle=use_eagle,
            enable_partial_hash_hits=partial_hash_hits_enabled(
                kv_cache_config, self.block_size
            ),
            num_hash_lookahead_tokens=(
                getattr(vllm_config, "num_hash_lookahead_tokens", 0)
                if manager_groups
                else 0
            ),
            retention_interval=getattr(
                kv_cache_config, "prefix_cache_retention_interval", None
            ),
        )
        # One ChunkedTokenDatabase per group; addresses populated in
        # register_kv_caches once the kv-cache layout is known.
        self.token_dbs: list[ChunkedTokenDatabase] = [
            ChunkedTokenDatabase(
                dataclasses.replace(
                    self.metadata,
                    group_id=self._store_group_ids[g_idx],
                    # Old state objects used attention's collapsed rank
                    # coordinates. Keep them outside the corrected state pool,
                    # including legacy numeric GQA/DCP coordinate collisions.
                    pool_name="mamba" if g_idx in self._tp_sharded_groups else "kv",
                    tp_rank=(
                        self.tp_rank
                        if g_idx in self._tp_sharded_groups
                        else self.head_or_tp_rank
                    ),
                ),
                group_block_size(g),
                hash_block_size=self.hash_block_size,
                cacheable=prefix_cacheable(g),
            )
            for g_idx, g in enumerate(self._kv_cache_groups)
        ]

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a unified cross-layer tensor against its real cache group."""
        assert len(self._kv_cache_groups) == 1, (
            "Cross-layer KV cache is supported only for a single cache group"
        )
        layer_name = next(
            (name for name in self._kv_cache_groups[0].layer_names if name),
            None,
        )
        if layer_name is None:
            raise RuntimeError(
                "Cannot register cross-layer KV cache: cache group 0 has no "
                "real layer names"
            )
        self.register_kv_caches({layer_name: kv_cache})

    def _ensure_client_for_load(self) -> Any:
        """Lazily un-elide: create the dfkv client on an elided producer rank
        the first time a real load reaches it (cross-instance prefix reuse —
        get_transfer_results has no role gate). Keeps phase 2a's connection savings for
        the common P-instance case while never trading a whole-span recompute
        for them. Thread-safe; returns None (load misses, vLLM recomputes) if
        creation fails or the rank was never elided-with-kwargs."""
        if getattr(self, "_closed", False):
            return None
        if self.client is not None:
            return self.client
        if self._lazy_client_kwargs is None:
            return None
        with self._lazy_client_lock:
            if getattr(self, "_closed", False):
                return None
            if self.client is None:
                client = None
                try:
                    client = DfkvDeviceClient(**self._lazy_client_kwargs)
                    self._validate_cache_block_sizes(client)
                    for base, ln in self._kv_pool_regions:
                        client.register_memory(base, ln)
                except Exception:
                    if client is not None:
                        client.close()
                    logger.exception(
                        "dfkv lazy un-elide failed; loads on this rank miss")
                    return None
                self.client = client
                logger.info(
                    "dfkv client un-elided: a load reached this elided "
                    "producer rank (tp_rank=%d)", self.tp_rank)
        return self.client

    def _validate_cache_block_sizes(self, client) -> None:
        """Each cache group's scheduler block is one logical dfkv object."""
        for group, db in enumerate(self.token_dbs):
            if db.cacheable:
                client.validate_block_sizes(
                    (db.geometry.logical_bytes_per_block,),
                    context=f"vLLM cache group {group}")

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        """Register KV cache tensors and start transfer threads."""
        if getattr(self, "_closed", False):
            raise RuntimeError("dfkv worker is closed")
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        # An attention layer is one tensor. A recurrent-state layer (Mamba,
        # GDN, KDA, ...) is, in vLLM 0.26.x and earlier, a list of tensors --
        # conv state, SSM / recurrent state -- that are views of one page
        # buffer at successive offsets; from 0.27 it is one view over the
        # whole page.
        # Every tensor's bytes belong to the object: a hit that restored only
        # the first would leave the others' bytes stale in the target block.
        def _layer_tensors(v) -> list:
            return list(v) if isinstance(v, list | tuple) else [v]

        # The committed block count. Under vLLM's extensible KV cache the
        # connector registers inside extend_kv_cache, before the worker
        # updates cache_config.num_gpu_blocks from the reserved capacity.
        self.num_blocks = self._kv_cache_config.num_blocks

        # dfkv: map each layer name to its kv_cache_group so that each group's
        # ChunkedTokenDatabase addresses ONLY its own group's layer segments.
        # The Mooncake template handed the SAME flat addrs list to every group's
        # token_db -- correct for single-group models, but for a multi-group
        # model (DeepSeek-V4 MLA main + lightning-indexer = 2 groups) it makes
        # group g's block_ids index the OTHER group's layer regions, scattering
        # the loaded KV into the wrong blocks (load succeeds but is corrupt ->
        # vLLM resumes over garbage). Partition per group.
        layer_to_group: dict[str, int] = {}
        for g_idx, group in enumerate(self._kv_cache_groups):
            for ln in group.layer_names:
                layer_to_group[ln] = g_idx

        # dfkv: decouple MR registration (dedup by physical storage) from
        # per-layer address collection (NEVER dedup). Multiple kv_cache_groups
        # can ALIAS the same storage -- on DeepSeek-V4-Flash the bs4 partial-
        # state group (g3) is 168 views of the main-MLA group (g0)'s blocks
        # (same 8640B/block, off=0, different logical shape). The Mooncake
        # template deduped by storage in the SAME loop that collected addrs, so
        # every aliased layer got NO segment -> its group was never offloaded
        # (observed: segments_per_group=[62,1,0,0,0]). Register each storage
        # once, but give EVERY layer (aliased or not) its segment in its group.
        registered_ptrs: set[int] = set()
        group_addrs: list[list[int]] = [[] for _ in self.token_dbs]
        group_block_lens: list[list[int]] = [[] for _ in self.token_dbs]
        # Exact (base, block_stride, block_content) triples per group for
        # blocks-first layers; deduped because hybrid groups can alias the
        # same physical slots (DeepSeek-V4 g3 views of g0 blocks).
        group_seg_layouts: list[list[tuple[int, int, int]]] = [
            [] for _ in self.token_dbs
        ]
        # Groups whose layers came as several state tensors: see below.
        multi_tensor_groups: set[int] = set()

        # Dict insertion order is not a storage-layout contract. Canonicalize by
        # semantic layer name so independently constructed producer/consumer
        # mappings gather and scatter the same ordered byte stream.
        for layer_name in sorted(kv_caches):
            g_idx = layer_to_group.get(layer_name)
            if g_idx is None or not self.token_dbs[g_idx].cacheable:
                continue
            tensors = _layer_tensors(kv_caches[layer_name])
            if len(tensors) > 1:
                multi_tensor_groups.add(g_idx)
            for cache in tensors:
                cache_storage = cache.untyped_storage()
                base_addr = cache_storage.data_ptr()
                region_len = cache_storage.nbytes()

                # register_memory raises on failure (vs Mooncake's int return), so
                # a failed GPUDirect MR registration aborts bring-up loudly. Register
                # each physical storage region ONCE (aliased groups reuse the MR).
                if base_addr not in registered_ptrs:
                    registered_ptrs.add(base_addr)
                    # Recorded even when the client is elided: a lazy un-elide
                    # (load reaching an elided rank) must replay these
                    # registrations before its first GET.
                    self._kv_pool_regions.append((base_addr, region_len))
                    if self.client is not None:
                        self.client.register_memory(base_addr, region_len)

                layer_addr = cache.data_ptr()
                el = cache.element_size()
                page_size_bytes = region_len // self.num_blocks
                outer_dims = [
                    d for d in range(cache.ndim)
                    if cache.stride(d) * el > page_size_bytes
                ]
                if not outer_dims:
                    # Blocks-first layout (FlashInfer / MLA): one segment.
                    group_addrs[g_idx].append(layer_addr)
                    group_block_lens[g_idx].append(page_size_bytes)
                    # Exact geometry for blocks-first layouts. A layer may contain
                    # several disjoint contiguous runs inside one block (padding /
                    # transpose); represent every run explicitly rather than
                    # silently transferring the whole stride and crossing slots.
                    block_stride, runs = split_block_contiguous_runs(
                        cache.shape, cache.stride(), el, logical_blocks=self.num_blocks
                    )
                    for offset, content in runs:
                        group_seg_layouts[g_idx].append(
                            (layer_addr + offset, block_stride, content)
                        )
                else:
                    # K/V-first layout (FlashAttn / ROCm): split segments.
                    seg_stride = cache.stride(outer_dims[0]) * el
                    for idx in range(cache.shape[outer_dims[0]]):
                        group_addrs[g_idx].append(layer_addr + idx * seg_stride)
                        group_block_lens[g_idx].append(seg_stride // self.num_blocks)
                        # Packed triple: per-block dense content equals the
                        # legacy block_len for K/V-first layouts.
                        packed = seg_stride // self.num_blocks
                        group_seg_layouts[g_idx].append(
                            (layer_addr + idx * seg_stride, packed, packed)
                        )

        logger.info(
            "Registered KV caches: num_groups=%d, segments_per_group=%s, num_blocks=%d",
            len(self.token_dbs),
            [len(a) for a in group_seg_layouts],
            self.num_blocks,
        )

        for g_idx, db in enumerate(self.token_dbs):
            db.set_kv_caches_base_addr(group_addrs[g_idx])
            db.set_block_len(group_block_lens[g_idx])
            # Exact layout wins over the legacy flat tables when present.
            seen: set[tuple[int, int, int]] = set()
            seg_layout = [
                e for e in group_seg_layouts[g_idx] if not (e in seen or seen.add(e))
            ]
            if seg_layout:
                db.set_seg_layout(seg_layout)
            if g_idx in multi_tensor_groups:
                # Objects saved before every state tensor was included hold
                # the first one only, under the same chunk hashes and of a
                # shorter length. A load would reject them on length, but a
                # SAVE finds them present and never replaces them: keep them
                # out of reach under a pool of their own.
                db.metadata = dataclasses.replace(
                    db.metadata, pool_name=f"{db.metadata.pool_name}-states"
                )
        if self.client is not None:
            self._validate_cache_block_sizes(self.client)
        # Start transfer threads
        if self.kv_role in ["kv_producer", "kv_both"]:
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.client,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                self.stripe_idx,
                self.stripe_step,
                self.kv_role,
                ready_event_sending,
                self.enable_kv_events,
                record_operation=self._record_kv_connector_operation,
                queue_capacity=getattr(
                    self, "transfer_queue_capacity", DEFAULT_TRANSFER_QUEUE_CAPACITY
                ),
                tp_sharded_groups=self._tp_sharded_groups,
                segment_affinity_pages=getattr(
                    self, "segment_affinity_pages", 0
                ),
            )
            if self.client_ranks < self.tp_size:
                # Converged mode: re-stripe stores over the participant set.
                self.kv_send_thread.stripe_idx = participant(
                    self.tp_rank, self.tp_size, self.client_ranks
                )
                self.kv_send_thread.stripe_step = self.client_ranks
            self.kv_send_thread.start()

        ready_event_recving = threading.Event()
        self.kv_recv_thread = KVCacheStoreRecvingThread(
            self.client,
            self.coord,
            self.token_dbs,
            self.block_size,
            self.tp_rank,
            ready_event_recving,
            record_operation=self._record_kv_connector_operation,
            record_observation=self._record_kv_connector_observation,
            client_provider=self._ensure_client_for_load,
            queue_capacity=getattr(
                self, "transfer_queue_capacity", DEFAULT_TRANSFER_QUEUE_CAPACITY
            ),
            recv_workers=getattr(self, "recv_workers", DEFAULT_RECV_WORKERS),
            record_pool_sample=self._record_kv_connector_pool_sample,
            load_window_keys=getattr(
                self, "load_window_keys", DEFAULT_LOAD_WINDOW_KEYS
            ),
            load_window_min_keys=getattr(
                self,
                "load_window_min_keys",
                DEFAULT_LOAD_WINDOW_MIN_KEYS,
            ),
            request_level_loads=self.request_level_loads,
            segment_affinity_pages=getattr(
                self, "segment_affinity_pages", 0
            ),
        )
        self.kv_recv_thread.start()
        ready_event_recving.wait()

    def handle_preemptions(
        self,
        metadata: DfkvStoreConnectorMetadata,
    ) -> None:
        """Fence preempted block sources before the next forward pass."""
        if not metadata.preempted_req_ids:
            return
        if self.kv_recv_thread is not None:
            self.kv_recv_thread.cancel_requests(
                metadata.preempted_req_ids,
                wait=True,
                fail_closed=False,
            )
        send_thread = self.kv_send_thread
        if send_thread is None:
            return
        # Drop queued entries first, then join whichever entry is active.
        for req_id in metadata.preempted_req_ids:
            send_thread.delete_finished_stored_request(req_id)
        for req_id in metadata.preempted_req_ids:
            wait_start = time.perf_counter()
            while not send_thread.wait_for_inflight_put(req_id):
                logger.error(
                    "preemption fence still waiting for in-flight save "
                    "of request %s after %.1fs; GPU block reuse remains "
                    "fenced until the native save exits",
                    req_id,
                    time.perf_counter() - wait_start,
                )

    def start_load_kv(
        self,
        metadata: DfkvStoreConnectorMetadata,
    ):
        """Serialize native loads on the model thread, never a pool thread."""
        if self.load_async or getattr(self, "_last_load_metadata", None) is metadata:
            return
        self._last_load_metadata = metadata
        assert self.kv_recv_thread is not None
        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:
                continue
            load_spec.token_len = load_spec.kvpool_cached_tokens
            self.kv_recv_thread.load_request_sync(request)


    def get_transfer_results(
        self,
        finished_req_ids: set[str],
        meta: DfkvStoreConnectorMetadata,
    ) -> TransferResults:
        """Submit post-forward I/O and atomically collect receive outcomes.

        Mutable and windowed stores finish before the next model step can
        overwrite their sources; full-attention stores retain async overlap.
        """
        # Aborted/finished requests may have an outstanding remote load.  Their
        # blocks cannot be freed until that request's own GPUDirect write exits.
        if finished_req_ids and self.kv_recv_thread is not None:
            self.kv_recv_thread.cancel_requests(
                finished_req_ids,
                wait=True,
                fail_closed=False,
            )
        if getattr(self, "_last_transfer_metadata", None) is not meta:
            self._last_transfer_metadata = meta
            self._submit_transfers(meta)

        done_sending = (
            self._get_and_clear_finished_sending(finished_req_ids)
            if self.kv_role in ["kv_producer", "kv_both"]
            else set()
        )
        done_recving, failed_recving = set(), set()
        if self.kv_recv_thread is not None and (
            self.load_async or self.request_level_loads
        ):
            done_recving, failed_recving = (
                self.kv_recv_thread.get_and_clear_receive_results()
            )
        return TransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
        )

    def _submit_transfers(self, meta: DfkvStoreConnectorMetadata) -> None:
        # Async mode overlaps loads with unrelated model work. Synchronous mode
        # executes them inline in start_load_kv, never in the receive pool.
        if self.load_async:
            for request in meta.requests:
                load_spec = request.load_spec
                if load_spec is None or not load_spec.can_load:
                    continue
                load_spec.token_len = load_spec.kvpool_cached_tokens
                assert self.kv_recv_thread is not None
                self.kv_recv_thread.add_request(request)

        # Issue stores with CUDA event synchronization
        if self.kv_role in ["kv_producer", "kv_both"]:
            current_event = None
            for request in meta.requests:
                if request.can_save:
                    current_event = torch.cuda.Event()
                    current_event.record()
                    break

            send_thread = self.kv_send_thread
            fenced = self._fenced_store_groups if current_event is not None else ()
            for request in meta.requests:
                if not request.can_save:
                    continue
                request.current_event = current_event
                assert send_thread is not None
                if not fenced:
                    send_thread.add_stored_request(request)
                    send_thread.add_request(request)
                    continue
                # A following step may update recurrent state or free/reuse
                # a window block even while this request is still running:
                # those PUTs finish before this step returns. Attention blocks
                # stay with the request until it is reported finished sending,
                # so their PUT overlaps the following steps. Each part counts
                # as one pending save of the request.
                part = dataclasses.replace(request, save_groups=fenced)
                send_thread.add_stored_request(part)
                send_thread.add_fenced_request(part)
                unfenced = frozenset(
                    g for g, db in enumerate(self.token_dbs) if db.cacheable
                ) - fenced
                if unfenced:
                    rest = dataclasses.replace(request, save_groups=unfenced)
                    send_thread.add_stored_request(rest)
                    send_thread.add_request(rest)

            if fenced:
                send_thread.fence_queue.join()


    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.kv_recv_thread is None:
            return set()
        errs = self.kv_recv_thread.get_and_clear_block_ids_with_load_errors()
        if errs:
            logger.warning("dfkv load_errors: %d blocks flagged tp=%d", len(errs), self.tp_rank)
        return errs

    def _record_kv_connector_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_logical_keys: int | None = None,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        with self._kv_connector_stats_lock:
            self.kv_connector_stats.record_operation(
                operation=operation,
                duration_seconds=duration_seconds,
                num_keys=num_keys,
                num_logical_keys=num_logical_keys,
                num_bytes=num_bytes,
                status=status,
                num_failed_keys=num_failed_keys,
            )

    def _record_kv_connector_observation(
        self,
        name: str,
        duration_seconds: float,
    ) -> None:
        with self._kv_connector_stats_lock:
            self.kv_connector_stats.record_observation(name, duration_seconds)

    def _record_kv_connector_pool_sample(
        self,
        name: str,
        value: int,
    ) -> None:
        with self._kv_connector_stats_lock:
            self.kv_connector_stats.record_pool_sample(name, value)

    def get_kv_connector_stats(self) -> DfkvStoreConnectorStats | None:
        with self._kv_connector_stats_lock:
            if self.kv_connector_stats.is_empty():
                return None
            kv_connector_stats = self.kv_connector_stats
            self.kv_connector_stats = DfkvStoreConnectorStats()
            return kv_connector_stats

    def _get_and_clear_finished_sending(
        self,
        finished_req_ids: set[str],
    ) -> set[str]:
        assert self.kv_send_thread is not None
        finished_sending: set[str] = set()

        for req_id in self.kv_send_thread.stored_requests.copy():
            if (
                self.kv_send_thread.stored_requests[req_id] == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(req_id)
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)

        return finished_sending

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        retain_ttl_s: int = 0,
    ) -> tuple[int, int]:
        """Return ``(hit_tokens, applied_ttl_s)``.

        ``hit_tokens`` is a prefix whose length-dependent load objects all
        exist. The probe is a retention REFRESH, so every object found (also
        ones served from GPU memory on this instance) has its protection
        renewed. ``applied_ttl_s`` is the TTL this request's writes will be
        protected with: 0 without a TTL, on a ring without retention, or when
        the whole-prefix precheck finds an owner node short of headroom.
        """
        first_candidate_keys: list[bytes] | None = None
        first_candidate_routes: list[bytes] | None = None
        affinity_pages = getattr(self, "segment_affinity_pages", 0)

        def probe(
            candidate_keys: list[bytes], num_logical_keys: int,
            route_keys: list[bytes] | None = None,
        ) -> list[int] | None:
            """Per-key presence (1/0) under a retention REFRESH; None on error."""
            lookup_start = time.perf_counter()
            try:
                if route_keys is None:
                    codes = self.client.batch_retain(
                        candidate_keys, 0, dfkv_cabi.DFKV_RETAIN_REFRESH)
                else:
                    codes = self.client.batch_retain(
                        candidate_keys, 0, dfkv_cabi.DFKV_RETAIN_REFRESH,
                        route_keys=route_keys)
                res = [1 if c in dfkv_cabi.RETAIN_HIT_CODES else 0
                       for c in codes]
                if len(res) != len(candidate_keys):
                    raise RuntimeError(
                        "batch_exist returned incomplete per-key results: "
                        f"keys={len(candidate_keys)} statuses={len(res)}"
                    )
            except Exception as e:
                self._record_kv_connector_operation(
                    "lookup_exists",
                    time.perf_counter() - lookup_start,
                    len(candidate_keys),
                    num_logical_keys=num_logical_keys,
                    status="error",
                    num_failed_keys=len(candidate_keys),
                )
                logger.error("Remote connection failed in lookup: %s", e)
                return None
    
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
                num_logical_keys=num_logical_keys,
            )
            return res

        # Presence of each physical object, by (group, chunk, tp, pcp, dcp),
        # as far as this lookup has probed it: a shorter candidate re-uses
        # what longer ones found instead of probing (and building the keys of)
        # the whole prefix again.
        presence: dict[tuple[int, int, int, int, int], int] = {}
        object_anchors: dict[tuple[int, int, int, int, int], bytes] = {}

        def probe_into(objects: dict, num_logical_keys: int) -> bool:
            """Probe ``objects`` (object -> key) and record their presence."""
            routes = (
                [object_anchors[physical] for physical in objects]
                if affinity_pages else None
            )
            res = probe(list(objects.values()), num_logical_keys, routes)
            if res is None:
                return False
            presence.update(zip(objects, res, strict=True))
            return True

        def group_namespaces(g_idx: int):
            # State pools need every physical TP shard; payload pools need
            # every head/PCP/DCP coordinate (MLA keeps its canonical -1).
            db = self.token_dbs[g_idx]
            return _key_namespaces(
                tp_size=self.tp_size,
                pcp_size=db.metadata.pcp_size,
                dcp_size=db.metadata.dcp_size,
                num_kv_heads=self.num_kv_head,
                use_mla=db.metadata.tp_rank < 0,
                tp_sharded=is_mamba_group(
                getattr(self, "_kv_cache_group_views", self._kv_cache_groups)[g_idx]
            ),
                dcp_group_ranks=getattr(self, "_dcp_group_ranks", None),
                dcp_global_rank_base=getattr(self, "_dcp_global_rank_base", 0),
            )

        # A key is its namespace's prefix, the hash field and the suffix;
        # each namespace of a group is encoded once per lookup.
        affixes: dict[tuple[int, int, int, int], tuple[bytes, bytes]] = {}

        def object_key(g_idx: int, h, tp: int, pcp: int, dcp: int) -> bytes:
            # Keep this worker's own pp_rank (db.metadata.pp_rank): PP
            # partitions layers, so each (group, chunk) lives under a single
            # stage's pp_rank, matching the SAVE/LOAD keys. Probing every
            # pp_rank matched nothing and forced present=0 on PP>1.
            namespace = (g_idx, tp, pcp, dcp)
            parts = affixes.get(namespace)
            if parts is None:
                parts = affixes[namespace] = key_affixes(dataclasses.replace(
                    self.token_dbs[g_idx].metadata,
                    tp_rank=tp, pcp_rank=pcp, dcp_rank=dcp,
                ))
            return parts[0] + pool_key_hash_field(h.hex()) + parts[1]

        def object_anchor_key(
            g_idx: int, chunk_id: int, current_key: bytes,
            tp: int, pcp: int, dcp: int,
        ) -> bytes:
            first = chunk_id // affinity_pages * affinity_pages
            if first == chunk_id:
                # This also preserves a boundary/partial object's actual key.
                return current_key
            first_hash = self.token_dbs[g_idx].chunk_hash(
                first, block_hashes
            )
            return object_key(g_idx, first_hash, tp, pcp, dcp)

        def prefetch_moving_groups(candidate: int, hit: int) -> bool:
            """After a failed candidate: probe, in one call, every chunk below
            ``hit`` of the groups whose load mask moves with the candidate
            (a recurrent state's tail, a window), so the shorter candidates
            the loop tries next are answered locally. A group whose mask at
            ``hit`` is the prefix of its mask at ``candidate`` (full
            attention) needs nothing new: every one of its chunks below
            ``hit`` was probed already. One linear pass over those groups'
            chunks; False if the probe failed."""
            long_masks = self.coord.load_mask(block_hashes, candidate)
            short_masks = self.coord.load_mask(block_hashes, hit)
            pending: dict[tuple[int, int, int, int, int], bytes] = {}
            for g_idx, db in enumerate(self.token_dbs):
                short = list(short_masks[g_idx])
                if short == list(long_masks[g_idx])[: len(short)]:
                    continue
                group_hashes = self.coord.block_hashes_for_group(
                    block_hashes, self._kv_cache_groups[g_idx]
                )
                namespaces = group_namespaces(g_idx)
                for chunk_id in range(min(hit // db.block_size, len(group_hashes))):
                    h = group_hashes[chunk_id]
                    for tp, pcp, dcp in namespaces:
                        physical = (g_idx, chunk_id, tp, pcp, dcp)
                        if physical not in presence and physical not in pending:
                            key = object_key(g_idx, h, tp, pcp, dcp)
                            pending[physical] = key
                            if affinity_pages:
                                object_anchors[physical] = object_anchor_key(
                                    g_idx, chunk_id, key, tp, pcp, dcp
                                )
            return not pending or probe_into(pending, len(pending))

        def lookup_candidate(token_len: int) -> int:
            nonlocal first_candidate_keys, first_candidate_routes, eagle_admission_pending
            """Check how many prefix tokens exist in the store.

            Checks across all TP ranks at this worker's own PP rank.
            """
            if not block_hashes or token_len <= 0:
                return 0
    
            # Build every physical object required by each logical LCM chunk.
            # Object metadata retains the logical chunk index so repeated hashes
            # cannot collapse accounting across distinct prefix positions.
            candidate_objects: list[tuple[int, int, int, int, int]] = []
            unknown: dict[tuple[int, int, int, int, int], bytes] = {}
            candidate_meta: list[tuple[int, int, int, bytes]] = []
            expected_per_object: dict[tuple[int, int, int, bytes], int] = {}
            # SAVE omits volatile draft tails. Admission must instead prove
            # every object LOAD needs at this exact candidate boundary.
            aligned_token_len = (
                token_len // self.coord.lcm_block_size * self.coord.lcm_block_size
            )
            load_masks = self.coord.load_mask(block_hashes, aligned_token_len)
            if aligned_token_len == 0:
                return 0
            num_prefix_chunks = aligned_token_len // self.coord.lcm_block_size
            missing_required_per_chunk = [False] * num_prefix_chunks
            for g_idx, db in enumerate(self.token_dbs):
                spec_block_size = db.block_size
                mask = load_masks[g_idx]
                if not mask:
                    continue
                group_hashes = self.coord.block_hashes_for_group(
                    block_hashes, self._kv_cache_groups[g_idx]
                )
                # Key metadata carries this group's CP geometry; a negative
                # tp_rank is exactly the canonical replicated-MLA coordinate.
                namespaces = group_namespaces(g_idx)
                rank_probes = len(namespaces)
                processed_chunks = 0
                for chunk_id, h in enumerate(group_hashes):
                    start_idx = chunk_id * spec_block_size
                    if start_idx >= aligned_token_len:
                        break
                    processed_chunks = chunk_id + 1
                    if chunk_id >= len(mask) or not mask[chunk_id]:
                        continue
                    logical_chunk_idx = start_idx // self.coord.lcm_block_size
                    object_meta = (logical_chunk_idx, g_idx, chunk_id, bytes(h))
                    expected_per_object[object_meta] = rank_probes
                    for tp, pcp, dcp in namespaces:
                        physical = (g_idx, chunk_id, tp, pcp, dcp)
                        candidate_objects.append(physical)
                        candidate_meta.append(object_meta)
                        if physical not in presence and physical not in unknown:
                            key = object_key(g_idx, h, tp, pcp, dcp)
                            unknown[physical] = key
                            if affinity_pages:
                                object_anchors[physical] = object_anchor_key(
                                    g_idx, chunk_id, key, tp, pcp, dcp
                                )
                # A truncated hash vector is malformed metadata. Mark only its
                # ungenerated required suffix chunks incomplete; the normal path
                # has no second full mask scan.
                for chunk_id in range(
                    processed_chunks,
                    min(len(mask), aligned_token_len // spec_block_size),
                ):
                    if mask[chunk_id]:
                        logical_chunk_idx = (
                            chunk_id * spec_block_size // self.coord.lcm_block_size
                        )
                        missing_required_per_chunk[logical_chunk_idx] = True
    
            if not candidate_objects:
                return 0
            if first_candidate_keys is None:
                first_candidate_keys = list(unknown.values())
                if affinity_pages:
                    first_candidate_routes = [
                        object_anchors[physical] for physical in unknown
                    ]
            if unknown and not probe_into(unknown, len(expected_per_object)):
                return 0
            res = [presence[physical] for physical in candidate_objects]
    
            # A semantic object exists only if every TP*PP coordinate exists. A
            # logical LCM chunk is complete only if all of its semantic objects do.
            # Scan logical chunks in order and cap the semantic coordinator at the
            # first incomplete one, so an isolated later hit can never be loaded.
            present_per_object: dict[tuple[int, int, int, bytes], int] = {}
            for object_meta, exists in zip(candidate_meta, res, strict=True):
                if exists == 1:
                    present_per_object[object_meta] = (
                        present_per_object.get(object_meta, 0) + 1
                    )
    
            complete_chunks = [
                not missing_required_per_chunk[idx]
                for idx in range(num_prefix_chunks)
            ]
            for object_meta, expected in expected_per_object.items():
                logical_chunk_idx = object_meta[0]
                if present_per_object.get(object_meta, 0) != expected:
                    complete_chunks[logical_chunk_idx] = False
    
            prefix_chunks = 0
            for complete in complete_chunks:
                if not complete:
                    break
                prefix_chunks += 1
            complete_prefix_tokens = prefix_chunks * self.coord.lcm_block_size
            if (
                eagle_admission_pending
                and self.coord._manager_api
                and self.coord.eagle_block_drop
                and complete_prefix_tokens > 0
                and len(self.coord.attention_groups) == 1
                and is_full_attention(self.coord.attention_groups[0].view)
            ):
                # Admission must re-evaluate the tail under the real manager
                # contract. Do not change the subsequent load/prefetch mask:
                # once h1 is excluded here, h0 remains the stable prefix.
                admission_exists = {
                    (object_meta[1], object_meta[3])
                    for object_meta, expected in expected_per_object.items()
                    if present_per_object.get(object_meta, 0) == expected
                }
                admission_pool = ExternalCachedBlockPool(
                    self.coord.hash_block_size, admission_exists
                )
                complete_prefix_tokens = self.coord.find_longest_cache_hit(
                    block_hashes, complete_prefix_tokens, admission_pool
                )
                eagle_admission_pending = False

            # This is only a candidate boundary. The outer loop checks the
            # mask at that exact length before admitting it for LOAD.
            hit_length = complete_prefix_tokens
            logger.debug(
                "dfkv lookup: token_len=%d candidates=%d complete_chunks=%d/%d "
                "-> hit_length=%d",
                token_len,
                len(candidate_objects),
                prefix_chunks,
                num_prefix_chunks,
                hit_length,
            )
            return hit_length
    
        def lookup_partial(token_len: int) -> int:
            """Hash-block-granular hit, converged as vLLM core over the
            objects present under every rank namespace of their group.

            A hit inside an attention block needs that block stored under
            the hit's own hash, so only these are probed: each attention
            group's whole blocks and the interior hash boundaries of its
            first block not stored whole, and each recurrent state at the
            scheduler block boundaries below that and at those interior
            boundaries. A second probe follows only when a whole block is
            missing below ``token_len``.
            """
            coord = self.coord
            hbs = coord.hash_block_size
            lcm = coord.lcm_block_size
            token_len = min(token_len, len(block_hashes) * hbs) // hbs * hbs
            if token_len <= 0:
                return 0
            cacheable = sorted(
                g for entry in coord.attention_groups for g in entry.group_ids
            )
            state_groups = [
                g for g in cacheable
                if is_mamba_group(
                    getattr(self, "_kv_cache_group_views", self._kv_cache_groups)[g]
                )
            ]
            attention_groups = [g for g in cacheable if g not in state_groups]
            block_sizes = {g: self.token_dbs[g].block_size for g in cacheable}
            views = {
                g: coord.block_hashes_for_group(
                    block_hashes, self._kv_cache_groups[g]
                )
                for g in attention_groups
            }
            exists: set[tuple[int, bytes]] = set()
            probed: set[tuple[int, int, BlockHash]] = set()

            def probe_objects(
                objects: list[tuple[int, int, BlockHash]],
            ) -> bool:
                nonlocal first_candidate_keys, first_candidate_routes
                objects = [o for o in dict.fromkeys(objects) if o not in probed]
                probed.update(objects)
                keys: list[bytes] = []
                routes: list[bytes] | None = [] if affinity_pages else None
                owners: list[tuple[int, BlockHash]] = []
                for g_idx, chunk_id, h in objects:
                    db = self.token_dbs[g_idx]
                    for tp, pcp, dcp in _key_namespaces(
                        tp_size=self.tp_size,
                        pcp_size=db.metadata.pcp_size,
                        dcp_size=db.metadata.dcp_size,
                        num_kv_heads=self.num_kv_head,
                        use_mla=db.metadata.tp_rank < 0,
                        tp_sharded=g_idx in state_groups,
                        dcp_group_ranks=getattr(self, "_dcp_group_ranks", None),
                        dcp_global_rank_base=getattr(self, "_dcp_global_rank_base", 0),
                    ):
                        key = object_key(g_idx, h, tp, pcp, dcp)
                        keys.append(key)
                        if routes is not None:
                            routes.append(object_anchor_key(
                                g_idx, chunk_id, key, tp, pcp, dcp
                            ))
                        owners.append((g_idx, h))
                if not keys:
                    return True
                if first_candidate_keys is None:
                    first_candidate_keys = keys
                    first_candidate_routes = (
                        None if routes is None else list(routes)
                    )
                res = probe(keys, len(objects), routes)
                if res is None:
                    return False
                missing = {o for o, present in zip(owners, res, strict=True)
                           if present != 1}
                exists.update(
                    (g_idx, bytes(h)) for g_idx, _chunk_id, h in objects
                    if (g_idx, h) not in missing
                )
                return True

            def candidates(full_blocks: dict[int, int], *, with_full: bool):
                objects: list[tuple[int, int, BlockHash]] = []
                boundaries: set[int] = set()
                for g in attention_groups:
                    if with_full:
                        objects.extend(
                            (g, i, views[g][i]) for i in range(full_blocks[g])
                        )
                    start = full_blocks[g] * block_sizes[g]
                    end = min(start + block_sizes[g], token_len + 1)
                    for c in range(start + hbs, end, hbs):
                        objects.append((
                            g, (c - 1) // block_sizes[g],
                            block_hashes[c // hbs - 1],
                        ))
                        boundaries.add(c)
                if with_full:
                    whole = min(
                        (full_blocks[g] * block_sizes[g] for g in attention_groups),
                        default=token_len,
                    )
                    boundaries.update(range(lcm, whole // lcm * lcm + 1, lcm))
                objects.extend(
                    (
                        g, (c - 1) // block_sizes[g],
                        block_hashes[c // hbs - 1],
                    )
                    for g in state_groups for c in sorted(boundaries)
                )
                return objects

            assumed = {g: token_len // block_sizes[g] for g in attention_groups}
            if not probe_objects(candidates(assumed, with_full=True)):
                return 0
            leading: dict[int, int] = {}
            for g in attention_groups:
                n = 0
                while n < assumed[g] and (g, bytes(views[g][n])) in exists:
                    n += 1
                leading[g] = n
            if leading != assumed and not probe_objects(
                candidates(leading, with_full=False)
            ):
                return 0

            pool = ExternalCachedBlockPool(hbs, exists)
            max_length = token_len
            while max_length > 0:
                hit = coord.find_longest_cache_hit(block_hashes, max_length, pool)
                # LOAD reads a hit inside an attention block from the object
                # stored under the hit's own hash; core may pair a state with
                # a longer tail in that block.
                if all(
                    hit % block_sizes[g] == 0
                    or (g, bytes(block_hashes[hit // hbs - 1])) in exists
                    for g in attention_groups
                ):
                    logger.debug(
                        "dfkv partial lookup: token_len=%d objects=%d -> "
                        "hit_length=%d", token_len, len(exists), hit,
                    )
                    return hit
                max_length = hit - hbs
            return 0

        def applied_ttl() -> int:
            if retain_ttl_s <= 0 or not first_candidate_keys:
                return 0
            try:
                supported, max_ttl = self.client.retain_limits()
                if supported != 1 or max_ttl <= 0:
                    return 0
                # Decision 9: one owner short of headroom downgrades the whole
                # request to best-effort rather than protecting a holed prefix.
                if first_candidate_routes is None:
                    precheck_ok = self.client.retain_precheck(
                        first_candidate_keys
                    )
                else:
                    precheck_ok = self.client.retain_precheck(
                        first_candidate_keys,
                        route_keys=first_candidate_routes,
                    )
                if not precheck_ok:
                    return 0
                return min(retain_ttl_s, max_ttl)
            except Exception as e:  # never fail a lookup over retention
                logger.warning("dfkv retention limits/precheck failed: %s", e)
                return 0

        if self.coord.enable_partial_hash_hits:
            return lookup_partial(token_len), applied_ttl()
        lcm = self.coord.lcm_block_size
        candidate = token_len // lcm * lcm
        eagle_admission_pending = True
        prefetched = False
        while candidate > 0:
            hit = lookup_candidate(candidate)
            if hit == candidate:
                return hit, applied_ttl()
            if not prefetched and hit > 0:
                prefetched = True
                if not prefetch_moving_groups(candidate, hit):
                    return 0, applied_ttl()
            # Tail-only state masks move when the candidate prefix shrinks.
            # Revalidate that boundary rather than loading unprobed checkpoints.
            candidate = hit
        return 0, applied_ttl()

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            return self.kv_send_thread.get_kv_events()
        return []

    def close(self) -> None:
        """Cancel queued transfers, join workers, and close native state once.

        Shutdown is fail-closed for new submissions. Queued loads are reported
        as load errors/completed so vLLM recomputes them; queued saves decrement
        their per-request counters so delayed block frees cannot remain pinned.
        An operation already inside the native client is allowed to finish
        before the client handle is closed.
        """
        with self._close_lock:
            close_done = getattr(self, "_close_done", None)
            if close_done is None:
                close_done = self._close_done = threading.Event()
            if self._closed:
                wait_for_close = True
            else:
                self._closed = True
                wait_for_close = False
        if wait_for_close:
            close_done.wait()
            return

        errors: list[Exception] = []
        try:
            recv_thread = self.kv_recv_thread
            if recv_thread is not None:
                try:
                    recv_thread.stop(cancel_pending=True)
                except Exception as e:
                    errors.append(e)
                finally:
                    self.kv_recv_thread = None

            send_thread = self.kv_send_thread
            if send_thread is not None:
                try:
                    send_thread.stop(cancel_pending=True)
                except Exception as e:
                    errors.append(e)
                finally:
                    self.kv_send_thread = None

            lookup_server = self.lookup_server
            if lookup_server is not None:
                try:
                    lookup_server.close()
                except Exception as e:
                    errors.append(e)
                finally:
                    self.lookup_server = None

            # Serialize against lazy un-elision. Clear the reference before
            # close so repeated cleanup or DfkvDeviceClient.__del__ cannot
            # close the native handle twice.
            with self._lazy_client_lock:
                self._lazy_client_kwargs = None
                client = self.client
                self.client = None
            if client is not None:
                try:
                    client.close()
                except Exception as e:
                    errors.append(e)
        finally:
            close_done.set()

        if errors:
            raise RuntimeError(
                f"dfkv worker shutdown had {len(errors)} cleanup error(s)"
            ) from errors[0]


# ============================================================
# Lookup Key Server
# ============================================================


class LookupKeyServer:
    """ZMQ server on worker rank 0 for the LookupKey admin channel.

    Handles two request types, tagged at frame 0:
    - ``LOOKUP_MSG``: prefix-cache hit query, returns hit count.
    - ``RESET_MSG``: drains the send thread queue, then attempts a global
      store wipe. Caller must have paused the scheduler first.
    """

    def __init__(
        self,
        store_worker: DfkvStoreWorker,
        vllm_config: VllmConfig,
    ):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self._ipc_path = socket_path.removeprefix("ipc://")
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # type: ignore[attr-defined]
        self._close_lock = threading.Lock()
        self._closed = False

        self.store_worker = store_worker
        self.running = True

        def process_request():
            while self.running:
                try:
                    all_frames = self.socket.recv_multipart(copy=False)
                except zmq.Again:  # type: ignore[attr-defined]
                    continue
                except zmq.ZMQError:  # type: ignore[attr-defined]
                    if not self.running:
                        return
                    raise

                if not all_frames:
                    logger.warning("LookupKeyServer received an empty request")
                    self.socket.send(RESP_ERR)
                    continue

                msg_type = bytes(all_frames[0])
                if msg_type == LOOKUP_MSG:
                    # A malformed request is an external-cache miss, never a
                    # reason to terminate the rank-0 admin server.
                    lookup_ipc_start = time.perf_counter()
                    try:
                        token_len, hash_bytes, retain_ttl_s = (
                            _decode_lookup_request(
                                [frame.buffer for frame in all_frames]
                            )
                        )
                        block_hashes = [BlockHash(value) for value in hash_bytes]
                        result, applied = self.store_worker.lookup(
                            token_len, block_hashes, retain_ttl_s)
                        if not 0 <= result < 1 << 32:
                            raise ValueError(
                                f"lookup result is outside uint32: {result}"
                            )
                    except Exception as e:
                        logger.warning(
                            "LookupKeyServer rejected lookup request: %s", e
                        )
                        result, applied = 0, 0
                    self.socket.send(_encode_lookup_response(result, applied))
                    self.store_worker._record_kv_connector_observation(
                        "lookup_ipc", time.perf_counter() - lookup_ipc_start
                    )

                elif msg_type == RESET_MSG:
                    if len(all_frames) != 1:
                        logger.warning(
                            "LookupKeyServer rejected reset with %d frames",
                            len(all_frames),
                        )
                        self.socket.send(RESP_ERR)
                        continue
                    # dfkv: DfkvDeviceClient exposes no remove_all / global wipe
                    # primitive. Entries expire by the server's own policy. We
                    # still drain in-flight puts to honor the ordering contract,
                    # then NACK so callers know the hard reset was not applied.
                    try:
                        send_thread = self.store_worker.kv_send_thread
                        if send_thread is not None:
                            send_thread.fence_queue.join()
                            send_thread.request_queue.join()
                    except Exception as e:
                        logger.error("Dfkv reset drain failed: %s", e)
                    logger.warning(
                        "Dfkv store has no remove_all; reset request NACKed "
                        "(send queue drained)."
                    )
                    self.socket.send(RESP_ERR)

                else:
                    logger.warning(
                        "LookupKeyServer received unknown msg_type: %r",
                        msg_type,
                    )
                    self.socket.send(RESP_ERR)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.running = False
        self.thread.join()
        self.socket.close(linger=0)
        self.ctx.term()
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)


# ============================================================
# Lookup Key Client
# ============================================================


class LookupKeyClient:
    """ZMQ client for the LookupKey admin channel.

    The single-worker executor is also the sole owner of socket I/O. Async
    scheduler calls retain one future per request and poll it on later steps.

    vLLM re-polls ``get_num_new_matched_tokens`` for every waiting request on
    every scheduler step. A completed lookup is therefore reused for the same
    request and prefix length for ``lookup_cache_ttl_ms``; each reuse skips the
    rank-0 round trip and the whole-prefix retention REFRESH it carries. The
    entry is dropped by ``discard`` (finish, abort, preemption, failed load),
    and a changed prefix length or retention TTL is a different lookup.
    """

    def __init__(self, vllm_config: VllmConfig):
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._cache_ttl_s = _parse_lookup_cache_ttl_ms(
            extra_config.get("lookup_cache_ttl_ms", DEFAULT_LOOKUP_CACHE_TTL_MS)
        ) / 1000.0
        self._clock: Callable[[], float] = time.monotonic
        # req_id -> ((token_len, retain_ttl_s), result, expires_at)
        self._results: dict[
            str, tuple[tuple[int, int], tuple[int, int], float]
        ] = {}
        self._timeout_ms = _parse_lookup_timeout_ms(
            extra_config.get("lookup_timeout_ms", DEFAULT_LOOKUP_TIMEOUT_MS)
        )
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        self._close_lock = threading.Lock()
        self._closed = False
        # After an unanswered lookup, rank 0 is presumed unavailable until
        # this clock value: lookups are misses at once instead of each
        # waiting a full timeout in turn on the single socket thread.
        self._unavailable_until = 0.0
        self._socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = self._connect()
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="DfkvLookupClient",
        )
        self.futures: dict[str, Future[tuple[int, int]]] = {}

    def _connect(self):
        socket = make_zmq_socket(
            self.ctx,
            self._socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )
        if self._timeout_ms:
            socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)  # type: ignore[attr-defined]
            socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)  # type: ignore[attr-defined]
        return socket

    def _call(self, send: Callable[[Any], None]) -> bytes:
        """One request/reply on the socket-owner thread. A request rank 0
        does not answer within lookup_timeout_ms raises TimeoutError; the
        REQ socket is then stuck waiting for that reply, so it is replaced
        (a late reply goes to the closed one and is dropped)."""
        try:
            send(self.socket)
            return bytes(self.socket.recv())
        except zmq.Again as e:  # type: ignore[attr-defined]
            self.socket.close(linger=0)
            self.socket = self._connect()
            self._unavailable_until = self._clock() + self._timeout_ms / 1000.0
            raise TimeoutError(
                f"rank-0 lookup server did not answer in {self._timeout_ms} ms"
            ) from e

    def _unavailable(self) -> bool:
        return self._clock() < self._unavailable_until

    def _lookup(self, token_len: int, block_hashes: list[BlockHash],
                retain_ttl_s: int) -> tuple[int, int] | None:
        if self._unavailable():
            # Queued behind the lookup that timed out: a miss, not another
            # full timeout. None tells lookup() not to log it as a failure.
            return None
        all_frames = _encode_lookup_request(token_len, block_hashes,
                                            retain_ttl_s)
        return _decode_lookup_response(
            self._call(lambda socket: socket.send_multipart(all_frames, copy=False))
        )

    def lookup(
        self,
        req_id: str,
        token_len: int,
        block_hashes: list[BlockHash],
        non_block: bool = False,
        retain_ttl_s: int = 0,
    ) -> tuple[int, int] | None:
        """Return ``(hit_tokens, applied_ttl_s)``, or ``None`` while a
        non-blocking lookup runs."""
        signature = (token_len, retain_ttl_s)
        with self._close_lock:
            if self._closed:
                return 0, 0
            cached = self._results.get(req_id)
            if cached is not None:
                if cached[0] == signature and self._clock() < cached[2]:
                    return cached[1]
                del self._results[req_id]
            future = self.futures.get(req_id)
            if future is None and self._unavailable():
                # Rank 0 left a lookup unanswered within the last timeout:
                # an uncached miss until then; the first lookup after it
                # probes rank 0 again.
                return 0, 0
            if future is None:
                future = self.executor.submit(
                    self._lookup,
                    token_len,
                    list(block_hashes),
                    retain_ttl_s,
                )
                self.futures[req_id] = future

        if non_block and not future.done():
            return None
        try:
            result = future.result()
        except Exception as e:
            logger.error("Dfkv lookup failed for %s: %s", req_id, e)
            return 0, 0
        finally:
            with self._close_lock:
                if self.futures.get(req_id) is future:
                    self.futures.pop(req_id)
        if result is None:
            return 0, 0
        # Only an answered lookup is reusable: an RPC error stays a one-step
        # miss so the next step probes again.
        if self._cache_ttl_s > 0:
            with self._close_lock:
                if not self._closed:
                    self._results[req_id] = (
                        signature, result, self._clock() + self._cache_ttl_s
                    )
        return result

    def discard(self, req_id: str) -> None:
        """Drop any cached or queued lookup for an aborted request."""
        with self._close_lock:
            self._results.pop(req_id, None)
            future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()

    def _reset(self) -> bool:
        """Send the reset admin request from the socket-owner thread."""
        return self._call(lambda socket: socket.send(RESET_MSG)) == RESP_OK

    def reset(self) -> bool:
        """Trigger the worker's best-effort global store reset."""
        with self._close_lock:
            if self._closed:
                return False
            future = self.executor.submit(self._reset)
        return future.result()

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            futures = tuple(self.futures.values())
            self.futures.clear()
            self._results.clear()
        for future in futures:
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.socket.close(linger=0)
        self.ctx.term()


def pipeline_rank(rank: int, tp_size: int, pcp_size: int, pp_size: int) -> int:
    """PP stage of a rank in vLLM's ``DP x PP x PCP x TP`` layout."""
    return rank // (tp_size * pcp_size) % pp_size


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for ZMQ lookup socket."""
    assert vllm_config.kv_transfer_config is not None
    # The global DP index: externally load-balanced engines on one host all
    # have local DP rank 0 and would otherwise share one socket path.
    dp_rank = vllm_config.parallel_config.data_parallel_index
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    hostname = socket.gethostname()
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return (
        f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_host_{hostname}_dp_rank{dp_rank}"
    )
