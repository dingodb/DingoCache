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
from typing import Any, TypeVar

import torch
from dfkv_common import (
    canonical_namespace,
    reject_namespace_override,
)
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
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
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

# dfkv: get_dp_engine_index replaces mooncake_utils.get_mooncake_dp_engine_index;
# dfkv handles its own RDMA bootstrap so the transfer-engine helpers are dropped.
from ._determinism import ensure_deterministic_block_hashing
from .client_ranks import ENV_NAME as CLIENT_RANKS_ENV
from .client_ranks import (ELIDE_ENV, participant, resolve_client_ranks,
                           should_create_client)
from .coordinator import (
    ExternalCachedBlockPool,
    DfkvStoreCoordinator,
)
from .data import (
    VLLM_MULTIWR_V2,
    ChunkedTokenDatabase,
    DfkvStoreConnectorMetadata,
    KeyMetadata,
    PoolKey,
    ReqMeta,
    key_diagnostic_label,
    split_block_contiguous_runs,
)
from .dfkv_client import DfkvDeviceClient, SgDescriptorBatch
from .dfkv_utils import get_dp_engine_index
from .metrics import DfkvStoreConnectorStats
from ._telemetry import config as _tcfg  # connector identity + client_register switch
from .protocol import (
    LOOKUP_MSG,
    LOOKUP_RESPONSE_BYTES,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
    _decode_lookup_request,
    _encode_lookup_request,
)

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
) -> _KeyStripeIdentity:
    """Map a vLLM TP/PCP/DCP rank to its key namespace and store stripe.

    vLLM forms DCP groups after transposing the ``PCP x TP`` rank grid, so
    DCP rank is ``(tp_rank * pcp_size + pcp_rank) % dcp_size``.  Workers with
    the same TP-head, PCP, and DCP key coordinate are payload replicas.  They
    split that namespace's chunks by their position in that actual replica
    set; raw TP parity is not a replica identity when DCP is enabled.
    """
    replica_step = tp_size // num_kv_heads if num_kv_heads < tp_size else 1
    key_tp_rank = tp_rank // replica_step if replica_step > 1 else tp_rank
    expected_dcp_rank = (tp_rank * pcp_size + pcp_rank) % dcp_size
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
            and (candidate * pcp_size + pcp_rank) % dcp_size == dcp_rank
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


# ============================================================
# Transfer Threads
# ============================================================

# Each direction owns one queue. ReqMeta objects retain block/hash lists and
# CUDA-event references, so an unbounded queue can grow host memory and keep
# scheduler-owned GPU blocks pinned indefinitely when the native client slows.
# Reject-new is deliberately non-blocking: blocking get_finished() on queue
# capacity would prevent vLLM from observing completions and freeing blocks.
DEFAULT_TRANSFER_QUEUE_CAPACITY = 256
MAX_TRANSFER_QUEUE_CAPACITY = 65536
DEFAULT_RECV_WORKERS = 1
MAX_RECV_WORKERS = 32
_TRANSFER_STOP = object()


def _parse_transfer_queue_capacity(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("transfer_queue_capacity must be an integer")
    if isinstance(value, int):
        capacity = value
    elif isinstance(value, str) and value.strip().isdigit():
        capacity = int(value.strip())
    else:
        raise ValueError("transfer_queue_capacity must be an integer")
    if not 1 <= capacity <= MAX_TRANSFER_QUEUE_CAPACITY:
        raise ValueError(
            "transfer_queue_capacity must be in "
            f"[1, {MAX_TRANSFER_QUEUE_CAPACITY}], got {capacity}"
        )
    return capacity

def _parse_recv_workers(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("recv_workers must be an integer")
    if isinstance(value, int):
        workers = value
    elif isinstance(value, str) and value.strip().isdigit():
        workers = int(value.strip())
    else:
        raise ValueError("recv_workers must be an integer")
    if not 1 <= workers <= MAX_RECV_WORKERS:
        raise ValueError(
            f"recv_workers must be in [1, {MAX_RECV_WORKERS}], got {workers}"
        )
    return workers


@dataclasses.dataclass
class _ReceiveRequestState:
    request: ReqMeta | None
    block_ids: tuple[int, ...]
    phase: str = "queued"
    cancel_requested: bool = False
    fail_closed_on_cancel: bool = False
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
        reason = "closed"
        with self._stop_lock:
            if self._accepting:
                try:
                    self.request_queue.put_nowait(request)
                    return True
                except queue.Full:
                    reason = "saturated"
        self._cancel_request(request)
        logger.warning(
            "%s rejected request %s: transfer queue %s (capacity=%d)",
            self.name,
            getattr(request, "req_id", "<unknown>"),
            reason,
            self.request_queue.maxsize,
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
        while True:
            request_data = self.request_queue.get()
            try:
                if request_data is _TRANSFER_STOP:
                    return
                self._handle_request(request_data)
            except Exception as e:
                logger.error("Error in %s: %s", self.name, e)
            finally:
                self.request_queue.task_done()

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
        self.coord = coord
        self.kv_role = kv_role
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.enable_kv_event = enable_kv_event
        # Which request _handle_request is executing right now. Published under
        # done_task_lock together with the dequeue gate below, so a request is
        # either dropped at the gate or visibly in flight to
        # wait_for_inflight_put() -- never invisibly both.
        self._active_req_id: str | None = None
        self._active_cv = threading.Condition(self.done_task_lock)

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def wait_for_inflight_put(self, req_id: str, timeout_s: float = 30.0) -> bool:
        """Block until no store for ``req_id`` is currently executing.

        Queued-but-not-started entries are handled by
        delete_finished_stored_request() + the dequeue gate (they get dropped
        before any GPU read); the one entry the thread may already be
        executing cannot be cancelled -- it holds an RDMA read against the
        request's GPU blocks -- so the preemption path must wait it out
        before those blocks can be handed to another request. Returns False
        when the diagnostic interval expires; callers that protect GPU block
        ownership must keep waiting until this returns True.
        """
        with self._active_cv:
            return self._active_cv.wait_for(
                lambda: self._active_req_id != req_id, timeout_s
            )

    def _handle_request(self, req_meta: ReqMeta):
        # Cache hits are always a multiple of ``lcm_block_size`` tokens, which
        # is also ``store_mask``'s precondition.
        lcm_block_size = self.coord.lcm_block_size
        token_len = req_meta.token_len_chunk // lcm_block_size * lcm_block_size
        block_ids_per_group = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        # Publish the active request and take the dequeue gate under ONE lock
        # acquisition: a concurrent preemption fence either deletes the counter
        # first (we drop the entry here) or sees _active_req_id == req_id and
        # waits for the finally below. No window where the put runs invisibly.
        with self.done_task_lock:
            self._active_req_id = req_id
            gate_ok = req_id in self.stored_requests
        if not gate_ok:
            with self._active_cv:
                self._active_req_id = None
                self._active_cv.notify_all()
            return

        # Decrement the in-flight counter in ``finally`` (the base run loop
        # always signals ``task_done``) so the scheduler can release the GPU
        # blocks it pinned for this request even when the store path raises.
        try:
            if token_len == 0:
                return

            # Within each lcm region only per-spec relevant chunks are loaded
            # (e.g., SWA or linear attn), so mask out irrelevant chunks
            store_masks = self.coord.store_mask(token_len)
            starts: list[int] = []
            ends: list[int] = []
            keys: list[bytes] = []
            block_hashes: list[BlockHash] = []
            group_indices: list[int] = []
            for g_idx, db in enumerate(self.token_databases):
                mask = store_masks[g_idx]
                for chunk_idx, (start, end, key) in enumerate(
                    db.process_tokens(token_len, req_meta.block_hashes)
                ):
                    if chunk_idx >= len(mask) or not mask[chunk_idx]:
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.to_bytes())
                    block_hashes.append(BlockHash(bytes.fromhex(key.chunk_hash)))
                    group_indices.append(g_idx)

            # Split chunks across the workers that share this exact key
            # coordinate. A converged CLIENT_RANKS non-participant selects
            # nothing while preserving request bookkeeping and completion.
            if self.stripe_idx is None:
                sl = slice(0, 0)
            else:
                sl = slice(self.stripe_idx, None, self.stripe_step)
            starts = starts[sl]
            ends = ends[sl]
            keys = keys[sl]
            block_hashes = block_hashes[sl]
            group_indices = group_indices[sl]

            if not keys:
                return

            # A multiwr-v2 chunk is exactly one dfkv object. The native client
            # owns any HCA-width windowing, so dedup probes the logical keys
            # directly and a missing object is rewritten as a whole.
            save_exists_start = time.perf_counter()
            try:
                exists_states = self.client.batch_exist(keys)
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

            if not missing_indices:
                return

            starts = [starts[i] for i in missing_indices]
            ends = [ends[i] for i in missing_indices]
            keys = [keys[i] for i in missing_indices]
            block_hashes = [block_hashes[i] for i in missing_indices]
            group_indices = [group_indices[i] for i in missing_indices]

            logger.debug(
                "Storing KV cache for %d blocks (groups=%s) for request %s",
                len(keys),
                set(group_indices),
                req_id,
            )

            descriptor_chunks = [
                (
                    self.token_databases[g_idx],
                    start,
                    end,
                    block_ids_per_group[g_idx],
                )
                for start, end, g_idx in zip(
                    starts, ends, group_indices, strict=True
                )
            ]
            descriptor_batch = SgDescriptorBatch.from_chunks(descriptor_chunks)
            stored_events: list[BlockStored] = []
            # parent_block_hash chains live within a group, not across.
            prev_key_per_group: dict[int, Any] = {}
            new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]

            for idx, (s, e, g_idx) in enumerate(
                zip(starts, ends, group_indices, strict=True)
            ):
                db = self.token_databases[g_idx]

                if self.enable_kv_event:
                    token_ids = (
                        req_meta.token_ids[s:e]
                        if req_meta.token_ids is not None
                        else None
                    )
                    stored_event = BlockStored(
                        block_hashes=[new_block_hashes[idx]],
                        parent_block_hash=prev_key_per_group.get(g_idx),
                        token_ids=token_ids,
                        block_size=db.block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                        group_idx=g_idx,
                    )
                    stored_events.append(stored_event)
                    prev_key_per_group[g_idx] = new_block_hashes[idx]

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
                    keys, descriptor_batch.ptrs, descriptor_batch.sizes
                )
                if len(res) != len(keys):
                    raise RuntimeError(
                        "batch_put_sg returned incomplete per-key results: "
                        f"keys={len(keys)} statuses={len(res)}"
                    )
                failed = _put_failed_indices(res)
                if stored_events:
                    successful_events = [
                        event
                        for event, rc in zip(stored_events, res, strict=True)
                        if rc == 0
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
                self._active_req_id = None
                self._active_cv.notify_all()
            self.dec_stored_request(req_id)

    def _cancel_request(self, req_meta: Any) -> None:
        # Keep a zero-valued entry so _get_and_clear_finished_sending can
        # acknowledge a finished request whose save was rejected/cancelled.
        self.dec_stored_request(req_meta.req_id)


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
        record_pool_sample: Callable[[str, int], None] | None = None,
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
        self.recv_workers = _parse_recv_workers(recv_workers)
        self.client_provider = client_provider
        self._invalid_block_ids_lock = threading.Lock()
        self._invalid_block_ids: set[int] = set()
        self.coord = coord
        self._request_states_lock = threading.Lock()
        self._request_states: dict[str, _ReceiveRequestState] = {}
        self._active_workers = 0
        self._pool_stop_lock = threading.Lock()
        self._pool_stop_started = False
        self._pool_stop_done = threading.Event()
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
        block_ids = tuple(
            block_id for group_ids in request.block_ids for block_id in group_ids
        )
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
                    state = _ReceiveRequestState(request=request, block_ids=block_ids)
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
                state = _ReceiveRequestState(request=request, block_ids=block_ids)
                with self._request_states_lock:
                    existing = self._request_states.get(request.req_id)
                    if existing is not None:
                        return False
                    self._request_states[request.req_id] = state
                    self._terminalize_locked(state, failed=True)
                    self._request_states.pop(request.req_id, None)
        logger.warning(
            "%s rejected request %s: transfer queue %s (capacity=%d)",
            self.name,
            request.req_id,
            reason,
            self.request_queue.maxsize,
        )
        return False

    def get_and_clear_finished_requests(self) -> set[str]:
        with self._request_states_lock:
            with self.done_task_lock:
                finished = self.finished_requests.copy()
                self.finished_requests.clear()
            for req_id in finished:
                state = self._request_states.get(req_id)
                if state is not None and state.phase == "terminal":
                    self._request_states.pop(req_id, None)
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
        if failed and state.block_ids:
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(state.block_ids)
        state.block_ids = ()
        if req_id is not None:
            with self.done_task_lock:
                self.finished_requests.add(req_id)
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
            self._pool_stop_done.set()

    close = stop


    def _add_load_error_block_ids(self, block_ids: list[int]) -> None:
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def get_and_clear_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_block_ids = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_block_ids

    def _handle_request(self, req_meta: ReqMeta):
        req_id = req_meta.req_id
        enqueued_at = getattr(req_meta, "_dfkv_receive_enqueued_at", None)
        if enqueued_at is not None:
            self._record_observation("receive_queue_wait", enqueued_at)
        geometry_start = time.perf_counter()
        try:
            token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
            mask_num = (
                req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
                // self.block_size
                * self.block_size
            )

            # Skip chunks the consumer's per-group spec wouldn't populate
            # locally (e.g. SWA pre-window) even if the producer stored them.
            load_mask_per_group = self.coord.load_mask(req_meta.block_hashes, token_len)

            descriptor_chunks: list[
                tuple[object, int, int, Sequence[int]]
            ] = []
            key_list: list[bytes] = []
            block_id_list: list[int] = []
            for g_idx, db in enumerate(self.token_databases):
                mask = load_mask_per_group[g_idx]
                for start, end, key in db.process_tokens(
                    token_len, req_meta.block_hashes, mask_num
                ):
                    chunk_idx = start // db.block_size
                    if chunk_idx >= len(mask) or not mask[chunk_idx]:
                        continue
                    block_id = int(
                        req_meta.block_ids[g_idx][start // db.block_size]
                    )
                    key_list.append(key.to_bytes())
                    descriptor_chunks.append(
                        (db, start, end, req_meta.block_ids[g_idx])
                    )
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
            rotated_chunks = _rotate_list(descriptor_chunks, rotation)
            rotated_block_ids = _rotate_list(block_id_list, rotation)
            descriptor_batch = SgDescriptorBatch.from_chunks(rotated_chunks)

            # dfkv: one logical chunk is one batch key with its complete
            # destination segment vector. libdfkv performs any HCA-width
            # multi-WR windowing without changing object identity.
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
                # The native logical operation publishes a hit only after every
                # ordered WR window completes. Require the exact stored length;
                # misses and either short or oversized objects fail the chunk
                # closed and force vLLM to recompute it.
                if client is None:  # no client (un-elide failed): miss -> recompute
                    hits, lens = [False] * len(rotated_keys), [0] * len(rotated_keys)
                else:
                    hits, lens = client.batch_get_auto_sg(
                        rotated_keys,
                        descriptor_batch.ptrs,
                        descriptor_batch.caps,
                    )
                if len(hits) != len(rotated_keys) or len(lens) != len(rotated_keys):
                    raise RuntimeError(
                        "batch_get_auto_sg returned incomplete per-key results: "
                        f"keys={len(rotated_keys)} hits={len(hits)} lens={len(lens)}"
                    )
            except Exception as e:
                self._add_load_error_block_ids(rotated_block_ids)
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
                self._add_load_error_block_ids(failed_block_ids)
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

        except Exception as e:
            # Any unexpected failure in the load path -> recompute this
            # request's blocks (never hang vLLM's WAITING_FOR_REMOTE_KVS).
            logger.error("dfkv recv thread failed for req %s: %s", req_id, e)
            try:
                self._add_load_error_block_ids(
                    [b for ids in req_meta.block_ids for b in ids]
                )
            except Exception:
                pass
    def _cancel_request(self, req_meta: Any) -> None:
        self.cancel_requests(
            (req_meta.req_id,), wait=False, fail_closed=True
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
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0

        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.transfer_queue_capacity = _parse_transfer_queue_capacity(
            extra.get(
                "transfer_queue_capacity",
                DEFAULT_TRANSFER_QUEUE_CAPACITY,
            )
        )
        self.recv_workers = _parse_recv_workers(
            extra.get("recv_workers", DEFAULT_RECV_WORKERS)
        )
        logger.info(
            "dfkv transfer queues: capacity=%d per direction, recv_workers=%d, "
            "overload=reject-new, shutdown=cancel-pending",
            self.transfer_queue_capacity,
            self.recv_workers,
        )
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_done = threading.Event()
        # Store keys embed block_hashes: refuse to start with process-local
        # hashing (silent 0% cross-instance/cross-restart hit rate otherwise).
        ensure_deterministic_block_hashing(vllm_config.cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = extra.get("load_async", True)
        if not self.load_async:
            # Fail at construction with an actionable message: the synchronous
            # load mode was never implemented, and the old hot-path assert in
            # get_finished() crashed the worker mid-serving instead.
            raise ValueError(
                "dfkv connector: kv_connector_extra_config.load_async=false is "
                "not supported (loads are handled by the async recv thread); "
                "remove the key or set it to true")
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
        )
        self.head_or_tp_rank = key_stripe.tp_rank
        self.dcp_rank = key_stripe.dcp_rank
        self.stripe_idx = key_stripe.stripe_idx
        self.stripe_step = key_stripe.stripe_step

        # CLIENT_RANKS (issue #111): converge store-side dfkv clients onto a
        # subset of ranks when the KV object is fully TP-replicated.
        # Layout-clamped, never rejects (one fleet-wide env template must be
        # safe everywhere).
        replicated = self.head_or_tp_rank < 0
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

        self.metadata = KeyMetadata(
            model_name=model_config.model.rstrip("/").split("/")[-1],
            dp_size=self.dp_size,
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
        _hf_config = getattr(model_config, "hf_config", None)
        _model_revision = (
            extra.get("model_revision")
            or getattr(model_config, "revision", None)
            or getattr(_hf_config, "_commit_hash", None)
            or model_identity
        )
        _cache_dtype = str(
            getattr(self.cache_config, "cache_dtype", "unknown"))
        _group_layout = "|".join(
            f"{idx}:{type(group.kv_cache_spec).__name__}:"
            f"{int(getattr(group.kv_cache_spec, 'block_size', 0))}:"
            f"{','.join(sorted(group.layer_names))}"
            for idx, group in enumerate(kv_cache_config.kv_cache_groups)
        )
        key_namespace = canonical_namespace(
            model_identity,
            VLLM_MULTIWR_V2.decode("ascii"),
            tenant_id=str(extra.get("tenant_id", "default")),
            model_revision=str(_model_revision),
            dtype=_cache_dtype,
            block_tokens=max(1, self.block_size),
            layer_count=max(1, self.num_layers),
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            layout_fields={
                "storage_layout": VLLM_MULTIWR_V2.decode("ascii"),
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
        )
        # Phase 2a (issue #111): producer non-participants skip the client
        # (opt-in via DFKV_CONNECTOR_CLIENT_ELIDE=1; layout-clamped). The saved
        # kwargs let the rank UN-elide lazily if a load ever reaches it — a
        # producer DOES load on cross-instance prefix reuse (get_finished has
        # no role gate), and failing those loads into whole-span recomputes
        # would cost more than the elided connections save.
        self._lazy_client_kwargs: dict | None = None
        self._lazy_client_lock = threading.Lock()
        self._kv_pool_regions: list[tuple[int, int]] = []
        create_client, elide_reason = should_create_client(
            self.kv_role, self.tp_rank, self.tp_size, self.client_ranks,
            _tcfg.truthy(os.environ.get(ELIDE_ENV, "0")))
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
        # PCP/DCP > 1 (single- OR multi-group): scale EACH group's
        # spec.block_size to self.block_size (= scheduler_block_size) so the
        # coordinator's ``block_size % hash_block_size == 0`` and
        # ``scheduler_block_size % block_size == 0`` invariants hold. Under CP
        # different groups shard differently (e.g. GLM-5.2 DSA: the MLA group is
        # DCP-sharded while the indexer group is not), but normalizing every
        # group to scheduler_block_size keeps the per-group hashing / store-load
        # masks uniform; each rank still only holds (and stores under its
        # @pcp{r}@dcp{r} key) its own physical shard of the cache.
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

        def _scale_spec(spec):
            # UniformTypeKVCacheSpecs wraps inner per-layer specs; the
            # coordinator unwraps to the inner spec (see _unwrap_spec), so the
            # inner block_size(s) must be scaled too, not just the wrapper's.
            if isinstance(spec, UniformTypeKVCacheSpecs):
                inner = {
                    name: (
                        dataclasses.replace(s, block_size=self.block_size)
                        if s.block_size != self.block_size
                        else s
                    )
                    for name, s in spec.kv_cache_specs.items()
                }
                if spec.block_size == self.block_size and all(
                    s.block_size == self.block_size for s in inner.values()
                ):
                    return spec
                return dataclasses.replace(
                    spec, block_size=self.block_size, kv_cache_specs=inner
                )
            if spec.block_size != self.block_size:
                return dataclasses.replace(spec, block_size=self.block_size)
            return spec

        groups = [
            dataclasses.replace(g, kv_cache_spec=_scale_spec(g.kv_cache_spec))
            for g in kv_cache_config.kv_cache_groups
        ]
        self._kv_cache_groups: list[KVCacheGroupSpec] = groups
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        use_eagle = bool(
            spec_cfg.use_eagle()
            if spec_cfg is not None and callable(getattr(spec_cfg, "use_eagle", None))
            else False
        )
        self.coord = DfkvStoreCoordinator(
            self._kv_cache_groups,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            use_eagle=use_eagle,
        )
        # One ChunkedTokenDatabase per group; addresses populated in
        # register_kv_caches once the kv-cache layout is known.
        self.token_dbs: list[ChunkedTokenDatabase] = [
            ChunkedTokenDatabase(
                dataclasses.replace(self.metadata, group_id=g_idx),
                g.kv_cache_spec.block_size,
                hash_block_size=self.hash_block_size,
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
        get_finished has no role gate). Keeps phase 2a's connection savings for
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
                try:
                    client = DfkvDeviceClient(**self._lazy_client_kwargs)
                    for base, ln in self._kv_pool_regions:
                        client.register_memory(base, ln)
                except Exception:
                    logger.exception(
                        "dfkv lazy un-elide failed; loads on this rank miss")
                    return None
                self.client = client
                logger.info(
                    "dfkv client un-elided: a load reached this elided "
                    "producer rank (tp_rank=%d)", self.tp_rank)
        return self.client

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

        # Resolve each entry to a representative tensor for storage
        # deduplication. For attention layers the value is already a tensor;
        # for Mamba layers it is a list of tensors that all share the same
        # underlying raw storage, so we take the first one.
        def _repr_tensor(v: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
            assert isinstance(v, torch.Tensor | list)
            return v if isinstance(v, torch.Tensor) else v[0]

        assert self.cache_config.num_gpu_blocks is not None
        self.num_blocks = self.cache_config.num_gpu_blocks

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

        # Dict insertion order is not a storage-layout contract. Canonicalize by
        # semantic layer name so independently constructed producer/consumer
        # mappings gather and scatter the same ordered byte stream.
        for layer_name in sorted(kv_caches):
            cache = _repr_tensor(kv_caches[layer_name])
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

            g_idx = layer_to_group[layer_name]
            layer_addr = cache.data_ptr()
            el = cache.element_size()
            page_size_bytes = region_len // self.num_blocks
            outer_dims = [
                d for d in range(cache.ndim) if cache.stride(d) * el > page_size_bytes
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
                    cache.shape, cache.stride(), el
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
        )
        self.kv_recv_thread.start()
        ready_event_recving.wait()

    def start_load_kv(
        self,
        metadata: DfkvStoreConnectorMetadata,
    ):
        """Fence every preempted request's active GPU transfer.

        A queued receive is cancelled without a native call.  An active receive
        cannot be interrupted safely, so this waits for only that request's
        GPUDirect write before its blocks may be reused.  The send-side fence
        likewise waits for its request's active RDMA read.
        """
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
        # Drop queued entries first (the dequeue gate re-checks per entry),
        # then join whichever entry is already executing. The timeout is only
        # a diagnostic interval: returning from this hook while native PUT can
        # still read the blocks would let the scheduler reuse them.
        for req_id in metadata.preempted_req_ids:
            send_thread.delete_finished_stored_request(req_id)
        for req_id in metadata.preempted_req_ids:
            wait_start = time.perf_counter()
            while not send_thread.wait_for_inflight_put(req_id):
                logger.error(
                    "preemption fence still waiting for in-flight save "
                    "of request %s after %.1fs; GPU block reuse remains fenced "
                    "until the native save exits",
                    req_id,
                    time.perf_counter() - wait_start,
                )

    def wait_for_save(
        self,
        metadata: DfkvStoreConnectorMetadata,
    ):
        """No-op: stores are issued in get_finished() for overlap."""
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
        meta: DfkvStoreConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        """Issue all I/O and get completed send/recv request IDs.

        All load and store I/O requests are issued here (after model
        compute is launched on the compute stream) for better
        compute-I/O overlap.
        """
        # Aborted/finished requests may have an outstanding remote load.  Their
        # blocks cannot be freed until that request's own GPUDirect write exits.
        if finished_req_ids and self.kv_recv_thread is not None:
            self.kv_recv_thread.cancel_requests(
                finished_req_ids,
                wait=True,
                fail_closed=False,
            )
        # Issue async loads
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

            for request in meta.requests:
                if not request.can_save:
                    continue
                request.current_event = current_event
                assert self.kv_send_thread is not None
                self.kv_send_thread.add_stored_request(request.req_id)
                self.kv_send_thread.add_request(request)

        # Check completion of previously queued transfers
        done_sending = (
            self._get_and_clear_finished_sending(finished_req_ids, meta)
            if self.kv_role in ["kv_producer", "kv_both"]
            else set()
        )

        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests()
            if self.load_async and self.kv_recv_thread is not None
            else set()
        )

        if done_sending or done_recving:
            logger.debug(
                "dfkv get_finished: done_recving=%s done_sending=%s tp=%d",
                done_recving, done_sending, self.tp_rank,
            )
        return done_sending, done_recving

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
        meta: DfkvStoreConnectorMetadata,
    ) -> set[str]:
        assert self.kv_send_thread is not None
        finished_sending: set[str] = set()

        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)

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

    def lookup(self, token_len: int, block_hashes: list[BlockHash]) -> int:
        """Check how many prefix tokens exist in the store.

        Checks across all TP ranks at this worker's own PP rank.
        """
        if not block_hashes or token_len <= 0:
            return 0

        # Build every physical object required by each logical LCM chunk.
        # Object metadata retains the logical chunk index so repeated hashes
        # cannot collapse accounting across distinct prefix positions.
        candidate_keys: list[bytes] = []
        candidate_meta: list[tuple[int, int, int, bytes]] = []
        expected_per_object: dict[tuple[int, int, int, bytes], int] = {}
        tp_count = min(self.tp_size, self.num_kv_head)
        # dfkv: gate candidates by store_mask -- the SAME per-(group,chunk) set
        # the SAVE path stores (worker.py save gate) and the LOAD path reads
        # (load_mask delegates to store_mask). For SlidingWindow groups (V4-Flash
        # has 4 of them besides the full-MLA group) store_mask keeps only the
        # in-window tail chunks; without this gate the lookup enumerated every
        # chunk (4830 vs the 1058 actually stored), so find_longest_cache_hit's
        # SWA walk demanded never-stored pre-window chunks and collapsed to 0.
        aligned_token_len = (
            token_len // self.coord.lcm_block_size * self.coord.lcm_block_size
        )
        store_masks = self.coord.store_mask(aligned_token_len)
        if aligned_token_len == 0:
            return 0
        num_prefix_chunks = aligned_token_len // self.coord.lcm_block_size
        required_objects_per_chunk = [0] * num_prefix_chunks
        missing_required_per_chunk = [False] * num_prefix_chunks
        for g_idx, db in enumerate(self.token_dbs):
            spec_block_size = db.block_size
            mask = store_masks[g_idx]
            group_hashes = self.coord.block_hashes_for_spec(
                block_hashes, self._kv_cache_groups[g_idx].kv_cache_spec
            )
            # Replicated MLA stores every SAVE under the single canonical
            # tp_rank=-1 coordinate (see metadata init): expanding to
            # tp_rank=0..tp_count-1 would never match it. Probe the stored
            # coordinate as-is when negative; else expand across the per-TP
            # head coordinates used by GQA/DCP saves.
            tp_candidates = (
                [db.metadata.tp_rank] if db.metadata.tp_rank < 0 else range(tp_count)
            )
            rank_probes = max(1, len(tp_candidates))
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
                required_objects_per_chunk[logical_chunk_idx] += 1
                for tp in tp_candidates:
                    # Keep this worker's own pp_rank (db.metadata.pp_rank): PP
                    # partitions layers, so each (group, chunk) lives under a
                    # single stage's pp_rank, matching the SAVE/LOAD keys. Probing
                    # every pp_rank matched nothing and forced present=0 on PP>1.
                    md = dataclasses.replace(db.metadata, tp_rank=tp)
                    candidate_keys.append(PoolKey(md, h.hex()).to_bytes())
                    candidate_meta.append(object_meta)
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

        if not candidate_keys:
            return 0

        lookup_start = time.perf_counter()
        try:
            res = self.client.batch_exist(candidate_keys)
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
                num_logical_keys=len(expected_per_object),
                status="error",
                num_failed_keys=len(candidate_keys),
            )
            logger.error("Remote connection failed in lookup: %s", e)
            return 0

        self._record_kv_connector_operation(
            "lookup_exists",
            time.perf_counter() - lookup_start,
            len(candidate_keys),
            num_logical_keys=len(expected_per_object),
        )

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
            required > 0 and not missing_required_per_chunk[idx]
            for idx, required in enumerate(required_objects_per_chunk)
        ]
        exists_set: set[tuple[int, bytes]] = set()
        for object_meta, expected in expected_per_object.items():
            logical_chunk_idx, g_idx, _chunk_id, chunk_hash = object_meta
            if present_per_object.get(object_meta, 0) != expected:
                complete_chunks[logical_chunk_idx] = False
            else:
                exists_set.add((g_idx, chunk_hash))

        prefix_chunks = 0
        for chunk_idx, complete in enumerate(complete_chunks):
            if required_objects_per_chunk[chunk_idx] == 0 or not complete:
                break
            prefix_chunks += 1
        complete_prefix_tokens = prefix_chunks * self.coord.lcm_block_size

        if complete_prefix_tokens == 0:
            hit_length = 0
        else:
            _masks, semantic_hit_length = self.coord.find_longest_cache_hit(
                block_hashes,
                complete_prefix_tokens,
                ExternalCachedBlockPool(exists_set),
            )
            hit_length = min(complete_prefix_tokens, semantic_hit_length)
        logger.debug(
            "dfkv lookup: token_len=%d candidates=%d complete_chunks=%d/%d "
            "-> hit_length=%d",
            token_len,
            len(candidate_keys),
            prefix_chunks,
            num_prefix_chunks,
            hit_length,
        )
        return hit_length

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
                        token_len, hash_bytes = _decode_lookup_request(
                            [frame.buffer for frame in all_frames]
                        )
                        block_hashes = [BlockHash(value) for value in hash_bytes]
                        result = self.store_worker.lookup(token_len, block_hashes)
                        if not 0 <= result < 1 << (8 * LOOKUP_RESPONSE_BYTES):
                            raise ValueError(
                                f"lookup result is outside uint32: {result}"
                            )
                    except Exception as e:
                        logger.warning(
                            "LookupKeyServer rejected lookup request: %s", e
                        )
                        result = 0
                    self.socket.send(
                        result.to_bytes(LOOKUP_RESPONSE_BYTES, byteorder="big")
                    )
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
                        if self.store_worker.kv_send_thread is not None:
                            self.store_worker.kv_send_thread.request_queue.join()
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
    """

    def __init__(self, vllm_config: VllmConfig):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        self._close_lock = threading.Lock()
        self._closed = False
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="DfkvLookupClient",
        )
        self.futures: dict[str, Future[int]] = {}

    def _lookup(self, token_len: int, block_hashes: list[BlockHash]) -> int:
        all_frames = _encode_lookup_request(token_len, block_hashes)
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        if len(resp) != LOOKUP_RESPONSE_BYTES:
            raise ValueError(
                f"lookup response has {len(resp)} bytes, "
                f"expected {LOOKUP_RESPONSE_BYTES}"
            )
        return int.from_bytes(resp, byteorder="big")

    def lookup(
        self,
        req_id: str,
        token_len: int,
        block_hashes: list[BlockHash],
        non_block: bool = False,
    ) -> int | None:
        """Return a hit count, or ``None`` while a non-blocking lookup runs."""
        with self._close_lock:
            if self._closed:
                return 0
            future = self.futures.get(req_id)
            if future is None:
                future = self.executor.submit(
                    self._lookup,
                    token_len,
                    list(block_hashes),
                )
                self.futures[req_id] = future

        if non_block and not future.done():
            return None
        try:
            return future.result()
        except Exception as e:
            logger.error("Dfkv lookup failed for %s: %s", req_id, e)
            return 0
        finally:
            with self._close_lock:
                if self.futures.get(req_id) is future:
                    self.futures.pop(req_id)

    def discard(self, req_id: str) -> None:
        """Drop any cached or queued lookup for an aborted request."""
        with self._close_lock:
            future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()

    def _reset(self) -> bool:
        """Send the reset admin request from the socket-owner thread."""
        self.socket.send(RESET_MSG)
        resp = self.socket.recv()
        return bytes(resp) == RESP_OK

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
        for future in futures:
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.socket.close(linger=0)
        self.ctx.term()


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for ZMQ lookup socket."""
    assert vllm_config.kv_transfer_config is not None
    dp_rank = get_dp_engine_index(vllm_config.parallel_config)
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
