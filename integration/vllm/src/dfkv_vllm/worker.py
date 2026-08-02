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
import hashlib
import os
import queue
import socket
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

import torch
from dfkv_common import VLLM_RAW_V1, canonical_namespace, sg_key
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
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

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
    ChunkedTokenDatabase,
    DfkvStoreConnectorMetadata,
    KeyMetadata,
    PoolKey,
    ReqMeta,
    split_block_contiguous_runs,
)
from .dfkv_client import DfkvDeviceClient
from .dfkv_utils import get_dp_engine_index
from .metrics import DfkvStoreConnectorStats
from ._telemetry import config as _tcfg  # connector identity + client_register switch
from .protocol import (
    LOOKUP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
)

logger = init_logger(__name__)

_T = TypeVar("_T")


def _rotate_list(values: list[_T], offset: int) -> list[_T]:
    return values[offset:] + values[:offset]


def _sum_batch_bytes(sizes: list[list[int]]) -> int:
    return sum(sum(size) for size in sizes)


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


# dfkv: batch_put returns a per-key rc (0 == ok, non-zero == failure); a GET
# returns (hits, lens) with hit in {0, 1}. These helpers normalize the two
# dfkv return shapes into the (failed-index list) shape the threads expect.
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
        queue_capacity: int = DEFAULT_TRANSFER_QUEUE_CAPACITY,
    ):
        super().__init__(daemon=False, name=name)
        self.client = client
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.token_databases = token_databases
        self._record_operation_cb = record_operation
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
            num_bytes=num_bytes,
            status=status,
            num_failed_keys=num_failed_keys,
        )

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
        put_step: int,
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
        self.put_step = put_step
        # CLIENT_RANKS store convergence (issue #111): stripe index/step over
        # the participating ranks. stripe_idx None = this rank stores nothing
        # (a converged peer stores on its behalf; KV is TP-replicated).
        # Defaults reproduce the legacy tp_rank % put_step striping exactly.
        self.stripe_idx: int | None = tp_rank % put_step
        self.stripe_step: int = put_step
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
        on timeout (put ops are client-side bounded, so this only happens if
        the store path is badly wedged).
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

            # Apply store striding: legacy = tp_rank % put_step over put_step;
            # under CLIENT_RANKS convergence = index over the participant set,
            # and non-participants select nothing (empty slice keeps the
            # request lifecycle -- bookkeeping/finish -- identical).
            if self.stripe_idx is None:
                sl = slice(0, 0)
            else:
                sl = slice(self.stripe_idx % self.stripe_step, None, self.stripe_step)
            starts = starts[sl]
            ends = ends[sl]
            keys = keys[sl]
            block_hashes = block_hashes[sl]
            group_indices = group_indices[sl]

            if not keys:
                return

            # Stored values use explicit scatter-group coordinates (see
            # _group_segments_sg), so probe group 0 as the block-present proxy.
            # Lookup uses the same proxy. Probing the bare pool key would miss
            # every stored group and defeat deduplication.
            save_exists_start = time.perf_counter()
            try:
                exists_states = self.client.batch_exist(
                    [_sg_group_key(k, 0, _sg_segs_of(self.client)) for k in keys]
                )
            except Exception:
                self._record_operation(
                    "save_exists",
                    save_exists_start,
                    len(keys),
                    status="error",
                    num_failed_keys=len(keys),
                )
                raise
            self._record_operation(
                "save_exists",
                save_exists_start,
                len(keys),
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

            addrs: list[list[int]] = []
            sizes: list[list[int]] = []
            stored_events: list[BlockStored] = []
            # parent_block_hash chains live within a group, not across.
            prev_key_per_group: dict[int, Any] = {}
            new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]

            for idx, (s, e, g_idx) in enumerate(
                zip(starts, ends, group_indices, strict=True)
            ):
                db = self.token_databases[g_idx]
                addr, size, _ = db.prepare_value(s, e, block_ids_per_group[g_idx])
                addrs.append(addr)
                sizes.append(size)

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

            # Coalesce each chunk's per-layer segments into one raw dfkv value
            # per explicit scatter group. This cuts the key count ~20x
            # (25392 -> ~1242), making loads bandwidth-bound rather than
            # per-key-bound.
            sg_keys, sg_ptrs, sg_sizes, _ = _group_segments_sg(
                keys, addrs, sizes, _sg_segs_of(self.client)
            )
            batch_bytes = sum(sum(s) for s in sg_sizes)
            put_start = time.perf_counter()
            try:
                res = self.client.batch_put_sg(sg_keys, sg_ptrs, sg_sizes)
                failed = _put_failed_indices(res)
                self._record_operation(
                    "save_put",
                    put_start,
                    len(sg_keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                logger.debug(
                    "dfkv save_put: keys=%d bytes=%d ms=%.1f failed=%d",
                    len(sg_keys), batch_bytes,
                    (time.perf_counter() - put_start) * 1000.0, len(failed),
                )
                if failed:
                    failed_codes = set(res[i] for i in failed)
                    # dfkv: write failure is non-fatal -- count + drop this
                    # request's save; dfkv's peer-health marks the bad peer and
                    # short-circuits. Never block inference.
                    logger.warning(
                        "batch_put_sg failed: %d/%d keys failed "
                        "(codes=%s, batch_bytes=%d), first_key=%s",
                        len(failed),
                        len(sg_keys),
                        failed_codes,
                        batch_bytes,
                        _key_label(sg_keys[0]) if sg_keys else "N/A",
                    )
            except Exception as e:
                self._record_operation(
                    "save_put",
                    put_start,
                    len(sg_keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(sg_keys),
                )
                logger.error(
                    "Failed to put keys %s, error: %s",
                    [_key_label(key) for key in sg_keys[:3]],
                    e,
                )

            if self.enable_kv_event and stored_events:
                self.update_kv_event(stored_events)
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
    """Background thread for loading KV cache blocks from the store."""

    def __init__(
        self,
        client: Any,
        coord: DfkvStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        record_operation: Callable[..., None] | None = None,
        client_provider: Callable[[], Any] | None = None,
        queue_capacity: int = DEFAULT_TRANSFER_QUEUE_CAPACITY,
    ):
        super().__init__(
            client,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
            record_operation=record_operation,
            queue_capacity=queue_capacity,
        )
        # Elided ranks start with client=None; a real load arriving there calls
        # this back into the worker to lazily un-elide (create + register the
        # client) instead of failing the span into a recompute.
        self.client_provider = client_provider
        # _invalid_block_ids can be access by both the Worker and RecvingThread
        self._invalid_block_ids_lock = threading.Lock()
        self._invalid_block_ids: set[int] = set()
        self.coord = coord

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

            addr_list: list[list[int]] = []
            size_list: list[list[int]] = []
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
                    addr, size, block_id = db.prepare_value(
                        start, end, req_meta.block_ids[g_idx]
                    )
                    key_list.append(key.to_bytes())
                    addr_list.append(addr)
                    size_list.append(size)
                    block_id_list.append(block_id)

            # Nothing to load (e.g. every chunk masked out for this group) -> finish
            # the request now, else `tp_rank % 0` below would raise and the req would
            # never be reported done, hanging vLLM's WAITING_FOR_REMOTE_KVS wait.
            if not key_list:
                return

            # Rotate aligned lists by tp_rank for load balancing.
            rotation = self.tp_rank % len(key_list)
            key_list_c = _rotate_list(key_list, rotation)
            addr_list_c = _rotate_list(addr_list, rotation)
            size_list_c = _rotate_list(size_list, rotation)
            block_id_list_c = _rotate_list(block_id_list, rotation)

            # dfkv: A GET is one batch -- dfkv has no owner-side DirectIO staging
            # budget, so the Mooncake disk-offload sub-batch split is dropped.
            # Flatten per-key scatter-gather to one (ptr, cap) per dfkv key, then
            # map each flattened result back to its originating block id for error
            # reporting (see _group_segments_sg / seg_owner).
            # Resolve the client BEFORE grouping: the SG width (and with it the
            # on-wire key names) comes from the live transport.
            client = self.client
            if client is None and self.client_provider is not None:
                client = self.client_provider()  # lazy un-elide
                if client is not None:
                    self.client = client
            sg_keys, sg_ptrs, sg_caps, sg_owner = _group_segments_sg(
                key_list_c, addr_list_c, size_list_c, _sg_segs_of(client)
            )
            sg_totals = [sum(c) for c in sg_caps]
            batch_bytes = sum(sg_totals)

            load_get_start = time.perf_counter()
            try:
                # dfkv: one scatter-gather GET per coalesced key -> (hits, lens). The
                # server returns one blob; the RDMA recv SGE list scatters it back
                # into this chunk's per-layer segments. hit == 1 + len == sum(caps)
                # means the chunk loaded; a miss or short read marks the originating
                # block as a load error so vLLM recomputes that span (never fatal).
                if client is None:  # no client (un-elide failed): miss -> recompute
                    hits, lens = [False] * len(sg_keys), [0] * len(sg_keys)
                else:
                    hits, lens = client.batch_get_auto_sg(sg_keys, sg_ptrs, sg_caps)
                failed_block_ids: list[int] = []
                failed_detail: list[tuple[str, int]] = []
                for i, (hit, got_len) in enumerate(zip(hits, lens, strict=True)):
                    if hit != 1 or got_len < sg_totals[i]:
                        failed_block_ids.append(block_id_list_c[sg_owner[i]])
                        failed_detail.append((_key_label(sg_keys[i]), hit))
                self._record_operation(
                    "load_get",
                    load_get_start,
                    len(sg_keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed_block_ids else "ok",
                    num_failed_keys=len(failed_block_ids),
                )
                logger.debug(
                    "dfkv load_get: req=%s keys=%d bytes=%d ms=%.1f failed=%d",
                    req_id, len(sg_keys), batch_bytes,
                    (time.perf_counter() - load_get_start) * 1000.0,
                    len(failed_block_ids),
                )
                if failed_block_ids:
                    self._add_load_error_block_ids(failed_block_ids)
                    logger.warning(
                        "Failed to get %d dfkv keys from batch "
                        "(batch_keys=%d, first_failures=%s)",
                        len(failed_block_ids),
                        len(sg_keys),
                        failed_detail[:3],
                    )
            except Exception as e:
                self._add_load_error_block_ids(block_id_list_c)
                self._record_operation(
                    "load_get",
                    load_get_start,
                    len(sg_keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(sg_keys),
                )
                logger.warning(
                    "Failed to get dfkv batch %s, error: %s",
                    [_key_label(key) for key in sg_keys[:3]],
                    e,
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
        finally:
            self.set_finished_request(req_id)

    def _cancel_request(self, req_meta: Any) -> None:
        # A queued/rejected load must finish as a visible recompute, never leave
        # the request stuck in WAITING_FOR_REMOTE_KVS.
        try:
            self._add_load_error_block_ids(
                [block_id for ids in req_meta.block_ids for block_id in ids]
            )
        except Exception:
            logger.exception(
                "%s could not enumerate cancelled load blocks for request %s",
                self.name,
                getattr(req_meta, "req_id", "<unknown>"),
            )
        finally:
            self.set_finished_request(req_meta.req_id)


# Maximum raw-payload segments gathered into one scatter-group object.
# The ConnectX-era default is max_sge=30 minus one request-prefix SGE. The
# negotiated value comes from ``dfkv_max_sg_segs``; the fallback is encoded in
# the binary SG object-key coordinate (see ``_sg_group_key``).
SG_MAX_SEGS = 29


# Live SG width of a client's transport, cached on the client object (device
# caps don't change under a running process). None / old libs => the default.
def _sg_segs_of(client: Any) -> int:
    if client is None:
        return SG_MAX_SEGS
    v = getattr(client, "_sg_segs_cache", None)
    if v is None:
        try:
            v = int(client.max_sg_segs()) or SG_MAX_SEGS
        except Exception:  # older libdfkv without dfkv_max_sg_segs
            v = SG_MAX_SEGS
        client._sg_segs_cache = v
    return v

def _key_label(key: bytes) -> str:
    """Return a deterministic JSON/text-safe representation of a binary key."""
    return f"len={len(key)} sha256={hashlib.sha256(key).hexdigest()[:16]}"


# Scatter geometry is always explicit in the physical key.
def _sg_group_key(key: bytes, grp: int, max_segs: int) -> bytes:
    return sg_key(key, max_segs, grp)


# Coalesce each chunk's per-layer segments into explicit scatter-group keys.
# Each group is one raw dfkv value written through one RDMA multi-SGE operation.
# owner[i] maps each physical key back to its source chunk for error attribution.
def _group_segments_sg(
    keys: list[bytes],
    addrs: list[list[int]],
    sizes: list[list[int]],
    max_segs: int = SG_MAX_SEGS,
) -> tuple[list[bytes], list[list[int]], list[list[int]], list[int]]:
    sg_keys: list[bytes] = []
    sg_ptrs: list[list[int]] = []
    sg_sizes: list[list[int]] = []
    sg_owner: list[int] = []
    for key_idx, (key, addr, size) in enumerate(zip(keys, addrs, sizes, strict=True)):
        grp = 0
        for off in range(0, len(addr), max_segs):
            sg_keys.append(_sg_group_key(key, grp, max_segs))
            sg_ptrs.append(list(addr[off:off + max_segs]))
            sg_sizes.append(list(size[off:off + max_segs]))
            sg_owner.append(key_idx)
            grp += 1
    return sg_keys, sg_ptrs, sg_sizes, sg_owner


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
        logger.info(
            "dfkv transfer queues: capacity=%d per direction, "
            "overload=reject-new, shutdown=cancel-pending",
            self.transfer_queue_capacity,
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

        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        # DCP shards the (otherwise TP-replicated) MLA KV across dcp ranks, so
        # each rank holds a UNIQUE shard (separated by the @dcp{r} key field),
        # not a replica. put_step is a dedup stride that assumes replication:
        # only 1/put_step of the identical-across-TP keys are stored. Under DCP
        # that assumption is wrong -- the stride would drop ~(1 - 1/put_step) of
        # each rank's own unique shard, so cross-instance (PD) consumers miss
        # most of the KV (observed ~7% hit at TP8/DCP8, i.e. ~1/8). Shrink the
        # stride by dcp_size so every rank stores its full shard. head_or_tp_rank
        # is left unchanged: the `dcp=S:R` coordinate already separates shards,
        # and both P and D derive it identically.
        if self.dcp_size > 1 and self.put_step > 1:
            self.put_step = max(1, self.put_step // self.dcp_size)

        # CLIENT_RANKS (issue #111): converge store-side dfkv clients onto a
        # subset of ranks when the KV object is TP-replicated. Layout-clamped,
        # never rejects (one fleet-wide env template must be safe everywhere).
        replicated = (
            self.use_mla
            and self.put_step == self.tp_size
            and self.dcp_size <= 1
            and getattr(self, "pcp_size", 1) <= 1
        )
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
            tp_rank=-1 if replicated else self.head_or_tp_rank,
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
        # The automatic binary namespace captures model revision, runtime
        # layout contract, cache dtype/block geometry and topology. Pool/rank
        # scope remains in each object key. key_namespace is an explicit
        # byte-compatibility override across runtimes.
        _tcfg.require_ring_endpoint(extra.get("members", ""), mds_endpoints)
        model_identity = str(model_config.model)
        if not extra.get("key_namespace"):
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
            VLLM_RAW_V1,
            extra.get("key_namespace"),
            tenant_id=str(extra.get("tenant_id", "default")),
            model_revision=str(_model_revision),
            dtype=_cache_dtype,
            block_tokens=max(1, self.block_size),
            layer_count=max(1, self.num_layers),
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            layout_fields={
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

        for layer_name, value in kv_caches.items():
            cache = _repr_tensor(value)
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
                self.put_step,
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
            client_provider=self._ensure_client_for_load,
            queue_capacity=getattr(
                self, "transfer_queue_capacity", DEFAULT_TRANSFER_QUEUE_CAPACITY
            ),
        )
        self.kv_recv_thread.start()
        ready_event_recving.wait()

    def start_load_kv(
        self,
        metadata: DfkvStoreConnectorMetadata,
    ):
        """Pre-forward hook: fence preempted requests' in-flight saves.

        Loads/stores are issued in get_finished() for overlap; the ONLY work
        here is the preemption fence, and it must be here. The scheduler
        frees a preempted request's GPU blocks with no connector hook
        (request_finished()'s delay-free covers only the normal finish path)
        and can hand them to another request scheduled in this very step.
        This hook runs before this step's forward pass writes into those
        blocks, so it is the last point where the race can still be closed:
        drop the queued-not-started saves, then wait out the one save the
        send thread may currently be executing (an uncancellable RDMA read
        of the request's old blocks). Without the fence that read races the
        new request's prefill writes and stores mixed bytes under a
        content-addressed key -- and the save path's exists-dedup then
        prevents the correct bytes from ever replacing them (persistent
        cache poison, spread across instances by PD reuse).
        """
        if self.kv_send_thread is None or not metadata.preempted_req_ids:
            return
        # Drop queued entries first (the dequeue gate re-checks per entry),
        # then join whichever entry is already executing.
        for req_id in metadata.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)
        for req_id in metadata.preempted_req_ids:
            wait_start = time.perf_counter()
            if not self.kv_send_thread.wait_for_inflight_put(req_id):
                logger.error(
                    "preemption fence timed out waiting for in-flight save "
                    "of request %s (%.1fs); its old GPU blocks may be read "
                    "while reused -- possible poisoned store keys",
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
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        with self._kv_connector_stats_lock:
            self.kv_connector_stats.record_operation(
                operation=operation,
                duration_seconds=duration_seconds,
                num_keys=num_keys,
                num_bytes=num_bytes,
                status=status,
                num_failed_keys=num_failed_keys,
            )

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

        Checks across all TP ranks and PP ranks.
        """
        if not block_hashes or token_len <= 0:
            return 0

        # Build per-(group, hash) candidate keys expanded across TP/PP.
        # candidate_meta[i] is the (group_id, hash_bytes) for candidate_keys[i].
        candidate_keys: list[bytes] = []
        candidate_meta: list[tuple[int, bytes]] = []
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
        for g_idx, db in enumerate(self.token_dbs):
            spec_block_size = db.block_size
            mask = store_masks[g_idx]
            group_hashes = self.coord.block_hashes_for_spec(
                block_hashes, self._kv_cache_groups[g_idx].kv_cache_spec
            )
            for chunk_id, h in enumerate(group_hashes):
                start_idx = chunk_id * spec_block_size
                if start_idx >= token_len:
                    break
                if chunk_id >= len(mask) or not mask[chunk_id]:
                    continue
                for tp in range(tp_count):
                    for pp in range(self.pp_size):
                        md = dataclasses.replace(db.metadata, tp_rank=tp, pp_rank=pp)
                        # Probe scatter group 0 so lookup agrees with the exact
                        # SAVE/LOAD object key.
                        candidate_keys.append(
                            _sg_group_key(PoolKey(md, h.hex()).to_bytes(), 0,
                                          _sg_segs_of(self.client))
                        )
                        candidate_meta.append((g_idx, bytes(h)))

        if not candidate_keys:
            return 0

        lookup_start = time.perf_counter()
        try:
            res = self.client.batch_exist(candidate_keys)
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
            )
        except Exception as e:
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
                status="error",
                num_failed_keys=len(candidate_keys),
            )
            logger.error("Remote connection failed in lookup: %s", e)
            return 0

        # A (group, hash) is "present" only when every TP*PP rank has it.
        expected_per_key = max(1, tp_count * self.pp_size)
        present_count: dict[tuple[int, bytes], int] = {}
        for gh, exists in zip(candidate_meta, res, strict=True):
            if exists == 1:
                present_count[gh] = present_count.get(gh, 0) + 1
        exists_set = {gh for gh, c in present_count.items() if c >= expected_per_key}

        _masks, hit_length = self.coord.find_longest_cache_hit(
            block_hashes, token_len, ExternalCachedBlockPool(exists_set)
        )
        logger.debug(
            "dfkv lookup: token_len=%d candidates=%d present=%d -> hit_length=%d",
            token_len, len(candidate_keys), len(exists_set), hit_length,
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
        self.decoder = MsgpackDecoder()
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
                msg_type = bytes(all_frames[0])

                if msg_type == LOOKUP_MSG:
                    token_len = int.from_bytes(all_frames[1], byteorder="big")
                    hash_frames = all_frames[2:]
                    hashes_str = self.decoder.decode(hash_frames)
                    block_hashes = [BlockHash(bytes.fromhex(s)) for s in hashes_str]
                    result = self.store_worker.lookup(token_len, block_hashes)
                    self.socket.send(result.to_bytes(4, "big"))

                elif msg_type == RESET_MSG:
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

    Routes both prefix-cache lookups and admin commands (currently:
    ``reset``) to ``LookupKeyServer`` on worker rank 0. The first frame
    of every request is a named tag from ``protocol.py``.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.encoder = MsgpackEncoder()
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

    def lookup(self, token_len: int, block_hashes: list[BlockHash]) -> int:
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        token_len_bytes = token_len.to_bytes(4, byteorder="big")
        all_frames = [LOOKUP_MSG, token_len_bytes] + list(hash_frames)
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def reset(self) -> bool:
        """Trigger a global store wipe on worker rank 0.

        Ordering assumption: caller MUST ensure no in-flight Dfkv
        lookups or transfers when invoking reset. In RL workflows this
        holds naturally at the step boundary after weight updates and
        rollout drain. Returns True on ACK, False on NACK.

        dfkv: dfkv has no remove_all primitive, so the worker drains the
        send queue and NACKs; this returns False.
        """
        self.socket.send(RESET_MSG)
        resp = self.socket.recv()
        return bytes(resp) == RESP_OK

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
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
