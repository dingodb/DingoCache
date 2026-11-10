# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Scheduler-side logic for DfkvStoreConnector."""

import time
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from ._determinism import ensure_deterministic_block_hashing
from .coordinator import partial_hash_hits_enabled
from .data import (
    LoadSpec,
    DfkvStoreConnectorMetadata,
    ReqMeta,
    RequestTracker,
    requires_request_level_loads,
)
from .transfer_protocol import failed_requests_from_output
from .worker import (
    LookupKeyClient,
)

logger = init_logger(__name__)

# kv_transfer_params key carrying the per-request retention TTL (seconds),
# injected by the gateway; absent or 0 = best-effort.
RETENTION_TTL_PARAM = "dfkv_retention_ttl_s"
USAGE_PARAM = "dfkv_cache_usage"


def _requested_retention_ttl(request: Request) -> int:
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return 0
    value = params.get(RETENTION_TTL_PARAM, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return 0
    return min(value, 0xFFFFFFFF)


def _new_req_prefill_tokens(request: NewRequestData) -> list[int]:
    """Tokens this prefill will compute KV for.

    Under the v2 model runner, resumed-from-preemption requests appear in
    ``scheduled_new_reqs`` with ``prefill_token_ids`` set to the request's full
    token list (prompt + previously-generated). For all other cases this falls
    back to the original prompt.
    """
    if request.prefill_token_ids is not None:
        return request.prefill_token_ids
    assert request.prompt_token_ids is not None
    return request.prompt_token_ids


# Re-emit the misaligned state hand-off warning at most this often.
_MISALIGNED_STATE_WARN_INTERVAL_S = 600.0


class DfkvStoreScheduler:
    """Scheduler-side component for DfkvStoreConnector."""

    # Recurrent-state boundaries the engine handed off off the scheduler
    # block grid (never saveable), and when that was last reported.
    _misaligned_state_boundaries = 0
    _misaligned_state_warned_at: float | None = None
    # vLLM core lands prefix hits on hash blocks inside a scheduler block.
    _partial_hash_hits = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        assert vllm_config.kv_transfer_config is not None
        # Store keys embed block_hashes: refuse to start with process-local
        # hashing (silent 0% cross-instance/cross-restart hit rate otherwise).
        ensure_deterministic_block_hashing(vllm_config.cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        self.load_async = extra_config.get("load_async", True)
        if not isinstance(self.load_async, bool):
            raise ValueError("dfkv connector: load_async must be a boolean")
        self.request_level_loads = requires_request_level_loads(kv_cache_config)
        self.lookup_async = extra_config.get("lookup_async", False)
        self.client = LookupKeyClient(vllm_config)
        self._closed = False

        # Align with the engine's own scheduler_block_size and hash_block_size.
        self._block_size, self._hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self._partial_hash_hits = partial_hash_hits_enabled(
            kv_cache_config, self._block_size
        )

        # Per-request state
        self.load_specs: dict[str, LoadSpec] = {}  # to be loaded
        self._request_trackers: dict[str, RequestTracker] = {}  # scheduled new requests
        self._preempted_req_ids: set[str] = set()  # preempted requests
        self._unfinished_requests: dict[str, tuple[Request, tuple[list[int], ...]]] = {}
        self._unfinished_request_ids: set[str] = set()
        self._allocated_req_ids: set[str] = set()
        self._failed_load_req_ids: set[str] = set()
        # req_id -> (requested_ttl_s, applied_ttl_s, cached_tokens) for TTL
        # requests; the applied TTL comes back from the worker's lookup.
        self._retention: dict[str, tuple[int, int, int]] = {}

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check for an external KV cache hit.

        Returns ``(None, False)`` while an asynchronous lookup is pending so
        vLLM retries the request on a later scheduler step.
        """
        if (
            request.request_id in self._failed_load_req_ids
            or getattr(request, "skip_reading_prefix_cache", False)
        ):
            self.client.discard(request.request_id)
            self.load_specs.pop(request.request_id, None)
            return 0, False

        # Leave the sampling tail before lookup: shortening an already
        # validated hit can select different recurrent-state checkpoint keys.
        # Partial hash hits land on hash blocks inside a scheduler block.
        align = self._hash_block_size if self._partial_hash_hits else self._block_size
        token_len = (request.num_tokens - 1) // align * align
        requested_ttl = _requested_retention_ttl(request)
        # A local partial hit ends inside a block that core copies on write;
        # an external load into the same block would race that copy.
        if token_len < align or (
            self._partial_hash_hits and num_computed_tokens % self._block_size
        ):
            if requested_ttl:
                self._retention[request.request_id] = (
                    requested_ttl, 0, num_computed_tokens)
            return 0, False

        _lk0 = time.perf_counter()
        lookup = self.client.lookup(
            request.request_id,
            token_len,
            request.block_hashes,
            non_block=self.lookup_async,
            retain_ttl_s=requested_ttl,
        )
        if lookup is None:
            return None, False
        num_external_hit_tokens, applied_ttl = lookup
        if requested_ttl:
            self._retention[request.request_id] = (
                requested_ttl,
                applied_ttl if self.kv_role != "kv_consumer" else 0,
                max(num_computed_tokens, num_external_hit_tokens),
            )
        logger.debug(
            "dfkv matched: req=%s tokens=%d computed=%d lookup_ms=%.1f hit_tokens=%d",
            request.request_id, request.num_tokens, num_computed_tokens,
            (time.perf_counter() - _lk0) * 1000.0, num_external_hit_tokens,
        )

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.debug(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        if need_to_allocate <= 0:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        # Request-level failure recovery only applies to waiting requests.
        # Parking admission is independent of whether the worker overlaps I/O.
        return need_to_allocate, self.load_async or self.request_level_loads

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        """Update state after block allocation."""
        local_block_ids = blocks.get_block_ids()

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)
        self._allocated_req_ids.add(request.request_id)

        if request.request_id not in self.load_specs:
            return

        if num_external_tokens == 0:
            self.load_specs[request.request_id].can_load = False
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        """Prevent a failed remote source from being re-admitted on retry."""
        if not self.request_level_loads:
            return
        for req_id in failed_requests_from_output(output):
            # A receive can finish after an aborted request's final metadata.
            # Such an output must not recreate scheduler-side request state.
            if req_id not in self._unfinished_request_ids:
                continue
            self._failed_load_req_ids.add(req_id)
            self.client.discard(req_id)
            self.load_specs.pop(req_id, None)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)
            self._allocated_req_ids.discard(req_id)
            # Upstream frees failed-load blocks and re-admits without emitting
            # a new preemption. MRV1 may use scheduled_cached_reqs on that retry;
            # rebuild its tracker from the new allocation and computed prefix,
            # never from the failed load's advertised token count/block table.
            self._preempted_req_ids.add(req_id)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build connector metadata for this scheduler step."""
        force_skip_save = self.kv_role == "kv_consumer"

        for finished_req_id in scheduler_output.finished_req_ids:
            self.client.discard(finished_req_id)
            self.load_specs.pop(finished_req_id, None)
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)
            self._preempted_req_ids.discard(finished_req_id)
            self._failed_load_req_ids.discard(finished_req_id)
            self._retention.pop(finished_req_id, None)

        preempted_ids = scheduler_output.preempted_req_ids or set()
        self._preempted_req_ids.update(preempted_ids)
        for req_id in preempted_ids:
            self.client.discard(req_id)
            # Allocation callbacks can already describe a same-step resume.
            # Discard only the preemption's old allocation/load state.
            if req_id not in self._allocated_req_ids:
                self.load_specs.pop(req_id, None)
                self._unfinished_requests.pop(req_id, None)
            if request_tracker := self._request_trackers.get(req_id):
                request_tracker.reset()

        meta = DfkvStoreConnectorMetadata(
            self._unfinished_request_ids,
            preempted_ids,
        )

        # Handle new requests
        for request in scheduler_output.scheduled_new_reqs:
            self._preempted_req_ids.discard(request.req_id)
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            assert request.req_id in self._unfinished_requests
            request_real, allocated_block_ids = self._unfinished_requests[request.req_id]
            # NewRequestData.block_ids contains only blocks allocated in this
            # scheduler step. Keep the complete table captured after allocation
            # so later chunks can address from sequence offset zero.
            unfolded_block_ids = tuple(
                group_ids.copy() for group_ids in allocated_block_ids
            )

            prefill_tokens = _new_req_prefill_tokens(request)
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=0,
                token_ids=prefill_tokens[:num_tokens_to_compute],
                prefill_end_tokens=len(prefill_tokens),
            )
            self._request_trackers[request.req_id] = request_tracker

            last_chunk_tokens_num = (
                len(prefill_tokens) // self._block_size * self._block_size
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                block_hashes=request_real.block_hashes,
                is_last_chunk=(request_tracker.token_len >= last_chunk_tokens_num),
                num_prompt_tokens=self._tail_prompt_tokens(request_real),
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        # Handle cached (running, or MRV1 resumed-from-preemption) requests.
        # Consumer-only mode still needs LOAD metadata; skip_save only suppresses
        # the independent SAVE side of each request.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            new_block_ids = cached_reqs.new_block_ids[i]

            req_meta = None
            if req_id in self._preempted_req_ids:
                # Resumed after preemption
                self._preempted_req_ids.discard(req_id)
                load_spec = self.load_specs.pop(req_id, None)
                request_real, allocated_block_ids = self._unfinished_requests[req_id]
                num_tokens_to_compute = (
                    request_real.num_computed_tokens
                    + scheduler_output.num_scheduled_tokens[req_id]
                )
                # On resume, the request re-prefills prompt + previously
                # generated tokens (all_token_ids).
                prefill_tokens = list(request_real.all_token_ids)
                request_tracker = RequestTracker(
                    req_id=req_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids=tuple(b.copy() for b in allocated_block_ids),
                    num_saved_tokens=0,
                    token_ids=prefill_tokens[:num_tokens_to_compute].copy(),
                    prefill_end_tokens=len(prefill_tokens),
                )
                self._request_trackers[req_id] = request_tracker

                last_chunk_tokens_num = (
                    len(prefill_tokens) // self._block_size * self._block_size
                )
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=force_skip_save,
                    block_hashes=request_real.block_hashes,
                    is_last_chunk=(
                        request_tracker.token_len >= last_chunk_tokens_num
                    ),
                    num_prompt_tokens=self._tail_prompt_tokens(request_real),
                )
            else:
                # Decode/chunked request
                request_tracker = self._request_trackers[req_id]
                num_computed_token = cached_reqs.num_computed_tokens[i]
                # Use the tracker's snapshot of the prefill range so resumed
                # requests keep saving past the original prompt boundary.
                prefill_end = request_tracker.prefill_end_tokens
                if num_computed_token >= prefill_end:
                    continue
                req_tuple = self._unfinished_requests.get(req_id)
                if req_tuple is None:
                    raise ValueError(
                        f"Request {req_id} is not in _unfinished_requests"
                    )
                unfinished_req = req_tuple[0]
                num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                has_new_blocks = bool(new_block_ids) and any(new_block_ids)
                if num_new_tokens <= 0 and not has_new_blocks:
                    continue
                # Prefill can advance within already allocated blocks. Account
                # every scheduled chunk, even when new_block_ids is None, or
                # SAVE can lag behind and read retired/reused state pages.
                # Use scheduler progress rather than accumulating tracker state;
                # never include decode/speculative tokens in the prefill SAVE.
                old_len = request_tracker.token_len
                new_len = min(
                    num_computed_token + num_new_tokens,
                    prefill_end,
                    len(unfinished_req.all_token_ids),
                )
                if old_len < num_computed_token:
                    logger.warning(
                        "Resynchronizing DFKV tracker for request %s: "
                        "tracker_len=%d computed=%d",
                        req_id,
                        old_len,
                        num_computed_token,
                    )
                token_ids = request_tracker.token_ids
                if (
                    new_len < old_len
                    or token_ids is None
                    or len(token_ids) != old_len
                ):
                    token_ids = list(unfinished_req.all_token_ids[:new_len])
                    request_tracker.token_ids = token_ids
                elif new_len > old_len:
                    token_ids.extend(
                        unfinished_req.all_token_ids[old_len:new_len]
                    )
                request_tracker.token_len = new_len
                if has_new_blocks:
                    request_tracker.update(new_block_ids)

                last_chunk_tokens_num = (
                    prefill_end // self._block_size * self._block_size
                )
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=None,
                    skip_save=force_skip_save,
                    block_hashes=unfinished_req.block_hashes,
                    is_last_chunk=(
                        request_tracker.token_len >= last_chunk_tokens_num
                    ),
                    num_prompt_tokens=self._tail_prompt_tokens(unfinished_req),
                )

            if req_meta is not None:
                meta.add_request(req_meta)

        # Handle requests with pending load specs not yet scheduled
        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (
            unfinished_req,
            block_ids,
        ) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                load_spec = self.load_specs.pop(request_id, None)
                # MultiConnector may reject this connector's proposed load in
                # favor of another connector (can_load=False). Only the chosen
                # connector may issue it; saving here would read blocks that
                # the other connector is still filling.
                if load_spec is None or not load_spec.can_load:
                    continue
                num_tokens_to_compute = load_spec.kvpool_cached_tokens
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids=block_ids,
                    num_saved_tokens=0,
                )
                self._request_trackers[request_id] = request_tracker
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=unfinished_req.block_hashes,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)

        self._attach_state_blocks(meta, scheduler_output)
        self._allocated_req_ids.clear()
        for req_meta in meta.requests:
            retention = self._retention.get(req_meta.req_id)
            if retention is not None:
                req_meta.retain_ttl_s = retention[1]
        return meta

    def _tail_prompt_tokens(self, request: Request) -> int | None:
        """Prompt length whose partial tail a SAVE may publish (partial hash
        hits only)."""
        return request.num_prompt_tokens if self._partial_hash_hits else None

    def _attach_state_blocks(
        self, meta: DfkvStoreConnectorMetadata, scheduler_output: SchedulerOutput
    ) -> None:
        """Give each SAVE the exact recurrent-state blocks vLLM committed.

        A Mamba "align" block table is not append-only: interior states are
        nulled, speculative blocks relocate, and a prompt's last chunk keeps
        its running state in the slot of the boundary it crossed unless the
        engine exported a checkpoint there. Engines that hand off committed
        boundary states (``kv_connector_block_state``) name the exact block
        per boundary; the hand-off is drained every step and the state is
        written by this step's forward, which the worker fences before reading.

        With partial hash hits, a state off the scheduler block grid (the
        prompt's checkpoint) is saved even in a step without a SAVE, by an
        offload-only SAVE (``token_len_chunk=0``) like Mooncake's.
        """
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        if block_state is None:
            return
        offloads = block_state.boundary_state_offloads
        misaligned = 0
        saving: set[str] = set()
        for req_meta in meta.requests:
            if not req_meta.can_save:
                continue
            saving.add(req_meta.req_id)
            blocks: dict[int, dict[int, int]] = {}
            for group_id, block_id, boundary in offloads.get(req_meta.req_id, ()):
                blocks.setdefault(group_id, {})[boundary] = block_id
                if boundary % self._block_size:
                    misaligned += 1
            req_meta.state_blocks = blocks
        if not self._partial_hash_hits:
            if misaligned:
                self._warn_misaligned_state_boundaries(misaligned)
            return
        if self.kv_role == "kv_consumer":
            return
        for req_id, entries in offloads.items():
            tracker = self._request_trackers.get(req_id)
            req_tuple = self._unfinished_requests.get(req_id)
            if req_id in saving:
                continue
            if (
                tracker is None
                or req_tuple is None
                or req_id in self._preempted_req_ids
            ):
                # Finished or preempted within this step: its blocks are
                # going away, so the offload is dropped.
                continue
            blocks = {}
            for group_id, block_id, boundary in entries:
                if boundary % self._block_size:
                    blocks.setdefault(group_id, {})[boundary] = block_id
            if not blocks:
                continue
            tracker.has_pending_offload = True
            meta.add_request(ReqMeta(
                req_id=req_id,
                token_len_chunk=0,
                block_ids=tracker.allocated_block_ids,
                block_hashes=req_tuple[0].block_hashes,
                can_save=True,
                state_blocks=blocks,
            ))

    def _warn_misaligned_state_boundaries(self, count: int) -> None:
        """Report state hand-offs that can never reach the external cache.

        Without partial hash hits, saves and lookups only happen on
        ``scheduler_block_size`` multiples (the LCM of the group block sizes).
        With decode context parallelism that LCM exceeds the recurrent-state
        block, so a state handed off at, say, a 256-aligned prompt end is
        never saved and hybrid prefixes never hit. Engines that align the
        hand-off to the LCM, or that export hash-block hit alignment, never
        trigger this.
        """
        self._misaligned_state_boundaries += count
        now = time.monotonic()
        last = self._misaligned_state_warned_at
        if last is not None and now - last < _MISALIGNED_STATE_WARN_INTERVAL_S:
            return
        self._misaligned_state_warned_at = now
        logger.warning(
            "dfkv: %d recurrent-state boundaries handed off off the %d-token "
            "scheduler block grid so far; they cannot be saved, so hybrid "
            "(linear-attention) prefixes will not hit the external cache. "
            "This happens with decode context parallelism unless the engine "
            "aligns state hand-off to the scheduler block size.",
            self._misaligned_state_boundaries,
            self._block_size,
        )

    def request_finished(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Determine whether to delay freeing blocks for async save."""
        self.client.discard(request.request_id)
        self._failed_load_req_ids.discard(request.request_id)
        self.load_specs.pop(request.request_id, None)
        usage = self._retention_usage(request)
        if self.kv_role == "kv_consumer":
            return False, usage
        tracker = self._request_trackers.get(request.request_id)
        # Missing tracker can happen when the request is aborted before the
        # connector observes the normal finished lifecycle or is preempted
        # before finishing. A partial tail or offload-only SAVE reads blocks
        # even when no scheduler block was saved (a prompt shorter than one).
        if tracker is None or not (
            tracker.num_saved_tokens > 0
            or tracker.partial_tail_sent
            or tracker.has_pending_offload
        ):
            return False, usage
        total_blocks = sum(len(g) for g in block_ids)
        delay_free_blocks = total_blocks > 0
        if delay_free_blocks:
            logger.debug(
                "Delaying free of %d blocks for request %s",
                total_blocks,
                request.request_id,
            )
        return delay_free_blocks, usage

    def _retention_usage(self, request: Request) -> dict[str, Any] | None:
        """Usage for a TTL request, returned in the response's
        kv_transfer_params. ``cache_write_tokens`` is the block-aligned part of
        the prompt not served from cache (what the gateway bills as a write);
        ``applied_ttl_s`` is 0 when the writes stayed best-effort."""
        retention = self._retention.pop(request.request_id, None)
        if retention is None:
            return None
        _, applied_ttl, cached = retention
        prompt = getattr(request, "num_prompt_tokens", 0) or 0
        aligned = prompt // self._block_size * self._block_size
        return {
            USAGE_PARAM: {
                "cached_tokens": int(cached),
                "cache_write_tokens": max(0, aligned - int(cached)),
                "applied_ttl_s": int(applied_ttl),
            }
        }

    def close(self) -> None:
        """Close the scheduler-side lookup channel once."""
        if self._closed:
            return
        self._closed = True
        self.client.close()

    def reset_store(self) -> bool:
        """Request a global store reset via worker rank 0.

        Routes through the existing LookupKey ZMQ admin channel to worker
        rank 0, which owns the ``DfkvDeviceClient`` handle.

        Ordering assumption: caller (typically
        ``Scheduler.reset_connector_cache``, invoked via
        ``reset_prefix_cache(reset_connector=True)``) MUST ensure no
        in-flight Dfkv lookups or transfers. For RL workflows this is
        satisfied at the step boundary after weight updates and rollout
        drain. Violating this can allow stale KV to be served on the next
        request, defeating the hard-reset guarantee.

        dfkv: dfkv has no remove_all primitive, so the worker drains its
        send queue and NACKs; this returns False (best-effort, never raises
        into the scheduler hot path).

        Returns True on ACK from worker, False on NACK or RPC error.
        """
        try:
            ok = self.client.reset()
            if ok:
                logger.info("Dfkv store reset succeeded.")
            else:
                # dfkv: expected -- no global wipe primitive, queue was drained.
                logger.warning("Dfkv store reset returned NACK from worker.")
            return ok
        except Exception as e:
            logger.error("Dfkv reset_store RPC failed: %s", e)
            return False
