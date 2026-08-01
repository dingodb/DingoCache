# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Data classes for DfkvStoreConnector."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
)

logger = init_logger(__name__)


@dataclass
class KeyMetadata:
    """Metadata for constructing pool keys."""

    model_name: str
    tp_rank: int
    pcp_rank: int
    dcp_rank: int
    pp_rank: int
    group_id: int = 0


@dataclass(order=True)
class PoolKey:
    """Key for addressing KV cache blocks in the distributed store."""

    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.key_metadata.group_id,
                self.chunk_hash,
            )
        )

    def to_string(self) -> str:
        return (
            f"{self.key_metadata.model_name}"
            f"@tp_rank:{self.key_metadata.tp_rank}"
            f"@pcp{self.key_metadata.pcp_rank}"
            f"@dcp{self.key_metadata.dcp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}"
            f"@group:{self.key_metadata.group_id}"
            f"@{self.chunk_hash}"
        )


class ChunkedTokenDatabase:
    """Maps token positions to store keys and GPU memory addresses."""

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int | None = None,
    ):
        self.metadata = metadata
        self.block_size = block_size
        self.hash_block_size = hash_block_size or block_size
        if self.block_size % self.hash_block_size != 0:
            raise ValueError(
                f"block_size ({self.block_size}) must be a multiple of "
                f"hash_block_size ({self.hash_block_size})"
            )
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []
        # Per-layer segment layout: (base_addr, block_stride, block_content).
        # block_stride is the layer's dim0 byte stride; block_content is the
        # trailing-dense byte run that actually holds this layer's tokens
        # within one block. They differ for interleaved pools (DeepSeek-V4
        # hybrid: all layers live in one pool, each block is a 1MiB page with
        # per-layer slots inside). A fully-dense layer has stride == content
        # and stays packed into one segment per chunk (legacy behavior).
        self._seg_layout: list[tuple[int, int, int]] | None = None

    def _make_key_by_hash(self, chunk_hash: str) -> PoolKey:
        return PoolKey(self.metadata, chunk_hash)

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        self.kv_caches_base_addr = kv_caches_base_addr

    def set_block_len(self, block_len: list[int]):
        self.block_len = block_len

    def set_seg_layout(self, seg_layout: list[tuple[int, int, int]]):
        """Install exact per-layer (base, block_stride, block_content).

        When set, prepare_value derives addresses from this layout instead of
        the legacy flat block_len table, splitting interleaved layers into
        one segment per block. With stride == content the entry stays packed
        exactly like the legacy path.
        """
        for base, stride, content in seg_layout:
            if stride <= 0 or content <= 0 or content > stride:
                raise ValueError(
                    f"invalid seg layout entry: base={base:#x} "
                    f"stride={stride} content={content}"
                )
        self._seg_layout = seg_layout

    def prepare_value(
        self, start: int, end: int, block_ids: list[int]
    ) -> tuple[list[int], list[int], int]:
        """Compute memory addresses and sizes for a token range.

        Returns:
            (addr_list, size_list, block_id)
        """
        addr_list = []
        size_list = []
        block_id = block_ids[start // self.block_size]
        nblocks = cdiv(end - start, self.block_size)
        if self._seg_layout is not None:
            for base, stride, content in self._seg_layout:
                if stride == content:
                    # Fully dense layer: one packed segment for the chunk.
                    addr_list.append(base + block_id * stride)
                    size_list.append(content * nblocks)
                else:
                    # Interleaved pool layer: its per-block slot is not
                    # contiguous with the next block, so one segment per
                    # block is required. Anything wider would read/write
                    # bytes owned by other layers (or by the NEXT chunk's
                    # first block) and corrupt them on load.
                    for i in range(nblocks):
                        addr_list.append(base + (block_id + i) * stride)
                        size_list.append(content)
            return addr_list, size_list, block_id
        length = len(self.block_len)
        for index, base_addr in enumerate(self.kv_caches_base_addr):
            addr = base_addr + block_id * self.block_len[index % length]
            assert (end - start) % self.block_size == 0
            size = self.block_len[index % length] * nblocks
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list, block_id

    def process_tokens(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        mask_num: int = 0,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        """Process tokens and yield (start_idx, end_idx, pool_key) tuples.

        Args:
            token_len: Total number of tokens.
            block_hashes: Block hashes computed at ``hash_block_size`` granularity.
                When ``block_size > hash_block_size`` consecutive hashes are merged
                up to the group's ``block_size`` via ``BlockHashListWithBlockSize``.
            mask_num: Number of tokens to skip from the beginning.
        """
        if not block_hashes:
            return
        if self.block_size == self.hash_block_size:
            chunk_hashes: Iterable[BlockHash] = block_hashes
        else:
            chunk_hashes = BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, self.block_size
            )
        for chunk_id, h in enumerate(chunk_hashes):
            start_idx = chunk_id * self.block_size
            if start_idx >= token_len:
                break
            end_idx = min(start_idx + self.block_size, token_len)
            if start_idx < mask_num:
                continue
            yield start_idx, end_idx, self._make_key_by_hash(h.hex())


@dataclass
class LoadSpec:
    """Specification for loading KV cache from external store."""

    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    can_load: bool
    token_len: int = 0


@dataclass
class RequestTracker:
    """Tracks per-request state across scheduler ticks."""

    req_id: str
    token_len: int
    allocated_block_ids: tuple[list[int], ...]
    num_saved_tokens: int = 0
    token_ids: list[int] | None = None
    # Snapshot of the prefill range length at tracker creation time.
    # For a fresh request this is len(prompt). For a resumed-from-preemption
    # request it includes previously-generated tokens, which are re-prefilled.
    prefill_end_tokens: int = 0

    def reset(self) -> None:
        self.token_len = 0
        self.allocated_block_ids = ()
        self.num_saved_tokens = 0
        self.token_ids = None
        self.prefill_end_tokens = 0

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
    ) -> None:
        # Backward-compat: accept a single list (broadcast to single group).
        if isinstance(new_block_ids, list):
            new_block_ids = (new_block_ids,)
        if len(new_block_ids) != len(self.allocated_block_ids):
            raise ValueError(
                f"Group count mismatch: tracker has "
                f"{len(self.allocated_block_ids)} groups, update has "
                f"{len(new_block_ids)}"
            )
        for existing, new in zip(self.allocated_block_ids, new_block_ids, strict=True):
            if new:
                existing.extend(new)


@dataclass
class ReqMeta:
    """Per-request metadata for store put/get operations."""

    req_id: str
    token_len_chunk: int
    block_ids: tuple[list[int], ...]
    block_hashes: list[BlockHash]

    can_save: bool | None = None
    load_spec: LoadSpec | None = None
    is_last_chunk: bool | None = None
    current_event: torch.cuda.Event | None = None

    token_ids: list[int] | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
    ) -> "ReqMeta | None":
        """Create ReqMeta from a RequestTracker."""
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        chunk_boundary = cdiv(tracker.num_saved_tokens + 1, block_size) * block_size
        num_tokens_to_save = input_token_len // block_size * block_size

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        # A ReqMeta must never carry both a save AND a load.
        # The save would also be wasted work — the bytes are being looked up
        # in the store right now. Later cached_reqs steps save new tokens
        # normally.
        if load_spec is not None and load_spec.can_load:
            skip_save = True
        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        token_ids = None
        if tracker.token_ids:
            token_ids = tracker.token_ids

        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            load_spec = None

        logger.debug(
            "request:%s, meta save spec:%s, meta load spec:%s",
            tracker.req_id,
            not skip_save,
            load_spec,
        )
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
        )


class DfkvStoreConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker."""

    def __init__(
        self,
        unfinished_request_ids: set[str],
        preempted_req_ids: set[str],
    ):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)
