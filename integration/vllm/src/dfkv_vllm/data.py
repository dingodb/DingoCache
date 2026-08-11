# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Data classes for DfkvStoreConnector."""

import hashlib

from collections.abc import Iterable, Sequence
from dfkv_common import pool_key
from dataclasses import dataclass
from itertools import product

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

# Source-controlled object-layout discriminator. It is framed into every pool
# key and also binds the native namespace, so sg-v1 and multiwr-v2 objects can
# never collide or be read through a compatibility fallback.
VLLM_MULTIWR_V2 = b"vllm-multiwr-v2"

def key_diagnostic_label(key: bytes) -> str:
    """Return the standard non-reversible diagnostic label for a store key."""
    try:
        return f"len={len(key)} sha256={hashlib.sha256(key).hexdigest()[:16]}"
    except Exception:
        # Diagnostics must never turn a successfully transferred key into an
        # application-visible failure.
        return "<key unavailable>"


def split_block_contiguous_runs(
    shape: Sequence[int],
    strides: Sequence[int],
    element_size: int,
) -> tuple[int, list[tuple[int, int]]]:
    """Return ``(block_stride, [(offset, size), ...])`` in bytes.

    Dimension 0 indexes KV blocks. Expand padded or transposed in-block
    dimensions into deterministic contiguous runs; overlapping views fail
    closed instead of transferring the whole block stride.
    """
    if len(shape) != len(strides) or not shape:
        raise ValueError("shape and strides must have the same non-zero rank")
    if element_size <= 0 or any(int(n) <= 0 for n in shape):
        raise ValueError("shape dimensions and element_size must be positive")
    if any(int(s) < 0 for s in strides):
        raise ValueError("negative KV-cache strides are not supported")

    block_stride = int(strides[0]) * element_size
    if block_stride <= 0:
        raise ValueError("KV-cache block stride must be positive")

    run_elems = 1
    run_start = len(shape)
    for dim in range(len(shape) - 1, 0, -1):
        if int(strides[dim]) != run_elems:
            break
        run_elems *= int(shape[dim])
        run_start = dim

    run_size = run_elems * element_size
    prefix_ranges = [range(int(shape[d])) for d in range(1, run_start)]
    offsets: list[int] = []
    seen_offsets: set[int] = set()
    for indices in product(*prefix_ranges):
        offset = sum(
            index * int(strides[dim])
            for dim, index in enumerate(indices, start=1)
        ) * element_size
        if offset not in seen_offsets:
            seen_offsets.add(offset)
            offsets.append(offset)
    if not offsets:
        offsets.append(0)

    runs = [(offset, run_size) for offset in offsets]
    previous_end = 0
    for offset, size in sorted(runs):
        if offset < previous_end or offset + size > block_stride:
            raise ValueError(
                "KV-cache in-block runs overlap or exceed block stride: "
                f"offset={offset} size={size} block_stride={block_stride}"
            )
        previous_end = offset + size
    return block_stride, runs



@dataclass
class KeyMetadata:
    """Metadata for constructing pool keys."""

    model_name: str
    dp_size: int
    dp_rank: int
    tp_size: int
    tp_rank: int
    pcp_size: int
    pcp_rank: int
    dcp_size: int
    dcp_rank: int
    pp_size: int
    pp_rank: int
    group_id: int = 0


@dataclass(order=True)
class PoolKey:
    """Key for addressing KV cache blocks in the distributed store."""

    key_metadata: KeyMetadata
    chunk_hash: str | bytes

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.dp_size,
                self.key_metadata.dp_rank,
                self.key_metadata.tp_size,
                self.key_metadata.tp_rank,
                self.key_metadata.pcp_size,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_size,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_size,
                self.key_metadata.pp_rank,
                self.key_metadata.group_id,
                self.chunk_hash,
            )
        )

    def to_bytes(self) -> bytes:
        return pool_key(
            self.chunk_hash,
            pool="kv",
            dp_size=self.key_metadata.dp_size,
            dp_rank=self.key_metadata.dp_rank,
            tp_size=self.key_metadata.tp_size,
            tp_rank=self.key_metadata.tp_rank,
            pcp_size=self.key_metadata.pcp_size,
            pcp_rank=self.key_metadata.pcp_rank,
            dcp_size=self.key_metadata.dcp_size,
            dcp_rank=self.key_metadata.dcp_rank,
            pp_size=self.key_metadata.pp_size,
            pp_rank=self.key_metadata.pp_rank,
            group_id=self.key_metadata.group_id,
            component=VLLM_MULTIWR_V2.decode("ascii"),
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
        # Multiple entries may represent disjoint contiguous runs of one
        # strided layer. prepare_value emits them in canonical layer/run/block
        # order and always addresses the real physical block IDs.
        self._seg_layout: list[tuple[int, int, int]] | None = None

    def _make_key_by_hash(self, chunk_hash: str | bytes) -> PoolKey:
        return PoolKey(self.metadata, chunk_hash)

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        self.kv_caches_base_addr = kv_caches_base_addr

    def set_block_len(self, block_len: list[int]):
        self.block_len = block_len

    def set_seg_layout(self, seg_layout: list[tuple[int, int, int]]):
        """Install exact per-run ``(base, block_stride, block_content)``."""
        for base, stride, content in seg_layout:
            if base < 0 or stride <= 0 or content <= 0 or content > stride:
                raise ValueError(
                    f"invalid seg layout entry: base={base:#x} "
                    f"stride={stride} content={content}"
                )
        self._seg_layout = list(seg_layout)

    def prepare_value(
        self, start: int, end: int, block_ids: list[int]
    ) -> tuple[list[int], list[int], int]:
        """Return one logical chunk's complete, canonically ordered SG vector."""
        if (
            start < 0
            or end <= start
            or start % self.block_size != 0
            or (end - start) % self.block_size != 0
        ):
            raise ValueError(
                f"token range [{start}, {end}) must align to "
                f"block_size={self.block_size}"
            )

        start_block = start // self.block_size
        nblocks = (end - start) // self.block_size
        chunk_block_ids = block_ids[start_block:start_block + nblocks]
        if len(chunk_block_ids) != nblocks:
            raise ValueError(
                f"block table has {len(chunk_block_ids)} ids for "
                f"{nblocks} requested blocks at index {start_block}"
            )

        if self._seg_layout is None:
            if len(self.kv_caches_base_addr) != len(self.block_len):
                raise ValueError("legacy KV base and block-length tables differ")
            layout = [
                (base, length, length)
                for base, length in zip(
                    self.kv_caches_base_addr, self.block_len, strict=True
                )
            ]
        else:
            layout = self._seg_layout

        addr_list: list[int] = []
        size_list: list[int] = []
        for base, stride, content in layout:
            for block_id in chunk_block_ids:
                addr_list.append(base + block_id * stride)
                size_list.append(content)
        return addr_list, size_list, chunk_block_ids[0]

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
