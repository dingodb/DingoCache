# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-group persistence and restoration masks for DfkvStoreConnector."""

from typing import cast

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

# Dummy placeholder hash for store_mask's template computation.
_DUMMY_BLOCK_HASH = BlockHash(b"\x00" * 32)


def _unwrap_hit_blocks(res):
    """vLLM >= 0.26: SingleTypeKVCacheManager.find_longest_cache_hit returns
    ``(blocks_per_group, hit_length_tokens)``; older versions return just
    ``blocks_per_group``. Accept both. hit_length is always re-derived from
    the returned block lists, so the manager-side value is not needed here.
    """
    if (
        isinstance(res, tuple)
        and len(res) == 2
        and isinstance(res[0], tuple)
        and isinstance(res[1], int)
    ):
        return res[0]
    return res


class _TemplateBlockPool:
    """All-present keys for deriving a manager's required state window."""

    def __init__(self, hash_block_size: int) -> None:
        self.hash_block_size = hash_block_size
        self.null_block = KVCacheBlock(block_id=0)
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock]:
        return [self._present_block] * len(group_ids)


class ExternalCachedBlockPool(_TemplateBlockPool):
    """Duck-typed BlockPool backed by a ``(group_id, hash)`` exists set."""

    def __init__(
        self, hash_block_size: int, exists: set[tuple[int, bytes]]
    ) -> None:
        super().__init__(hash_block_size)
        self._exists = exists

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        # Mirrors BlockPool.get_cached_block: hit only when every group_id
        # (groups sharing a spec) has the hash cached.
        h = bytes(block_hash)
        if all((g, h) in self._exists for g in group_ids):
            return [self._present_block] * len(group_ids)
        return None


def partial_hash_hits_enabled(kv_cache_config, scheduler_block_size: int) -> bool:
    """Whether vLLM core lands prefix hits on hash blocks inside a scheduler
    block. Engines that do not export ``cache_hit_alignment_tokens`` never do.
    """
    alignment = getattr(kv_cache_config, "cache_hit_alignment_tokens", None)
    return alignment is not None and alignment < scheduler_block_size


def _prefix_cacheable(spec):
    """Older engines name this participates_in_prefix_caching."""
    for name in ("prefix_cacheable", "participates_in_prefix_caching"):
        value = getattr(spec, name, None)
        if value is not None:
            return bool(value)
    return True


class DfkvStoreCoordinator:
    """Build per-group cache masks using the running engine's managers."""
    def __init__(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
        use_eagle: bool = False,
        enable_partial_hash_hits: bool = False,
    ) -> None:

        participating = [
            (_prefix_cacheable(g.kv_cache_spec), g.kv_cache_spec.block_size)
            for g in kv_cache_groups
        ]
        assert all(
            size % hash_block_size == 0
            for participates, size in participating
            if participates
        ), (
            f"block_size must be divisible by hash_block_size "
            f"(hash={hash_block_size}, scheduler={scheduler_block_size}, "
            f"groups={participating})"
        )
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of "
            f"hash_block_size ({hash_block_size})"
        )
        assert all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_groups
        ), "scheduler_block_size must be a multiple of each group's block_size"
        self.kv_cache_groups = kv_cache_groups
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.use_eagle = use_eagle
        # Core's decision: whether hits land on the hash block, below the
        # scheduler block. Regular saves stay on the scheduler block grid.
        self.enable_partial_hash_hits = enable_partial_hash_hits
        self.hit_alignment = (
            hash_block_size if enable_partial_hash_hits else scheduler_block_size
        )
        self._verify_and_split_kv_cache_groups()

    def _verify_and_split_kv_cache_groups(self) -> None:
        """Group equal specs and resolve their engine-owned managers."""
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []
        for i, g in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(g.kv_cache_spec)
            if not _prefix_cacheable(g.kv_cache_spec):
                continue
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None, (
                f"No manager registered for KVCacheSpec {spec}"
            )
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))
        self.attention_groups = attention_groups
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if self.use_eagle
            and any(self.kv_cache_groups[gid].is_eagle_group for gid in group_ids)
        }
        if self.use_eagle and not self.eagle_attn_group_indices:
            self.eagle_attn_group_indices = set(range(len(self.attention_groups)))


    def load_mask(
        self,
        block_hashes: list[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...]:
        """Restore every group required at the already-admitted hit boundary."""
        aligned = token_len // self.hit_alignment * self.hit_alignment
        return self._cache_mask(
            aligned, drop_eagle=False, alignment=self.hit_alignment
        )

    def store_mask(self, aligned_token_len: int) -> tuple[list[bool], ...]:
        """Persist only stable context, never a volatile draft/lookahead tail."""
        return self._cache_mask(
            aligned_token_len, drop_eagle=True, alignment=self.lcm_block_size
        )

    def _cache_mask(
        self,
        aligned_token_len: int,
        *,
        drop_eagle: bool,
        alignment: int,
    ) -> tuple[list[bool], ...]:
        assert aligned_token_len % alignment == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of "
            f"the hit alignment ({alignment})"
        )
        if aligned_token_len == 0:
            return tuple([] for _ in self.kv_cache_groups)

        dummy_hashes: list[BlockHash] = [_DUMMY_BLOCK_HASH] * (
            aligned_token_len // self.hash_block_size
        )
        block_pool = _TemplateBlockPool(hash_block_size=self.hash_block_size)
        masks: list[list[bool]] = [[] for _ in self.kv_cache_groups]
        for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
            # Fine-grained managers probe interior hash boundaries, so they
            # take the raw hash-block list and view whole blocks themselves.
            hashes = (
                dummy_hashes
                if alignment < self.lcm_block_size
                else self.block_hashes_for_spec(dummy_hashes, spec)
            )
            hit_blocks = _unwrap_hit_blocks(
                manager_cls.find_longest_cache_hit(
                    block_hashes=hashes,
                    max_length=aligned_token_len,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, block_pool),
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle and idx in self.eagle_attn_group_indices,
                    alignment_tokens=alignment,
                )
            )
            for group_id, blocks in zip(group_ids, hit_blocks, strict=True):
                masks[group_id] = [
                    block is not block_pool.null_block for block in blocks
                ]
        return tuple(masks)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_length: int,
        block_pool: ExternalCachedBlockPool,
    ) -> int:
        """Hash-block-granular hit length over ``block_pool``.

        Converges the registered managers as vLLM core's
        ``HybridKVCacheCoordinator`` does (ported from the Mooncake store
        connector); requires ``enable_partial_hash_hits``. No Eagle block is
        dropped, matching ``load_mask``.
        """
        groups = sorted(
            self.attention_groups,
            key=lambda g: not isinstance(g[0], FullAttentionSpec),
        )
        is_simple_hybrid = len(groups) == 2 and isinstance(
            groups[0][0], FullAttentionSpec
        )
        full_attention_hits: dict[int, int] = {}
        hit_length = max_length
        while True:
            curr_hit_length = hit_length
            for idx, (spec, group_ids, manager_cls) in enumerate(groups):
                if idx in full_attention_hits:
                    # A full-attention hit only shrinks with the bound.
                    curr_hit_length = min(curr_hit_length, full_attention_hits[idx])
                    continue
                _, curr_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=curr_hit_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, block_pool),
                    kv_cache_spec=spec,
                    drop_eagle_block=False,
                    alignment_tokens=self.hash_block_size,
                )
                if isinstance(spec, FullAttentionSpec):
                    full_attention_hits[idx] = curr_hit_length
            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break
        return hit_length

    def partial_tail_boundary(self, num_prompt_tokens: int) -> int | None:
        """The prompt's recurrent-state checkpoint when it ends inside a
        scheduler block, else None.

        Mirrors core's ``get_mamba_prefill_checkpoint_position``: vLLM hands
        off the state there, and a resend can hit at most up to it.
        """
        if not self.enable_partial_hash_hits:
            return None
        hbs = self.hash_block_size
        if self.use_eagle:
            boundary = num_prompt_tokens // hbs * hbs - hbs
        else:
            boundary = (num_prompt_tokens - 1) // hbs * hbs
        if boundary <= 0 or boundary % self.lcm_block_size == 0:
            return None
        return boundary

    def block_hashes_for_spec(
        self, block_hashes: list[BlockHash], spec: KVCacheSpec
    ) -> BlockHashList:
        if spec.block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes, self.hash_block_size, spec.block_size
        )



def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec
