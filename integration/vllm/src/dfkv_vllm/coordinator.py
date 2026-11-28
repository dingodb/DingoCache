# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-group persistence and restoration masks for DfkvStoreConnector."""

import dataclasses
from typing import NamedTuple, cast

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)

from .cache_api import (
    CacheGroupView,
    accepts_keyword,
    cache_group_view,
    group_block_size,
    is_full_attention,
    prefix_cacheable as _prefix_cacheable,
)


class _AttentionGroup(NamedTuple):
    view: CacheGroupView
    group_ids: list[int]


def _coverage_config(config):
    """Match core's external-store SWA replay coverage, if configured."""
    if config is None:
        return None
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    if (
        isinstance(config.params, SlidingWindowManager.Params)
        and config.params.extra_retained_tokens
    ):
        return config.replace_params(
            sliding_window=(
                config.params.sliding_window + config.params.extra_retained_tokens
            ),
            extra_retained_tokens=0,
        )
    return config


def _find_hit_blocks(
    group: _AttentionGroup,
    *,
    block_hashes,
    max_length: int,
    block_pool: BlockPool,
    alignment_tokens: int,
    drop_eagle_block: bool = False,
):
    """Call the running engine's manager without hiding either signature."""
    view = group.view
    kwargs = dict(
        block_hashes=block_hashes,
        max_length=max_length,
        kv_cache_group_ids=group.group_ids,
        block_pool=block_pool,
        alignment_tokens=alignment_tokens,
    )
    if view.manager_config is not None:
        kwargs.update(
            block_size=view.block_size,
            manager_config=view.manager_config,
        )
        if accepts_keyword(
            view.manager_cls.find_longest_cache_hit, "drop_eagle_block"
        ):
            kwargs["drop_eagle_block"] = drop_eagle_block
    else:
        kwargs.update(
            kv_cache_spec=view.registry_spec,
            drop_eagle_block=drop_eagle_block,
        )
    return view.manager_cls.find_longest_cache_hit(**kwargs)

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
        kv_cache_groups: list,
        scheduler_block_size: int,
        hash_block_size: int,
        use_eagle: bool = False,
        enable_partial_hash_hits: bool = False,
        num_hash_lookahead_tokens: int = 0,
        retention_interval: int | None = None,
    ) -> None:
        views = [cache_group_view(group) for group in kv_cache_groups]
        participating = [
            (view.prefix_cacheable, view.block_size) for view in views
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
            scheduler_block_size % view.block_size == 0
            for view in views
            if view.prefix_cacheable
        ), "scheduler_block_size must be a multiple of each group's block_size"
        self.kv_cache_groups = kv_cache_groups
        self._views = views
        self._manager_api = bool(views and views[0].manager_config is not None)
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        # Modern engines prove draft KV with lookahead hashes instead of
        # dropping a trailing block. Legacy engines keep the old indication.
        self.use_eagle = use_eagle and not self._manager_api
        self.num_hash_lookahead_tokens = (
            num_hash_lookahead_tokens if self._manager_api else 0
        )
        # Keep the global drop decision separate from planning's optional group
        # markers. Lookahead hashes suppress drops without clearing markers.
        self.eagle_block_drop = use_eagle and not self.num_hash_lookahead_tokens
        self.drop_eagle_group_ids: set[int] = set()
        if self.eagle_block_drop:
            self.drop_eagle_group_ids = {
                index
                for index, group in enumerate(self.kv_cache_groups)
                if getattr(group, "is_eagle_group", False)
            }
            if not self.drop_eagle_group_ids:
                self.drop_eagle_group_ids = set(range(len(self.kv_cache_groups)))
        self.retention_interval = retention_interval if self._manager_api else None
        self.enable_partial_hash_hits = enable_partial_hash_hits
        self.hit_alignment = (
            hash_block_size if enable_partial_hash_hits else scheduler_block_size
        )
        self._verify_and_split_kv_cache_groups()

    def _verify_and_split_kv_cache_groups(self) -> None:
        """Resolve the same joint lookups as the running engine core."""
        attention_groups: list[_AttentionGroup] = []
        if self._manager_api:
            from vllm.v1.kv_cache_interface import prefix_lookup_groups

            for lookup in prefix_lookup_groups(self.kv_cache_groups):
                first = dataclasses.replace(
                    self._views[lookup.group_ids[0]],
                    manager_config=_coverage_config(
                        self._views[lookup.group_ids[0]].manager_config
                    ),
                )
                attention_groups.append(_AttentionGroup(first, lookup.group_ids))
        else:
            for index, view in enumerate(self._views):
                if not view.prefix_cacheable:
                    continue
                for existing in attention_groups:
                    if existing.view.registry_spec == view.registry_spec:
                        assert existing.view.manager_cls is view.manager_cls
                        existing.group_ids.append(index)
                        break
                else:
                    attention_groups.append(_AttentionGroup(view, [index]))
        self.attention_groups = attention_groups
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids) in enumerate(self.attention_groups)
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

    def store_mask(
        self, aligned_token_len: int, *, num_prompt_tokens: int | None = None
    ) -> tuple[list[bool], ...]:
        """Persist stable blocks using the running engine's retention policy."""
        if self._manager_api:
            return self._modern_store_mask(
                aligned_token_len, num_prompt_tokens=num_prompt_tokens
            )
        return self._cache_mask(
            aligned_token_len, drop_eagle=True, alignment=self.lcm_block_size
        )

    def _modern_store_mask(
        self, aligned_token_len: int, *, num_prompt_tokens: int | None
    ) -> tuple[list[bool], ...]:
        if aligned_token_len == 0:
            return tuple([] for _ in self.kv_cache_groups)
        assert aligned_token_len % self.lcm_block_size == 0
        from vllm.v1.core.kv_cache_coordinator import replay_boundaries
        from .cache_api import is_mamba_group

        replay = ()
        if num_prompt_tokens is not None:
            replay_kwargs = {
                "num_hash_lookahead_tokens": self.num_hash_lookahead_tokens
            }
            if accepts_keyword(replay_boundaries, "use_eagle"):
                replay_use_eagle = bool(self.drop_eagle_group_ids)
                replay_kwargs.update(
                    scheduler_block_size=self.lcm_block_size,
                    use_eagle=replay_use_eagle,
                )
            replay = replay_boundaries(num_prompt_tokens, **replay_kwargs)
        masks: list[list[bool]] = [[] for _ in self.kv_cache_groups]
        for index, view in enumerate(self._views):
            if not view.prefix_cacheable:
                continue
            end_block = aligned_token_len // view.block_size
            if is_mamba_group(view):
                # Modern core exports recurrent states only through exact
                # boundary hand-offs, never by position in the live table.
                masks[index] = [False] * end_block
                continue
            mask_kwargs = dict(
                start_block=0,
                end_block=end_block,
                alignment_tokens=self.lcm_block_size,
                block_size=view.block_size,
                manager_config=_coverage_config(view.manager_config),
                retention_interval=self.retention_interval,
                reachable_boundaries=replay,
                dcp_world_size=1,
            )
            if accepts_keyword(
                view.manager_cls.reachable_block_mask, "use_eagle"
            ):
                mask_kwargs["use_eagle"] = index in self.drop_eagle_group_ids
            mask = view.manager_cls.reachable_block_mask(**mask_kwargs)
            masks[index] = (
                [True] * end_block if mask is None else mask
            )
        return tuple(masks)

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
        for index, group in enumerate(self.attention_groups):
            hashes = (
                dummy_hashes
                if alignment < self.lcm_block_size
                else self._block_hashes_for_size(dummy_hashes, group.view.block_size)
            )
            hit_blocks = _unwrap_hit_blocks(
                _find_hit_blocks(
                    group,
                    block_hashes=hashes,
                    max_length=aligned_token_len,
                    block_pool=cast(BlockPool, block_pool),
                    alignment_tokens=alignment,
                    drop_eagle_block=(
                        drop_eagle and index in self.eagle_attn_group_indices
                    ),
                )
            )
            for group_id, blocks in zip(group.group_ids, hit_blocks, strict=True):
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

        Converges managers and applies the same drop-enabled group decisions
        as core. Newer managers omit the argument by signature; load masks
        continue to use their separate non-dropping path.
        """
        groups = sorted(
            self.attention_groups,
            key=lambda group: not is_full_attention(group.view),
        )
        from .cache_api import is_mamba_group

        is_simple_hybrid = len(groups) == 2 and is_full_attention(groups[0].view)
        full_attention_hits: dict[int, int] = {}
        # Verify each drop-enabled group at most once per candidate length.
        eagle_verified: set[int] = set()
        hit_length = max_length
        while True:
            curr_hit_length = hit_length
            for index, group in enumerate(groups):
                if index in full_attention_hits:
                    curr_hit_length = min(curr_hit_length, full_attention_hits[index])
                    continue
                drop_eagle_block = (
                    self._manager_api
                    and index not in eagle_verified
                    and accepts_keyword(
                        group.view.manager_cls.find_longest_cache_hit,
                        "drop_eagle_block",
                    )
                    and any(
                        gid in self.drop_eagle_group_ids
                        for gid in group.group_ids
                    )
                )
                candidate_max_length = curr_hit_length
                if drop_eagle_block and not is_mamba_group(group.view):
                    from vllm.v1.core.kv_cache_utils import eagle_proof_margin

                    margin = eagle_proof_margin(
                        group.view.block_size,
                        self.hash_block_size,
                        self.enable_partial_hash_hits
                        and group.view.manager_cls.supports_fine_grained_hash_lookup,
                    )
                    # Probe the block that proves the current candidate, then
                    # let the manager drop back to it instead of shortening twice.
                    candidate_max_length = min(
                        curr_hit_length + margin,
                        len(block_hashes) * self.hash_block_size,
                    )
                _, new_hit_length = _find_hit_blocks(
                    group,
                    block_hashes=block_hashes,
                    max_length=candidate_max_length,
                    block_pool=cast(BlockPool, block_pool),
                    alignment_tokens=self.hash_block_size,
                    drop_eagle_block=drop_eagle_block,
                )
                if drop_eagle_block:
                    eagle_verified.add(index)
                elif new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = new_hit_length
                if is_full_attention(group.view):
                    full_attention_hits[index] = curr_hit_length
            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break
        return hit_length

    def partial_tail_boundary(self, num_prompt_tokens: int) -> int | None:
        """Return the prompt's recurrent-state checkpoint inside an LCM block."""
        if not self.enable_partial_hash_hits:
            return None
        hbs = self.hash_block_size
        if self._manager_api:
            from .cache_api import reusable_prefix_length

            boundary = reusable_prefix_length(
                num_prompt_tokens,
                num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
                alignment_tokens=hbs,
            )
        elif self.use_eagle:
            boundary = num_prompt_tokens // hbs * hbs - hbs
        else:
            boundary = (num_prompt_tokens - 1) // hbs * hbs
        if boundary <= 0 or boundary % self.lcm_block_size == 0:
            return None
        return boundary

    def _block_hashes_for_size(
        self, block_hashes: list[BlockHash], block_size: int
    ) -> BlockHashList:
        if block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes, self.hash_block_size, block_size
        )

    def block_hashes_for_group(
        self, block_hashes: list[BlockHash], group
    ) -> BlockHashList:
        return self._block_hashes_for_size(
            block_hashes, group_block_size(group)
        )
