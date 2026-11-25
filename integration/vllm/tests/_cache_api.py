"""Real legacy-spec and manager-owned vLLM fixtures for connector tests.

Callers describe the desired cache policy once. Legacy groups keep the policy
on their spec; main groups receive real KVCacheLayerConfig/ManagerConfig
objects and never import the policy classes removed by vLLM #61452.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)

try:
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
except ImportError:
    KVCacheGroupSpec = None

_LEGACY = KVCacheGroupSpec is not None

if _LEGACY:
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MLAAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )
else:
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        KVCacheGroup,
        KVCacheLayerConfig,
        MLAAttentionSpec,
        MambaSpec,
    )

    class _NonCachingFullAttentionManager(FullAttentionManager):
        @classmethod
        def prefix_cacheable(cls, config):
            return False


@dataclass(frozen=True)
class _LayerRecipe:
    """Main layout plus the manager policy that owns it."""

    spec: Any
    manager_config: Any

    def __getattr__(self, name):
        return getattr(self.spec, name)


@dataclass(frozen=True)
class _UniformRecipe:
    block_size: int
    kv_cache_specs: dict[str, _LayerRecipe]


def _recipe(spec, manager_cls, **manager_params):
    if _LEGACY:
        return spec
    return _LayerRecipe(spec, manager_cls.config(**manager_params))


def full_attention_spec(**kwargs):
    spec = (FullAttentionSpec if _LEGACY else AttentionSpec)(**kwargs)
    return _recipe(spec, FullAttentionManager)


def sliding_window_spec(
    *,
    sliding_window,
    extra_retained_tokens=0,
    bounded_replay=False,
    **kwargs,
):
    if _LEGACY:
        assert not bounded_replay
        return SlidingWindowSpec(
            sliding_window=sliding_window,
            extra_retained_tokens=extra_retained_tokens,
            **kwargs,
        )
    return _recipe(
        AttentionSpec(**kwargs),
        SlidingWindowManager,
        sliding_window=sliding_window,
        extra_retained_tokens=extra_retained_tokens,
        bounded_replay=bounded_replay,
    )


def mla_attention_spec(**kwargs):
    return _recipe(MLAAttentionSpec(**kwargs), FullAttentionManager)


def mamba_spec(
    *,
    mamba_cache_mode="none",
    num_speculative_blocks=0,
    num_prefill_checkpoint_blocks=0,
    prefill_checkpoint_alignment=None,
    **kwargs,
):
    spec = MambaSpec(
        **kwargs,
        **(
            dict(
                mamba_cache_mode=mamba_cache_mode,
                num_speculative_blocks=num_speculative_blocks,
                num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
                prefill_checkpoint_alignment=prefill_checkpoint_alignment,
            )
            if _LEGACY
            else {}
        ),
    )
    return _recipe(
        spec,
        MambaManager,
        enable_prefix_caching=mamba_cache_mode == "align",
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        prefill_checkpoint_alignment=prefill_checkpoint_alignment,
    )


def uniform_specs(*, block_size, kv_cache_specs):
    if _LEGACY:
        return UniformTypeKVCacheSpecs(
            block_size=block_size, kv_cache_specs=kv_cache_specs
        )
    return _UniformRecipe(block_size, dict(kv_cache_specs))


def _uncacheable_legacy_spec(spec):
    """The old non-caching scratch layout, retaining every policy field."""
    if not isinstance(spec, MLAAttentionSpec):
        raise TypeError("non-cacheable test groups use an MLA layout")

    class NonCachingMLAAttentionSpec(MLAAttentionSpec):
        @property
        def prefix_cacheable(self):
            return False

        @property
        def participates_in_prefix_caching(self):
            return False

    return NonCachingMLAAttentionSpec(**{
        field.name: getattr(spec, field.name)
        for field in dataclasses.fields(spec)
    })


def cache_group(
    layer_names,
    spec,
    *,
    is_eagle_group=False,
    prefix_cacheable=True,
    enable_kv_transfer=True,
):
    """Build the running engine's genuine cache-group object."""
    if _LEGACY:
        if not prefix_cacheable:
            spec = _uncacheable_legacy_spec(spec)
        return KVCacheGroupSpec(
            list(layer_names),
            spec,
            is_eagle_group=is_eagle_group,
            enable_kv_transfer=enable_kv_transfer,
        )

    if isinstance(spec, _UniformRecipe):
        recipes = spec.kv_cache_specs
    else:
        if not isinstance(spec, _LayerRecipe):
            raise TypeError("main cache_group needs a shared spec constructor")
        recipes = dict.fromkeys(layer_names, spec)
    layer_configs = {}
    for name in layer_names:
        recipe = recipes[name]
        manager_config = recipe.manager_config
        if not prefix_cacheable:
            if manager_config.manager_cls is not FullAttentionManager:
                raise TypeError("non-cacheable test groups use full attention")
            manager_config = _NonCachingFullAttentionManager.config()
        layer_configs[name] = KVCacheLayerConfig(manager_config, recipe.spec)
    if not layer_configs:
        if not isinstance(spec, _LayerRecipe):
            raise TypeError("an empty main group needs one layer recipe")
        return KVCacheGroup(
            {}, spec.manager_config, spec.spec.block_size, spec.spec.dcp_sharded
        )
    return KVCacheGroup.from_layers(
        layer_configs, enable_kv_transfer=enable_kv_transfer
    )


def cache_manager(
    group,
    block_pool,
    *,
    enable_caching,
    kv_cache_group_id,
    scheduler_block_size,
):
    """Instantiate the real manager using its generation's constructor."""
    from dfkv_vllm.cache_api import cache_group_view

    view = cache_group_view(group)
    policy = group if view.manager_config is not None else group.kv_cache_spec
    return view.manager_cls(
        policy,
        block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=kv_cache_group_id,
        scheduler_block_size=scheduler_block_size,
    )
