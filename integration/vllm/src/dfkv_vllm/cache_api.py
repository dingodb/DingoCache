# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalize vLLM's legacy and manager-owned KV-cache group APIs.

vLLM #61452 moved allocation policy from policy-typed specs to the manager.
This module resolves that boundary once during connector/coordinator setup;
request and block hot paths use the resulting view instead of version probes.
Only APIs present in both generations are imported at module scope.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


@lru_cache(maxsize=1)
def _manager_api_available() -> bool:
    try:
        from vllm.v1.kv_cache_interface import KVCacheGroup  # noqa: F401
    except ImportError:
        return False
    return True


def manager_api(group_or_config: Any | None = None) -> bool:
    """Whether this object uses manager-owned allocation policy.

    Real groups decide locally so test doubles cannot accidentally select the
    running package's other API. With no object, report the installed engine's
    native generation for fixture/setup code.
    """
    if group_or_config is not None:
        groups = getattr(group_or_config, "kv_cache_groups", None)
        group = groups[0] if groups else group_or_config
        return getattr(group, "manager_config", None) is not None
    return _manager_api_available()


def _spec_layers(group: Any) -> tuple[tuple[str, KVCacheSpec], ...]:
    if manager_api(group):
        return tuple(sorted(group.kv_cache_specs.items()))
    raw = group.kv_cache_spec
    wrapped = getattr(raw, "kv_cache_specs", None)
    if wrapped is not None:
        return tuple(sorted(wrapped.items()))
    return tuple(
        (name, raw) for name in sorted(getattr(group, "layer_names", ("",)))
    )


def _registry_spec(raw: KVCacheSpec) -> KVCacheSpec:
    wrapped = getattr(raw, "kv_cache_specs", None)
    if wrapped is not None:
        return next(iter(wrapped.values()))
    return raw


def _legacy_prefix_cacheable(spec: KVCacheSpec) -> bool:
    """Older engines name this participates_in_prefix_caching."""
    for name in ("prefix_cacheable", "participates_in_prefix_caching"):
        value = getattr(spec, name, None)
        if value is not None:
            return bool(value)
    return True


@dataclass(frozen=True, slots=True)
class CacheGroupView:
    """An initialized group's layout and manager policy on either API."""

    group: Any
    block_size: int
    manager_cls: type
    manager_config: Any | None
    registry_spec: KVCacheSpec | None
    layer_specs: tuple[tuple[str, KVCacheSpec], ...]
    distinct_specs: tuple[KVCacheSpec, ...]
    prefix_cacheable: bool


def cache_group_view(group: Any) -> CacheGroupView:
    if isinstance(group, CacheGroupView):
        return group
    config = getattr(group, "manager_config", None)
    layers = _spec_layers(group)
    if config is not None:
        distinct = tuple(group.distinct_specs)
        manager_cls = config.manager_cls
        registry_spec = None
        cacheable = bool(group.prefix_cacheable)
    else:
        raw = group.kv_cache_spec
        registry_spec = _registry_spec(raw)
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        manager_cls = KVCacheSpecRegistry.get_manager_class(registry_spec)
        assert manager_cls is not None, (
            f"No manager registered for KVCacheSpec {registry_spec}"
        )
        distinct = tuple(dict.fromkeys(spec for _, spec in layers))
        cacheable = _legacy_prefix_cacheable(raw)
    return CacheGroupView(
        group=group,
        block_size=group.block_size if config is not None else group.kv_cache_spec.block_size,
        manager_cls=manager_cls,
        manager_config=config,
        registry_spec=registry_spec,
        layer_specs=layers,
        distinct_specs=distinct,
        prefix_cacheable=cacheable,
    )


def group_specs(group: Any) -> tuple[KVCacheSpec, ...]:
    """All distinct layouts, without manager registry side effects."""
    if isinstance(group, CacheGroupView):
        return group.distinct_specs
    if manager_api(group):
        return tuple(group.distinct_specs)
    return tuple(
        dict.fromkeys(spec for _, spec in _spec_layers(group))
    )


def group_block_size(group: Any) -> int:
    if isinstance(group, CacheGroupView):
        return group.block_size
    return group.block_size if manager_api(group) else group.kv_cache_spec.block_size


def prefix_cacheable(obj: Any) -> bool:
    if isinstance(obj, CacheGroupView):
        return obj.prefix_cacheable
    if manager_api(obj):
        return bool(obj.prefix_cacheable)
    return _legacy_prefix_cacheable(getattr(obj, "kv_cache_spec", obj))


def store_group_ids(kv_cache_config: Any) -> tuple[int, ...]:
    """Engine group IDs this hash-addressed connector may restore.

    Main can mark a transferable-prefix-cache group out of external transfer;
    its projection is normative. Lightweight stand-ins without the property
    retain the legacy all-group shape.
    """
    if manager_api(kv_cache_config):
        ids = getattr(kv_cache_config, "prefix_cacheable_group_ids", None)
        if ids is not None:
            return tuple(ids)
    return tuple(range(len(kv_cache_config.kv_cache_groups)))


def store_groups(kv_cache_config: Any) -> list[Any]:
    return [
        kv_cache_config.kv_cache_groups[index]
        for index in store_group_ids(kv_cache_config)
    ]


def is_full_attention(group: Any) -> bool:
    view = group if isinstance(group, CacheGroupView) else cache_group_view(group)
    if view.manager_config is not None:
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

        return issubclass(view.manager_cls, FullAttentionManager)
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return isinstance(view.registry_spec, FullAttentionSpec)


def is_mamba_group(group: Any) -> bool:
    return any(isinstance(spec, MambaSpec) for spec in group_specs(group))


def is_cross_attention(group: Any) -> bool:
    view = group if isinstance(group, CacheGroupView) else cache_group_view(group)
    if view.manager_config is not None:
        from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager

        return issubclass(view.manager_cls, CrossAttentionManager)
    from vllm.v1.kv_cache_interface import CrossAttentionSpec

    return isinstance(view.registry_spec, CrossAttentionSpec)


def resolve_dcp_group(group: Any, dcp_size: int) -> Any:
    """Return the engine-native group with DCP-resolved block geometry."""
    if manager_api(group):
        from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size

        return dataclasses.replace(
            group, block_size=resolve_dcp_kv_block_size(group, dcp_size)
        )

    from vllm.v1.core import kv_cache_utils

    resolver = getattr(kv_cache_utils, "resolve_dcp_kv_cache_spec", None)
    if resolver is not None:
        return dataclasses.replace(
            group,
            kv_cache_spec=resolver(group.kv_cache_spec, dcp_size),
        )
    spec = group.kv_cache_spec
    from vllm.v1.kv_cache_interface import AttentionSpec

    if dcp_size != 1 and isinstance(spec, AttentionSpec):
        spec = dataclasses.replace(spec, block_size=spec.block_size * dcp_size)
    return dataclasses.replace(group, kv_cache_spec=spec)


def logical_bytes_per_block(group: Any) -> int:
    """Bytes in one logical block across this stage's real group layers."""
    view = group if isinstance(group, CacheGroupView) else cache_group_view(group)
    total = 0
    for _, spec in view.layer_specs:
        effective = (
            spec.copy_with_new_block_size(view.block_size)
            if spec.block_size != view.block_size
            else spec
        )
        total += effective.page_size_bytes
    return total
