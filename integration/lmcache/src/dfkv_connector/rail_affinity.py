from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LMCacheRailAffinityConfig:
    enabled: bool = False
    fallback_count: int = 1


def parse_plugin_rail_affinity(
    extra_config: Mapping[str, Any], prefix: str
) -> LMCacheRailAffinityConfig:
    enabled_key = f"{prefix}.rail_affinity"
    fallback_key = f"{prefix}.rail_affinity_fallbacks"

    enabled = extra_config.get(enabled_key, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"extra_config[{enabled_key!r}] must be a boolean")

    fallback_count = extra_config.get(fallback_key, 1)
    if isinstance(fallback_count, bool) or not isinstance(fallback_count, int):
        raise ValueError(f"extra_config[{fallback_key!r}] must be an integer")
    if fallback_count < 0:
        raise ValueError(f"extra_config[{fallback_key!r}] must be non-negative")

    return LMCacheRailAffinityConfig(enabled, fallback_count)


def physical_affinity_rank(metadata: Any) -> int:
    """Return LMCache's explicit per-host worker/GPU coordinate.

    ``worker_id`` is the global/logical rank used in object identity. It can
    exceed the host-local GPU count and must never drive HCA affinity. Current
    LMCache v0.4.7+ metadata exposes both ``local_worker_id`` and
    ``local_world_size``; fail closed instead of guessing when either is absent.
    """
    if not hasattr(metadata, "local_worker_id") or not hasattr(
        metadata, "local_world_size"
    ):
        raise ValueError(
            "LMCache rail affinity requires metadata.local_worker_id and "
            "metadata.local_world_size"
        )

    local_rank = getattr(metadata, "local_worker_id")
    local_world_size = getattr(metadata, "local_world_size")
    if isinstance(local_rank, bool) or isinstance(local_world_size, bool):
        raise ValueError("LMCache local rank metadata must be integer-valued")
    try:
        local_rank = int(local_rank)
        local_world_size = int(local_world_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("LMCache local rank metadata must be integer-valued") from exc

    if local_world_size <= 0:
        raise ValueError("LMCache metadata.local_world_size must be positive")
    if local_rank < 0 or local_rank >= local_world_size:
        raise ValueError(
            "LMCache metadata.local_worker_id must be within local_world_size"
        )
    return local_rank
