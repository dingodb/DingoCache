from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class RailAffinitySelection:
    enabled: bool
    applied: bool
    rank: int
    available: tuple[str, ...]
    selected: tuple[str, ...]
    primary: Optional[str]
    fallback_count: int
    reason: str


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def _fallback_count(value: Any, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError("rail_affinity_fallbacks must be an integer, not boolean")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("rail_affinity_fallbacks must be an integer") from exc
    return max(0, min(count, maximum))


def apply_rank_local_rail_affinity(
    cfg: Mapping[str, Any],
    rank: int,
    environ: MutableMapping[str, str],
) -> RailAffinitySelection:
    """Apply one primary plus bounded ordered fallback rails before native open.

    ``rank`` is a connector-owned physical process/GPU coordinate.  This helper
    deliberately does not derive it from TP/DCP/PCP/PP metadata: HiCache and
    vLLM expose different parallel layouts, while HCA affinity is a per-host
    physical property.
    """
    try:
        physical_rank = int(rank)
    except (TypeError, ValueError) as exc:
        raise ValueError("rail affinity rank must be an integer") from exc
    if physical_rank < 0:
        raise ValueError("rail affinity rank must be non-negative")

    enabled = _truthy(cfg.get("rail_affinity"))
    rails = tuple(
        rail.strip()
        for rail in environ.get("DFKV_RDMA_DEV", "").split(",")
        if rail.strip()
    )

    if not enabled:
        return RailAffinitySelection(
            False, False, physical_rank, rails, rails, None, 0, "disabled"
        )
    if len(set(rails)) != len(rails):
        raise ValueError("DFKV_RDMA_DEV contains duplicate rail names")
    if len(rails) <= 1:
        return RailAffinitySelection(
            True,
            False,
            physical_rank,
            rails,
            rails,
            rails[0] if rails else None,
            0,
            "fewer_than_two_rails",
        )

    fallback_count = _fallback_count(
        cfg.get("rail_affinity_fallbacks", 1), len(rails) - 1
    )
    primary_index = physical_rank % len(rails)
    selected = tuple(
        rails[(primary_index + offset) % len(rails)]
        for offset in range(fallback_count + 1)
    )

    environ["DFKV_RDMA_DEV"] = ",".join(selected)
    environ["DFKV_RDMA_PRIMARY_DEV"] = selected[0]
    environ["DFKV_RDMA_NUMA"] = "0"
    environ.pop("DFKV_RDMA_RAIL_TIERS", None)

    return RailAffinitySelection(
        True,
        True,
        physical_rank,
        rails,
        selected,
        selected[0],
        fallback_count,
        "applied",
    )
