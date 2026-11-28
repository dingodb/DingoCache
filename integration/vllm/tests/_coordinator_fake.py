"""Shared DfkvStoreCoordinator stand-in for tests that script its masks.

The real coordinator needs vLLM's block pools and managers. Tests that only
need scripted masks build this fake instead and pass the methods they script
as overrides. test/python/test_dfkv_vllm_worker.py::CoordinatorFakeDriftTest
fails by name when coordinator.py gains an attribute that is not defaulted
here; a missing attribute otherwise surfaces as an AttributeError inside a
worker thread, which can hang the test run.
"""
from __future__ import annotations

from types import SimpleNamespace


def fake_coordinator(lcm_block_size: int = 64, **overrides) -> SimpleNamespace:
    attributes = {
        "kv_cache_groups": [],
        "hash_block_size": lcm_block_size,
        "lcm_block_size": lcm_block_size,
        "use_eagle": False,
        "_manager_api": False,
        "_views": [],
        "num_hash_lookahead_tokens": 0,
        "retention_interval": None,
        "enable_partial_hash_hits": False,
        "hit_alignment": lcm_block_size,
        "attention_groups": [],
        "eagle_attn_group_indices": set(),
    }
    attributes.update(overrides)
    attributes.setdefault(
        "eagle_block_drop",
        bool(
            attributes["use_eagle"]
            and not attributes["num_hash_lookahead_tokens"]
        ),
    )
    if "drop_eagle_group_ids" not in attributes:
        if attributes["eagle_block_drop"]:
            drop_eagle_group_ids = {
                index
                for index, group in enumerate(attributes["kv_cache_groups"])
                if getattr(group, "is_eagle_group", False)
            }
            attributes["drop_eagle_group_ids"] = drop_eagle_group_ids or set(
                range(len(attributes["kv_cache_groups"]))
            )
        else:
            attributes["drop_eagle_group_ids"] = set()
    else:
        attributes["drop_eagle_group_ids"] = set(
            attributes["drop_eagle_group_ids"]
        )
    return SimpleNamespace(**attributes)
