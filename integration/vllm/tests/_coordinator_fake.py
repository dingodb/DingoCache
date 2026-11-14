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
        "enable_partial_hash_hits": False,
        "hit_alignment": lcm_block_size,
        "attention_groups": [],
        "eagle_attn_group_indices": set(),
    }
    attributes.update(overrides)
    return SimpleNamespace(**attributes)
