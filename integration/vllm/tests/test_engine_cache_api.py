"""Compatibility checks against the real vLLM API loaded by the test run."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from _cache_api import cache_group, full_attention_spec
from dfkv_vllm.cache_api import cache_group_view, group_block_size, manager_api
from dfkv_vllm.coordinator import _AttentionGroup, _find_hit_blocks


def test_production_modules_import_on_the_current_engine():
    for name in (
        "dfkv_vllm.connector",
        "dfkv_vllm.coordinator",
        "dfkv_vllm.worker",
    ):
        assert importlib.import_module(name).__name__ == name


def test_cache_fixture_builds_the_current_engines_real_group():
    group = cache_group(
        ["attention"],
        full_attention_spec(
            block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
        ),
    )
    view = cache_group_view(group)
    assert view.block_size == 16
    assert view.manager_cls.__name__ == "FullAttentionManager"
    if manager_api(group):
        from vllm.v1.kv_cache_interface import KVCacheGroup, KVCacheSpec, ManagerConfig

        assert isinstance(group, KVCacheGroup)
        assert isinstance(group.manager_config, ManagerConfig)
        assert isinstance(group.kv_cache_specs["attention"], KVCacheSpec)
    else:
        assert type(group).__name__ == "KVCacheGroupSpec"
        assert type(group).__module__ == "vllm.v1.kv_cache_interface"
        assert type(view.registry_spec).__name__ == "FullAttentionSpec"


@pytest.mark.skipif(not manager_api(), reason="manager-owned cache API required")
def test_main_manager_lookup_does_not_pass_legacy_eagle_arguments(monkeypatch):
    group = cache_group(
        ["attention"],
        full_attention_spec(
            block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
        ),
    )
    view = cache_group_view(group)
    calls = []
    result = object()

    def find_longest_cache_hit(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(
        view.manager_cls, "find_longest_cache_hit", find_longest_cache_hit
    )
    assert _find_hit_blocks(
        _AttentionGroup(view, [0]),
        block_hashes=["hash"],
        max_length=16,
        block_pool=SimpleNamespace(),
        alignment_tokens=16,
        drop_eagle_block=True,
    ) is result

    (call,) = calls
    assert call["block_size"] == group_block_size(group) == 16
    assert call["manager_config"] is group.manager_config
    assert "drop_eagle_block" not in call
    assert "kv_cache_spec" not in call


def test_dcp_membership_uses_global_stage_ranks():
    from dfkv_vllm.worker import _key_stripe_identity

    # PP stage 1 has global ranks 8..15, not local ordinals 0..7.
    identity = _key_stripe_identity(
        tp_rank=0,
        tp_size=8,
        pcp_rank=0,
        pcp_size=1,
        dcp_rank=0,
        dcp_size=8,
        num_kv_heads=1,
        use_mla=True,
        dcp_group_ranks=tuple(range(8, 16)),
        dcp_global_rank_base=8,
    )
    assert identity.dcp_rank == 0
    assert identity.stripe_idx == 0
    assert identity.stripe_step == 1


@pytest.mark.skipif(not manager_api(), reason="manager group projection required")
def test_external_store_honors_engine_group_projection():
    from dfkv_vllm.cache_api import store_group_ids, store_groups
    from dfkv_vllm.scheduler import DfkvStoreScheduler

    transferable = cache_group(
        ["attention"],
        full_attention_spec(
            block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
        ),
    )
    excluded = cache_group(
        ["aux"],
        full_attention_spec(
            block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
        ),
        enable_kv_transfer=False,
    )
    config = SimpleNamespace(
        kv_cache_groups=[transferable, excluded], prefix_cacheable_group_ids=(0,)
    )
    assert store_group_ids(config) == (0,)
    assert store_groups(config) == [transferable]

    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._store_group_ids = store_group_ids(config)
    scheduler._store_index_by_engine_group = {
        engine: store for store, engine in enumerate(scheduler._store_group_ids)
    }
    assert scheduler._store_index(0) == 0
    assert scheduler._store_index(1) is None
    assert scheduler._select_store_blocks(([4], [9])) == ([4],)


def test_skip_reading_external_kv_short_circuits_before_lookup():
    from unittest.mock import MagicMock

    from dfkv_vllm.scheduler import DfkvStoreScheduler

    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._failed_load_req_ids = set()
    scheduler.load_specs = {}
    scheduler.client = MagicMock()
    request = SimpleNamespace(
        request_id="skip-external", skip_reading_external_kv=True
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.client.discard.assert_called_once_with("skip-external")
    scheduler.client.lookup.assert_not_called()
