"""Compatibility checks against the real vLLM API loaded by the test run."""
import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch

from _cache_api import cache_group, full_attention_spec
from dfkv_vllm.cache_api import cache_group_view, group_block_size, manager_api
from dfkv_vllm.coordinator import DfkvStoreCoordinator, _AttentionGroup, _find_hit_blocks


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
def test_main_manager_lookup_passes_required_eagle_argument(monkeypatch):
    group = cache_group(
        ["attention"],
        full_attention_spec(
            block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
        ),
    )
    view = cache_group_view(group)
    result = object()
    parameters = inspect.signature(
        view.manager_cls.find_longest_cache_hit
    ).parameters
    find_longest_cache_hit = create_autospec(
        view.manager_cls.find_longest_cache_hit, return_value=result
    )
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

    (call,) = find_longest_cache_hit.call_args_list
    kwargs = call.kwargs
    assert kwargs["block_size"] == group_block_size(group) == 16
    assert kwargs["manager_config"] is group.manager_config
    if "drop_eagle_block" in parameters:
        assert kwargs["drop_eagle_block"] is True
    else:
        assert "drop_eagle_block" not in kwargs
    assert "kv_cache_spec" not in kwargs


@pytest.mark.parametrize(
    ("num_tokens", "lookahead", "alignment", "expected"),
    [
        (0, 0, 1, 0),
        (1, 0, 16, 0),
        (16, 0, 16, 0),
        (17, 0, 16, 16),
        (65, 16, 16, 48),
        (32, 64, 16, 0),
        (129, 64, 64, 64),
    ],
)
def test_reusable_prefix_length_contract(num_tokens, lookahead, alignment, expected):
    from dfkv_vllm.cache_api import reusable_prefix_length

    assert reusable_prefix_length(num_tokens, lookahead, alignment) == expected


@pytest.mark.skipif(not manager_api(), reason="manager-owned cache API required")
@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("num_hash_lookahead_tokens", [0, 16])
def test_modern_store_mask_passes_eagle_replay_and_group(
    monkeypatch, eagle, num_hash_lookahead_tokens
):
    group = cache_group(
        ["attention"],
        full_attention_spec(
            block_size=64, num_kv_heads=1, head_size=8, dtype=torch.float16
        ),
        is_eagle_group=eagle,
    )
    if hasattr(group, "is_eagle_group"):
        # Mirrors vLLM's post-planning assignment before coordinator creation.
        group.is_eagle_group = eagle
    coordinator = DfkvStoreCoordinator(
        [group],
        64,
        64,
        use_eagle=eagle,
        num_hash_lookahead_tokens=num_hash_lookahead_tokens,
    )
    drop_eagle = eagle and not num_hash_lookahead_tokens
    from vllm.v1.core.kv_cache_coordinator import replay_boundaries

    replay_parameters = inspect.signature(replay_boundaries).parameters
    replay = create_autospec(replay_boundaries, return_value=(255,))
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_coordinator.replay_boundaries", replay
    )
    mask_parameters = inspect.signature(
        coordinator._views[0].manager_cls.reachable_block_mask
    ).parameters
    reachable_block_mask = create_autospec(
        coordinator._views[0].manager_cls.reachable_block_mask,
        return_value=[True] * 4,
    )
    monkeypatch.setattr(
        coordinator._views[0].manager_cls,
        "reachable_block_mask",
        reachable_block_mask,
    )

    assert coordinator.store_mask(256, num_prompt_tokens=256) == ([True] * 4,)
    assert replay.call_args.args == (256,)
    replay_kwargs = replay.call_args.kwargs
    assert replay_kwargs["num_hash_lookahead_tokens"] == num_hash_lookahead_tokens
    if "use_eagle" in replay_parameters:
        assert replay_kwargs["scheduler_block_size"] == 64
        assert replay_kwargs["use_eagle"] is drop_eagle
    else:
        assert "scheduler_block_size" not in replay_kwargs
        assert "use_eagle" not in replay_kwargs

    (call,) = reachable_block_mask.call_args_list
    mask_kwargs = call.kwargs
    assert mask_kwargs["manager_config"] is group.manager_config
    assert mask_kwargs["reachable_boundaries"] == (255,)
    if "use_eagle" in mask_parameters:
        assert mask_kwargs["use_eagle"] is drop_eagle
    else:
        assert "use_eagle" not in mask_kwargs


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
