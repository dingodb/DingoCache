from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

from dfkv_vllm.coordinator import DfkvStoreCoordinator
from dfkv_vllm.data import KeyMetadata, PoolKey
from dfkv_vllm.worker import DfkvStoreWorker


def _coordinator(
    *, eagle: bool, drop: bool = True, scheduler_block: int = 64,
    dense: bool = True,
):
    groups = []
    if dense:
        groups.append(KVCacheGroupSpec(
            ["main"],
            FullAttentionSpec(
                block_size=scheduler_block, num_kv_heads=1,
                head_size=8, dtype=torch.float16,
            ),
        ))
    groups.append(KVCacheGroupSpec(
        ["window"],
        SlidingWindowSpec(
            block_size=64, num_kv_heads=1, head_size=8,
            dtype=torch.float16, sliding_window=128,
        ),
        is_eagle_group=eagle,
    ))
    return DfkvStoreCoordinator(
        groups, scheduler_block, 64, use_eagle=eagle and drop
    )


def _stored_prefix(coordinator, tokens=256):
    hashes = [BlockHash(bytes([i + 1]) * 32) for i in range(tokens // 64)]
    exists = set()
    for group, mask in enumerate(coordinator.store_mask(tokens)):
        group_hashes = coordinator.block_hashes_for_spec(
            hashes, coordinator.kv_cache_groups[group].kv_cache_spec
        )
        exists.update(
            (group, bytes(group_hashes[index]))
            for index, present in enumerate(mask)
            if present
        )
    return hashes, exists


def _lookup(coordinator, tokens, hashes, exists):
    metadata = [
        KeyMetadata(
            model_name="stable-draft", dp_size=1, dp_rank=-1,
            tp_size=1, tp_rank=-1, pcp_size=1, pcp_rank=0,
            dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0, group_id=index,
        )
        for index in range(len(coordinator.kv_cache_groups))
    ]
    objects = {PoolKey(metadata[group], value.hex()).to_bytes()
               for group, value in exists}
    worker = SimpleNamespace(
        coord=coordinator,
        token_dbs=[
            SimpleNamespace(metadata=md, block_size=group.kv_cache_spec.block_size)
            for md, group in zip(metadata, coordinator.kv_cache_groups, strict=True)
        ],
        client=SimpleNamespace(
            batch_exist=lambda keys: [int(key in objects) for key in keys],
            batch_retain=lambda keys, ttl_s=0, mode=1, **kw: [int(key in objects) for key in keys],
        ),
        tp_size=1,
        num_kv_head=1,
        _kv_cache_groups=coordinator.kv_cache_groups,
        _record_kv_connector_operation=lambda *args, **kwargs: None,
    )
    return DfkvStoreWorker.lookup(worker, tokens, hashes)[0]


def test_equal_block_size_window_excludes_unneeded_prefix():
    assert _coordinator(eagle=False).store_mask(256) == (
        [True, True, True, True], [False, False, True, True],
    )


def test_volatile_draft_tail_is_not_persisted():
    coordinator = _coordinator(eagle=True)
    hashes, exists = _stored_prefix(coordinator)
    assert (1, bytes(hashes[-1])) not in exists


def test_worker_admits_only_the_stored_stable_draft_prefix():
    coordinator = _coordinator(eagle=True)
    hashes, exists = _stored_prefix(coordinator)
    hit = _lookup(coordinator, 256, hashes, exists)
    assert hit == 192
    assert coordinator.load_mask(hashes, hit) == (
        [True, True, True], [False, True, True],
    )


def test_longer_writer_makes_the_previous_draft_tail_reusable():
    coordinator = _coordinator(eagle=True)
    hashes, exists = _stored_prefix(coordinator, tokens=320)
    hit = _lookup(coordinator, 256, hashes, exists)
    assert hit == 256
    assert coordinator.load_mask(hashes, hit)[1] == [False, False, True, True]


def test_disabled_eagle_drop_preserves_the_complete_prefix():
    coordinator = _coordinator(eagle=True, drop=False)
    hashes, exists = _stored_prefix(coordinator)
    assert _lookup(coordinator, 256, hashes, exists) == 256


def test_hybrid_alignment_revalidates_the_shorter_stable_window():
    coordinator = _coordinator(eagle=True, scheduler_block=256)
    hashes, exists = _stored_prefix(coordinator, tokens=512)
    hit = _lookup(coordinator, 512, hashes, exists)
    assert hit == 256
    assert coordinator.load_mask(hashes, hit) == (
        [True], [False, False, True, True],
    )


def test_window_only_model_can_reuse_its_required_tail():
    coordinator = _coordinator(eagle=False, dense=False)
    hashes, exists = _stored_prefix(coordinator)
    assert _lookup(coordinator, 256, hashes, exists) == 256
