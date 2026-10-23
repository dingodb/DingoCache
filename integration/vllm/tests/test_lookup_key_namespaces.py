"""Lookup must probe every key coordinate the TP x PCP workers write."""
import dataclasses
import hashlib
from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from dfkv_vllm.coordinator import DfkvStoreCoordinator
from dfkv_vllm.data import KeyMetadata, PoolKey
from dfkv_vllm.worker import DfkvStoreWorker, _key_namespaces

BLOCK = 64


def test_mla_dcp_needs_every_dcp_shard():
    assert set(_key_namespaces(
        tp_size=8, pcp_size=1, dcp_size=8, num_kv_heads=1, use_mla=True,
        tp_sharded=False,
    )) == {(0, 0, dcp) for dcp in range(8)}


def test_plain_tp_mla_uses_the_canonical_coordinate():
    assert _key_namespaces(
        tp_size=8, pcp_size=1, dcp_size=1, num_kv_heads=1, use_mla=True,
        tp_sharded=False,
    ) == ((-1, 0, 0),)


def test_gqa_dcp_pairs_each_head_with_its_dcp_rank():
    # TP=8 with 8 KV heads and DCP=2: head k is written only under dcp=k%2.
    assert set(_key_namespaces(
        tp_size=8, pcp_size=1, dcp_size=2, num_kv_heads=8, use_mla=False,
        tp_sharded=False,
    )) == {(head, 0, head % 2) for head in range(8)}


def test_tp_sharded_state_covers_every_tp_and_pcp_rank():
    assert set(_key_namespaces(
        tp_size=4, pcp_size=2, dcp_size=2, num_kv_heads=1, use_mla=True,
        tp_sharded=True,
    )) == {(tp, pcp, (tp * 2 + pcp) % 2) for tp in range(4) for pcp in range(2)}


def _gqa_dcp_lookup(store):
    spec = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.bfloat16,
    )
    groups = [KVCacheGroupSpec(["layer"], spec)]
    md = KeyMetadata(
        model_name="gqa", dp_size=1, dp_rank=-1, tp_size=8, tp_rank=0,
        pcp_size=1, pcp_rank=0, dcp_size=2, dcp_rank=0, pp_size=1, pp_rank=0,
        group_id=0,
    )
    worker = SimpleNamespace(
        coord=DfkvStoreCoordinator(
            groups, scheduler_block_size=BLOCK, hash_block_size=BLOCK
        ),
        token_dbs=[SimpleNamespace(metadata=md, block_size=BLOCK)],
        client=SimpleNamespace(
            batch_exist=lambda keys: [int(key in store) for key in keys]
        ),
        tp_size=8,
        num_kv_head=8,
        pp_size=1,
        _kv_cache_groups=groups,
        _record_kv_connector_operation=lambda *args, **kwargs: None,
    )
    return md, worker


def test_gqa_dcp_lookup_hits_what_the_workers_wrote():
    hashes = [BlockHash(hashlib.sha256(b"%d" % i).digest()) for i in range(4)]
    md, _ = _gqa_dcp_lookup(set())
    store = {
        PoolKey(
            dataclasses.replace(md, tp_rank=head, dcp_rank=head % 2), h.hex()
        ).to_bytes()
        for head in range(8)
        for h in hashes
    }
    _, worker = _gqa_dcp_lookup(store)
    assert DfkvStoreWorker.lookup(worker, 4 * BLOCK, hashes) == 4 * BLOCK
