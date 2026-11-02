"""Per-request retention TTL through the vLLM connector.

Scheduler: the TTL comes from ``kv_transfer_params``, travels with the lookup,
and the worker's applied TTL is stamped on ReqMeta and reported as usage.
Worker: lookup probes with REFRESH and grants the TTL only when the ring
supports retention and the whole-prefix precheck passes (decision 9).
"""
import hashlib
from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from dfkv_common import cabi as dfkv_cabi
from dfkv_vllm.coordinator import DfkvStoreCoordinator
from dfkv_vllm.data import KeyMetadata, PoolKey
from dfkv_vllm.scheduler import DfkvStoreScheduler, _requested_retention_ttl
from dfkv_vllm.worker import DfkvStoreWorker

BLOCK = 64


def test_requested_ttl_accepts_only_positive_integers():
    def req(params):
        return SimpleNamespace(kv_transfer_params=params)

    assert _requested_retention_ttl(req(None)) == 0
    assert _requested_retention_ttl(req({})) == 0
    assert _requested_retention_ttl(req({"dfkv_retention_ttl_s": 300})) == 300
    assert _requested_retention_ttl(req({"dfkv_retention_ttl_s": "300"})) == 0
    assert _requested_retention_ttl(req({"dfkv_retention_ttl_s": True})) == 0
    assert _requested_retention_ttl(req({"dfkv_retention_ttl_s": -5})) == 0


def _scheduler(lookup_result, kv_role="kv_both"):
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._retention = {}
    scheduler._failed_load_req_ids = set()
    scheduler._request_trackers = {}
    scheduler.load_specs = {}
    scheduler.lookup_async = False
    scheduler.load_async = False
    scheduler.request_level_loads = False
    scheduler.kv_role = kv_role
    scheduler._block_size = BLOCK
    calls = []

    def lookup(request_id, token_len, hashes, non_block=False,
               retain_ttl_s=0):
        calls.append(retain_ttl_s)
        return lookup_result

    scheduler.client = SimpleNamespace(lookup=lookup, discard=lambda _r: None)
    return scheduler, calls


def _request(ttl, num_tokens=200):
    return SimpleNamespace(
        request_id="r1", num_tokens=num_tokens, num_prompt_tokens=num_tokens,
        block_hashes=[], kv_transfer_params={"dfkv_retention_ttl_s": ttl},
    )


def test_scheduler_forwards_ttl_and_reports_usage():
    scheduler, calls = _scheduler((128, 300))
    request = _request(300)
    assert scheduler.get_num_new_matched_tokens(request, 64) == (64, False)
    assert calls == [300]
    # (requested, applied by the worker, cached = max(local, external)).
    assert scheduler._retention["r1"] == (300, 300, 128)
    delay, params = scheduler.request_finished(request, ([],))
    assert delay is False
    # prompt 200 -> 192 block-aligned; 128 served from cache.
    assert params == {"dfkv_cache_usage": {
        "cached_tokens": 128, "cache_write_tokens": 64, "applied_ttl_s": 300}}
    assert "r1" not in scheduler._retention


def test_best_effort_request_reports_nothing():
    scheduler, calls = _scheduler((128, 0))
    request = _request(0)
    scheduler.get_num_new_matched_tokens(request, 0)
    assert calls == [0]
    assert scheduler.request_finished(request, ([],)) == (False, None)


def test_consumer_never_applies_a_ttl():
    scheduler, _ = _scheduler((128, 300), kv_role="kv_consumer")
    request = _request(300)
    scheduler.get_num_new_matched_tokens(request, 0)
    _, params = scheduler.request_finished(request, ([],))
    assert params["dfkv_cache_usage"]["applied_ttl_s"] == 0


def _worker(store, *, supported=1, max_ttl=3600, precheck=True):
    spec = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.bfloat16,
    )
    groups = [KVCacheGroupSpec(["layer"], spec)]
    md = KeyMetadata(
        model_name="m", dp_size=1, dp_rank=-1, tp_size=1, tp_rank=0,
        pcp_size=1, pcp_rank=0, dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0,
        group_id=0,
    )
    probes = []

    def batch_retain(keys, ttl_s, mode, **kw):
        probes.append((mode, len(keys)))
        return [dfkv_cabi.DFKV_RET_REFRESHED if k in store
                else dfkv_cabi.DFKV_RET_MISS for k in keys]

    worker = SimpleNamespace(
        coord=DfkvStoreCoordinator(
            groups, scheduler_block_size=BLOCK, hash_block_size=BLOCK),
        token_dbs=[SimpleNamespace(metadata=md, block_size=BLOCK)],
        client=SimpleNamespace(
            batch_retain=batch_retain,
            retain_limits=lambda: (supported, max_ttl),
            retain_precheck=lambda keys: precheck,
        ),
        tp_size=1, num_kv_head=1, pp_size=1, _kv_cache_groups=groups,
        _record_kv_connector_operation=lambda *a, **k: None,
    )
    return md, worker, probes


def _hashes(n):
    return [BlockHash(hashlib.sha256(b"%d" % i).digest()) for i in range(n)]


def test_lookup_refreshes_and_applies_clamped_ttl():
    hashes = _hashes(4)
    md, _, _ = _worker(set())
    store = {PoolKey(md, h.hex()).to_bytes() for h in hashes[:2]}
    _, worker, probes = _worker(store, max_ttl=600)
    hit, applied = DfkvStoreWorker.lookup(worker, 4 * BLOCK, hashes, 3600)
    assert hit == 2 * BLOCK
    assert applied == 600
    assert all(mode == dfkv_cabi.DFKV_RETAIN_REFRESH for mode, _ in probes)


def test_lookup_without_ttl_or_support_or_headroom_applies_nothing():
    hashes = _hashes(2)
    _, worker, _ = _worker(set())
    assert DfkvStoreWorker.lookup(worker, 2 * BLOCK, hashes, 0)[1] == 0
    _, worker, _ = _worker(set(), supported=0)
    assert DfkvStoreWorker.lookup(worker, 2 * BLOCK, hashes, 300)[1] == 0
    _, worker, _ = _worker(set(), precheck=False)
    assert DfkvStoreWorker.lookup(worker, 2 * BLOCK, hashes, 300)[1] == 0
