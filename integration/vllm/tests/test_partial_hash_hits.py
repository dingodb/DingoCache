"""Partial hash hits: DCP + hybrid prefixes hit at the hash block.

Geometry of Kimi-Linear-48B-A3B at TP8 DCP8: three KDA state groups with
256-token blocks and one MLA group whose 256-token block covers 2048 tokens
under DCP8 (the scheduler block, LCM). vLLM hands off the recurrent state
only at the prompt checkpoint ``floor((P - 1) / 256) * 256``, e.g. 39936
for P = 40000, which lies inside attention block 19 = [38912, 40960).
"""
import ctypes
import hashlib
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec, MLAAttentionSpec

from dfkv_vllm import scheduler as scheduler_module
from dfkv_vllm.coordinator import (
    DfkvStoreCoordinator,
    ExternalCachedBlockPool,
    partial_hash_hits_enabled,
)
from dfkv_vllm.data import (
    ChunkedTokenDatabase,
    DfkvStoreConnectorMetadata,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
    RequestTracker,
)
from dfkv_vllm.scheduler import DfkvStoreScheduler
from dfkv_vllm.worker import (
    DfkvStoreWorker,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)

HASH = 256
LCM = 2048
TP = DCP = 8
PROMPT = 40000
B = 39936                     # prompt checkpoint, inside attention block 19
STATE_GROUPS = (0, 1, 2)
ATTN = 3
NUM_BLOCKS = 24
BYTES = 8                     # bytes per block in every test pool

needs_fine_grained = pytest.mark.skipif(
    not getattr(FullAttentionManager, "supports_fine_grained_hash_lookup", False),
    reason="engine core without hash-block (partial) prefix hits",
)


def _hashes(tokens=PROMPT):
    return [
        BlockHash(hashlib.sha256(str(i).encode()).digest())
        for i in range(tokens // HASH)
    ]


def _groups():
    state = MambaSpec(
        block_size=HASH, shapes=((16,),), dtypes=(torch.uint8,),
        mamba_cache_mode="align",
    )
    mla = MLAAttentionSpec(
        block_size=LCM, num_kv_heads=1, head_size=1, dtype=torch.uint8,
    )
    return [KVCacheGroupSpec([f"kda{g}"], state) for g in STATE_GROUPS] + [
        KVCacheGroupSpec(["mla"], mla)
    ]


def _coord(partial=True):
    return DfkvStoreCoordinator(
        _groups(), LCM, HASH, enable_partial_hash_hits=partial
    )


def _metadata(group, rank=0):
    """The key coordinate rank ``rank`` writes for ``group`` (worker rules)."""
    state = group in STATE_GROUPS
    return KeyMetadata(
        model_name="kimi-linear", dp_size=1, dp_rank=-1, tp_size=TP,
        tp_rank=rank if state else 0, pcp_size=1, pcp_rank=0,
        dcp_size=DCP, dcp_rank=rank, pp_size=1, pp_rank=0,
        group_id=group, pool_name="mamba" if state else "kv",
    )


def _key(group, rank, h):
    return PoolKey(_metadata(group, rank), h.hex()).to_bytes()


class _Store:
    """In-memory dfkv substitute that gathers/scatters real CPU buffers."""

    def __init__(self):
        self.objects: dict[bytes, bytes] = {}
        self.puts: list[bytes] = []
        self.gets: list[tuple[bytes, int]] = []
        self.probes: list[list[bytes]] = []

    def batch_exist(self, keys):
        return [int(key in self.objects) for key in keys]

    def batch_retain(self, keys, ttl_s=0, mode=1, **kw):
        self.probes.append(list(keys))
        return self.batch_exist(keys)

    def batch_put_sg(self, keys, pointers, sizes):
        for key, ptrs, lens in zip(keys, pointers, sizes, strict=True):
            self.objects[key] = b"".join(
                ctypes.string_at(p, n) for p, n in zip(ptrs, lens, strict=True)
            )
            self.puts.append(key)
        return [0] * len(keys)

    def batch_get_auto_sg(self, keys, pointers, capacities):
        hits, lengths = [], []
        for key, ptrs, caps in zip(keys, pointers, capacities, strict=True):
            self.gets.append((key, sum(caps)))
            value = self.objects.get(key)
            hits.append(value is not None)
            lengths.append(len(value) if value is not None else 0)
            offset = 0
            for ptr, cap in zip(ptrs, caps, strict=True):
                if value is not None:
                    ctypes.memmove(ptr, value[offset:offset + cap], cap)
                offset += cap
        return hits, lengths


def _databases(rank, fill):
    """Real per-group databases over CPU pools; block i holds fill(g, i)."""
    pools, dbs = [], []
    for g, group in enumerate(_groups()):
        pool = ctypes.create_string_buffer(NUM_BLOCKS * BYTES)
        for block in range(NUM_BLOCKS):
            ctypes.memset(ctypes.addressof(pool) + block * BYTES, fill(g, block), BYTES)
        db = ChunkedTokenDatabase(
            _metadata(g, rank), group.kv_cache_spec.block_size, hash_block_size=HASH,
        )
        db.set_seg_layout([(ctypes.addressof(pool), BYTES, BYTES)])
        pools.append(pool)
        dbs.append(db)
    return dbs, pools


def _sender(store, rank, dbs):
    return KVCacheStoreSendingThread(
        store, _coord(), dbs, LCM, rank, 0, 1, "kv_producer", threading.Event(),
        tp_sharded_groups=frozenset(STATE_GROUPS),
    )


def _save(store, rank, **meta):
    dbs, pools = _databases(rank, lambda g, block: 10 * g + block + rank)
    sender = _sender(store, rank, dbs)
    fields = dict(
        req_id=f"save-{rank}", token_len_chunk=0,
        block_ids=([9], [9], [9], list(range(1, 21))),
        block_hashes=_hashes(), can_save=True,
    )
    job = ReqMeta(**{**fields, **meta})
    sender.add_stored_request(job)
    sender._handle_request(job)
    return pools


def _lookup(store, token_len=B, partial=True, hashes=None):
    groups = _groups()
    worker = SimpleNamespace(
        coord=_coord(partial),
        token_dbs=[
            SimpleNamespace(metadata=_metadata(g), block_size=group.kv_cache_spec.block_size)
            for g, group in enumerate(groups)
        ],
        client=store, tp_size=TP, num_kv_head=1, _kv_cache_groups=groups,
        _record_kv_connector_operation=lambda *a, **k: None,
    )
    return DfkvStoreWorker.lookup(worker, token_len, hashes or _hashes())[0]


def _stored_checkpoint(boundary=B, *, states_at=(), skip=()):
    """Every object a hit at ``boundary`` needs, on every rank, minus skip."""
    hashes = _hashes()
    store = _Store()
    view = _coord().block_hashes_for_spec(hashes, _groups()[ATTN].kv_cache_spec)
    objects = [(ATTN, r, view[i]) for r in range(DCP) for i in range(boundary // LCM)]
    objects += [(ATTN, r, hashes[boundary // HASH - 1]) for r in range(DCP)]
    for c in (boundary, *states_at):
        objects += [(g, r, hashes[c // HASH - 1]) for g in STATE_GROUPS for r in range(TP)]
    for group, rank, h in objects:
        if (group, rank, h) not in skip:
            store.objects[_key(group, rank, h)] = b"x"
    return store, hashes


# -- feature detection and coordinator ----------------------------------------


def test_feature_detection_follows_the_exported_hit_alignment():
    assert not partial_hash_hits_enabled(SimpleNamespace(), LCM)
    assert not partial_hash_hits_enabled(
        SimpleNamespace(cache_hit_alignment_tokens=None), LCM)
    assert not partial_hash_hits_enabled(
        SimpleNamespace(cache_hit_alignment_tokens=LCM), LCM)
    assert partial_hash_hits_enabled(
        SimpleNamespace(cache_hit_alignment_tokens=HASH), LCM)


def test_prompt_checkpoint_matches_the_engine_hand_off():
    coord = _coord()
    assert coord.partial_tail_boundary(PROMPT) == B
    assert coord.partial_tail_boundary(19 * LCM + 1) is None   # on the grid
    assert coord.partial_tail_boundary(200) is None             # no full hash block
    assert coord.partial_tail_boundary(1000) == 768
    assert _coord(partial=False).partial_tail_boundary(PROMPT) is None


@needs_fine_grained
def test_load_mask_lands_on_the_hash_block():
    masks = _coord().load_mask(_hashes(), B)
    for g in STATE_GROUPS:
        assert masks[g] == [False] * (B // HASH - 1) + [True]
    assert masks[ATTN] == [True] * 20            # cdiv keeps the partial block


@needs_fine_grained
def test_store_mask_and_disabled_masks_are_unchanged():
    enabled, disabled = _coord(), _coord(partial=False)
    assert enabled.store_mask(19 * LCM) == disabled.store_mask(19 * LCM)
    assert disabled.load_mask(_hashes(), B) == disabled.load_mask(_hashes(), 19 * LCM)


@needs_fine_grained
def test_coordinator_converges_on_the_checkpoint():
    coord, hashes = _coord(), _hashes()
    view = coord.block_hashes_for_spec(hashes, _groups()[ATTN].kv_cache_spec)
    exists = {(ATTN, bytes(view[i])) for i in range(19)}
    exists.add((ATTN, bytes(hashes[B // HASH - 1])))
    states = {(g, bytes(hashes[B // HASH - 1])) for g in STATE_GROUPS}
    pool = ExternalCachedBlockPool(HASH, exists | states)
    assert coord.find_longest_cache_hit(hashes, B, pool) == B
    # One state group short: no other checkpoint, no hit.
    pool = ExternalCachedBlockPool(HASH, exists | set(list(states)[:2]))
    assert coord.find_longest_cache_hit(hashes, B, pool) == 0


# -- lookup ---------------------------------------------------------------------


@needs_fine_grained
def test_lookup_hits_the_checkpoint_inside_the_attention_block():
    store, _ = _stored_checkpoint()
    assert _lookup(store) == B
    # One probe: 19 whole + 4 interior attention objects, and each state group
    # at the 19 grid boundaries + those 4 interior ones, under every namespace.
    (keys,) = store.probes
    assert len(keys) == (19 + 4) * DCP + 3 * (19 + 4) * TP


@needs_fine_grained
@pytest.mark.parametrize("missing,fallback", [
    ((1, 5, "state"), 19 * LCM),        # one state shard at the checkpoint
    ((ATTN, 0, "tail"), 19 * LCM),      # the attention tail
    ((ATTN, DCP - 1, "tail"), 19 * LCM),  # one DCP namespace of the tail
])
def test_lookup_falls_back_to_the_grid_checkpoint(missing, fallback):
    hashes = _hashes()
    group, rank, _ = missing
    skip = {(group, rank, hashes[B // HASH - 1])}
    store, _ = _stored_checkpoint(states_at=(19 * LCM,), skip=skip)
    assert _lookup(store) == fallback
    # Without a grid checkpoint there is no shorter hit.
    store, _ = _stored_checkpoint(skip=skip)
    assert _lookup(store) == 0


@needs_fine_grained
def test_lookup_requires_every_dcp_namespace_of_a_whole_block():
    hashes = _hashes()
    view = _coord().block_hashes_for_spec(hashes, _groups()[ATTN].kv_cache_spec)
    store, _ = _stored_checkpoint(skip={(ATTN, 3, view[7])})
    assert _lookup(store) == 0
    # A second probe covers only the first block not stored whole.
    assert len(store.probes) == 2
    assert len(store.probes[1]) == 7 * DCP + 3 * 7 * TP


@needs_fine_grained
def test_lookup_rejects_a_state_paired_with_another_boundarys_tail():
    """Core may pair the state at 39680 with the tail at 39936; LOAD would
    read the tail under H(39680), so the hit must not end there."""
    hashes = _hashes()
    skip = {(g, r, hashes[B // HASH - 1]) for g in STATE_GROUPS for r in range(TP)}
    store, _ = _stored_checkpoint(states_at=(B - HASH,), skip=skip)
    assert _lookup(store) == 0


def test_disabled_lookup_stays_on_the_grid():
    store, _ = _stored_checkpoint(states_at=(19 * LCM,))
    assert _lookup(store, token_len=19 * LCM, partial=False) == 19 * LCM


# -- save -----------------------------------------------------------------------


@needs_fine_grained
def test_save_publishes_one_attention_tail_per_rank_plus_states():
    store = _Store()
    hashes = _hashes()
    state_blocks = {g: {B: 5 + g} for g in STATE_GROUPS}
    for rank in range(DCP):
        _save(store, rank, token_len_chunk=19 * LCM, num_prompt_tokens=PROMPT,
              step_end=PROMPT, state_blocks=state_blocks)
    tail = hashes[B // HASH - 1]
    for rank in range(DCP):
        assert store.puts.count(_key(ATTN, rank, tail)) == 1
        for g in STATE_GROUPS:
            assert store.puts.count(_key(g, rank, tail)) == 1
        # The tail is the whole attention block 19 (table entry 20).
        assert store.objects[_key(ATTN, rank, tail)] == bytes([30 + 20 + rank]) * BYTES
        assert store.objects[_key(1, rank, tail)] == bytes([10 + 6 + rank]) * BYTES
    assert len(store.puts) == DCP * (19 + 1 + 3)
    assert _lookup(store) == B


@needs_fine_grained
def test_offload_only_save_stores_just_the_handed_off_states():
    store = _Store()
    for rank in range(TP):
        _save(store, rank, state_blocks={g: {B: 5 + g} for g in STATE_GROUPS})
    tail = _hashes()[B // HASH - 1]
    assert sorted(store.puts) == sorted(
        _key(g, rank, tail) for g in STATE_GROUPS for rank in range(TP)
    )


@needs_fine_grained
def test_short_prompt_publishes_its_tail_without_a_whole_block():
    store = _Store()
    _save(store, 0, num_prompt_tokens=1000, block_ids=([9], [9], [9], [4]))
    assert store.puts == [_key(ATTN, 0, _hashes()[768 // HASH - 1])]
    assert store.objects[store.puts[0]] == bytes([30 + 4]) * BYTES


# -- load -----------------------------------------------------------------------


@needs_fine_grained
def test_load_reads_the_tail_into_a_whole_block(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    store = _Store()
    state_blocks = {g: {B: 5 + g} for g in STATE_GROUPS}
    sources = _save(store, 0, token_len_chunk=19 * LCM, num_prompt_tokens=PROMPT,
                    step_end=PROMPT, state_blocks=state_blocks)
    dbs, pools = _databases(0, lambda g, block: 0)
    receiver = KVCacheStoreRecvingThread(
        store, _coord(), dbs, LCM, tp_rank=0, ready_event=threading.Event(),
    )
    destination = list(range(2, 22))
    receiver.load_request_sync(ReqMeta(
        req_id="load", token_len_chunk=B,
        block_ids=([1], [1], [1], destination), block_hashes=_hashes(),
        load_spec=LoadSpec(0, B, True, token_len=B),
    ))
    assert receiver.get_and_clear_block_ids_with_load_errors() == set()
    tail = _key(ATTN, 0, _hashes()[B // HASH - 1])
    assert (tail, BYTES) in store.gets                  # one whole-block object
    assert len(store.gets) == 20 + 3

    def block(pool, block_id):
        return ctypes.string_at(ctypes.addressof(pool) + block_id * BYTES, BYTES)

    for i, block_id in enumerate(destination):
        assert block(pools[ATTN], block_id) == block(sources[ATTN], i + 1)
    for g in STATE_GROUPS:
        assert block(pools[g], 1) == block(sources[g], 5 + g)


# -- scheduler ------------------------------------------------------------------


def _scheduler(partial=True, **fields):
    scheduler = object.__new__(DfkvStoreScheduler)
    for name, value in dict(
        kv_role="kv_producer", _block_size=LCM, _hash_block_size=HASH,
        _partial_hash_hits=partial, client=MagicMock(), load_specs={},
        _failed_load_req_ids=set(), _preempted_req_ids=set(), _retention={},
        _request_trackers={},
        _unfinished_requests={}, lookup_async=False, load_async=True,
        request_level_loads=True,
    ).items():
        setattr(scheduler, name, value)
    for name, value in fields.items():
        setattr(scheduler, name, value)
    return scheduler


def _request(num_tokens=PROMPT):
    return SimpleNamespace(
        request_id="r", num_tokens=num_tokens, num_prompt_tokens=num_tokens,
        block_hashes=[], kv_transfer_params=None,
    )


def test_lookup_length_is_hash_aligned():
    scheduler = _scheduler()
    scheduler.client.lookup.return_value = (B, 0)
    assert scheduler.get_num_new_matched_tokens(_request(), LCM) == (B - LCM, True)
    assert scheduler.client.lookup.call_args.args[1] == B


def test_disabled_lookup_length_is_unchanged():
    scheduler = _scheduler(partial=False)
    scheduler.client.lookup.return_value = (19 * LCM, 0)
    assert scheduler.get_num_new_matched_tokens(_request(), 0) == (19 * LCM, True)
    assert scheduler.client.lookup.call_args.args[1] == 19 * LCM


def test_external_hit_is_declined_over_a_local_partial_block():
    """Core copies a locally hit partial block on write; a load must not race."""
    scheduler = _scheduler()
    assert scheduler.get_num_new_matched_tokens(_request(), LCM + HASH) == (0, False)
    scheduler.client.lookup.assert_not_called()


def test_first_save_covering_the_prompt_publishes_the_tail_once():
    tracker = RequestTracker("r", 1000, ([1],))
    job = ReqMeta.from_request_tracker(tracker, LCM, num_prompt_tokens=1000)
    assert (job.can_save, job.token_len_chunk, job.num_prompt_tokens) == (True, 0, 1000)
    assert ReqMeta.from_request_tracker(tracker, LCM, num_prompt_tokens=1000) is None
    # Before the prompt is covered nothing is published.
    early = RequestTracker("r", 999, ([1],))
    assert ReqMeta.from_request_tracker(early, LCM, num_prompt_tokens=1000) is None


def test_short_prompt_tail_delays_block_free():
    tracker = RequestTracker("r", 1000, ([1],))
    scheduler = _scheduler(_request_trackers={"r": tracker})
    assert scheduler.request_finished(_request(1000), ([1],))[0] is False
    ReqMeta.from_request_tracker(tracker, LCM, num_prompt_tokens=1000)
    assert tracker.num_saved_tokens == 0
    assert scheduler.request_finished(_request(1000), ([1],))[0] is True


def _offloads(offloads):
    return SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(boundary_state_offloads=offloads))


def test_hand_off_without_a_save_gets_an_offload_only_meta():
    tracker = RequestTracker("r", PROMPT, ([9], [9], [9], [1]))
    request = SimpleNamespace(block_hashes=_hashes())
    scheduler = _scheduler(
        _request_trackers={"r": tracker}, _unfinished_requests={"r": (request, ())},
    )
    meta = DfkvStoreConnectorMetadata(set(), set())
    scheduler._attach_state_blocks(
        meta, _offloads({"r": [(g, 5 + g, B) for g in STATE_GROUPS]}))
    (job,) = meta.requests
    assert (job.req_id, job.token_len_chunk, job.can_save) == ("r", 0, True)
    assert job.state_blocks == {g: {B: 5 + g} for g in STATE_GROUPS}
    assert tracker.has_pending_offload
    assert scheduler.request_finished(_request(), ([1],))[0] is True
    tracker.reset()
    assert not tracker.has_pending_offload


def test_consumer_and_grid_hand_offs_get_no_offload_only_meta():
    tracker = RequestTracker("r", PROMPT, ([9], [9], [9], [1]))
    unfinished = {"r": (SimpleNamespace(block_hashes=[]), ())}
    for scheduler, offload in (
        (_scheduler(kv_role="kv_consumer"), B),
        (_scheduler(), 19 * LCM),
        (_scheduler(partial=False), B),
    ):
        scheduler._request_trackers = {"r": tracker}
        scheduler._unfinished_requests = unfinished
        meta = DfkvStoreConnectorMetadata(set(), set())
        scheduler._attach_state_blocks(meta, _offloads({"r": [(0, 5, offload)]}))
        assert meta.requests == []


def test_misaligned_warning_is_silent_with_partial_hash_hits(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(scheduler_module, "logger", logger)
    meta = DfkvStoreConnectorMetadata(set(), set())
    meta.add_request(ReqMeta("r", LCM, ([9], [9], [9], [1]), [], can_save=True))
    _scheduler()._attach_state_blocks(meta, _offloads({"r": [(0, 5, B)]}))
    assert meta.requests[0].state_blocks == {0: {B: 5}}
    logger.warning.assert_not_called()
    _scheduler(partial=False)._attach_state_blocks(meta, _offloads({"r": [(0, 5, B)]}))
    logger.warning.assert_called_once()
