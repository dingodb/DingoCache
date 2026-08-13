"""DCP key-space hit-geometry regression test (issue #70 follow-up).

Under DCP (Decode Context Parallelism) the MLA KV is SHARDED across dcp
ranks:  rank r stores its shard chunks under keys tagged ``@dcp{r}``
(worker.py KeyMetadata), while the scheduler-side lookup (worker.py
``DfkvStoreWorker.lookup``) probes candidates from worker-rank-0's metadata,
i.e. ``@dcp0``, with tp_count=min(tp_size, num_kv_head)=1 probes per chunk.

This file drives the REAL lookup against synthetic dfkv key-space contents
modelling the two save geometries, so the coverage contract survives
refactors instead of living only in tribal knowledge:

* geometry A (post-#70, default cp_kv_cache_interleave_size=1):
  every rank stores every chunk under its own ``@dcp{r}`` namespace (each
  rank holds 1/dcp of every block's tokens - the shard is per-chunk, not
  per-token-range).  ``@dcp0`` keys therefore exist for every chunk and the
  lookup reports the FULL prefix.
* geometry B (pre-#70 put_step=tp_size stride, fixed by #70/v1.10.0):
  chunk c is stored only by the single rank c%dcp under ``@dcp{c%dcp}``.
  The lookup sees only the dcp=0 quarter/eighth -> the external hit
  collapses to ~1/dcp of the prompt.  This is the field-reported
  "low dfkv hit rate with DCP"; deploys on < v1.10.0 show exactly it.

Variation B is asserted as a DOCUMENTED NEGATIVE: if a future change
reintroduces a store stride (or block-grain interleave keyed per owning
rank) without teaching lookup about dcp ownership, A must not silently
degrade into B-like coverage.

Runs in the engine image (vllm + torch provided by it -- see
integration/vllm/pyproject.toml). No GPU or dfkv server needed: the client
is faked, only the pure-python key/lookup math is exercised.
"""

import hashlib
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import torch  # noqa: F401  (spec dtype; provided by the runtime image)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from dfkv_vllm.coordinator import DfkvStoreCoordinator
from dfkv_vllm.data import ChunkedTokenDatabase, KeyMetadata, PoolKey, ReqMeta
from dfkv_vllm.worker import (
    DfkvStoreWorker,
    KVCacheStoreSendingThread,
    SG_MAX_SEGS,
    _sg_group_key,
)

BLOCK = 64
NCHUNK = 64
DCP = 8

_METADATA = {
    "model_name": "m",
    "dp_size": 1,
    "dp_rank": -1,
    "tp_size": 8,
    "tp_rank": 0,
    "pcp_size": 1,
    "pcp_rank": 0,
    "dcp_size": DCP,
    "pp_size": 1,
    "pp_rank": 0,
    "group_id": 0,
}


def _md(dcp_rank: int) -> KeyMetadata:
    return KeyMetadata(**{**_METADATA, "dcp_rank": dcp_rank})


def _onewire_key(md: KeyMetadata, h: BlockHash) -> str:
    # Same on-wire form lookup probes and the save path store: group-0 SG key.
    return _sg_group_key(PoolKey(md, h.hex()).to_bytes(), 0, SG_MAX_SEGS)


def _make_store(geometry: str, hashes: list[BlockHash]) -> set[bytes]:
    store: set[bytes] = set()
    for r in range(DCP):
        for c, h in enumerate(hashes):
            if geometry == "A" or (geometry == "B" and r == c % DCP):
                store.add(_onewire_key(_md(r), h))
    return store


def _lookup_hit_tokens(
    store: set[bytes], hashes: list[BlockHash], md: KeyMetadata | None = None
) -> int:
    class _FakeClient:
        _sg_segs_cache = SG_MAX_SEGS

        def __init__(self, present: set[bytes]):
            self._present = present

        def batch_exist(self, keys):
            return [1 if k in self._present else 0 for k in keys]

    spec = FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    groups = [KVCacheGroupSpec([f"layer{i}" for i in range(43)], spec)]
    coord = DfkvStoreCoordinator(
        groups, scheduler_block_size=BLOCK, hash_block_size=BLOCK
    )
    self_ = SimpleNamespace(
        coord=coord,
        token_dbs=[
            SimpleNamespace(metadata=md if md is not None else _md(0),
                            block_size=BLOCK)
        ],
        client=_FakeClient(store),
        tp_size=8,
        num_kv_head=1,
        pp_size=1,
        _kv_cache_groups=groups,
        _record_kv_connector_operation=lambda *a, **k: None,
    )
    return DfkvStoreWorker.lookup(self_, NCHUNK * BLOCK, hashes)


def _hashes() -> list[BlockHash]:
    return [BlockHash(hashlib.sha256(f"{i}".encode()).digest()) for i in range(NCHUNK)]


def test_post_70_geometry_lookup_full_prefix():
    """Every rank stores every chunk (@dcp{r}): rank-0 lookup must report
    the complete prefix, i.e. DCP on >= v1.10.0 has full L3 hit coverage."""
    hashes = _hashes()
    store = _make_store("A", hashes)
    assert len(store) == DCP * NCHUNK
    hit = _lookup_hit_tokens(store, hashes)
    assert hit == NCHUNK * BLOCK, f"expected full prefix, got {hit}/{NCHUNK * BLOCK}"


def test_pre_70_geometry_lookup_collapses():
    """Negative control: chunk c only under @dcp{c%dcp} (the pre-#70 stride
    or a block-grain interleave) truncates the lookup to ~1/dcp. If this
    ever turns into a full hit the incident class deserves a fresh look."""
    hashes = _hashes()
    store = _make_store("B", hashes)
    hit = _lookup_hit_tokens(store, hashes)
    assert 0 < hit < NCHUNK * BLOCK // 2, (
        f"documented-negative geometry should truncate badly, got {hit}"
    )


# Replicated MLA (plain-TP MLA -- put_step==tp_size, no DCP/PCP, e.g.
# GLM-5.2 TP16 / V4-Flash TP8): the SAVE/dedup/LOAD paths encode the single
# canonical tp_rank=-1 (worker.py metadata init), NOT a per-TP coordinate.
# The pre-fix lookup probed tp_rank=0 unconditionally
# (tp_count = min(tp_size, num_kv_head) = 1), so external L3 hit detection
# was constantly 0 for exactly this topology.
_MLA_METADATA = {**_METADATA, "dcp_size": 1, "tp_rank": -1}


def _mla_md() -> KeyMetadata:
    return KeyMetadata(**{**_MLA_METADATA, "dcp_rank": 0})


def _legacy_tp0_md() -> KeyMetadata:
    # The coordinate the pre-fix lookup probed for ANY metadata: tp_rank=0.
    return KeyMetadata(**{**_METADATA, "dcp_size": 1, "dcp_rank": 0})


def test_replicated_mla_lookup_full_prefix():
    """Every chunk stored under the canonical tp_rank=-1 (group 0): lookup
    with replicated-MLA metadata must probe that same coordinate and report
    the complete prefix, i.e. plain-TP MLA has full external L3 coverage."""
    hashes = _hashes()
    md = _mla_md()
    store = {_onewire_key(md, h) for h in hashes}
    assert len(store) == NCHUNK
    hit = _lookup_hit_tokens(store, hashes, md=md)
    assert hit == NCHUNK * BLOCK, (
        f"replicated-MLA lookup must hit the tp_rank=-1 store, "
        f"got {hit}/{NCHUNK * BLOCK}"
    )


def test_replicated_mla_lookup_misses_legacy_tp0_coordinate():
    """Negative control: keys under the OLD probe coordinate (tp_rank=0)
    must NOT match a replicated-MLA lookup -- nothing in this geometry ever
    saves there. Locks the regression: probing coordinates the SAVE side
    never writes."""
    hashes = _hashes()
    store = {_onewire_key(_legacy_tp0_md(), h) for h in hashes}
    assert len(store) == NCHUNK
    hit = _lookup_hit_tokens(store, hashes, md=_mla_md())
    assert hit == 0, (
        f"lookup must not probe the legacy tp_rank=0 coordinate, got {hit}"
    )


# ---------------------------------------------------------------------------
# Partial scatter-group probe/heal (v2 review #14)
#
# A chunk is stored as G = ceil(per-layer segments / SG_MAX_SEGS) explicit
# SG objects (_group_segments_sg): a 40-layer model with 29-wide RDMA spans
# 2 groups (29 + 11). The pre-fix dedup/lookup probed group 0 only, so a
# partial write (group 0 in, a sibling group lost to a failed put or
# eviction) probed "stored": dedup skipped the rewrite, lookup counted it
# cached, and every load of that chunk then failed wholesale -- recomputed
# for the chunk's whole LRU lifetime while pinning ring capacity. The tests
# below drive the REAL lookup and the REAL save thread against the fixed
# all-group geometry.
# ---------------------------------------------------------------------------

_SG_LAYERS = SG_MAX_SEGS + 11  # segments/chunk -> exactly 2 SG groups
_SAVE_CHUNKS = 4


def _sg_onewire_key(md: KeyMetadata, h: BlockHash, grp: int) -> bytes:
    return _sg_group_key(PoolKey(md, h.hex()).to_bytes(), grp, SG_MAX_SEGS)


def _spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )


def _lookup_hit_tokens_sg(
    store: set[bytes],
    hashes: list[BlockHash],
    md: KeyMetadata,
    segs_per_block: int,
) -> int:
    # Same as _lookup_hit_tokens but the fake db carries a seg layout, so
    # _num_sg_groups resolves multi-group chunks exactly like the real
    # ChunkedTokenDatabase does after register_kv_caches.
    class _FakeClient:
        _sg_segs_cache = SG_MAX_SEGS

        def __init__(self, present: set[bytes]):
            self._present = present

        def batch_exist(self, keys):
            return [1 if k in self._present else 0 for k in keys]

    groups = [
        KVCacheGroupSpec([f"layer{i}" for i in range(segs_per_block)], _spec())
    ]
    coord = DfkvStoreCoordinator(
        groups, scheduler_block_size=BLOCK, hash_block_size=BLOCK
    )
    self_ = SimpleNamespace(
        coord=coord,
        token_dbs=[
            SimpleNamespace(
                metadata=md,
                block_size=BLOCK,
                _seg_layout=[(0, 4096, 4096)] * segs_per_block,
            )
        ],
        client=_FakeClient(store),
        tp_size=8,
        num_kv_head=1,
        pp_size=1,
        _kv_cache_groups=groups,
        _record_kv_connector_operation=lambda *a, **k: None,
    )
    return DfkvStoreWorker.lookup(self_, NCHUNK * BLOCK, hashes)


def test_multigroup_lookup_hits_when_all_groups_present():
    """Both SG groups of every chunk stored -> full prefix hit. Guard: the
    all-group probe must not LOSE coverage the group-0 probe used to see."""
    hashes = _hashes()
    md = _mla_md()
    store = {_sg_onewire_key(md, h, grp) for h in hashes for grp in (0, 1)}
    assert len(store) == 2 * NCHUNK
    hit = _lookup_hit_tokens_sg(store, hashes, md, _SG_LAYERS)
    assert hit == NCHUNK * BLOCK, (
        f"all-group store must yield a full prefix, got {hit}/{NCHUNK * BLOCK}"
    )


def test_multigroup_lookup_misses_when_either_group_missing():
    """Root regression (lookup side): with only ONE of the two SG groups
    stored the chunk must NOT count as cached. The pre-fix group-0 probe
    reported the group-0-only case as a full hit while every load of the
    chunk failed."""
    hashes = _hashes()
    md = _mla_md()
    for present_grp in (0, 1):
        store = {_sg_onewire_key(md, h, present_grp) for h in hashes}
        assert len(store) == NCHUNK
        hit = _lookup_hit_tokens_sg(store, hashes, md, _SG_LAYERS)
        assert hit == 0, (
            f"group{present_grp}-only partial store must probe as uncached, "
            f"got {hit}/{NCHUNK * BLOCK} tokens"
        )


class _FakeSaveClient:
    """In-memory dfkv stand-in for the save thread: scripted per-key put
    failures and a recording remove RPC."""

    _sg_segs_cache = SG_MAX_SEGS

    def __init__(
        self,
        present: set[bytes] | None = None,
        fail_keys: set[bytes] | None = None,
    ):
        self._present = set(present or ())
        self._fail_keys = set(fail_keys or ())
        self.exist_calls: list[list[bytes]] = []
        self.put_calls: list[list[bytes]] = []
        self.removed: list[bytes] = []

    def batch_exist(self, keys):
        keys = list(keys)
        self.exist_calls.append(keys)
        return [1 if k in self._present else 0 for k in keys]

    def batch_put_sg(self, keys, seg_ptrs, seg_sizes):
        keys = list(keys)
        self.put_calls.append(keys)
        res = []
        for k in keys:
            if k in self._fail_keys:
                res.append(1)
            else:
                res.append(0)
                self._present.add(k)
        return res

    def supports_remove(self):
        return True

    def batch_remove(self, keys):
        keys = list(keys)
        self.removed.extend(keys)
        for k in keys:
            self._present.discard(k)
        return [1] * len(keys)


def _run_save_request(
    client,
    md: KeyMetadata,
    hashes: list[BlockHash],
    segs_per_block: int = _SG_LAYERS,
    record_ops: list | None = None,
) -> None:
    """Drive the REAL KVCacheStoreSendingThread._handle_request over a real
    1-group coordinator + a real ChunkedTokenDatabase whose legacy layout
    tables emulate a ``segs_per_block``-layer model (prepare_value emits one
    segment per entry per block, so _num_sg_groups sees the legacy branch)."""
    groups = [
        KVCacheGroupSpec([f"layer{i}" for i in range(segs_per_block)], _spec())
    ]
    coord = DfkvStoreCoordinator(
        groups, scheduler_block_size=BLOCK, hash_block_size=BLOCK
    )
    db = ChunkedTokenDatabase(md, block_size=BLOCK, hash_block_size=BLOCK)
    db.set_kv_caches_base_addr([0x1000 + i * 4096 * 1024 for i in range(segs_per_block)])
    db.set_block_len([4096] * segs_per_block)
    thread = KVCacheStoreSendingThread(
        client=client,
        coord=coord,
        token_databases=[db],
        block_size=BLOCK,
        tp_rank=0,
        stripe_idx=0,
        stripe_step=1,
        kv_role="kv_both",
        ready_event=threading.Event(),
        enable_kv_event=False,
        record_operation=(
            (lambda **kw: record_ops.append(kw))
            if record_ops is not None
            else None
        ),
    )
    nblocks = len(hashes)
    meta = ReqMeta(
        req_id="req-sg",
        token_len_chunk=nblocks * BLOCK,
        block_ids=(list(range(nblocks)),),
        block_hashes=list(hashes),
    )
    thread.add_stored_request("req-sg")
    thread._handle_request(meta)
    assert thread.stored_requests.get("req-sg", 0) == 0


def test_save_dedup_rewrites_when_sibling_group_missing():
    """Root regression (dedup side): group 0 of every chunk is stored but
    its sibling group is gone. The pre-fix gate probed group 0 only and
    skipped the save entirely, so the partial chunk was never rewritten.
    The gate must now probe both groups and re-store the chunks."""
    hashes = _hashes()[:_SAVE_CHUNKS]
    md = _mla_md()
    present = {_sg_onewire_key(md, h, 0) for h in hashes}
    client = _FakeSaveClient(present=present)
    _run_save_request(client, md, hashes)
    assert client.exist_calls, "dedup gate must probe first"
    assert len(client.exist_calls[0]) == 2 * len(hashes), (
        f"probe must cover both SG groups per chunk: "
        f"{len(client.exist_calls[0])} keys for {len(hashes)} chunks"
    )
    assert client.put_calls, (
        "group-0-only partial chunks must be rewritten, not skipped"
    )
    expected_keys = {
        _sg_onewire_key(md, h, grp) for h in hashes for grp in (0, 1)
    }
    assert set(client.put_calls[0]) == expected_keys
    assert expected_keys <= client._present


def test_save_dedup_skips_when_all_groups_present():
    """Guard: with BOTH SG groups of every chunk stored the dedup gate keeps
    its skip-fast behavior (no put at all)."""
    hashes = _hashes()[:_SAVE_CHUNKS]
    md = _mla_md()
    present = {_sg_onewire_key(md, h, grp) for h in hashes for grp in (0, 1)}
    client = _FakeSaveClient(present=present)
    _run_save_request(client, md, hashes)
    assert client.put_calls == [], (
        f"fully-stored chunks must stay dedup-skipped, got puts {client.put_calls}"
    )


def test_save_partial_put_failure_removes_sibling_groups():
    """Cleanup: when one group put fails, the chunk's sibling group keys
    (including the successfully written one) must receive a best-effort
    remove so no half-stored ghost lingers in the ring."""
    hashes = _hashes()[:_SAVE_CHUNKS]
    md = _mla_md()
    victim_g1 = _sg_onewire_key(md, hashes[1], 1)
    ops: list[dict] = []
    client = _FakeSaveClient(fail_keys={victim_g1})
    _run_save_request(client, md, hashes, record_ops=ops)
    assert set(client.removed) == {
        _sg_onewire_key(md, hashes[1], 0),
        victim_g1,
    }, (
        "both group keys of the failed chunk must be removed, got "
        f"{[k.hex()[-8:] for k in client.removed]}"
    )
    assert not {_sg_onewire_key(md, hashes[1], grp) for grp in (0, 1)} & client._present
    rm_ops = [op for op in ops if op.get("operation") == "save_remove_partial"]
    assert rm_ops and rm_ops[0].get("status") == "ok", (
        f"cleanup metric missing or wrong: {ops}"
    )


def test_save_partial_put_failure_without_remove_rpc_is_inert():
    """Degradation: a client/lib without the remove RPC (older libdfkv)
    must skip cleanup silently -- never raise into the save path -- and the
    metric records the unsupported skip."""

    class _OldLibClient(_FakeSaveClient):
        def supports_remove(self):
            return False

        def batch_remove(self, keys):  # pragma: no cover - must never run
            raise AssertionError("old lib without remove RPC must be skipped")

    hashes = _hashes()[:_SAVE_CHUNKS]
    md = _mla_md()
    ops: list[dict] = []
    client = _OldLibClient(fail_keys={_sg_onewire_key(md, hashes[0], 1)})
    _run_save_request(client, md, hashes, record_ops=ops)  # must not raise
    assert client.put_calls, "put was attempted before the scripted failure"
    rm_ops = [op for op in ops if op.get("operation") == "save_remove_partial"]
    assert rm_ops and rm_ops[0].get("status") == "unsupported", (
        f"unsupported-cleanup metric missing or wrong: {ops}"
    )


def test_single_group_chunk_probe_geometry_unchanged():
    """G=1 guard: a chunk that fits ONE SG group keeps the byte-identical
    pre-fix geometry -- exactly one group-0 key per chunk in the dedup probe
    and in the rewrite, and unchanged skip behavior."""
    hashes = _hashes()[:_SAVE_CHUNKS]
    md = _mla_md()
    single_layers = SG_MAX_SEGS - 9  # 20 segments -> exactly 1 SG group
    # Fully present -> still skipped, probe list identical to the pre-fix
    # group-0-only form.
    present = {_sg_onewire_key(md, h, 0) for h in hashes}
    client = _FakeSaveClient(present=present)
    _run_save_request(client, md, hashes, segs_per_block=single_layers)
    assert client.put_calls == []
    old_probe_form = [_sg_onewire_key(md, h, 0) for h in hashes]
    assert client.exist_calls[0] == old_probe_form, (
        "single-group chunk probe list must be byte-identical to the "
        "pre-fix group-0-only form"
    )
    # Absent -> rewrite puts exactly the old group-0 keys.
    client2 = _FakeSaveClient()
    _run_save_request(client2, md, hashes, segs_per_block=single_layers)
    assert len(client2.put_calls) == 1
    assert client2.put_calls[0] == old_probe_form
