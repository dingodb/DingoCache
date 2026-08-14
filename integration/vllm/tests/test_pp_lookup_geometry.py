"""PP key-space hit-geometry regression test (PP>1 external-cache miss fix).

Under PP (pipeline parallelism) the layers are PARTITIONED across pp stages:
each (group, chunk) is SAVED and LOADED under the owning stage's own pp_rank
(worker.py process_tokens and the recv path both use db.metadata as-is).
The lookup must therefore probe only this worker's own pp_rank.

The pre-fix lookup expanded candidates across every pp_rank
(``for pp in range(pp_size)``) and multiplied the per-chunk quorum by
pp_size (``rank_probes *= pp_size``).  Cross-pp probes never match --
nothing ever saves under a foreign stage's pp_rank -- so present stayed 0,
hit_length collapsed to 0 and the external prefix hit rate was 0% on every
pp_size>1 deployment (field-verified on GLM TP8/PP2).

Same harness style as test_dcp_lookup_geometry.py: synthetic dfkv key-space
contents drive the REAL ``DfkvStoreWorker.lookup``.  Runs in the engine
image (vllm + torch provided by it).  No GPU or dfkv server needed: the
client is faked, only the pure-python key/lookup math is exercised.
"""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch  # noqa: F401  (spec dtype; provided by the runtime image)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from dfkv_vllm.coordinator import DfkvStoreCoordinator
from dfkv_vllm.data import KeyMetadata, PoolKey
from dfkv_vllm.worker import DfkvStoreWorker

BLOCK = 64
NCHUNK = 64
PP = 2

_METADATA = {
    "model_name": "m",
    "dp_size": 1,
    "dp_rank": -1,
    "tp_size": 8,
    "tp_rank": 0,
    "pcp_size": 1,
    "pcp_rank": 0,
    "dcp_size": 1,
    "dcp_rank": 0,
    "pp_size": PP,
    "group_id": 0,
}


def _md(pp_rank: int) -> KeyMetadata:
    return KeyMetadata(**{**_METADATA, "pp_rank": pp_rank})


def _onewire_key(md: KeyMetadata, h: BlockHash) -> bytes:
    # Same on-wire form lookup probes and the save path stores.
    return PoolKey(md, h.hex()).to_bytes()


def _hashes() -> list[BlockHash]:
    return [BlockHash(hashlib.sha256(f"{i}".encode()).digest()) for i in range(NCHUNK)]


class _RecordingClient:
    """Fake client that also records every probed key, so a test can assert
    the probe SET itself, not just the resulting hit length."""

    def __init__(self, present: set[bytes]):
        self._present = present
        self.probed: list[bytes] = []

    def batch_exist(self, keys):
        self.probed.extend(keys)
        return [1 if k in self._present else 0 for k in keys]


def _run_lookup(
    md: KeyMetadata, store: set[bytes], hashes: list[BlockHash]
) -> tuple[int, list[bytes]]:
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
    client = _RecordingClient(store)
    self_ = SimpleNamespace(
        coord=coord,
        token_dbs=[SimpleNamespace(metadata=md, block_size=BLOCK)],
        client=client,
        tp_size=8,
        num_kv_head=1,
        pp_size=PP,
        _kv_cache_groups=groups,
        _record_kv_connector_operation=lambda *a, **k: None,
    )
    hit = DfkvStoreWorker.lookup(self_, NCHUNK * BLOCK, hashes)
    return hit, client.probed


def test_pp2_lookup_probes_only_own_pp_rank():
    """The probe set must be exactly this worker's own-pp_rank keys: one
    probe per chunk (tp_count=min(tp_size, num_kv_head)=1), with no foreign
    pp_rank coordinate and no pp_size multiplication of the candidate list.
    Locks the SAVE/LOAD <-> lookup key-space contract for both stages."""
    hashes = _hashes()
    for rank in range(PP):
        own = {_onewire_key(_md(rank), h) for h in hashes}
        foreign = {_onewire_key(_md((rank + 1) % PP), h) for h in hashes}
        _, probed = _run_lookup(_md(rank), own, hashes)
        assert set(probed) == own, (
            f"pp_rank={rank}: lookup must probe exactly its own-rank keys"
        )
        assert len(probed) == NCHUNK, (
            f"pp_rank={rank}: expected 1 probe/chunk, got {len(probed)} "
            f"(a pp_size-multiplied candidate list probes {PP}x too many)"
        )
        assert not (set(probed) & foreign), (
            f"pp_rank={rank}: lookup probed foreign pp_rank coordinates"
        )


def test_pp2_own_rank_store_full_hit_without_other_rank():
    """Chunks stored ONLY under this worker's own pp_rank -- the other
    stage's keys entirely absent, as is always the case since stages save
    disjoint layer sets -- must report the FULL prefix.  This fails both
    pre-fix defects at once: a pp_size-multiplied per-chunk quorum
    (present=1 < expected=2) and foreign-rank probing, either of which
    zeroes hit_length."""
    hashes = _hashes()
    for rank in range(PP):
        own = {_onewire_key(_md(rank), h) for h in hashes}
        assert len(own) == NCHUNK
        hit, _ = _run_lookup(_md(rank), own, hashes)
        assert hit == NCHUNK * BLOCK, (
            f"pp_rank={rank}: other-stage absence must not zero hit_length, "
            f"got {hit}/{NCHUNK * BLOCK}"
        )


def test_pp2_lookup_misses_foreign_rank_only_store():
    """Negative control: keys stored only under the OTHER stage's pp_rank
    must NOT match -- nothing in this geometry is loadable by this stage.
    Locks the regression class: probing coordinates the SAVE side of this
    stage never writes."""
    hashes = _hashes()
    foreign_only = {_onewire_key(_md(1), h) for h in hashes}
    hit, _ = _run_lookup(_md(0), foreign_only, hashes)
    assert hit == 0, (
        f"pp_rank=0 lookup must not hit @pp1-only contents, got {hit}"
    )
