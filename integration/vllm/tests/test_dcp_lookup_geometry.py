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
from pathlib import Path
from types import SimpleNamespace

import torch  # noqa: F401  (spec dtype; provided by the runtime image)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from dfkv_vllm.coordinator import DfkvStoreCoordinator
from dfkv_vllm.data import KeyMetadata, PoolKey
from dfkv_vllm.worker import DfkvStoreWorker, SG_MAX_SEGS, _sg_group_key

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


def _lookup_hit_tokens(store: set[bytes], hashes: list[BlockHash]) -> int:
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
        token_dbs=[SimpleNamespace(metadata=_md(0), block_size=BLOCK)],
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
