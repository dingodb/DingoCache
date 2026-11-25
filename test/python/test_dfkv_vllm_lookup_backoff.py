"""A lookup that backs off probes each object once.

A recurrent-state group's load mask is true only at the candidate's tail, so
when the state at the tail is missing the lookup tries a shorter candidate.
Each round used to probe the whole prefix again: with 128 LCM chunks and no
states, 128 probes of about 8 400 keys in all. Now the first candidate is
probed, then, if it fails, every shorter candidate's remaining objects in one
probe, and the rounds are answered from what was found. Runs on the
stand-ins of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integration" / "vllm" / "tests"))
from _coordinator_fake import fake_coordinator  # noqa: E402

from dfkv_vllm.data import KeyMetadata, PoolKey  # noqa: E402
import dfkv_vllm.worker as worker_module  # noqa: E402
from dfkv_vllm.worker import DfkvStoreWorker  # noqa: E402

LCM = 64


def _md(group: int) -> KeyMetadata:
    return KeyMetadata(
        model_name="m", dp_size=1, dp_rank=-1, tp_size=1, tp_rank=-1,
        pcp_size=1, pcp_rank=0, dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0,
        group_id=group, pool_name="kv" if group == 0 else "mamba",
    )


class _Ring:
    def __init__(self, present_keys) -> None:
        self.present = set(present_keys)
        self.calls: list[int] = []

    def batch_retain(self, keys, ttl_s=0, mode=1, **_kw):
        self.calls.append(len(keys))
        return [1 if k in self.present else 0 for k in keys]


def _worker(n_chunks: int, states_at: set[int]):
    hashes = [hashlib.sha256(f"b{i}".encode()).digest() for i in range(n_chunks)]
    present = {PoolKey(_md(0), h.hex()).to_bytes() for h in hashes}
    present |= {PoolKey(_md(1), hashes[c].hex()).to_bytes() for c in states_at}
    coord = fake_coordinator(
        LCM,
        # Attention: every chunk; state: the candidate's last chunk only.
        load_mask=lambda _b, n: [[True] * (n // LCM), [False] * (n // LCM - 1) + [True]],
        block_hashes_for_group=lambda b, _spec: b,
    )
    worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
    worker.client = _Ring(present)
    worker.coord = coord
    worker.token_dbs = [
        SimpleNamespace(block_size=LCM, metadata=_md(0)),
        SimpleNamespace(block_size=LCM, metadata=_md(1)),
    ]
    worker._kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=object()), SimpleNamespace(kv_cache_spec=object())
    ]
    worker.tp_size = 1
    worker.num_kv_head = 1
    worker.pp_size = 1
    worker._record_kv_connector_operation = lambda *a, **k: None
    return worker, hashes


class LookupBackoff(unittest.TestCase):
    def test_no_state_anywhere_takes_two_probes(self) -> None:
        worker, hashes = _worker(128, set())
        self.assertEqual(worker.lookup(128 * LCM, hashes)[0], 0)
        self.assertEqual(worker.client.calls, [129, 127])

    def test_the_longest_prefix_with_its_state_is_found(self) -> None:
        # States were handed off at chunks 9 and 40 (0-based): the longest
        # candidate ending at a saved state is 41 chunks.
        worker, hashes = _worker(64, {9, 40})
        self.assertEqual(worker.lookup(64 * LCM, hashes)[0], 41 * LCM)
        self.assertEqual(len(worker.client.calls), 2)

    def test_a_full_hit_takes_one_probe(self) -> None:
        worker, hashes = _worker(16, {15})
        self.assertEqual(worker.lookup(16 * LCM, hashes)[0], 16 * LCM)
        self.assertEqual(worker.client.calls, [17])

    def test_a_full_attention_partial_hit_does_linear_work(self) -> None:
        # Pure full attention: a shorter candidate needs nothing the first
        # probe did not cover, so there is no prefetch. The work must stay
        # linear in the prefix: bounded load_mask evaluations and key builds.
        n, present = 4096, 2048
        hashes = [hashlib.sha256(f"b{i}".encode()).digest() for i in range(n)]
        ring = _Ring({PoolKey(_md(0), h.hex()).to_bytes() for h in hashes[:present]})
        masks = []

        def load_mask(_b, length):
            masks.append(length)
            return [[True] * (length // LCM)]

        worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
        worker.client = ring
        worker.coord = fake_coordinator(
            LCM, load_mask=load_mask, block_hashes_for_group=lambda b, _s: b)
        worker.token_dbs = [SimpleNamespace(block_size=LCM, metadata=_md(0))]
        worker._kv_cache_groups = [SimpleNamespace(kv_cache_spec=object())]
        worker.tp_size = 1
        worker.num_kv_head = 1
        worker.pp_size = 1
        worker._record_kv_connector_operation = lambda *a, **k: None
        builds = [0]
        hash_field = worker_module.pool_key_hash_field

        def counted(page_hash):
            builds[0] += 1
            return hash_field(page_hash)

        with mock.patch.object(worker_module, "pool_key_hash_field", counted):
            self.assertEqual(worker.lookup(n * LCM, hashes)[0], present * LCM)
        self.assertLessEqual(len(masks), 4, f"{len(masks)} load_mask evaluations")
        self.assertEqual(builds[0], n, "keys were built more than once")
        self.assertEqual(ring.calls, [n])


if __name__ == "__main__":
    unittest.main(verbosity=2)
