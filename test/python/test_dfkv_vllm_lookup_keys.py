"""The lookup encodes each key namespace once, not once per key.

Building a lookup key used to copy the key metadata and validate and encode
every namespace field for each object: with 15 624 chunks under 8 rank
namespaces that was about 85% of a full-hit lookup. Each (group, rank)
namespace is now encoded once per lookup and every key is that prefix, the
chunk hash field and the suffix. The keys are unchanged. Runs on the
stand-ins of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import dataclasses
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
from dfkv_vllm.worker import DfkvStoreWorker  # noqa: E402

LCM = 64
TP = 4


def _md(tp_rank: int) -> KeyMetadata:
    return KeyMetadata(
        model_name="m", dp_size=1, dp_rank=-1, tp_size=TP, tp_rank=tp_rank,
        pcp_size=1, pcp_rank=0, dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0,
        group_id=0, pool_name="kv",
    )


class _Ring:
    def __init__(self, present) -> None:
        self.present = set(present)
        self.keys: list[bytes] = []

    def batch_retain(self, keys, ttl_s=0, mode=1, **_kw):
        self.keys.extend(keys)
        return [1 if k in self.present else 0 for k in keys]


def _worker(hashes, present):
    worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
    worker.client = _Ring(present)
    worker.coord = fake_coordinator(
        LCM, load_mask=lambda _b, n: [[True] * (n // LCM)],
        block_hashes_for_spec=lambda b, _s: b)
    worker.token_dbs = [SimpleNamespace(block_size=LCM, metadata=_md(0))]
    worker._kv_cache_groups = [SimpleNamespace(kv_cache_spec=object())]
    worker.tp_size = TP
    worker.num_kv_head = TP
    worker.pp_size = 1
    worker._record_kv_connector_operation = lambda *a, **k: None
    return worker


class LookupKeys(unittest.TestCase):
    def setUp(self) -> None:
        self.hashes = [hashlib.sha256(f"b{i}".encode()).digest() for i in range(256)]
        self.expected = {
            PoolKey(_md(tp), h.hex()).to_bytes()
            for h in self.hashes for tp in range(TP)
        }

    def test_keys_are_the_pool_keys_of_every_rank(self) -> None:
        worker = _worker(self.hashes, self.expected)
        self.assertEqual(worker.lookup(256 * LCM, self.hashes)[0], 256 * LCM)
        self.assertEqual(set(worker.client.keys), self.expected)
        self.assertEqual(len(worker.client.keys), len(self.expected))

    def test_each_namespace_is_encoded_once(self) -> None:
        worker = _worker(self.hashes, self.expected)
        replace = dataclasses.replace
        copies = []

        def counting_replace(obj, **changes):
            if isinstance(obj, KeyMetadata):
                copies.append(changes)
            return replace(obj, **changes)

        with mock.patch.object(dataclasses, "replace", counting_replace):
            self.assertEqual(worker.lookup(256 * LCM, self.hashes)[0], 256 * LCM)
        self.assertEqual(len(copies), TP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
