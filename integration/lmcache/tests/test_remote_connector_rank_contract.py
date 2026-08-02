# SPDX-License-Identifier: Apache-2.0
"""Rank-ownership regressions for the in-process LMCache connector.

LMCache's ``RemoteBackend`` owns replicated-layout detection and writer
selection. The dfkv connector must encode every key it receives verbatim and
must never apply a second rank-folding or chunk-striping policy.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest

# Make ``import dfkv_connector`` work when run from the repo without install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dfkv_connector.exists_cache import ExistsLRU  # noqa: E402
from dfkv_connector.remote_connector import DfkvConnector  # noqa: E402


class _FakeObj:
    def __init__(self, payload: bytes):
        self._buf = bytearray(payload)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._buf)


class _FakeClient:
    def __init__(self):
        self.set_calls: list[tuple[list[bytes], list[bytes]]] = []

    async def batch_set(self, keys, bufs):
        self.set_calls.append((list(keys), [bytes(buf) for buf in bufs]))
        return True, [True] * len(keys)


def _key(chunk_hash: int, worker_id: int = 0):
    return types.SimpleNamespace(
        model_name="unit-test",
        world_size=8,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
    )


def _connector() -> DfkvConnector:
    connector = object.__new__(DfkvConnector)
    connector._client = _FakeClient()
    connector._exists_lru = ExistsLRU(32)
    connector._batch_max_keys = 512

    # Recreate the state of the retired connector-side MLA policy. The fixed
    # connector deliberately ignores these fields; retaining them here makes
    # each test fail against the old rank-folding/striping implementation.
    connector._canonical_keys = True
    connector._world_size = 8
    connector._worker_id = 0
    return connector


class TestFrameworkOwnedRankContract(unittest.TestCase):
    def test_scalar_put_never_drops_a_framework_selected_key(self):
        connector = _connector()
        asyncio.run(connector.put(_key(1), _FakeObj(b"scalar")))

        self.assertEqual(len(connector._client.set_calls), 1)
        self.assertEqual(connector._client.set_calls[0][1], [b"scalar"])

    def test_batched_put_writes_every_framework_selected_key(self):
        connector = _connector()
        keys = [_key(1), _key(2), _key(8)]
        payloads = [_FakeObj(b"one"), _FakeObj(b"two"), _FakeObj(b"eight")]

        asyncio.run(connector.batched_put(keys, payloads))

        self.assertEqual(len(connector._client.set_calls), 1)
        stored_keys, stored_payloads = connector._client.set_calls[0]
        self.assertEqual(len(stored_keys), 3)
        self.assertEqual(stored_payloads, [b"one", b"two", b"eight"])

    def test_connector_preserves_distinct_framework_ranks(self):
        connector = _connector()
        rank0 = connector._key(_key(7, worker_id=0))
        rank1 = connector._key(_key(7, worker_id=1))
        self.assertNotEqual(rank0, rank1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
