# SPDX-License-Identifier: Apache-2.0
"""Fail-closed rank-isolation tests for the LMCache MP-server L2 adapter.

``ObjectKey.kv_rank`` packs
``(world_size<<24 | global_rank<<16 | local_world_size<<8 | local_rank)``.
The MP-server adapter receives no model or pipeline-parallel metadata, so it
cannot prove that rank payloads are byte-identical MLA replicas. It therefore
preserves world-size/global-rank identity and rejects the retired manual
``mla_canonical_keys`` switch.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import unittest

# Make ``import dfkv_connector`` work when run from the repo without install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dfkv_connector.l2_adapter as l2mod  # noqa: E402
from dfkv_connector.l2_adapter import (  # noqa: E402
    DfkvL2Adapter,
    DfkvL2AdapterConfig,
    _object_key_to_bytes,
)
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402


# --------------------------------------------------------------------------
# Fakes (same pattern as test_l2_adapter.py)
# --------------------------------------------------------------------------


class _FakeObj:
    def __init__(self, data: bytes):
        self._buf = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._buf)

    def get_size(self) -> int:
        return len(self._buf)

    def data(self) -> bytes:
        return bytes(self._buf)


class _FakeDfkvClient:
    """In-memory async dfkv client with store-call instrumentation."""

    def __init__(self, **kwargs):
        self.transport_mode = "fake"
        self.store: dict[bytes, bytes] = {}
        self.set_calls: list[list[bytes]] = []
        self.set_bytes = 0

    async def batch_set(self, keys, bufs):
        self.set_calls.append(list(keys))
        for k, b in zip(keys, bufs):
            self.store[k] = bytes(b)
            self.set_bytes += len(b)
        return True, [True] * len(keys)

    async def batch_get(self, keys, bufs):
        per_key, lengths = [], []
        for k, b in zip(keys, bufs):
            data = self.store.get(k)
            if data is None:
                per_key.append(False)
                lengths.append(0)
            else:
                n = min(len(data), len(b))
                b[:n] = data[:n]
                per_key.append(True)
                lengths.append(n)
        return all(per_key) if per_key else True, per_key, lengths

    async def batch_exists(self, keys):
        return [k in self.store for k in keys]

    def supports_remove(self) -> bool:
        return True

    async def batch_remove(self, keys):
        return [self.store.pop(k, None) is not None or True for k in keys]

    def close(self):
        pass


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _kv_rank(ws: int, grank: int, lws: int, lrank: int) -> int:
    return (ws << 24) | (grank << 16) | (lws << 8) | lrank


def _mk_adapter() -> DfkvL2Adapter:
    _install_fake_client()
    cfg = DfkvL2AdapterConfig.from_dict(
        {
            "url": "dfkv://127.0.0.1:28150/glm",
            "membership": "mds",
            "model_name": "unit-test",
        }
    )
    return DfkvL2Adapter(cfg)


def _rank_key(tag: int, grank: int, ws: int = 8) -> ObjectKey:
    return ObjectKey(
        chunk_hash=tag.to_bytes(8, "little"),
        model_name="unit-test",
        kv_rank=_kv_rank(ws, grank, ws, grank),
    )


def _wait_for(fn, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.01)
    return None


def _install_fake_client() -> None:
    l2mod.DfkvNativeClient = _FakeDfkvClient  # type: ignore[assignment]


_install_fake_client()


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestConfigGating(unittest.TestCase):
    def test_manual_rank_folding_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "does not expose enough model/PP metadata"
        ):
            DfkvL2AdapterConfig.from_dict(
                {
                    "url": "dfkv://127.0.0.1:1/g",
                    "model_name": "m",
                    "mla_canonical_keys": True,
                }
            )


class TestRankIsolatedKeyForm(unittest.TestCase):
    def test_same_chunk_stays_isolated_across_ranks(self):
        k3 = _rank_key(1, grank=3)
        k5 = _rank_key(1, grank=5)
        self.assertNotEqual(_object_key_to_bytes(k3), _object_key_to_bytes(k5))

    def test_topology_dims_keep_shapes_apart(self):
        """TP4 and TP8 deployments must not collapse onto one key."""
        k4 = _rank_key(1, grank=3, ws=4)
        k8 = _rank_key(1, grank=3, ws=8)
        self.assertNotEqual(_object_key_to_bytes(k4), _object_key_to_bytes(k8))


class TestStoreIsolation(unittest.TestCase):
    def test_same_chunk_on_two_ranks_writes_two_isolated_objects(self):
        adapter = _mk_adapter()
        try:
            payload = b"potentially-sharded-chunk" * 64
            first = adapter.submit_store_task(
                [_rank_key(42, grank=3)], [_FakeObj(payload)]
            )
            self.assertIsNotNone(
                _wait_for(lambda: adapter.pop_completed_store_tasks().get(first))
            )

            second = adapter.submit_store_task(
                [_rank_key(42, grank=5)], [_FakeObj(payload)]
            )
            result = _wait_for(lambda: adapter.pop_completed_store_tasks().get(second))
            self.assertIsNotNone(result)
            self.assertEqual(len(adapter._client.set_calls), 2)
            self.assertEqual(len(adapter._client.store), 2)
            self.assertEqual(adapter._client.set_bytes, 2 * len(payload))
        finally:
            adapter.close()


class TestCrossRankRead(unittest.TestCase):
    def test_one_rank_cannot_read_another_ranks_payload(self):
        adapter = _mk_adapter()
        try:
            payload = b"rank-isolated-kv"
            store_task = adapter.submit_store_task(
                [_rank_key(7, grank=7)], [_FakeObj(payload)]
            )
            _wait_for(lambda: adapter.pop_completed_store_tasks().get(store_task))

            lookup_task = adapter.submit_lookup_and_lock_task([_rank_key(7, grank=3)])
            bitmap = _wait_for(
                lambda: adapter.query_lookup_and_lock_result(lookup_task)
            )
            self.assertIsNotNone(bitmap)
            self.assertEqual(bitmap.popcount(), 0)
        finally:
            adapter.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
