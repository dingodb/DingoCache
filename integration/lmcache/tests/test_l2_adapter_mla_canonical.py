# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MLA canonical keys in the dfkv LMCache MP-server L2 adapter.

Historical KNOWN GAP (now opt-in fixed): ``ObjectKey.kv_rank`` packs
``(world_size<<24 | global_rank<<16 | local_world_size<<8 | local_rank)`` —
every TP worker wrote MLA's replicated KV under its own key → 8x storage,
8x write traffic, and the same-host rendezvous could never match.

With ``mla_canonical_keys=true`` the adapter folds the rank fields to zero
(topology dims kept) and dedups stores with an exists probe. These tests
verify: config gating (default OFF), key sharing/isolation, store dedup,
cross-rank hit, and the cold-cache flip semantics.

Run (inside an env with lmcache installed):
    python3 -m pytest integration/lmcache/tests/test_l2_adapter_mla_canonical.py -v
or standalone:
    python3 integration/lmcache/tests/test_l2_adapter_mla_canonical.py
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
    _canonical_kv_rank,
    _object_key_to_string,
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
        self.store: dict[str, bytes] = {}
        self.set_calls: list[list[str]] = []
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


def _mk_adapter(canonical: bool) -> DfkvL2Adapter:
    cfg = DfkvL2AdapterConfig.from_dict(
        {
            "url": "dfkv://127.0.0.1:28150/glm",
            "membership": "mds",
            "model_name": "unit-test",
            "mla_canonical_keys": canonical,
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
    def test_default_off(self):
        cfg = DfkvL2AdapterConfig.from_dict(
            {"url": "dfkv://127.0.0.1:1/g", "model_name": "m"}
        )
        self.assertFalse(cfg.mla_canonical_keys)

    def test_opt_in_via_dict(self):
        cfg = DfkvL2AdapterConfig.from_dict(
            {"url": "dfkv://127.0.0.1:1/g", "model_name": "m",
             "mla_canonical_keys": True}
        )
        self.assertTrue(cfg.mla_canonical_keys)

    def test_opt_in_via_env(self):
        os.environ["DFKV_L2ADAPTER_MLA_CANONICAL_KEYS"] = "1"
        try:
            cfg = DfkvL2AdapterConfig.from_dict(
                {"url": "dfkv://127.0.0.1:1/g", "model_name": "m"}
            )
            self.assertTrue(cfg.mla_canonical_keys)
        finally:
            del os.environ["DFKV_L2ADAPTER_MLA_CANONICAL_KEYS"]

    def test_dict_beats_env(self):
        os.environ["DFKV_L2ADAPTER_MLA_CANONICAL_KEYS"] = "1"
        try:
            cfg = DfkvL2AdapterConfig.from_dict(
                {"url": "dfkv://127.0.0.1:1/g", "model_name": "m",
                 "mla_canonical_keys": False}
            )
            self.assertFalse(cfg.mla_canonical_keys)
        finally:
            del os.environ["DFKV_L2ADAPTER_MLA_CANONICAL_KEYS"]


class TestCanonicalKeyForm(unittest.TestCase):
    def test_canonical_kv_rank_zeroes_only_rank_fields(self):
        r = _kv_rank(ws=8, grank=3, lws=8, lrank=3)
        c = _canonical_kv_rank(r)
        self.assertEqual(c, (8 << 24) | (8 << 8))  # ws/lws kept, ranks zeroed

    def test_same_chunk_shared_key_across_ranks_when_on(self):
        a = _mk_adapter(True)
        try:
            k3 = a._canon(_rank_key(1, grank=3))
            k5 = a._canon(_rank_key(1, grank=5))
            self.assertEqual(k3, k5)
            self.assertEqual(
                _object_key_to_string(k3), _object_key_to_string(k5))
        finally:
            a.close()

    def test_legacy_mode_keeps_per_rank_keys(self):
        a = _mk_adapter(False)
        try:
            k3 = a._canon(_rank_key(1, grank=3))
            k5 = a._canon(_rank_key(1, grank=5))
            self.assertNotEqual(k3, k5)
            self.assertNotEqual(
                _object_key_to_string(k3), _object_key_to_string(k5))
        finally:
            a.close()

    def test_topology_dims_keep_shapes_apart(self):
        """TP4 and TP8 deployments must NOT collapse onto one key."""
        a = _mk_adapter(True)
        try:
            k4 = a._canon(_rank_key(1, grank=3, ws=4))
            k8 = a._canon(_rank_key(1, grank=3, ws=8))
            self.assertNotEqual(
                _object_key_to_string(k4), _object_key_to_string(k8))
        finally:
            a.close()

    def test_world_size_1_passthrough(self):
        a = _mk_adapter(True)
        try:
            k = _rank_key(1, grank=0, ws=1)
            self.assertEqual(a._canon(k), k)
        finally:
            a.close()

    def test_flip_is_cold_cache(self):
        """Canonical on/off produce different key strings — flipping the flag
        for a live ring must be understood as a cache reset (documented)."""
        on = _mk_adapter(True)
        off = _mk_adapter(False)
        try:
            k = _rank_key(1, grank=3)
            self.assertNotEqual(
                _object_key_to_string(on._canon(k)),
                _object_key_to_string(off._canon(k)),
            )
        finally:
            on.close()
            off.close()


class TestStoreDedup(unittest.TestCase):
    def test_replicated_chunk_written_once_across_ranks(self):
        a = _mk_adapter(True)
        try:
            payload = b"mlA-Replicated-Chunk" * 64  # 1.2 KiB
            # rank 3 stores chunk 42
            k3 = _rank_key(42, grank=3)
            t1 = a.submit_store_task([k3], [_FakeObj(payload)])
            r1 = _wait_for(lambda: a.pop_completed_store_tasks().get(t1))
            self.assertIsNotNone(r1)
            self.assertEqual(r1.bytes_transferred(), len(payload))
            self.assertEqual(len(a._client.set_calls), 1)

            # rank 5 submits the SAME chunk (old behavior: 8 separate keys)
            k5 = _rank_key(42, grank=5)
            t2 = a.submit_store_task([k5], [_FakeObj(payload)])
            r2 = _wait_for(lambda: a.pop_completed_store_tasks().get(t2))
            self.assertIsNotNone(r2)
            # exists-probe deduped the write: no second batch_set call, 0 bytes
            self.assertEqual(len(a._client.set_calls), 1)
            self.assertEqual(r2.bytes_transferred(), 0)
            self.assertEqual(a._client.set_bytes, len(payload))
        finally:
            a.close()

    def test_missing_subset_only_transfers_missing(self):
        a = _mk_adapter(True)
        try:
            p = b"xxxx"
            k0 = _rank_key(10, grank=0)
            k1 = _rank_key(11, grank=0)
            a.submit_store_task([k0], [_FakeObj(p)])
            _wait_for(lambda: len(a._client.set_calls) == 1)

            # batch of two, one already present → only the missing goes on wire
            t = a.submit_store_task(
                [_rank_key(10, grank=2), _rank_key(11, grank=2)],
                [_FakeObj(p), _FakeObj(p)])
            r = _wait_for(lambda: a.pop_completed_store_tasks().get(t))
            self.assertIsNotNone(r)
            self.assertEqual(len(a._client.set_calls), 2)
            # the second set call carried exactly one key (the missing k1)
            self.assertEqual(len(a._client.set_calls[1]), 1)
            self.assertIn(_object_key_to_string(a._canon(k1)),
                          a._client.set_calls[1][0])
        finally:
            a.close()


class TestCrossRankRead(unittest.TestCase):
    def test_write_by_one_rank_read_by_another(self):
        a = _mk_adapter(True)
        try:
            payload = b"rank-shared-kv" * 32
            writer = _rank_key(7, grank=7)
            t = a.submit_store_task([writer], [_FakeObj(payload)])
            _wait_for(lambda: a.pop_completed_store_tasks().get(t))

            # Another rank's own kv_rank address must still find the chunk.
            reader = _rank_key(7, grank=3)
            lk = a.submit_lookup_and_lock_task([reader])
            bm = _wait_for(lambda: a.query_lookup_and_lock_result(lk))
            self.assertIsNotNone(bm)
            self.assertTrue(bm.test(0))

            dst = _FakeObj(bytes(len(payload)))
            ld = a.submit_load_task([reader], [dst])
            bm2 = _wait_for(lambda: a.query_load_result(ld))
            self.assertIsNotNone(bm2)
            self.assertTrue(bm2.test(0))
            self.assertEqual(dst.data(), payload)
        finally:
            a.close()

    def test_legacy_mode_has_no_cross_rank_hit(self):
        a = _mk_adapter(False)
        try:
            payload = b"per-rank"
            a_task = a.submit_store_task([_rank_key(1, grank=7)],
                                         [_FakeObj(payload)])
            _wait_for(lambda: a.pop_completed_store_tasks().get(a_task))

            lk = a.submit_lookup_and_lock_task([_rank_key(1, grank=3)])
            bm = _wait_for(lambda: a.query_lookup_and_lock_result(lk))
            self.assertIsNotNone(bm)
            self.assertEqual(bm.popcount(), 0)
        finally:
            a.close()


class TestOperationalHint(unittest.TestCase):
    def test_hint_fires_once_when_off_and_world_gt1(self):
        a = _mk_adapter(False)
        try:
            self.assertFalse(a._warned_mla_hint)
            a._hint_canonical_if_mla([_rank_key(1, grank=1)])
            self.assertTrue(a._warned_mla_hint)
            # second call is a no-op (hint once)
            a._warned_mla_hint = False  # reset then verify `True` guards
            a._hint_canonical_if_mla([_rank_key(1, grank=1)])
            self.assertTrue(a._warned_mla_hint)
        finally:
            a.close()

    def test_hint_silent_when_canonical_on(self):
        a = _mk_adapter(True)
        try:
            a._hint_canonical_if_mla([_rank_key(1, grank=1)])
            self.assertFalse(a._warned_mla_hint)
        finally:
            a.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
