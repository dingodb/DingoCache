# SPDX-License-Identifier: Apache-2.0
"""A positive answer of the exists cache must not outlive the object.

The server evicts on its own. A key the cache still calls present is then
neither found (the GET misses) nor stored again (LMCache's put path asks
``contains`` first), for as long as the process lives. Two bounds: entries
expire, and a GET that misses discards the entry at once.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import _lmcache_stub  # noqa: E402

_lmcache_stub.install()

from dfkv_connector.exists_cache import ExistsLRU  # noqa: E402
from dfkv_connector.remote_connector import DfkvConnector  # noqa: E402

CHUNK = 16


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class ExistsCacheExpiry(unittest.TestCase):
    def test_a_positive_expires_after_the_ttl(self):
        clock = _Clock()
        lru = ExistsLRU(8, ttl_s=60, clock=clock)
        lru.add(b"k")
        self.assertTrue(lru.has(b"k"))
        clock.now += 59.9
        self.assertTrue(lru.has(b"k"))
        clock.now += 0.2
        self.assertFalse(lru.has(b"k"), "still positive after its TTL")
        self.assertEqual(len(lru), 0, "an expired entry is dropped")

    def test_being_asked_does_not_extend_the_ttl_but_a_confirmation_does(self):
        clock = _Clock()
        lru = ExistsLRU(8, ttl_s=60, clock=clock)
        lru.add(b"k")
        for _ in range(10):
            clock.now += 10
            if clock.now < 1060:
                self.assertTrue(lru.has(b"k"))
        self.assertFalse(lru.has(b"k"), "lookups alone kept it alive")
        lru.add(b"k")
        clock.now += 50
        lru.add(b"k")  # the server confirmed it again
        clock.now += 50
        self.assertTrue(lru.has(b"k"))

    def test_prefix_stops_at_an_expired_key(self):
        clock = _Clock()
        lru = ExistsLRU(8, ttl_s=60, clock=clock)
        lru.add_many([b"a", b"b"])
        clock.now += 30
        lru.add(b"c")
        self.assertEqual(lru.prefix_len([b"c", b"a", b"b"]), 3)
        clock.now += 40  # a and b are 70 s old, c is 40 s old
        self.assertEqual(lru.prefix_len([b"c", b"a", b"b"]), 1)
        self.assertEqual(lru.prefix_len([b"a", b"c"]), 0)

    def test_no_ttl_keeps_entries_as_before(self):
        clock = _Clock()
        for ttl in (None, 0):
            lru = ExistsLRU(8, ttl_s=ttl, clock=clock)
            lru.add(b"k")
            clock.now += 10**9
            self.assertTrue(lru.has(b"k"))
            self.assertEqual(lru.prefix_len([b"k"]), 1)

    def test_capacity_eviction_and_discard_still_work(self):
        lru = ExistsLRU(2, ttl_s=60, clock=_Clock())
        lru.add_many([b"a", b"b"])
        self.assertTrue(lru.has(b"a"))  # a is now the most recently used
        lru.add(b"c")
        self.assertFalse(lru.has(b"b"))
        self.assertTrue(lru.has(b"a"))
        lru.discard_many([b"a", b"missing"])
        self.assertFalse(lru.has(b"a"))
        with self.assertRaises(ValueError):
            ExistsLRU(2, ttl_s=-1)


class _Obj:
    def __init__(self) -> None:
        self._buf = bytearray(CHUNK)
        self.released = False

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._buf)

    def ref_count_down(self) -> None:
        self.released = True


class _Backend:
    def allocate(self, *_args, **_options):
        return _Obj()


class _Server:
    """The ring as a fake client: a set of keys it holds."""

    def __init__(self) -> None:
        self.stored: set[bytes] = set()
        self.exists_calls = 0
        self.short_length: set[bytes] = set()

    async def batch_exists(self, keys):
        self.exists_calls += 1
        return [k in self.stored for k in keys]

    async def batch_get(self, keys, bufs, *, release_buffers=None):
        found = [k in self.stored for k in keys]
        lengths = [
            (CHUNK - 1 if k in self.short_length else CHUNK) if f else 0
            for k, f in zip(keys, found)
        ]
        return True, found, lengths

    async def batch_set(self, keys, bufs):
        self.stored.update(keys)
        return True, None


def _key(i: int):
    return types.SimpleNamespace(
        model_name="unit-test", world_size=1, worker_id=0, chunk_hash=i
    )


def _connector(server: _Server, ttl_s=None, clock=None) -> DfkvConnector:
    connector = object.__new__(DfkvConnector)
    connector._client = server
    connector._exists_lru = (
        ExistsLRU(64, ttl_s=ttl_s, clock=clock) if clock else ExistsLRU(64)
    )
    connector._batch_max_keys = 512
    connector._get_parallelism = 1
    connector._assume_exists = False
    connector._local_cpu_backend = _Backend()
    connector.meta_shapes = [(CHUNK,)]
    connector.meta_dtypes = ["uint8"]
    connector.meta_fmt = "raw"
    connector.full_chunk_size_bytes = CHUNK
    connector.single_token_size = 4  # CHUNK - 1 is not a whole token: bad length

    def reshape_partial_chunk(memory_obj, bytes_read):
        raise ValueError("not a whole number of tokens")

    connector.reshape_partial_chunk = reshape_partial_chunk
    return connector


class StalePositiveIsForgottenWhenAGetMisses(unittest.TestCase):
    """The server evicted the keys; the cache still calls them present."""

    def _stale(self, n: int = 3):
        server = _Server()
        connector = _connector(server)
        keys = [_key(i) for i in range(n)]
        asyncio.run(connector.batched_put(keys, [_Obj() for _ in keys]))
        server.stored.clear()  # evicted by the server
        self.assertTrue(asyncio.run(connector.exists(keys[0])))
        self.assertEqual(server.exists_calls, 0, "answered from the cache")
        return server, connector, keys

    def test_scalar_get_miss(self):
        server, connector, keys = self._stale()
        self.assertIsNone(asyncio.run(connector.get(keys[0])))
        self.assertFalse(asyncio.run(connector.exists(keys[0])),
                         "still reported present after the GET missed")
        self.assertEqual(server.exists_calls, 1, "the server was asked again")

    def test_batched_get_miss(self):
        server, connector, keys = self._stale()
        self.assertEqual(asyncio.run(connector.batched_get(keys)), [None] * 3)
        self.assertEqual(
            asyncio.run(connector.batched_async_contains("id", keys)), 0
        )
        self.assertEqual(server.exists_calls, 1)

    def test_batched_get_non_blocking_miss(self):
        server, connector, keys = self._stale()
        self.assertEqual(
            asyncio.run(connector.batched_get_non_blocking("id", keys)), []
        )
        self.assertEqual(
            asyncio.run(connector.batched_async_contains("id", keys)), 0
        )

    def test_only_the_missing_keys_are_forgotten(self):
        server, connector, keys = self._stale(4)
        server.stored = {connector._key(keys[0]), connector._key(keys[2]),
                         connector._key(keys[3])}
        got = asyncio.run(connector.batched_get_non_blocking("id", keys))
        self.assertEqual(len(got), 1, "the prefix ends at the missing key")
        lru = connector._exists_lru
        self.assertTrue(lru.has(connector._key(keys[0])))
        self.assertFalse(lru.has(connector._key(keys[1])))
        # The server answered "present" for these two: nothing says otherwise.
        self.assertTrue(lru.has(connector._key(keys[2])))
        self.assertTrue(lru.has(connector._key(keys[3])))

    def test_a_hit_with_an_unusable_length_is_forgotten_too(self):
        server, connector, keys = self._stale(1)
        key_bytes = connector._key(keys[0])
        server.stored = {key_bytes}
        server.short_length = {key_bytes}
        self.assertIsNone(asyncio.run(connector.get(keys[0])))
        self.assertFalse(connector._exists_lru.has(key_bytes))

    def test_after_the_miss_the_value_is_stored_again(self):
        # What LMCache's put path does: skip a key that "exists".
        server, connector, keys = self._stale(1)

        async def put_if_absent():
            if await connector.exists(keys[0]):
                return False
            await connector.put(keys[0], _Obj())
            return True

        self.assertFalse(asyncio.run(put_if_absent()), "stale positive: skipped")
        self.assertIsNone(asyncio.run(connector.get(keys[0])))
        self.assertTrue(asyncio.run(put_if_absent()),
                        "the chunk is never stored again")
        self.assertIn(connector._key(keys[0]), server.stored)


class StalePositiveExpiresWithoutAGet(unittest.TestCase):
    def test_expired_positive_is_checked_remotely(self):
        clock = _Clock()
        server = _Server()
        connector = _connector(server, ttl_s=60, clock=clock)
        key = _key(1)
        asyncio.run(connector.put(key, _Obj()))
        server.stored.clear()
        self.assertTrue(asyncio.run(connector.exists(key)))
        clock.now += 61
        self.assertFalse(asyncio.run(connector.exists(key)))
        self.assertEqual(server.exists_calls, 1)

    def test_ttl_setting(self):
        name = DfkvConnector._ENV_EXISTS_TTL_S
        cases = {None: 60.0, "": 60.0, "0": 0.0, "5": 5.0, "2.5": 2.5,
                 "-1": 60.0, "soon": 60.0, "nan": 60.0, "inf": 60.0}
        for raw, expected in cases.items():
            if raw is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = raw
            try:
                self.assertEqual(DfkvConnector._resolve_exists_ttl_s(), expected, raw)
            finally:
                os.environ.pop(name, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
