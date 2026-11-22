# SPDX-License-Identifier: Apache-2.0
"""Lookups of the in-process LMCache connector.

* ``batched_contains`` (synchronous, as LMCache's RemoteBackend calls it) is
  supported: without it RemoteBackend asked ``contains`` key by key, one
  round trip per chunk of the prefix.
* Existence probes and writes have workers of their own: a long PUT does not
  hold up lookups.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import _lmcache_stub  # noqa: E402

_lmcache_stub.install()

import dfkv_connector.native_client as nc  # noqa: E402
from dfkv_connector.exists_cache import ExistsLRU  # noqa: E402
from dfkv_connector.remote_connector import DfkvConnector  # noqa: E402


class _Ring:
    def __init__(self, present) -> None:
        self.present = set(present)
        self.calls: list[list[bytes]] = []

    def batch_exists_sync(self, keys):
        self.calls.append(list(keys))
        return [k in self.present for k in keys]


def _connector(present, batch_max_keys=4):
    c = object.__new__(DfkvConnector)
    c._client = _Ring(present)
    c._exists_lru = ExistsLRU(64)
    c._assume_exists = False
    c._batch_max_keys = batch_max_keys
    c._key = lambda key: key
    return c


class BatchedContains(unittest.TestCase):
    def test_supported(self) -> None:
        self.assertTrue(_connector([]).support_batched_contains())

    def test_prefix_in_one_probe_per_slice(self) -> None:
        keys = [b"k%d" % i for i in range(10)]
        c = _connector(keys[:6])
        self.assertEqual(c.batched_contains(keys), 6)
        self.assertEqual([len(call) for call in c._client.calls], [4, 4])
        # The confirmed prefix is remembered; a second check asks only past it.
        c._client.calls.clear()
        self.assertEqual(c.batched_contains(keys), 6)
        self.assertEqual(c._client.calls, [keys[6:10]])

    def test_all_present_and_empty(self) -> None:
        keys = [b"a", b"b"]
        self.assertEqual(_connector(keys).batched_contains(keys), 2)
        self.assertEqual(_connector(keys).batched_contains([]), 0)
        self.assertEqual(_connector([]).batched_contains(keys), 0)


class _Lib:
    def dfkv_open_v2(self, _options):
        return 0x1

    def dfkv_transport_mode(self, _h):
        return b"tcp"

    def dfkv_close(self, _h):
        pass

    def dfkv_batch_exist(self, _h, _keys, _lens, n, out):
        for i in range(n):
            out[i] = 1
        return 0


class LookupsDoNotWaitForWrites(unittest.TestCase):
    def test_an_exists_probe_runs_while_a_put_is_in_flight(self) -> None:
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        orig = nc.load_lib
        nc.load_lib = lambda _path: _Lib()
        self.addCleanup(setattr, nc, "load_lib", orig)
        client = nc.DfkvNativeClient(
            raw_endpoint="127.0.0.1:1", group="g", membership="static",
            key_namespace=b"ns", loop=loop)
        release = threading.Event()

        def held_put(keys, _bufs):
            assert release.wait(10)
            return True, [True] * len(keys)

        client._batch_set_blocking = held_put

        async def scenario():
            put = asyncio.ensure_future(client.batch_set([b"p"], [memoryview(b"x")]))
            await asyncio.sleep(0.02)
            start = time.monotonic()
            found = await asyncio.wait_for(client.batch_exists([b"q"]), 2)
            waited = time.monotonic() - start
            release.set()
            await put
            return found, waited

        found, waited = loop.run_until_complete(scenario())
        client.close()
        self.assertEqual(found, [True])
        self.assertLess(waited, 0.5, "the lookup waited for the PUT")


class _BlockingLib(_Lib):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed_while_busy = None
        self.busy = False
        self.calls_after_close = 0
        self.is_closed = False

    def dfkv_batch_exist(self, h, keys, lens, n, out):
        if self.is_closed:
            self.calls_after_close += 1
        self.busy = True
        self.entered.set()
        assert self.release.wait(10)
        self.busy = False
        return super().dfkv_batch_exist(h, keys, lens, n, out)

    def dfkv_exist(self, _h, _key, _n):
        return 1

    def dfkv_close(self, _h):
        self.closed_while_busy = self.busy
        self.is_closed = True


class SynchronousCallsAndClose(unittest.TestCase):
    def _client(self, lib, **kwargs):
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        orig = nc.load_lib
        nc.load_lib = lambda _path: lib
        self.addCleanup(setattr, nc, "load_lib", orig)
        return nc.DfkvNativeClient(
            raw_endpoint="127.0.0.1:1", group="g", membership="static",
            key_namespace=b"ns", loop=loop, **kwargs)

    def test_close_waits_for_a_synchronous_call_in_flight(self) -> None:
        lib = _BlockingLib()
        client = self._client(lib)
        caller = threading.Thread(target=client.batch_exists_sync, args=([b"k"],))
        caller.start()
        self.assertTrue(lib.entered.wait(5))
        closer = threading.Thread(target=client.close)
        closer.start()
        closer.join(0.3)
        self.assertTrue(closer.is_alive(), "close() did not wait for the call")
        lib.release.set()
        caller.join(5)
        closer.join(5)
        self.assertFalse(lib.closed_while_busy, "dfkv_close ran under a native call")

    def test_no_synchronous_call_after_close(self) -> None:
        lib = _BlockingLib()
        lib.release.set()
        client = self._client(lib)
        client.close()
        for call in (lambda: client.batch_exists_sync([b"k"]),
                     lambda: client.exists_sync(b"k")):
            with self.assertRaises(RuntimeError):
                call()
        self.assertEqual(lib.calls_after_close, 0)

    def test_writes_keep_get_parallelism_workers(self) -> None:
        client = self._client(_Lib(), get_parallelism=4)
        self.addCleanup(client.close)
        self.assertEqual(client._write_executor._max_workers, 4)
        self.assertEqual(client._executor._max_workers, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
