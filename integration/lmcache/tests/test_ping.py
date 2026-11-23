"""LMCache's ping asks the ring, not only the handle.

ping_sync checked only that the client was open, so a ring with every node
down (or hung) answered LMCache's health ping as healthy. It now GETs a few
sentinel keys and needs one node's answer within its timeout; a probe that
hangs is waited for by the next ping, not repeated.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lmcache_stub  # noqa: E402

_lmcache_stub.install()

from dfkv_connector import native_client as nc  # noqa: E402
from dfkv_connector.native_client import DfkvNativeClient  # noqa: E402


class _Lib:
    """dfkv_batch_get_st answering every key with ``code``, after ``hold``."""

    def __init__(self, code: int, hold: threading.Event | None = None) -> None:
        self.code = code
        self.hold = hold
        self.calls = 0

    def dfkv_batch_get_st(self, _h, _k, _l, _p, _s, n, hits, status):
        self.calls += 1
        if self.hold is not None:
            self.hold.wait(10)
        for i in range(n):
            status[i] = self.code
            hits[i] = 0
        return 0


def _client(lib) -> DfkvNativeClient:
    c = object.__new__(DfkvNativeClient)
    c._lib = lib
    c._h = 1
    c._closed = False
    c._sync_cv = threading.Condition()
    c._sync_inflight = 0
    c._ping_lock = threading.Lock()
    c._ping_probe = None
    return c


class Ping(unittest.TestCase):
    def test_a_node_answer_is_alive(self) -> None:
        for code in (1, 0, 8):  # NOT_FOUND, OK, LENGTH_MISMATCH
            _client(_Lib(code)).ping_sync()

    def test_a_dead_ring_fails_the_ping(self) -> None:
        for code in (4, 9, 10):  # IO_ERROR, PEER_COOLING, NO_ROUTE
            with self.assertRaisesRegex(RuntimeError, "no node"):
                _client(_Lib(code)).ping_sync()

    def test_a_hung_ring_fails_within_the_timeout_and_is_probed_once(self) -> None:
        hold = threading.Event()
        lib = _Lib(1, hold)
        client = _client(lib)
        try:
            for _ in range(3):
                t0 = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "within 0.2 s"):
                    client.ping_sync(timeout_s=0.2)
                self.assertLess(time.monotonic() - t0, 1.0)
            self.assertEqual(lib.calls, 1)
        finally:
            hold.set()
        deadline = time.monotonic() + 5
        while not client._ping_probe.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        lib.hold = None
        client.ping_sync(timeout_s=0.2)  # a fresh probe once it answered
        self.assertEqual(lib.calls, 2)

    def test_the_probe_writes_nothing(self) -> None:
        self.assertTrue(all(k.startswith(b"dfkv-ping/") for k in nc._PING_KEYS))

    def test_close_does_not_wait_long_for_a_hung_probe(self) -> None:
        class _Executor:
            def shutdown(self, wait=True):
                pass

        hold = threading.Event()
        lib = _Lib(1, hold)
        lib.closed = []
        lib.dfkv_close = lib.closed.append
        client = _client(lib)
        client._executor = client._write_executor = _Executor()
        client._exists_executor = _Executor()
        client._telemetry_acquired = False
        try:
            with self.assertRaises(RuntimeError):
                client.ping_sync(timeout_s=0.1)
            saved = nc._CLOSE_PING_WAIT_S
            nc._CLOSE_PING_WAIT_S = 0.3
            try:
                t0 = time.monotonic()
                with self.assertLogs(nc.logger, "WARNING") as logs:
                    client.close()
                self.assertLess(time.monotonic() - t0, 2.0)
            finally:
                nc._CLOSE_PING_WAIT_S = saved
            # The native call is still running: its handle is not closed.
            self.assertEqual(lib.closed, [])
            self.assertIn("left open", logs.output[0])
        finally:
            hold.set()

    def test_close_closes_the_handle_after_an_answered_probe(self) -> None:
        class _Executor:
            def shutdown(self, wait=True):
                pass

        lib = _Lib(1)
        lib.closed = []
        lib.dfkv_close = lib.closed.append
        client = _client(lib)
        client._executor = client._write_executor = _Executor()
        client._exists_executor = _Executor()
        client._telemetry_acquired = False
        client.ping_sync()
        client.close()
        self.assertEqual(lib.closed, [1])

    def test_a_closed_client_fails(self) -> None:
        client = _client(_Lib(1))
        client._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client.ping_sync()


if __name__ == "__main__":
    unittest.main(verbosity=2)
