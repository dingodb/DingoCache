"""A scheduler lookup that rank 0 never answers ends as a miss.

The scheduler asks rank 0 over a ZMQ REQ socket. Without a receive timeout a
hung or dead rank-0 server blocked ``recv()`` forever: in synchronous mode the
scheduler loop stopped, and ``close()`` waited for the stuck call. Runs on the
stand-ins of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker as base  # noqa: E402  (installs the stand-ins)

import zmq  # noqa: E402

from dfkv_vllm.worker import LookupKeyClient  # noqa: E402
from vllm.v1.core.kv_cache_utils import BlockHash  # noqa: E402


class _Socket:
    """A REQ socket whose server answers only when told to. ``recv`` honours
    RCVTIMEO like ZMQ (raises zmq.Again); without one it blocks."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.options: dict = {}
        self.sent = 0
        self.closed = False
        self.unblock = threading.Event()

    def setsockopt(self, option, value) -> None:
        self.options[option] = value

    def send_multipart(self, frames, copy=False) -> None:
        self.sent += 1

    def recv(self) -> bytes:
        if self.answer:
            return (64).to_bytes(4, "big") + (0).to_bytes(4, "big")
        timeout_ms = self.options.get(zmq.RCVTIMEO, -1)
        if self.unblock.wait(None if timeout_ms < 0 else timeout_ms / 1000):
            raise RuntimeError("released by the test")
        raise zmq.Again()

    def close(self, linger=0) -> None:
        self.closed = True


def _config(**extra):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra)
    )


class LookupTimeout(unittest.TestCase):
    def setUp(self) -> None:
        self.sockets: list[_Socket] = []
        self.answers = [False, True]  # the first server call hangs
        patches = (
            patch("dfkv_vllm.worker.zmq.Context", return_value=base._MemoryContext()),
            patch("dfkv_vllm.worker.get_zmq_rpc_path_lookup",
                  return_value="ipc:///tmp/dfkv-test-timeout"),
            patch("dfkv_vllm.worker.make_zmq_socket", side_effect=self._socket),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self) -> None:
        for socket in self.sockets:
            socket.unblock.set()

    def _socket(self, *_args, **_kwargs) -> _Socket:
        socket = _Socket(self.answers.pop(0) if self.answers else True)
        self.sockets.append(socket)
        return socket

    @staticmethod
    def _hashes():
        return [BlockHash(hashlib.sha256(b"x").digest())]

    def _lookup(self, client, req_id="r"):
        result = []
        done = threading.Event()

        def run():
            result.append(client.lookup(req_id, 64, self._hashes()))
            done.set()

        threading.Thread(target=run, daemon=True).start()
        return done, result

    def test_an_unanswered_lookup_is_a_miss_and_the_next_one_works(self) -> None:
        client = LookupKeyClient(_config(lookup_timeout_ms=200, lookup_cache_ttl_ms=0))
        self.addCleanup(client.close)
        done, result = self._lookup(client)
        self.assertTrue(done.wait(5), "the lookup waited for rank 0 forever")
        self.assertEqual(result, [(0, 0)])
        # The stuck REQ socket was replaced; once the timeout period after
        # it is over, the next lookup is answered.
        self.assertTrue(self.sockets[0].closed)
        time.sleep(0.25)
        done, result = self._lookup(client, "s")
        self.assertTrue(done.wait(5))
        self.assertEqual(result, [(64, 0)])
        self.assertEqual(len(self.sockets), 2)

    def test_lookups_queued_behind_a_timeout_do_not_each_wait_it(self) -> None:
        client = LookupKeyClient(_config(lookup_timeout_ms=300, lookup_cache_ttl_ms=0))
        self.addCleanup(client.close)
        start = time.monotonic()
        lookups = [self._lookup(client, f"r{i}") for i in range(4)]
        for done, result in lookups:
            self.assertTrue(done.wait(5))
            self.assertEqual(result, [(0, 0)])
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.6, "each queued lookup waited a full timeout")
        sent = sum(socket.sent for socket in self.sockets)
        self.assertEqual(sent, 1, "only the first lookup asks rank 0")
        # Within the timeout period after it: a miss at once, nothing sent.
        done, result = self._lookup(client, "during")
        self.assertTrue(done.wait(1))
        self.assertEqual(result, [(0, 0)])
        self.assertEqual(sum(socket.sent for socket in self.sockets), 1)
        # After it: the next lookup asks rank 0 again (and is answered).
        time.sleep(0.35)
        done, result = self._lookup(client, "after")
        self.assertTrue(done.wait(5))
        self.assertEqual(result, [(64, 0)])

    def test_close_returns_while_rank_0_does_not_answer(self) -> None:
        client = LookupKeyClient(_config(lookup_timeout_ms=200))
        self._lookup(client)
        closed = threading.Event()
        threading.Thread(target=lambda: (client.close(), closed.set()),
                         daemon=True).start()
        self.assertTrue(closed.wait(5), "close() waited for the stuck lookup")

    def test_the_timeout_is_on_by_default_and_can_be_turned_off(self) -> None:
        client = LookupKeyClient(_config())
        self.addCleanup(client.close)
        self.assertEqual(self.sockets[0].options.get(zmq.RCVTIMEO), 10000)
        self.answers = [True]
        off = LookupKeyClient(_config(lookup_timeout_ms=0))
        self.addCleanup(off.close)
        self.assertNotIn(zmq.RCVTIMEO, self.sockets[-1].options)
        for bad in (-1, "x", True, 600001):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    LookupKeyClient(_config(lookup_timeout_ms=bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
