# SPDX-License-Identifier: Apache-2.0
"""Protections of the in-process LMCache connector.

* A PUT that stored not every key is a failed put in the metrics too (it was
  counted "ok", and only the access log, off by default, said otherwise).
* A chunk larger than the RDMA client can move is refused at startup, where
  every PUT of it would otherwise fail.

Request tags in the key are covered by test_key_mapper.py.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import _lmcache_stub  # noqa: E402

_lmcache_stub.install()

import dfkv_connector.native_client as nc  # noqa: E402


class _Op:
    def __init__(self, ops, name) -> None:
        self.name = name
        self.status = None
        self.bytes = 0
        ops.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.status is None:
            self.status = "fail" if exc[0] else "ok"
        return False

    def __bool__(self) -> bool:
        return True


def _client(put_result, transport="tcp", limit=None):
    client = object.__new__(nc.DfkvNativeClient)
    loop = asyncio.new_event_loop()
    client._loop = loop
    client._executor = ThreadPoolExecutor(max_workers=1)
    client._write_executor = client._exists_executor = client._executor
    client._batch_set_blocking = lambda keys, bufs: put_result
    client.transport_mode = transport

    def max_block_bytes(_handle):
        return limit

    client._lib = SimpleNamespace(dfkv_max_block_bytes=max_block_bytes)
    client._h = object()
    return client, loop


class PutFailureIsCounted(unittest.TestCase):
    def _put(self, result):
        ops = []
        client, loop = _client(result)
        with patch.object(nc._push_metrics, "op",
                          lambda name, num_keys=0: _Op(ops, name)):
            loop.run_until_complete(
                client.batch_set([b"a", b"b"], [memoryview(b"x"), memoryview(b"y")]))
        client._executor.shutdown()
        loop.close()
        return [op.status for op in ops if op.name == "put"]

    def test_a_partial_put_is_a_failed_put(self) -> None:
        self.assertEqual(self._put((False, [True, False])), ["fail"])

    def test_a_complete_put_is_ok(self) -> None:
        self.assertEqual(self._put((True, [True, True])), ["ok"])


class BlockSizePreflight(unittest.TestCase):
    def test_a_chunk_over_the_rdma_bound_is_refused(self) -> None:
        client, loop = _client(None, transport="rdma", limit=1 << 20)
        self.addCleanup(loop.close)
        self.addCleanup(client._executor.shutdown)
        with self.assertRaises(ValueError):
            client.validate_block_sizes((2 << 20,), context="LMCache chunk")
        client.validate_block_sizes((1 << 20,), context="LMCache chunk")

    def test_an_older_libdfkv_without_the_bound_skips_the_check(self) -> None:
        # Compatibility first: a library that worked before still starts.
        client, loop = _client(None, transport="rdma", limit=1 << 20)
        self.addCleanup(loop.close)
        self.addCleanup(client._executor.shutdown)
        client._lib = SimpleNamespace()
        with self.assertLogs(nc.logger, level="WARNING") as logs:
            client.validate_block_sizes((2 << 20,), context="LMCache chunk")
        self.assertIn("dfkv_max_block_bytes", logs.output[0])

    def test_tcp_has_no_bound(self) -> None:
        client, loop = _client(None, transport="tcp", limit=None)
        self.addCleanup(loop.close)
        self.addCleanup(client._executor.shutdown)
        client.validate_block_sizes((1 << 40,), context="LMCache chunk")


if __name__ == "__main__":
    unittest.main(verbosity=2)
