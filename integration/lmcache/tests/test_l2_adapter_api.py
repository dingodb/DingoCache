# SPDX-License-Identifier: Apache-2.0
"""The L2 adapter has to follow the L2AdapterInterface of every LMCache
release it supports.

Python's ABC checks only that a method exists, not how it can be called: an
adapter whose ``submit_lookup_and_lock_task`` took one argument fewer than
LMCache passes was constructed fine and failed with a TypeError at the first
prefetch. These tests check the signatures against the interface, and drive
the adapter the way LMCache's controllers call it.

Runs against the installed LMCache if there is one, else against the
stand-in for the release ``DFKV_TEST_LMCACHE_API`` names (``0.4.7``,
``0.5.0``, ``0.5.3``, ``0.5.4``; see ``_lmcache_stub.py``). ctest runs it once per
release.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import _lmcache_stub  # noqa: E402

API = _lmcache_stub.install()

import dfkv_connector.l2_adapter as l2mod  # noqa: E402
from dfkv_connector.l2_adapter import (  # noqa: E402
    DfkvL2Adapter,
    DfkvL2AdapterConfig,
)
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402
from lmcache.v1.distributed.l2_adapters.base import (  # noqa: E402
    L2AdapterInterface,
)


def _placeholders(params):
    """Arguments for ``params`` (after self), positional as LMCache passes
    them. The adapter ignores the layout hints, so any value does."""
    args = []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not p.empty:
            continue
        args.append(object())
    return args


class _Ring:
    store: dict = {}

    def __init__(self, **_kwargs) -> None:
        self.transport_mode = "fake"

    async def batch_set(self, keys, bufs):
        for k, b in zip(keys, bufs):
            _Ring.store[k] = bytes(b)
        return True, [True] * len(keys)

    async def batch_get(self, keys, bufs):
        per_key, lengths = [], []
        for k, b in zip(keys, bufs):
            data = _Ring.store.get(k)
            if data is None or len(data) > len(b):
                per_key.append(False)
                lengths.append(0)
            else:
                b[: len(data)] = data
                per_key.append(True)
                lengths.append(len(data))
        return True, per_key, lengths

    async def batch_exists(self, keys):
        return [k in _Ring.store for k in keys]

    def supports_remove(self) -> bool:
        return False

    def close(self) -> None:
        pass


class _Obj:
    def __init__(self, data: bytes) -> None:
        self._buf = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._buf)

    def get_size(self) -> int:
        return len(self._buf)


def _wait(fn, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value is not None:
            return value
        time.sleep(0.005)
    raise AssertionError("timed out")


def _key(tag: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=tag.to_bytes(8, "little"),
        model_name="m",
        kv_rank=ObjectKey.ComputeKVRank(1, 0, 1, 0),
    )


class FollowsTheInterface(unittest.TestCase):
    def test_every_abstract_method_takes_the_interface_arguments(self):
        abstract = sorted(L2AdapterInterface.__abstractmethods__)
        self.assertIn("submit_lookup_and_lock_task", abstract)
        for name in abstract:
            with self.subTest(api=API, method=name):
                base = inspect.signature(getattr(L2AdapterInterface, name))
                ours = inspect.signature(getattr(DfkvL2Adapter, name))
                params = list(base.parameters.values())[1:]
                args = _placeholders(params)
                # Positional, as the controllers call it ...
                ours.bind(None, *args)
                # ... and by the interface's parameter names.
                ours.bind(
                    None,
                    **{p.name: a for p, a in zip(
                        [p for p in params if p.default is p.empty
                         and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)],
                        args,
                    )},
                )

    def test_nothing_abstract_is_left(self):
        self.assertFalse(
            getattr(DfkvL2Adapter, "__abstractmethods__", frozenset()),
            f"LMCache {API} requires methods the adapter does not have",
        )


class DrivenLikeTheControllers(unittest.TestCase):
    def setUp(self) -> None:
        _Ring.store = {}
        l2mod.DfkvNativeClient = _Ring
        self.adapter = DfkvL2Adapter(DfkvL2AdapterConfig.from_dict(
            {"url": "dfkv://127.0.0.1:1/g", "model_name": "m"}))

    def tearDown(self) -> None:
        self.adapter.close()

    def test_store_lookup_load(self):
        a = self.adapter
        task = a.submit_store_task([_key(1)], [_Obj(b"x" * 64)])
        result = _wait(lambda: a.pop_completed_store_tasks().get(task))
        self.assertTrue(result.is_successful())

        params = list(inspect.signature(
            L2AdapterInterface.submit_lookup_and_lock_task).parameters.values())[1:]
        hints = _placeholders(params)[1:]  # what follows ``keys``
        task = a.submit_lookup_and_lock_task([_key(1), _key(2)], *hints)
        bitmap = _wait(lambda: a.query_lookup_and_lock_result(task))
        self.assertEqual([bitmap.test(0), bitmap.test(1)], [True, False])
        a.submit_unlock([_key(1)])

        obj = _Obj(b"\0" * 64)
        task = a.submit_load_task([_key(1)], [obj])
        bitmap = _wait(lambda: a.query_load_result(task))
        self.assertTrue(bitmap.test(0))
        self.assertEqual(bytes(obj.byte_array), b"x" * 64)


if __name__ == "__main__":
    print(f"LMCache API: {API}")
    unittest.main(verbosity=2)
