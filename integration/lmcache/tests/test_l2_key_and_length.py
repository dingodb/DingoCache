# SPDX-License-Identifier: Apache-2.0
"""The L2 adapter must not hand one model's bytes to another.

Two ways it could: the object key did not include ``ObjectKey.model_name``,
so two models served through one adapter shared keys; and a load accepted
whatever length came back, so a value stored for another layout under the
same key was reported loaded.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import _lmcache_stub  # noqa: E402

_lmcache_stub.install()

import dfkv_connector.l2_adapter as l2mod  # noqa: E402
from dfkv_connector.l2_adapter import (  # noqa: E402
    DfkvL2Adapter,
    DfkvL2AdapterConfig,
    _object_key_to_bytes,
)
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402


class _Obj:
    def __init__(self, data: bytes) -> None:
        self._buf = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._buf)

    def get_size(self) -> int:
        # As in LMCache: byte_array exposes get_size() bytes.
        return len(self._buf)

    def data(self) -> bytes:
        return bytes(self._buf)


class _Ring:
    """Shared by every adapter of a test: the ring."""

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


def _adapter(**params) -> DfkvL2Adapter:
    l2mod.DfkvNativeClient = _Ring
    config = {"url": "dfkv://127.0.0.1:1/g", "model_name": "served-name"}
    config.update(params)
    return DfkvL2Adapter(DfkvL2AdapterConfig.from_dict(config))


def _key(tag: int, model: str = "model-a", salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=tag.to_bytes(8, "little"),
        model_name=model,
        kv_rank=ObjectKey.ComputeKVRank(1, 0, 1, 0),
        cache_salt=salt,
    )


def _wait(fn, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value is not None:
            return value
        time.sleep(0.005)
    raise AssertionError("timed out")


def _store(adapter, key, data: bytes) -> None:
    task = adapter.submit_store_task([key], [_Obj(data)])
    _wait(lambda: adapter.pop_completed_store_tasks().get(task))


def _load(adapter, key, size: int):
    obj = _Obj(b"\0" * size)
    task = adapter.submit_load_task([key], [obj])
    bitmap = _wait(lambda: adapter.query_load_result(task))
    return bitmap.test(0), obj.data()


def _exists(adapter, key) -> bool:
    task = adapter.submit_lookup_and_lock_task([key])
    return _wait(lambda: adapter.query_lookup_and_lock_result(task)).test(0)


class KeyIncludesModelName(unittest.TestCase):
    def setUp(self) -> None:
        _Ring.store = {}
        self.adapters = []

    def tearDown(self) -> None:
        for adapter in self.adapters:
            adapter.close()

    def adapter(self, **params) -> DfkvL2Adapter:
        adapter = _adapter(**params)
        self.adapters.append(adapter)
        return adapter

    def test_key_layout(self):
        a, b = _key(1, "model-a"), _key(1, "model-b")
        # Without the switch the key is what it always was: no model in it.
        self.assertEqual(_object_key_to_bytes(a), _object_key_to_bytes(b))
        self.assertEqual(_object_key_to_bytes(a),
                         _object_key_to_bytes(a, include_model=False))
        # With it, the model tells the keys apart; everything else still does.
        with_a = _object_key_to_bytes(a, include_model=True)
        with_b = _object_key_to_bytes(b, include_model=True)
        self.assertNotEqual(with_a, with_b)
        self.assertNotEqual(with_a, _object_key_to_bytes(a))
        self.assertEqual(with_a, _object_key_to_bytes(_key(1, "model-a"), True))
        self.assertNotEqual(with_a, _object_key_to_bytes(_key(2, "model-a"), True))
        self.assertNotEqual(
            with_a, _object_key_to_bytes(_key(1, "model-a", salt="s"), True)
        )
        # A long or odd model name still makes a valid key.
        odd = _key(1, "org/模型 " + "x" * 500)
        self.assertNotEqual(_object_key_to_bytes(odd, True), with_a)

    def test_a_long_salt_fits_with_the_switch(self):
        # model part + "salt-" leave 212 bytes for the salt; beyond that the
        # salt goes in hashed, still telling salts apart.
        for n in (1, 212, 213, 300, 1000):
            long_a = _key(1, salt="a" * n)
            with self.subTest(salt_len=n):
                with_model = _object_key_to_bytes(long_a, include_model=True)
                self.assertEqual(with_model,
                                 _object_key_to_bytes(_key(1, salt="a" * n), True))
                self.assertNotEqual(
                    with_model, _object_key_to_bytes(_key(1, salt="b" * n), True))
                self.assertNotEqual(
                    with_model,
                    _object_key_to_bytes(_key(1, salt="a" * (n + 1)), True))
        # Multi-byte characters count in bytes.
        _object_key_to_bytes(_key(1, salt="模" * 80), include_model=True)
        # Without the switch keys are as before: a salt that fits is used as
        # it is, one that does not is refused as before.
        self.assertIn(b"salt-" + b"a" * 251,
                      _object_key_to_bytes(_key(1, salt="a" * 251)))
        with self.assertRaises(ValueError):
            _object_key_to_bytes(_key(1, salt="a" * 252))
        # Through the adapter: store, find and load with a 300-byte salt.
        adapter = self.adapter(key_includes_model=True)
        key = _key(5, salt="s" * 300)
        _store(adapter, key, b"Q" * 16)
        self.assertTrue(_exists(adapter, key))
        self.assertEqual(_load(adapter, key, 16), (True, b"Q" * 16))

    def test_two_models_do_not_share_objects_with_the_switch(self):
        adapter = self.adapter(key_includes_model=True)
        _store(adapter, _key(7, "model-a"), b"A" * 32)
        self.assertTrue(_exists(adapter, _key(7, "model-a")))
        self.assertFalse(_exists(adapter, _key(7, "model-b")),
                         "model-b was told model-a's object is its own")
        loaded, data = _load(adapter, _key(7, "model-b"), 32)
        self.assertFalse(loaded)
        self.assertNotEqual(data, b"A" * 32)
        _store(adapter, _key(7, "model-b"), b"B" * 32)
        self.assertEqual(_load(adapter, _key(7, "model-a"), 32), (True, b"A" * 32))
        self.assertEqual(_load(adapter, _key(7, "model-b"), 32), (True, b"B" * 32))

    def test_default_keeps_the_old_keys_and_warns_about_a_second_model(self):
        # An upgrade must find what the previous release stored.
        old_release = self.adapter()
        _store(old_release, _key(7, "model-a"), b"A" * 32)
        upgraded = self.adapter()
        self.assertEqual(_load(upgraded, _key(7, "model-a"), 32), (True, b"A" * 32))
        with self.assertLogs(l2mod.logger, level=logging.WARNING) as logs:
            self.assertTrue(_exists(upgraded, _key(7, "model-b")),
                            "this is the collision the warning is about")
            _exists(upgraded, _key(8, "model-b"))
            _exists(upgraded, _key(9, "model-c"))
        collisions = [m for m in logs.output if "more than one model" in m]
        self.assertEqual(len(collisions), 1, logs.output)
        self.assertIn("key_includes_model", collisions[0])
        self.assertIn("model-a", collisions[0])
        self.assertIn("model-b", collisions[0])

    def test_one_model_does_not_warn(self):
        adapter = self.adapter()
        logger = l2mod.logger
        handler = _Collect()
        logger.addHandler(handler)
        try:
            for tag in range(5):
                _store(adapter, _key(tag, "model-a"), b"A" * 8)
                _exists(adapter, _key(tag, "model-a"))
        finally:
            logger.removeHandler(handler)
        self.assertFalse([m for m in handler.messages if "more than one model" in m])

    def test_switching_it_on_does_not_find_objects_stored_without_it(self):
        _store(self.adapter(), _key(7, "model-a"), b"A" * 32)
        switched = self.adapter(key_includes_model=True)
        self.assertFalse(_exists(switched, _key(7, "model-a")),
                         "documented: the switch changes every key")

    def test_switch_is_validated(self):
        for bad in ("true", 1, None):
            with self.assertRaises(ValueError):
                DfkvL2AdapterConfig.from_dict({
                    "url": "dfkv://127.0.0.1:1/g", "model_name": "m",
                    "key_includes_model": bad,
                })
        config = DfkvL2AdapterConfig.from_dict(
            {"url": "dfkv://127.0.0.1:1/g", "model_name": "m"})
        self.assertFalse(config.key_includes_model)
        self.assertIn("key_includes_model", DfkvL2AdapterConfig.help())


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record) -> None:
        self.messages.append(record.getMessage())


class LoadChecksTheLength(unittest.TestCase):
    def setUp(self) -> None:
        _Ring.store = {}
        self.adapter = _adapter()

    def tearDown(self) -> None:
        self.adapter.close()

    def test_matching_length_loads(self):
        _store(self.adapter, _key(1), b"V" * 40)
        self.assertEqual(_load(self.adapter, _key(1), 40), (True, b"V" * 40))
        self.assertEqual(self.adapter.load_length_mismatches, 0)

    def test_shorter_stored_value_is_not_loaded(self):
        # Stored for another layout (say another dtype) under the same key.
        _store(self.adapter, _key(1), b"S" * 24)
        with self.assertLogs(l2mod.logger, level=logging.WARNING) as logs:
            loaded, _data = _load(self.adapter, _key(1), 40)
        self.assertFalse(loaded, "24 bytes were accepted for a 40-byte object")
        self.assertEqual(self.adapter.load_length_mismatches, 1)
        self.assertTrue(any("length other than" in m for m in logs.output))
        # Counted, but the warning is not repeated.
        handler = _Collect()
        l2mod.logger.addHandler(handler)
        try:
            self.assertFalse(_load(self.adapter, _key(1), 40)[0])
        finally:
            l2mod.logger.removeHandler(handler)
        self.assertEqual(self.adapter.load_length_mismatches, 2)
        self.assertFalse(handler.messages)

    def test_mixed_batch_reports_each_object_on_its_own(self):
        _store(self.adapter, _key(1), b"a" * 16)
        _store(self.adapter, _key(2), b"b" * 8)   # wrong length for the load
        accessed = []
        self.adapter._notify_keys_accessed = accessed.append
        objs = [_Obj(b"\0" * 16), _Obj(b"\0" * 16), _Obj(b"\0" * 16)]
        task = self.adapter.submit_load_task([_key(1), _key(2), _key(3)], objs)
        bitmap = _wait(lambda: self.adapter.query_load_result(task))
        self.assertEqual([bitmap.test(i) for i in range(3)], [True, False, False])
        self.assertEqual(objs[0].data(), b"a" * 16)
        # Only what was loaded is reported as accessed.
        self.assertEqual(accessed, [[_key(1)]])

    def test_serialized_values_are_not_held_to_the_buffer_length(self):
        # With a serde_config LMCache wraps the adapter: it stores serialized,
        # shorter values and loads them into buffers of the upper bound.
        config = DfkvL2AdapterConfig.from_dict(
            {"url": "dfkv://127.0.0.1:1/g", "model_name": "served-name"})
        config.serde_config = object()
        adapter = DfkvL2Adapter(config)
        try:
            _store(adapter, _key(1), b"z" * 24)
            loaded, data = _load(adapter, _key(1), 40)
            self.assertTrue(loaded)
            self.assertEqual(data[:24], b"z" * 24)
            self.assertEqual(adapter.load_length_mismatches, 0)
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
