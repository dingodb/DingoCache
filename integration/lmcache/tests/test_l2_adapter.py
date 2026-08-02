# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dfkv LMCache MP-server L2 adapter (DfkvL2Adapter).

These exercise the eventfd / task-id / Bitmap / byte-accounting *bridge* logic
with an in-memory fake dfkv client, so they need lmcache installed but NO real
dfkv ring or libdfkv.so. The real-ring end-to-end path is covered by the
integration test / RUNBOOK.

Run (inside an env with lmcache>=0.4.7):
    python3 -m pytest integration/lmcache/tests/test_l2_adapter.py -v
or standalone:
    python3 integration/lmcache/tests/test_l2_adapter.py
"""

from __future__ import annotations

import os
import select
import sys
import struct
import time

# Make ``import dfkv_connector`` work when run from the repo without install.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src")
)

import dfkv_connector.l2_adapter as l2mod  # noqa: E402
from dfkv_connector.l2_adapter import (  # noqa: E402
    DfkvL2Adapter,
    DfkvL2AdapterConfig,
    _object_key_to_bytes,
)
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeObj:
    """Minimal MemoryObj stand-in: the adapter only needs byte_array + get_size."""

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
    """In-memory async stand-in for DfkvNativeClient (binary key -> bytes)."""

    def __init__(self, **kwargs):
        self.transport_mode = "fake"
        self.closed = False
        self._store: dict[bytes, bytes] = {}

    async def batch_set(self, keys, bufs):
        for k, b in zip(keys, bufs):
            self._store[k] = bytes(b)
        return True, [True] * len(keys)

    async def batch_get(self, keys, bufs):
        per_key, lengths = [], []
        for k, b in zip(keys, bufs):
            data = self._store.get(k)
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
        return [k in self._store for k in keys]

    def supports_remove(self) -> bool:
        return True

    async def batch_remove(self, keys):
        out = []
        for k in keys:
            out.append(self._store.pop(k, None) is not None or True)
        return out

    def remove_sync(self, key) -> bool:
        self._store.pop(key, None)
        return True

    def close(self):
        self.closed = True


def _mk_adapter() -> DfkvL2Adapter:
    l2mod.DfkvNativeClient = _FakeDfkvClient  # type: ignore[assignment]
    cfg = DfkvL2AdapterConfig.from_dict(
        {
            "url": "dfkv://127.0.0.1:28150/glm",
            "membership": "mds",
            "model_name": "unit-test",
        }
    )
    return DfkvL2Adapter(cfg)


def _key(tag: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=tag.to_bytes(8, "little"),
        model_name="unit-test",
        kv_rank=0,
    )


def _wait_for(fn, timeout: float = 5.0):
    """Poll fn() until it returns a truthy value or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.01)
    return None


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_operator_namespace_override_is_rejected():
    try:
        DfkvL2AdapterConfig.from_dict({
            "url": "dfkv://127.0.0.1:28150/glm",
            "model_name": "unit-test",
            "key_namespace": "unreviewed-layout-alias",
        })
    except ValueError as exc:
        assert "namespace identity is derived automatically" in str(exc)
    else:
        raise AssertionError("key_namespace must fail closed")



def test_distinct_event_fds():
    a = _mk_adapter()
    try:
        fds = {
            a.get_store_event_fd(),
            a.get_lookup_and_lock_event_fd(),
            a.get_load_event_fd(),
        }
        assert len(fds) == 3, f"event fds must be distinct, got {fds}"
    finally:
        a.close()


def test_object_key_serialization_stable_and_salted():
    k = ObjectKey(chunk_hash=b"\x01\x02\x03", model_name="m", kv_rank=5)
    s1 = _object_key_to_bytes(k)
    assert s1 == _object_key_to_bytes(k), "serialization must be deterministic"
    assert s1.startswith(b"DFKVPOOL\x02")
    assert struct.pack("<I", 6) + b"010203" in s1
    assert s1.endswith(struct.pack("<I", 3) + b"all")
    ks = ObjectKey(
        chunk_hash=b"\x01", model_name="m", kv_rank=0, cache_salt="userA"
    )
    assert _object_key_to_bytes(ks).endswith(
        struct.pack("<I", 10) + b"salt-userA")


def test_store_then_lookup_then_load_roundtrip():
    a = _mk_adapter()
    try:
        k = _key(1)
        payload = b"hello-dfkv-kv-cache-payload"
        store_tid = a.submit_store_task([k], [_FakeObj(payload)])
        res = _wait_for(lambda: a.pop_completed_store_tasks().get(store_tid))
        assert res is not None and res.is_successful()
        assert res.bytes_transferred() == len(payload)

        # lookup hit
        lk_tid = a.submit_lookup_and_lock_task([k])
        bm = _wait_for(lambda: a.query_lookup_and_lock_result(lk_tid))
        assert bm is not None and bm.test(0) and bm.popcount() == 1

        # load writes the stored bytes back into the caller buffer
        dst = _FakeObj(bytes(len(payload)))
        ld_tid = a.submit_load_task([k], [dst])
        bm2 = _wait_for(lambda: a.query_load_result(ld_tid))
        assert bm2 is not None and bm2.test(0)
        assert dst.data() == payload, "loaded bytes must match stored payload"
    finally:
        a.close()


def test_lookup_miss_and_load_miss():
    a = _mk_adapter()
    try:
        miss = _key(999)
        lk_tid = a.submit_lookup_and_lock_task([miss])
        bm = _wait_for(lambda: a.query_lookup_and_lock_result(lk_tid))
        assert bm is not None and bm.popcount() == 0

        ld_tid = a.submit_load_task([miss], [_FakeObj(bytes(8))])
        bm2 = _wait_for(lambda: a.query_load_result(ld_tid))
        assert bm2 is not None and bm2.popcount() == 0
    finally:
        a.close()


def test_batch_partial_hit():
    a = _mk_adapter()
    try:
        k0, k1, k2 = _key(10), _key(11), _key(12)
        # store only k0 and k2
        st = a.submit_store_task([k0, k2], [_FakeObj(b"aaaa"), _FakeObj(b"cccc")])
        _wait_for(lambda: a.pop_completed_store_tasks().get(st))

        lk = a.submit_lookup_and_lock_task([k0, k1, k2])
        bm = _wait_for(lambda: a.query_lookup_and_lock_result(lk))
        assert bm is not None
        assert bm.test(0) and not bm.test(1) and bm.test(2)
        assert bm.popcount() == 2
    finally:
        a.close()


def test_store_byte_accounting_first_store_wins():
    a = _mk_adapter()
    try:
        k = _key(20)
        payload = b"x" * 4096
        t1 = a.submit_store_task([k], [_FakeObj(payload)])
        r1 = _wait_for(lambda: a.pop_completed_store_tasks().get(t1))
        assert r1.bytes_transferred() == 4096
        # re-store same key -> 0 new bytes (first-store-wins accounting)
        t2 = a.submit_store_task([k], [_FakeObj(payload)])
        r2 = _wait_for(lambda: a.pop_completed_store_tasks().get(t2))
        assert r2.is_successful() and r2.bytes_transferred() == 0
        usage = a.get_usage()
        assert usage.total_bytes_used == 4096
    finally:
        a.close()


def test_unlock_is_safe_noop_without_lock():
    a = _mk_adapter()
    try:
        # unlocking unknown keys must not raise
        a.submit_unlock([_key(31), _key(32)])
    finally:
        a.close()


def test_store_event_fd_is_signalled():
    a = _mk_adapter()
    try:
        efd = a.get_store_event_fd()
        a.submit_store_task([_key(40)], [_FakeObj(b"data")])
        # the store event fd must become readable on completion
        r, _, _ = select.select([efd], [], [], 5.0)
        assert efd in r, "store event fd was not signalled"
    finally:
        a.close()


def test_delete_removes_and_updates_byte_accounting():
    a = _mk_adapter()
    try:
        k = _key(70)
        payload = b"y" * 2048
        st = a.submit_store_task([k], [_FakeObj(payload)])
        _wait_for(lambda: a.pop_completed_store_tasks().get(st))
        assert a.get_usage().total_bytes_used == 2048

        # delete is synchronous; it removes from the backend and decrements the
        # base-class byte accounting via _notify_keys_deleted.
        a.delete([k])
        assert a.get_usage().total_bytes_used == 0

        # gone from the backend -> lookup miss
        lk = a.submit_lookup_and_lock_task([k])
        bm = _wait_for(lambda: a.query_lookup_and_lock_result(lk))
        assert bm is not None and bm.popcount() == 0
    finally:
        a.close()


def test_delete_empty_and_unknown_key_safe():
    a = _mk_adapter()
    try:
        a.delete([])            # no-op
        a.delete([_key(71)])    # never stored -> no accounting change, no raise
        assert a.get_usage().total_bytes_used == 0
    finally:
        a.close()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
