# SPDX-License-Identifier: Apache-2.0
"""Real-ring integration test for DfkvL2Adapter (LMCache MP-server L2 adapter).

Drives the adapter's store / lookup-and-lock / load path against a LIVE dfkv ring
(RDMA or TCP), verifying byte-exact round-trip through libdfkv.so — the same path
LMCache's MP store/prefetch controllers exercise, minus vLLM/LMCache themselves.

Requires lmcache (for ObjectKey/Bitmap/MemoryObj imports), libdfkv.so, and a
running dfkv ring. Opt-in via env; skipped otherwise (the C++ CI does not run it,
matching integration/vllm/tests/test_dfkv_client.py).

Env:
    DFKV_L2_URL          dfkv://<mds_ip:port,...>/<group>  (required to run)
    DFKV_L2_MEMBERSHIP   "mds" (default) | "static"
    DFKV_LIB             path to libdfkv.so
    DFKV_RDMA=1, DFKV_RDMA_DEV=<dev>   for the RDMA datapath (optional; TCP otherwise)

Run on a node with a ring:
    DFKV_L2_URL=dfkv://127.0.0.1:28150/glm DFKV_LIB=/dfkv/lib/libdfkv.so \
    DFKV_RDMA=1 DFKV_RDMA_DEV=mlx5_0 \
    python3 -m pytest integration/lmcache/tests/test_l2_adapter_integration.py -v -m integration
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("DFKV_L2_URL"),
        reason="needs DFKV_L2_URL + a running dfkv ring + libdfkv.so",
    ),
]

from dfkv_connector.l2_adapter import (  # noqa: E402
    DfkvL2Adapter,
    DfkvL2AdapterConfig,
)
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402


class _Buf:
    """Minimal MemoryObj stand-in: the adapter only needs byte_array + get_size.

    The buffer is real; the bytes still traverse the live dfkv datapath.
    """

    def __init__(self, data: bytes):
        self._buf = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._buf)

    def get_size(self) -> int:
        return len(self._buf)

    def data(self) -> bytes:
        return bytes(self._buf)


def _mk_adapter() -> DfkvL2Adapter:
    cfg = DfkvL2AdapterConfig.from_dict(
        {
            "url": os.environ["DFKV_L2_URL"],
            "membership": os.environ.get("DFKV_L2_MEMBERSHIP", "mds"),
            "lib": os.environ.get("DFKV_LIB"),
            "model_name": "l2-integration-test",
        }
    )
    return DfkvL2Adapter(cfg)


def _key(tag: bytes) -> ObjectKey:
    return ObjectKey(chunk_hash=tag, model_name="l2-integration-test", kv_rank=0)


def _wait_for(fn, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.02)
    return None


def test_real_ring_store_lookup_load_roundtrip():
    a = _mk_adapter()
    try:
        # Unique keys per run so the store always counts fresh and the
        # pre-store lookup is a clean miss.
        run = os.urandom(6)
        k0 = _key(b"k0-" + run)
        k1 = _key(b"k1-" + run)  # never stored -> miss
        payload = b"dfkv-L2-adapter real-ring integration payload " * 37  # ~1.7 KiB

        # lookup BEFORE store -> miss
        lk0 = a.submit_lookup_and_lock_task([k0])
        bm0 = _wait_for(lambda: a.query_lookup_and_lock_result(lk0))
        assert bm0 is not None and bm0.popcount() == 0, "key must be absent pre-store"

        # store
        st = a.submit_store_task([k0], [_Buf(payload)])
        res = _wait_for(lambda: a.pop_completed_store_tasks().get(st))
        assert res is not None and res.is_successful(), "store failed on real ring"
        assert res.bytes_transferred() == len(payload)

        # lookup AFTER store -> hit for k0, miss for k1
        lk = a.submit_lookup_and_lock_task([k0, k1])
        bm = _wait_for(lambda: a.query_lookup_and_lock_result(lk))
        assert bm is not None
        assert bm.test(0) and not bm.test(1) and bm.popcount() == 1

        # load into a fresh zeroed buffer -> bytes must match what we stored
        dst = _Buf(bytes(len(payload)))
        ld = a.submit_load_task([k0], [dst])
        bm2 = _wait_for(lambda: a.query_load_result(ld))
        assert bm2 is not None and bm2.test(0), "load missed on real ring"
        assert dst.data() == payload, "loaded bytes differ from stored payload"

        # transport sanity (rdma when DFKV_RDMA=1 on an IB host, else tcp)
        status = a.report_status()
        assert status["is_healthy"]
    finally:
        a.close()


def test_real_ring_remove_and_eviction_roundtrip():
    a = _mk_adapter()
    try:
        if not a._client.supports_remove():
            pytest.skip("libdfkv.so has no remove RPC (rebuild dfkv with it)")
        run = os.urandom(6)
        k = _key(b"rm-" + run)
        payload = b"dfkv remove + L2 eviction round-trip payload " * 20

        st = a.submit_store_task([k], [_Buf(payload)])
        res = _wait_for(lambda: a.pop_completed_store_tasks().get(st))
        assert res is not None and res.is_successful()
        assert a.get_usage().total_bytes_used == len(payload)

        # present before delete
        lk = a.submit_lookup_and_lock_task([k])
        bm = _wait_for(lambda: a.query_lookup_and_lock_result(lk))
        assert bm is not None and bm.test(0)

        # delete via the L2 eviction path (real dfkv_remove over the wire)
        a.delete([k])
        assert a.get_usage().total_bytes_used == 0  # accounting reclaimed

        # gone after delete
        lk2 = a.submit_lookup_and_lock_task([k])
        bm2 = _wait_for(lambda: a.query_lookup_and_lock_result(lk2))
        assert bm2 is not None and bm2.popcount() == 0

        # a load now misses (nothing to bring back)
        dst = _Buf(bytes(len(payload)))
        ld = a.submit_load_task([k], [dst])
        bm3 = _wait_for(lambda: a.query_load_result(ld))
        assert bm3 is not None and bm3.popcount() == 0
    finally:
        a.close()


def test_real_ring_batch_roundtrip():
    a = _mk_adapter()
    try:
        run = os.urandom(6)
        keys = [_key(f"b{i}-".encode() + run) for i in range(8)]
        payloads = [bytes([i]) * (512 * (i + 1)) for i in range(8)]
        st = a.submit_store_task(keys, [_Buf(p) for p in payloads])
        res = _wait_for(lambda: a.pop_completed_store_tasks().get(st))
        assert res is not None and res.is_successful()
        assert res.bytes_transferred() == sum(len(p) for p in payloads)

        dsts = [_Buf(bytes(len(p))) for p in payloads]
        ld = a.submit_load_task(keys, dsts)
        bm = _wait_for(lambda: a.query_load_result(ld))
        assert bm is not None and bm.popcount() == len(keys)
        for d, p in zip(dsts, payloads):
            assert d.data() == p
    finally:
        a.close()
