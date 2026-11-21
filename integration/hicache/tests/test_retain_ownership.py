# SPDX-License-Identifier: Apache-2.0
"""Retention covers exactly the objects a rank writes.

SGLang asks each owning rank, synchronously and before it reports usage,
which TTL the whole prefix can be promised (``retain_precheck``), and later
renews HBM/host hits asynchronously (``batch_touch``). Both must address the
same physical objects the write path stores on that rank:

* the elected MLA writer: main KV, replicated side pools, its sharded pools;
* an MLA follower: only rank-sharded side pools (e.g. KDA state);
* a follower with no sharded pool: nothing, and it does not vote;
* device-direct (L2-bypass) writes: the SG-chunked keys, not the bare key.

The tests capture the keys the real write entry points hand to the native put
and compare them with the retention key set, so a future change to either side
fails here instead of silently leaving pages unprotected or double-renewed.

Runs without SGLang, torch or libdfkv:
    python3 -m pytest integration/hicache/tests/test_retain_ownership.py -q
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "common", "src"))


def _install_sglang_shim() -> None:
    class HiCacheStorage:
        pass

    class HiCacheStorageConfig:
        pass

    hicache_storage = types.ModuleType("sglang.srt.mem_cache.hicache_storage")
    hicache_storage.HiCacheStorage = HiCacheStorage
    hicache_storage.HiCacheStorageConfig = HiCacheStorageConfig
    device_page_meta = types.ModuleType("sglang.srt.mem_cache.device_page_meta")
    # Fake device pools expose (layers, sub) and return one segment list per
    # sub-object, one segment per layer.
    def _segments(pool, indices):
        ptrs = [[1000 * i + l for l in range(pool.layers)]
                for i in range(len(indices) * pool.sub)]
        return ptrs, [[64] * pool.layers for _ in ptrs]
    device_page_meta.get_device_page_buffer_meta = _segments
    device_page_meta.get_device_sidecar_page_buffer_meta = _segments
    mem_cache = types.ModuleType("sglang.srt.mem_cache")
    mem_cache.hicache_storage = hicache_storage
    mem_cache.device_page_meta = device_page_meta
    srt = types.ModuleType("sglang.srt")
    srt.mem_cache = mem_cache
    sglang = types.ModuleType("sglang")
    sglang.srt = srt
    sys.modules.setdefault("sglang", sglang)
    sys.modules.setdefault("sglang.srt", srt)
    sys.modules.setdefault("sglang.srt.mem_cache", mem_cache)
    sys.modules["sglang.srt.mem_cache.hicache_storage"] = hicache_storage
    sys.modules["sglang.srt.mem_cache.device_page_meta"] = device_page_meta


_install_sglang_shim()

import dfkv_hicache as H  # noqa: E402


class _Ctx:
    hits = 0
    bytes = 0
    attrs = None
    result = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


H.access_log = lambda *a, **k: _Ctx()
H._tracing.span = lambda *a, **k: _Ctx()


class _Metrics:
    def __getattr__(self, name):
        return lambda *a, **k: None


class _Lib:
    def __init__(self, width):
        self.width = width

    def dfkv_max_sg_segs(self, _h):
        return self.width


class _HostPool:
    """Host pool whose page meta has ``sub`` components per page."""

    def __init__(self, sub):
        self.sub = sub

    def get_page_buffer_meta(self, indices):
        n = len(indices) * self.sub
        return list(range(1, n + 1)), [64] * n


class _DevicePool:
    def __init__(self, layers, sub=1):
        self.layers = layers
        self.sub = sub


PAGES = ["p0", "p1", "p2"]
TTL = {"retain_ttl_s": 3600}


def _backend(*, mla=True, writer=True, tp_rank=0, kda=True, indexer=True,
             device=False, width=29, layers=5):
    """A K3-like backend: MLA KV, sharded KDA state, replicated DSA indexer."""
    b = H.DfkvHiCache.__new__(H.DfkvHiCache)
    b._lib = _Lib(width)
    b._h = 1
    b.model = "TEST"
    b.is_mla = mla
    b.tp_rank = tp_rank
    b.tp_size = 8
    b.enable_pp = False
    b.pp_rank = 0
    b.pp_size = 1
    b.pcp_size = 1
    b.pcp_rank = 0
    b.dcp_size = 1
    b.dcp_rank = 0
    b.prefill_cp_storage_layout = None
    b._storage_pcp_rank = 0
    b._mla_replica_writer = writer
    b._metrics = _Metrics()
    b._alog_tag = "test"
    b._has_retain = False  # writes below must not issue native RETAIN
    b._backup_exist_gate = False
    b._put_retry_recovered = 0
    b._pool_component_names = {}
    b._pool_replicated = {}
    b.registered_pools = {}
    b.mem_pool_host = _HostPool(1 if mla else 2)
    if kda:
        b._pool_component_names["mamba"] = ("temporal", "conv0", "conv1")
        b._pool_replicated["mamba"] = False
        b.registered_pools["mamba"] = _HostPool(3)
    if indexer:
        b._pool_component_names["indexer"] = ("all",)
        b._pool_replicated["indexer"] = True
        b.registered_pools["indexer"] = _HostPool(1)
    if device:
        b.mem_pool_device = _DevicePool(layers, 1 if mla else 2)
        if indexer:
            b._sidecar_device_pools = {"indexer": _DevicePool(layers)}
            del b.registered_pools["indexer"]
    written = []

    def put(sks, *_):
        written.extend(sks)
        return [True] * len(sks)

    b._put_flat_impl = put
    b._put_sg_flat_impl = put
    b.written = written
    return b


def _write_prefix(b, device=False):
    """Back up PAGES through the real write entry points SGLang calls."""
    idx = list(range(len(PAGES)))
    side = [types.SimpleNamespace(name=n, keys=PAGES, host_indices=idx)
            for n in b.registered_pools]
    if device:
        b._kv_device_set(PAGES, idx)
        for name in getattr(b, "_sidecar_device_pools", {}):
            b._sidecar_device_set(name, PAGES, idx)
    else:
        b.batch_set_v1(PAGES, idx)
    if side:
        b._v2_io(side, putting=True)


class _Cabi:
    """Records native retention calls made through dfkv_cabi."""

    def __init__(self, supported=1, cap=3600, admit=True):
        self.supported, self.cap, self.admit = supported, cap, admit
        self.precheck_keys = None
        self.retain_calls = []

    def __enter__(self):
        self._saved = (H.dfkv_cabi.call_retain_limits,
                       H.dfkv_cabi.call_retain_precheck,
                       H.dfkv_cabi.call_batch_retain)
        H.dfkv_cabi.call_retain_limits = lambda lib, h: (self.supported, self.cap)

        def precheck(lib, h, keys, *a):
            self.precheck_keys = list(keys)
            if isinstance(self.admit, Exception):
                raise self.admit
            return self.admit

        def retain(lib, h, keys, ttl, mode, **k):
            self.retain_calls.append((list(keys), ttl, mode))
            return [H.dfkv_cabi.DFKV_RET_REFRESHED] * len(keys)

        H.dfkv_cabi.call_retain_precheck = precheck
        H.dfkv_cabi.call_batch_retain = retain
        return self

    def __exit__(self, *exc):
        (H.dfkv_cabi.call_retain_limits,
         H.dfkv_cabi.call_retain_precheck,
         H.dfkv_cabi.call_batch_retain) = self._saved
        return False


class RetainKeysMatchWrites(unittest.TestCase):
    def assertRetainMatchesWrites(self, b, device=False):
        _write_prefix(b, device=device)
        retained = b._retain_object_keys(PAGES)
        self.assertEqual(len(retained), len(set(retained)), "duplicate keys")
        self.assertEqual(set(retained), set(b.written))
        self.assertEqual(b.retain_owns_objects(), bool(b.written))

    def test_mla_writer(self):
        self.assertRetainMatchesWrites(_backend())

    def test_mla_follower_owns_only_sharded_state(self):
        b = _backend(writer=False, tp_rank=3)
        self.assertRetainMatchesWrites(b)
        self.assertTrue(b.written)
        expected = {sk for k in PAGES for sk in b._pool_keys("mamba", k)}
        self.assertEqual(set(b.written), expected)

    def test_mla_follower_without_sharded_pool_owns_nothing(self):
        b = _backend(writer=False, tp_rank=3, kda=False)
        self.assertRetainMatchesWrites(b)
        self.assertEqual(b.written, [])
        self.assertFalse(b.retain_owns_objects())

    def test_mha_rank_owns_its_kv_shard(self):
        self.assertRetainMatchesWrites(_backend(mla=False, tp_rank=5, indexer=False))

    def test_device_direct_writer_uses_sg_chunk_keys(self):
        # 5 layers at width 2 -> 3 SG chunks per object.
        b = _backend(device=True, width=2, layers=5)
        self.assertRetainMatchesWrites(b, device=True)
        self.assertNotIn(b._keys("p0")[0], b._retain_object_keys(PAGES))

    def test_device_direct_follower(self):
        b = _backend(writer=False, tp_rank=3, device=True, width=2, layers=5)
        self.assertRetainMatchesWrites(b, device=True)


class RetainPrecheck(unittest.TestCase):
    def setUp(self):
        self.b = _backend()
        self.b._has_retain = True

    def test_admits_min_of_request_and_ring_cap(self):
        with _Cabi(cap=300) as c:
            self.assertEqual(self.b.retain_precheck(PAGES, TTL), 300)
        self.assertEqual(c.precheck_keys, self.b._retain_object_keys(PAGES))

    def test_no_ttl_or_no_keys_is_zero(self):
        with _Cabi():
            self.assertEqual(self.b.retain_precheck(PAGES, None), 0)
            self.assertEqual(self.b.retain_precheck([], TTL), 0)

    def test_unknown_or_old_ring_is_zero(self):
        for supported in (-1, 0):
            with _Cabi(supported=supported):
                self.assertEqual(self.b.retain_precheck(PAGES, TTL), 0)

    def test_low_headroom_is_zero(self):
        with _Cabi(admit=False):
            self.assertEqual(self.b.retain_precheck(PAGES, TTL), 0)

    def test_native_error_is_zero_not_raised(self):
        with _Cabi(admit=RuntimeError("boom")):
            self.assertEqual(self.b.retain_precheck(PAGES, TTL), 0)

    def test_lib_without_retention_is_zero(self):
        self.b._has_retain = False
        with _Cabi():
            self.assertEqual(self.b.retain_precheck(PAGES, TTL), 0)


class BatchTouch(unittest.TestCase):
    def test_follower_without_objects_sends_nothing(self):
        b = _backend(writer=False, tp_rank=3, kda=False)
        b._has_retain = True
        with _Cabi() as c:
            b.batch_touch(PAGES, TTL)
            b.batch_touch(PAGES, None)
        self.assertEqual(c.retain_calls, [])
        self.assertIsNone(c.precheck_keys)

    def test_follower_renews_only_its_sharded_state(self):
        b = _backend(writer=False, tp_rank=3)
        b._has_retain = True
        with _Cabi() as c:
            b.batch_touch(PAGES, None)
        (keys, _ttl, mode), = c.retain_calls
        self.assertEqual(mode, H.dfkv_cabi.DFKV_RETAIN_REFRESH)
        self.assertEqual(keys, b._retain_object_keys(PAGES))


class Ownership(unittest.TestCase):
    def test_asking_before_pool_registration_fails_loudly(self):
        b = _backend(kda=False, indexer=False)
        b.mem_pool_host = None
        with self.assertRaises(RuntimeError):
            b.retain_owns_objects()


if __name__ == "__main__":
    unittest.main()
