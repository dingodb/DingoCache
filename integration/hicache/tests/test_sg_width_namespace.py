# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SG-width key namespacing in the HiCache backend.
The SG width is part of the canonical v2 key. Two instances on one ring that
negotiate different widths (for example, on a mixed-HCA fleet) must never read
each other's chunks and silently reassemble the wrong bytes.

These tests run WITHOUT SGLang/torch: a minimal shim module is installed so
the plugin's import surface is satisfied, and DfkvHiCache is exercised via
``__new__`` + attribute injection instead of the heavyweight ``__init__``.

Run:
    python3 -m pytest integration/hicache/tests/test_sg_width_namespace.py -v
or standalone:
    python3 integration/hicache/tests/test_sg_width_namespace.py
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import types
import unittest

# Make the plugin importable from the repo without install (sibling modules
# dfkv_access_log / dfkv_hot_config / dfkv_metrics / dfkv_telemetry live in
# integration/hicache/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "common", "src"))


# ---------------------------------------------------------------------------
# Minimal sglang shim (import-time surface only)
# ---------------------------------------------------------------------------


def _install_sglang_shim() -> None:
    class HiCacheStorage:  # base class stand-in
        pass

    class HiCacheStorageConfig:
        pass

    hicache_storage = types.ModuleType("sglang.srt.mem_cache.hicache_storage")
    hicache_storage.HiCacheStorage = HiCacheStorage
    hicache_storage.HiCacheStorageConfig = HiCacheStorageConfig
    mem_cache = types.ModuleType("sglang.srt.mem_cache")
    mem_cache.hicache_storage = hicache_storage
    srt = types.ModuleType("sglang.srt")
    srt.mem_cache = mem_cache
    sglang = types.ModuleType("sglang")
    sglang.srt = srt
    sys.modules.setdefault("sglang", sglang)
    sys.modules.setdefault("sglang.srt", srt)
    sys.modules.setdefault("sglang.srt.mem_cache", mem_cache)
    sys.modules["sglang.srt.mem_cache.hicache_storage"] = hicache_storage


_install_sglang_shim()

import dfkv_hicache as H  # noqa: E402


# ---------------------------------------------------------------------------
# Harness stubs (access log / tracing spans ride module-level functions)
# ---------------------------------------------------------------------------


class _FakeSpan:
    def __init__(self):
        self.hits = 0
        self.attrs = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeALCtx:
    def __init__(self):
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_observability() -> None:
    H.access_log = lambda name, fn: _FakeALCtx()
    H._tracing.span = lambda *a, **k: _FakeSpan()


_stub_observability()

@contextlib.contextmanager
def _parallel_runtime(**coordinates):
    module_name = "sglang.srt.runtime_context"
    previous = sys.modules.get(module_name)
    module = types.ModuleType(module_name)
    parallel = types.SimpleNamespace(**coordinates)
    module.get_parallel = lambda: parallel
    sys.modules[module_name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous




class _FakeLib:
    """dfkv_max_sg_segs is the only handle _sg_width() needs."""

    def __init__(self, width: int):
        self._width = width

    def dfkv_max_sg_segs(self, _h) -> int:
        return self._width


class _FakeMetrics:
    def __getattr__(self, name):
        return lambda *a, **k: None


def _mk_instance(width: int, mla: bool = True):
    inst = H.DfkvHiCache.__new__(H.DfkvHiCache)
    inst._lib = _FakeLib(width)
    inst._h = 1
    inst.model = "GLM-TEST"
    inst.is_mla = mla
    inst.tp_rank = 3
    inst.tp_size = 8
    inst.enable_pp = False
    inst.pp_rank = 0
    inst.pp_size = 1
    inst.pcp_size = 1
    inst.pcp_rank = 0
    inst.dcp_size = 1
    inst.dcp_rank = 0
    inst._mla_replica_writer = False
    inst.mem_pool_device = object()  # non-None → device (L2-bypass) mode
    inst._metrics = _FakeMetrics()
    inst._alog_tag = "test"
    inst._log_exist_probe = lambda *a, **k: None
    return inst


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelCoordinates(unittest.TestCase):
    def test_discovers_sglang_pcp_dcp_coordinates(self):
        with _parallel_runtime(
            attn_cp_size=8,
            attn_cp_rank=3,
            attn_dcp_size=2,
            attn_dcp_rank=1,
            attn_tp_rank=0,
        ):
            pcp, dcp, attn_tp_rank = H._resolve_parallel_coordinates({})
        self.assertEqual(pcp, (8, 3))
        self.assertEqual(dcp, (2, 1))
        self.assertEqual(attn_tp_rank, 0)

    def test_explicit_coordinates_override_runtime_axes(self):
        with _parallel_runtime(
            attn_cp_size=8,
            attn_cp_rank=3,
            attn_dcp_size=2,
            attn_dcp_rank=1,
            attn_tp_rank=2,
        ):
            pcp, dcp, attn_tp_rank = H._resolve_parallel_coordinates(
                {
                    "pcp_size": 4,
                    "pcp_rank": 1,
                    "dcp_size": 1,
                    "dcp_rank": 0,
                }
            )
        self.assertEqual(pcp, (4, 1))
        self.assertEqual(dcp, (1, 0))
        self.assertEqual(attn_tp_rank, 2)

    def test_cp_rank_changes_physical_key(self):
        left = _mk_instance(29)
        left.pcp_size = 8
        left.pcp_rank = 0
        right = _mk_instance(29)
        right.pcp_size = 8
        right.pcp_rank = 1
        self.assertNotEqual(left._keys("shared-page"), right._keys("shared-page"))

    def test_mla_writer_is_elected_per_physical_cp_shard(self):
        self.assertFalse(H._is_mla_replica_writer(True, 3, None, 1, 0))
        self.assertTrue(H._is_mla_replica_writer(True, 3, 0, 1, 0))
        self.assertTrue(H._is_mla_replica_writer(True, 3, 3, 8, 3))
        self.assertFalse(H._is_mla_replica_writer(True, 7, 7, 4, 3))
        self.assertTrue(H._is_mla_replica_writer(False, 3, 2, 8, 2))


class TestSgGroupKey(unittest.TestCase):
    def test_default_width_is_explicit(self):
        inst = _mk_instance(29)
        self.assertEqual(
            inst._sg_group_key(b"K\x00\xff", 0),
            H.sg_key(b"K\x00\xff", 29, 0))
        self.assertEqual(
            inst._sg_group_key(b"K\x00\xff", 2),
            H.sg_key(b"K\x00\xff", 29, 2))

    def test_nondefault_width_gets_own_namespace(self):
        inst = _mk_instance(4)
        self.assertEqual(
            inst._sg_group_key(b"K", 0), H.sg_key(b"K", 4, 0))
        inst63 = _mk_instance(63)
        self.assertEqual(
            inst63._sg_group_key(b"K", 1), H.sg_key(b"K", 63, 1))

    def test_width_detect_failure_uses_explicit_default_width(self):
        class _DeadLib:
            def dfkv_max_sg_segs(self, _h):
                raise RuntimeError("old lib")

        inst = _mk_instance(29)
        inst._lib = _DeadLib()
        self.assertEqual(
            inst._sg_group_key(b"K", 0), H.sg_key(b"K", 29, 0))

    def test_binary_key_log_label_is_digest_only(self):
        label = H._key_label(b"\xff\x00raw-secret")
        self.assertRegex(label, r"^len=12 sha256=[0-9a-f]{16}$")
        self.assertNotIn("raw-secret", label)

    def test_register_memory_checks_and_surfaces_native_failure(self):
        seen = []

        class FakeLib:
            def dfkv_register_memory(self, _h, base, size):
                seen.append((base.value, size.value))
                return -1

        inst = _mk_instance(29)
        inst._lib = FakeLib()
        with self.assertLogs(H._log.name, level="WARNING") as logs:
            self.assertFalse(inst.register_memory(0x1000, 4096))
        self.assertEqual(seen, [(0x1000, 4096)])
        self.assertIn("rc=-1", logs.output[0])


class TestFlattenDeviceKeyNames(unittest.TestCase):
    """layer_num=13 under width 4 → 4 chunks (4+4+4+1); under 29 → 1 chunk."""

    @staticmethod
    def _seglists(nlayers: int):
        ptrs = [1000 * (i + 1) for i in range(nlayers)]
        sizes = [512] * nlayers
        return ptrs, sizes

    def test_default_width_uses_canonical_v2_key(self):
        inst = _mk_instance(29)
        nl = 13
        ptrs, sizes = self._seglists(nl)
        stride, sks, sp, ss = inst._flatten_device(
            [b"hash\x00\xff"], [ptrs], [sizes])
        base = H.pool_key(
            b"hash\x00\xff", tp_size=8, tp_rank=-1, component="all")
        self.assertEqual(sks, [H.sg_key(base, 29, 0)])
        self.assertEqual(stride, 1)
        self.assertEqual(len(sp[0]), nl)

    def test_namespaced_chunking_when_narrow(self):
        inst = _mk_instance(4)
        nl = 13
        ptrs, sizes = self._seglists(nl)
        stride, sks, sp, ss = inst._flatten_device(
            ["hash0"], [ptrs], [sizes])
        base = H.pool_key(
            "hash0", tp_size=8, tp_rank=-1, component="all")
        self.assertEqual(
            sks, [H.sg_key(base, 4, group) for group in range(4)])
        self.assertEqual(stride, 4)
        # Segment slicing and the physical-key width coordinate must use the
        # same runtime limit; otherwise readers address different groups.
        self.assertEqual([len(p) for p in sp], [4, 4, 4, 1])
        self.assertEqual(sp[0], ptrs[0:4])
        self.assertEqual(sp[3], ptrs[12:13])

    def test_mha_pp_composition(self):
        inst = _mk_instance(4, mla=False)
        inst.enable_pp = True
        inst.pp_size = 2
        inst.pp_rank = 1
        ptrs, sizes = self._seglists(3)
        stride, sks, sp, ss = inst._flatten_device(
            ["hash0"], [ptrs, ptrs], [sizes, sizes])
        expected = [
            H.sg_key(
                H.pool_key(
                    "hash0", tp_size=8, tp_rank=3,
                    pp_size=2, pp_rank=1, component=component),
                4, 0,
            )
            for component in ("k", "v")
        ]
        self.assertEqual(sks, expected)
        self.assertEqual(stride, 2)

class TestSgBinaryAbi(unittest.TestCase):
    def test_sg_calls_keep_each_pointer_aligned_with_exact_length(self):
        key = H.sg_key(
            H.pool_key(b"hash\x00\xff", component=b"all\x00\xfe"), 4, 0)
        seen = []

        def assert_key(key_ptrs, key_lens):
            self.assertEqual(list(key_lens), [len(key)])
            self.assertEqual(
                ctypes.string_at(key_ptrs[0], key_lens[0]), key)

        class FakeLib:
            def dfkv_batch_put_sg(
                self, _h, key_ptrs, key_lens, ptrs, sizes,
                num_segs, n, out,
            ):
                self_test.assertEqual(n, 1)
                assert_key(key_ptrs, key_lens)
                self_test.assertIs(out._type_, ctypes.c_int)
                out[0] = 1
                seen.append("put")
                return 0

            def dfkv_batch_get_auto_sg(
                self, _h, key_ptrs, key_lens, ptrs, caps,
                num_segs, n, out_hit, out_len,
            ):
                self_test.assertEqual(n, 1)
                assert_key(key_ptrs, key_lens)
                self_test.assertIs(out_hit._type_, ctypes.c_int)
                out_hit[0], out_len[0] = 1, 8
                seen.append("get")
                return 0

        self_test = self
        inst = _mk_instance(4)
        inst._lib = FakeLib()
        self.assertEqual(inst._batch_put_sg(
            [key], [[0x1000]], [[8]]), [True])
        self.assertEqual(inst._batch_get_sg(
            [key], [[0x2000]], [[8]]), ([1], [8]))
        self.assertEqual(seen, ["put", "get"])


class TestCrossWidthIsolation(unittest.TestCase):
    """The actual failure mode: width-4 writer vs width-29 reader over a
    shared ring must MISS (cold cache), never return mis-split bytes."""

    @staticmethod
    def _ring_from(inst, hash_name: str, nlayers: int, width: int):
        """Simulate a ring containing the exact key bytes `inst` would write."""
        ptrs = [1000 * (i + 1) for i in range(nlayers)]
        sizes = [512] * nlayers
        _, sks, _, _ = inst._flatten_device([hash_name], [ptrs], [sizes])
        return set(sks)

    def _probe(self, inst, hashes):
        """Run batch_exists with a capture harness; return (probe_keys, hits)."""
        captured = {}
        ring = getattr(inst, "_ring", set())

        def fake_exist(sks):
            captured["sks"] = list(sks)
            return [k in ring for k in sks]

        inst._batch_exist_flat = fake_exist
        hits = inst.batch_exists(hashes)
        return captured["sks"], hits

    def test_cross_width_is_a_miss_not_a_corrupt_read(self):
        writer29 = _mk_instance(29)
        ring = self._ring_from(writer29, "hashX", 13, 29)
        reader4 = _mk_instance(4)
        reader4._ring = ring
        sks, hits = self._probe(reader4, ["hashX"])
        base = H.pool_key(
            "hashX", tp_size=8, tp_rank=-1, component="all")
        self.assertEqual(sks, [H.sg_key(base, 4, 0)])
        self.assertEqual(hits, 0)

    def test_same_width_hits(self):
        writer = _mk_instance(4)
        ring = self._ring_from(writer, "hashX", 13, 4)
        reader = _mk_instance(4)
        reader._ring = ring
        sks, hits = self._probe(reader, ["hashX"])
        base = H.pool_key(
            "hashX", tp_size=8, tp_rank=-1, component="all")
        self.assertEqual(sks, [H.sg_key(base, 4, 0)])
        self.assertEqual(hits, 1)

    def test_same_default_width_hits(self):
        writer = _mk_instance(29)
        ring = self._ring_from(writer, "hashX", 13, 29)
        reader = _mk_instance(29)
        reader._ring = ring
        sks, hits = self._probe(reader, ["hashX"])
        base = H.pool_key(
            "hashX", tp_size=8, tp_rank=-1, component="all")
        self.assertEqual(sks, [H.sg_key(base, 29, 0)])
        self.assertEqual(hits, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
