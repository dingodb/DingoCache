# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SG-width key namespacing in the HiCache backend.

Regression: `_flatten_device` used to always emit "@sg{n}" regardless of the
negotiated SG width. Two instances on one ring that negotiate DIFFERENT
widths (e.g. mixed HCA fleet) would read each other's chunk keys and
silently reassemble the wrong bytes — same key string, different layer
spans, no error signal. Non-default widths now use "@sgw{W}.{n}"; the
historical default (29) keeps "@sg{n}" so existing cached pages stay
reachable. Mirrors the vLLM connector's `_sg_group_key` policy.

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
import os
import sys
import types
import unittest

# Make the plugin importable from the repo without install (sibling modules
# dfkv_access_log / dfkv_hot_config / dfkv_metrics / dfkv_telemetry live in
# integration/hicache/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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
    inst.mem_pool_device = object()  # non-None → device (L2-bypass) mode
    inst._metrics = _FakeMetrics()
    inst._alog_tag = "test"
    inst._log_exist_probe = lambda *a, **k: None
    return inst


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSgGroupKey(unittest.TestCase):
    def test_default_width_keeps_legacy_suffix(self):
        inst = _mk_instance(29)
        self.assertEqual(inst._sg_group_key("K", 0), "K@sg0")
        self.assertEqual(inst._sg_group_key("K", 2), "K@sg2")

    def test_nondefault_width_gets_own_namespace(self):
        inst = _mk_instance(4)
        self.assertEqual(inst._sg_group_key("K", 0), "K@sgw4.0")
        inst63 = _mk_instance(63)
        self.assertEqual(inst63._sg_group_key("K", 1), "K@sgw63.1")

    def test_width_detect_failure_falls_back_to_legacy_namespace(self):
        class _DeadLib:
            def dfkv_max_sg_segs(self, _h):
                raise RuntimeError("old lib")

        inst = _mk_instance(29)
        inst._lib = _DeadLib()
        self.assertEqual(inst._sg_group_key("K", 0), "K@sg0")


class TestFlattenDeviceKeyNames(unittest.TestCase):
    """layer_num=13 under width 4 → 4 chunks (4+4+4+1); under 29 → 1 chunk."""

    @staticmethod
    def _seglists(nlayers: int):
        ptrs = [1000 * (i + 1) for i in range(nlayers)]
        sizes = [512] * nlayers
        return ptrs, sizes

    def test_legacy_namespace_when_default_width(self):
        inst = _mk_instance(29)
        nl = 13
        ptrs, sizes = self._seglists(nl)
        stride, sks, sp, ss = inst._flatten_device(["hash0"], [ptrs], [sizes])
        self.assertEqual(sks, ["GLM-TEST/hash0_k@sg0"])
        self.assertEqual(stride, 1)
        self.assertEqual(len(sp[0]), nl)  # single chunk holds all layers

    def test_namespaced_chunking_when_narrow(self):
        inst = _mk_instance(4)
        nl = 13
        ptrs, sizes = self._seglists(nl)
        stride, sks, sp, ss = inst._flatten_device(["hash0"], [ptrs], [sizes])
        self.assertEqual(
            sks,
            ["GLM-TEST/hash0_k@sgw4.0", "GLM-TEST/hash0_k@sgw4.1",
             "GLM-TEST/hash0_k@sgw4.2", "GLM-TEST/hash0_k@sgw4.3"],
        )
        self.assertEqual(stride, 4)
        # chunk segment slicing must follow the same width
        self.assertEqual([len(p) for p in sp], [4, 4, 4, 1])
        self.assertEqual(sp[0], ptrs[0:4])
        self.assertEqual(sp[3], ptrs[12:13])

    def test_mha_pp_composition(self):
        inst = _mk_instance(4, mla=False)
        inst.enable_pp = True
        inst.pp_rank = 1
        nl = 3
        ptrs, sizes = self._seglists(nl)
        stride, sks, sp, ss = inst._flatten_device(
            ["hash0"], [ptrs, ptrs], [sizes, sizes])
        self.assertEqual(
            sks,
            ["GLM-TEST/hash0_8_3_pp1_k@sgw4.0", "GLM-TEST/hash0_8_3_pp1_v@sgw4.0"],
        )
        self.assertEqual(stride, 2)  # sub=2 objects, one chunk each


class TestCrossWidthIsolation(unittest.TestCase):
    """The actual failure mode: width-4 writer vs width-29 reader over a
    shared ring must MISS (cold cache), never return mis-split bytes."""

    @staticmethod
    def _ring_from(inst, hash_name: str, nlayers: int, width: int):
        """Simulate a "ring": the set of key strings `inst` would have written."""
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
        reader4._ring = ring  # sole key universe contains only @sg{n} keys

        sks, hits = self._probe(reader4, ["hashX"])
        # the reader probes its own namespace (@sgw4.0) → nothing matches
        self.assertEqual(sks, ["GLM-TEST/hashX_k@sgw4.0"])
        self.assertEqual(hits, 0)

    def test_same_width_hits(self):
        writer = _mk_instance(4)
        ring = self._ring_from(writer, "hashX", 13, 4)

        reader = _mk_instance(4)
        reader._ring = ring
        sks, hits = self._probe(reader, ["hashX"])
        self.assertEqual(sks, ["GLM-TEST/hashX_k@sgw4.0"])
        self.assertEqual(hits, 1)

    def test_default_width_still_finds_legacy_cached_pages(self):
        writer = _mk_instance(29)
        ring = self._ring_from(writer, "hashX", 13, 29)

        reader = _mk_instance(29)
        reader._ring = ring
        sks, hits = self._probe(reader, ["hashX"])
        self.assertEqual(sks, ["GLM-TEST/hashX_k@sg0"])
        self.assertEqual(hits, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
