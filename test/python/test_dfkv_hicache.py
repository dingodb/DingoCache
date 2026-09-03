"""TDD R5 — DingoFS SGLang HiCache plugin, validated GPU/torch-free.

Spawns real dfkv_server cache nodes, drives the plugin's zero-copy v1 path with
numpy host buffers (CPU), and asserts MLA single-object keying, backup_skip,
batch_exists longest-prefix, and header-mismatch => miss.
"""
import ctypes
import logging
import os
import subprocess
import shutil
import sys
import tempfile
import time
import unittest
import warnings
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# shim 'sglang' (no torch) first on path, then the real plugin source dir.
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "integration", "hicache"))  # <repo>/integration/hicache

BUILD = os.environ.get("DFKV_BUILD", os.path.join(HERE, "..", "..", "build"))
SERVER_BIN = os.path.join(BUILD, "dfkv_server")

# These plugin tests are single-rank in one process, so the same-host
# rendezvous can leak published exist answers across tests when synthetic keys
# and namespaces collide inside the TTL. Pin it off so the datapath is
# deterministic; rendezvous behavior has dedicated node_dedup coverage.
os.environ.setdefault("DFKV_CLIENT_NODE_DEDUP", "0")


@contextmanager
def _env(key, value):
    """Set an env var for the duration of a block, restoring the prior value."""
    saved = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # noqa: E402
import dfkv_hicache  # noqa: E402  (RED until implemented)
import dfkv_access_log as alog  # noqa: E402


class FakeMlaPool:
    """Stand-in for MLATokenToKVPoolHost: one packed latent object per page."""

    def __init__(self, num_pages, page_bytes, page_size=64):
        self.page_size = page_size
        self.page_bytes = page_bytes
        self._num_pages = num_pages
        self.buf = np.zeros(num_pages * page_bytes, dtype=np.uint8)
        self._base = self.buf.ctypes.data

    def get_ksize_per_token(self):
        return self.page_bytes // self.page_size

    def get_page_buffer_meta(self, host_indices):
        # MLA: one pointer + size per page; host_indices stride = page_size.
        ptrs, sizes = [], []
        for i in range(0, len(host_indices), self.page_size):
            page_idx = host_indices[i] // self.page_size
            ptrs.append(self._base + page_idx * self.page_bytes)
            sizes.append(self.page_bytes)
        return ptrs, sizes

    def fill_page(self, page_idx, byte):
        s = page_idx * self.page_bytes
        self.buf[s:s + self.page_bytes] = byte

    def page_bytes_at(self, page_idx):
        s = page_idx * self.page_bytes
        return bytes(self.buf[s:s + self.page_bytes])

    def zero(self):
        self.buf[:] = 0


class FlatBuf:
    """Mimics a torch flat host page tensor (data_ptr/numel/element_size) over a
    numpy buffer, for exercising the generic get/batch_get path without torch."""

    def __init__(self, nbytes):
        self.arr = np.zeros(nbytes, dtype=np.uint8)

    def data_ptr(self):
        return self.arr.ctypes.data

    def numel(self):
        return int(self.arr.size)

    def element_size(self):
        return int(self.arr.itemsize)

    def tobytes(self):
        return bytes(self.arr)


class RegistrablePool(FakeMlaPool):
    """Fake v2 pool with a tensor-like backing buffer attribute (no hybrid
    accessor) — exercises the fixed-attribute fallback probe."""

    def __init__(self, num_pages, page_bytes, page_size=64):
        super().__init__(num_pages, page_bytes, page_size)
        self.data_buffer = FlatBuf(num_pages * page_bytes)


class FakeDSAIndexerPool(FakeMlaPool):
    """Stand-in for SGLang's DSAIndexerPoolHost: its backing tensor lives under
    `index_k_with_scale_buffer`, a name NOT in the old fixed-attr probe list, so
    it is reachable only through get_hybrid_pool_buffer(). This is the pool whose
    registration logged "no backing buffer found" for GLM-5.2 DSA."""

    def __init__(self, num_pages, page_bytes, page_size=64):
        super().__init__(num_pages, page_bytes, page_size)
        self.index_k_with_scale_buffer = FlatBuf(num_pages * page_bytes)

    def get_hybrid_pool_buffer(self):
        return [self.index_k_with_scale_buffer]


class FakeDSAPagedPool(FakeMlaPool):
    """Stand-in for DeepSeekV4PagedHostPool: kv_buffer is a LIST of per-layer
    tensors, self-reported via get_hybrid_pool_buffer() (mirrors the real pool)."""

    def __init__(self, num_pages, page_bytes, page_size=64, layers=2):
        super().__init__(num_pages, page_bytes, page_size)
        self.kv_buffer = [FlatBuf(num_pages * page_bytes) for _ in range(layers)]

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]


class FakeHybridStatePool:
    """Stand-in for SGLang's MambaPoolHost (Kimi-K3 KDA/mamba state): one page
    spans TWO independent tensors -- temporal (SSM) state plus conv state --
    and get_page_buffer_meta() emits one (ptr, size) pair per component per
    page in a stable order. Rank-sharded: every TP rank owns different bytes,
    so unlike the replicated MLA pools every rank must persist its own shard."""

    def __init__(self, num_pages, temporal_bytes, conv_bytes, page_size=64):
        self.page_size = page_size
        self.temporal_bytes = temporal_bytes
        self.conv_bytes = conv_bytes
        self.temporal_state_elem_size = temporal_bytes
        self.conv_state_elem_sizes = [conv_bytes]
        self.tbuf = np.zeros(num_pages * temporal_bytes, dtype=np.uint8)
        self.cbuf = np.zeros(num_pages * conv_bytes, dtype=np.uint8)

    def get_page_buffer_meta(self, host_indices):
        ptrs, sizes = [], []
        for i in range(0, len(host_indices), self.page_size):
            page_idx = int(host_indices[i]) // self.page_size
            ptrs.append(self.tbuf.ctypes.data + page_idx * self.temporal_bytes)
            sizes.append(self.temporal_bytes)
            ptrs.append(self.cbuf.ctypes.data + page_idx * self.conv_bytes)
            sizes.append(self.conv_bytes)
        return ptrs, sizes

    def fill_page(self, page_idx, byte):
        ts, cs = page_idx * self.temporal_bytes, page_idx * self.conv_bytes
        self.tbuf[ts:ts + self.temporal_bytes] = byte
        self.cbuf[cs:cs + self.conv_bytes] = (byte + 1) % 256

    def page_state_at(self, page_idx):
        ts, cs = page_idx * self.temporal_bytes, page_idx * self.conv_bytes
        return (bytes(self.tbuf[ts:ts + self.temporal_bytes]),
                bytes(self.cbuf[cs:cs + self.conv_bytes]))

    def zero(self):
        self.tbuf[:] = 0
        self.cbuf[:] = 0


class FakeLogicalAnchorPool:
    """Stand-in for SGLang's LogicalHostPool: the primary "kv" pool of a
    V4/DSA-compressed model (e.g. GLM-5.2). Holds NO KV buffer, so its
    get_page_buffer_meta() returns None — the real KV lives in side-pools moved
    over the v2 PoolTransfer path. The v1 anchor call must no-op on this."""

    def __init__(self, page_size=64):
        self.page_size = page_size

    def get_page_buffer_meta(self, host_indices):
        return None


class FakeTupleNoneAnchorPool:
    """SGLang v0.5.17+ DeepSeekV4PagedHostPool contract: the logical anchor
    declares "no host layout" as a tuple containing None instead of None
    itself. Semantics are identical to FakeLogicalAnchorPool."""

    def __init__(self, page_size=64):
        self.page_size = page_size

    def get_page_buffer_meta(self, host_indices):
        return (None, None)


def _spawn_node(tag):
    d = tempfile.mkdtemp(prefix=f"dfkv_py_{tag}_")
    p = subprocess.Popen(
        [SERVER_BIN, "--dir", d, "--port", "0", "--cap", str(1 << 30),
         "--store-engine", "file"],
        stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline().strip()
    assert line.startswith("PORT "), f"bad server greeting: {line!r}"
    port = int(line.split()[1])
    return p, d, port


def _count_objects(node_dir):
    n = 0
    for root, _dirs, files in os.walk(os.path.join(node_dir, "blocks")):
        n += sum(1 for f in files if not f.endswith(".tmp"))
    return n


class DingoFSHiCacheTest(unittest.TestCase):
    PAGE_SIZE = 64
    PAGE_BYTES = 4096  # small page for fast tests

    @classmethod
    def setUpClass(cls):
        cls.procs = []
        cls.dirs = []

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
            if p.stdout is not None:
                p.stdout.close()
        for d in cls.dirs:
            shutil.rmtree(d, ignore_errors=True)

    def _node(self, tag):
        p, d, port = _spawn_node(tag)
        self.procs.append(p)
        self.dirs.append(d)
        return f"{tag}=127.0.0.1:{port}", port, d

    def _cfg(self, members, tp_rank=0, tp_size=8, page_size=64, model="glm-5.1",
             pp_rank=0, pp_size=1, is_mla_model=True):
        return HiCacheStorageConfig(
            tp_rank=tp_rank, tp_size=tp_size, is_mla_model=is_mla_model,
            is_page_first_layout=False, model_name=model,
            pp_rank=pp_rank, pp_size=pp_size,
            extra_config={
                "members": members,
                "interface_v1": 1,
            })

    def _plugin(self, cfg, pool):
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_pool_host(pool)
        return st

    def test_instantiable_all_abstract_methods_present(self):
        members, _, _ = self._node("inst")
        pool = FakeMlaPool(4, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        self.assertTrue(hasattr(st, "get") and hasattr(st, "set"))

    def test_requires_interface_v1(self):
        # Missing interface_v1 must fail fast (enforces the zero-copy deploy
        # contract; otherwise SGLang silently uses the slower generic copy path).
        cfg = self._cfg("n=127.0.0.1:1")
        cfg.extra_config.pop("interface_v1")
        with self.assertRaises(ValueError):
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)

    def test_requires_model_identity(self):
        cfg = self._cfg("n=127.0.0.1:1", model="")
        with self.assertRaises(dfkv_hicache._tcfg.DfkvConfigError):
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)

    def test_operator_namespace_override_is_rejected(self):
        cfg = self._cfg("n=127.0.0.1:1")
        cfg.extra_config["key_namespace"] = "shared/model"
        with self.assertRaisesRegex(
                ValueError, "namespace identity is derived automatically"):
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)

    def test_requires_ring_endpoint(self):
        # Neither members nor mds_endpoints => no ring to connect to.
        cfg = self._cfg("")  # empty members
        with self.assertRaises(dfkv_hicache._tcfg.DfkvConfigError):
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)

    def test_namespace_is_derived_from_model_identity_and_layout(self):
        members, _, _ = self._node("namespace")
        cfg = self._cfg(members, model="org/model")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        self.assertEqual(
            st._key_namespace,
            dfkv_hicache.canonical_namespace(
                "org/model",
                dfkv_hicache.SGLANG_HICACHE_RAW_V1,
                dtype="unknown",
                block_tokens=64,
                tp_size=8,
                layout_fields={
                    "page_size": 64,
                    "dtype": "unknown",
                    "dtype_tag": 0,
                    "is_mla": True,
                    "layer_num": 0,
                    "head_num": 0,
                    "head_dim": 0,
                    "pcp_size": 1,
                    "dcp_size": 1,
                },
            ),
        )

    def test_prefill_cp_replicated_writer_key_cross_rank_roundtrip(self):
        members, _, _ = self._node("cpreplicated")
        cfg0 = self._cfg(members, tp_rank=0, tp_size=8, model="glm-cp")
        cfg7 = self._cfg(members, tp_rank=7, tp_size=8, model="glm-cp")
        for rank, cfg in ((0, cfg0), (7, cfg7)):
            cfg.extra_config.update({
                "pcp_size": 8,
                "pcp_rank": rank,
                "prefill_cp_storage_layout": "replicated",
            })
        writer_pool = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        reader_pool = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        writer_pool.fill_page(0, 0xA7)
        writer = self._plugin(cfg0, writer_pool)
        reader = self._plugin(cfg7, reader_pool)
        host_indices = list(range(self.PAGE_SIZE))
        self.assertEqual(writer.batch_set_v1(["shared"], host_indices), [True])
        self.assertEqual(reader.batch_get_v1(["shared"], host_indices), [True])
        self.assertEqual(reader_pool.page_bytes_at(0), bytes([0xA7]) * self.PAGE_BYTES)
        self.assertEqual(writer._keys("shared"), reader._keys("shared"))

    def test_prefill_cp_layout_is_explicit_and_sharded_fails_closed(self):
        for layout, message in ((None, "explicit"), ("sharded", "unsupported")):
            cfg = self._cfg(
                "n=127.0.0.1:1", tp_rank=0, tp_size=8, model="glm-cp")
            cfg.extra_config.update({"pcp_size": 8, "pcp_rank": 0})
            if layout is not None:
                cfg.extra_config["prefill_cp_storage_layout"] = layout
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(ValueError, message):
                    dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)

    def test_rdma_depth_is_preserved_for_connection_fanout(self):
        members, _, _ = self._node("rdepth")
        os.environ["DFKV_RDMA_DEPTH"] = "4"
        try:
            cfg = self._cfg(members)
            cfg.extra_config["rdma_depth"] = 8
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
            self.assertEqual(os.environ.get("DFKV_RDMA_DEPTH"), "8")
        finally:
            os.environ.pop("DFKV_RDMA_DEPTH", None)

    def _rail_for(self, members, rails_csv, rank, size, fallbacks=None):
        saved_dev = os.environ.get("DFKV_RDMA_DEV")
        saved_primary = os.environ.get("DFKV_RDMA_PRIMARY_DEV")
        saved_tiers = os.environ.get("DFKV_RDMA_RAIL_TIERS")
        os.environ["DFKV_RDMA_DEV"] = rails_csv
        os.environ.pop("DFKV_RDMA_NUMA", None)
        os.environ.pop("DFKV_RDMA_PRIMARY_DEV", None)
        os.environ.pop("DFKV_RDMA_RAIL_TIERS", None)
        try:
            cfg = self._cfg(members, tp_rank=0, tp_size=1)
            cfg.extra_config["rail_affinity"] = True
            if fallbacks is not None:
                cfg.extra_config["rail_affinity_fallbacks"] = fallbacks
            with patch.object(
                dfkv_hicache,
                "_resolve_local_physical_rank",
                return_value=rank,
            ):
                dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
            return (os.environ.get("DFKV_RDMA_DEV"),
                    os.environ.get("DFKV_RDMA_NUMA"),
                    os.environ.get("DFKV_RDMA_PRIMARY_DEV"),
                    os.environ.get("DFKV_RDMA_RAIL_TIERS"))
        finally:
            os.environ.pop("DFKV_RDMA_NUMA", None)
            for name, value in (
                    ("DFKV_RDMA_DEV", saved_dev),
                    ("DFKV_RDMA_PRIMARY_DEV", saved_primary),
                    ("DFKV_RDMA_RAIL_TIERS", saved_tiers)):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_rail_affinity_uses_world_group_local_rank(self):
        members, _, _ = self._node("rail8")
        rails = "ib0,ib1,ib2,ib3,ib4,ib5,ib6,ib7"
        dev, numa, primary, tiers = self._rail_for(
            members, rails, rank=3, size=8)
        self.assertEqual(dev, "ib3,ib4")
        self.assertEqual(numa, "0")
        self.assertEqual(primary, "ib3")
        self.assertIsNone(tiers)

    def test_rail_affinity_wraps_over_available_rails(self):
        members, _, _ = self._node("rail2")
        rails = "ibA,ibB"
        self.assertEqual(
            self._rail_for(members, rails, rank=2, size=8)[0], "ibA,ibB")
        self.assertEqual(
            self._rail_for(members, rails, rank=5, size=8)[0], "ibB,ibA")
        self.assertEqual(
            self._rail_for(members, rails, rank=7, size=8)[0], "ibB,ibA")

    def test_rail_affinity_can_add_one_ordered_fallback(self):
        members, _, _ = self._node("railfallback")
        rails = "ib0,ib1,ib2,ib3"
        dev, numa, primary, tiers = self._rail_for(
            members, rails, rank=2, size=4, fallbacks=1)
        self.assertEqual(dev, "ib2,ib3")
        self.assertEqual(numa, "0")
        self.assertEqual(primary, "ib2")
        self.assertIsNone(tiers)

    def test_rail_affinity_can_disable_default_fallback(self):
        members, _, _ = self._node("railnofallback")
        dev, _, primary, _ = self._rail_for(
            members, "ib0,ib1,ib2,ib3", rank=2, size=4, fallbacks=0)
        self.assertEqual(dev, "ib2")
        self.assertEqual(primary, "ib2")

    def test_rail_affinity_fallback_wraps_and_is_bounded(self):
        members, _, _ = self._node("railfallbackwrap")
        rails = "ib0,ib1,ib2,ib3"
        dev, _, primary, tiers = self._rail_for(
            members, rails, rank=3, size=4, fallbacks=99)
        self.assertEqual(dev, "ib3,ib0,ib1,ib2")
        self.assertEqual(primary, "ib3")
        self.assertIsNone(tiers)

    def test_rdma_numa_extra_config_sets_env(self):
        # The new rdma_numa knob opts into client-side NUMA-aware rail selection
        # by setting DFKV_RDMA_NUMA=1 (replaces rail_affinity's old side effect).
        members, _, _ = self._node("rnuma")
        os.environ.pop("DFKV_RDMA_NUMA", None)
        try:
            cfg = self._cfg(members)
            cfg.extra_config["rdma_numa"] = True
            dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
            self.assertEqual(os.environ.get("DFKV_RDMA_NUMA"), "1")
        finally:
            os.environ.pop("DFKV_RDMA_NUMA", None)

    def test_transport_mode_recorded(self):
        members, _, _ = self._node("tmode")
        saved_rdma = os.environ.get("DFKV_RDMA")
        os.environ.pop("DFKV_RDMA", None)
        try:
            cfg = self._cfg(members)
            st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
            self.assertTrue(st.transport_mode.startswith("tcp"), st.transport_mode)
        finally:
            if saved_rdma is None:
                os.environ.pop("DFKV_RDMA", None)
            else:
                os.environ["DFKV_RDMA"] = saved_rdma

    def test_generic_get_roundtrip(self):
        # Generic (non zero-copy) set/get round-trips a page through dfkv.
        members, _, _ = self._node("gget")
        cfg = self._cfg(members)
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        payload = bytes((i * 7) & 0xFF for i in range(self.PAGE_BYTES))
        self.assertTrue(st.set("g0", payload))
        tgt = FlatBuf(self.PAGE_BYTES)
        self.assertIs(st.get("g0", tgt), tgt)        # hit returns the buffer
        self.assertEqual(tgt.tobytes(), payload)     # bytes round-trip
        self.assertIsNone(st.get("g_missing", FlatBuf(self.PAGE_BYTES)))  # miss

    def test_generic_batch_get(self):
        members, _, _ = self._node("gbget")
        cfg = self._cfg(members)
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        p0, p1 = bytes([1]) * self.PAGE_BYTES, bytes([2]) * self.PAGE_BYTES
        self.assertTrue(st.set("b0", p0))
        self.assertTrue(st.set("b1", p1))
        bufs = [FlatBuf(self.PAGE_BYTES) for _ in range(3)]
        res = st.batch_get(["b0", "b1", "b_miss"], bufs)
        self.assertIs(res[0], bufs[0]); self.assertEqual(bufs[0].tobytes(), p0)
        self.assertIs(res[1], bufs[1]); self.assertEqual(bufs[1].tobytes(), p1)
        self.assertIsNone(res[2])

    def test_batch_set_get_v1_roundtrip(self):
        members, _, _ = self._node("rt")
        pool = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        keys = ["p0", "p1", "p2"]
        for i in range(3):
            pool.fill_page(i, 10 + i)
        host_indices = list(range(3 * self.PAGE_SIZE))  # 3 pages
        self.assertEqual(st.batch_set_v1(keys, host_indices), [True, True, True])
        expected = [pool.page_bytes_at(i) for i in range(3)]
        pool.zero()
        self.assertEqual(st.batch_get_v1(keys, host_indices), [True, True, True])
        for i in range(3):
            self.assertEqual(pool.page_bytes_at(i), expected[i])

    def test_client_metrics_count_set_and_get(self):
        # #5: plugin counts client-side read/write volume (pages/bytes/hits).
        members, _, _ = self._node("metrics")
        pool = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        keys = ["m0", "m1", "m2"]
        host_indices = list(range(3 * self.PAGE_SIZE))
        st.batch_set_v1(keys, host_indices)
        st.batch_get_v1(keys, host_indices)
        m = st._metrics.snapshot()
        self.assertEqual(m["set_calls"], 1)
        self.assertEqual(m["set_pages"], 3)
        self.assertEqual(m["set_ok_pages"], 3)
        self.assertEqual(m["set_bytes"], 3 * self.PAGE_BYTES)
        self.assertEqual(m["get_calls"], 1)
        self.assertEqual(m["get_pages"], 3)
        self.assertEqual(m["get_hit_pages"], 3)
        self.assertEqual(m["get_bytes"], 3 * self.PAGE_BYTES)
        # latency histograms observed one duration per batch call
        self.assertEqual(m["set_observations"], 1)
        self.assertEqual(m["get_observations"], 1)

    def test_client_metrics_count_v2_set_get_exists(self):
        # v2 (DSA side-pool) path metrics. For a V4/DSA model (GLM-5.2) the real
        # KV rides batch_set_v2/get_v2 while batch_set_v1 only writes one-byte
        # anchor markers, so the *_v1 counters stay blind — the *_v2 counters
        # express the L3 write/hit rate. Mirrors the anchor+side-pool v2 flow.
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer
        members, _, _ = self._node("metricsv2")
        cfg = self._cfg(members, model="glm-5.2")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        side = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        st.register_mem_host_pool_v2(side, "deepseek_v4_c4")
        keys = ["v0", "v1", "v2"]
        hi = list(range(3 * self.PAGE_SIZE))
        st.batch_set_v1(keys, hi)  # one-byte anchor markers -> v1 counter inert
        st.batch_set_v2([PoolTransfer(name="deepseek_v4_c4",
                                      host_indices=hi, keys=keys)])
        st.batch_get_v2([PoolTransfer(name="deepseek_v4_c4",
                                      host_indices=hi, keys=keys)])
        res = st.batch_exists_v2(
            keys, [PoolTransfer(name="deepseek_v4_c4", host_indices=hi, keys=keys)])
        m = st._metrics.snapshot()
        # set_v2: one call, 3 side-pool pages acked OK, bytes = 3 * page
        self.assertEqual(m["set_v2_calls"], 1)
        self.assertEqual(m["set_v2_pages"], 3)
        self.assertEqual(m["set_v2_ok_pages"], 3)
        self.assertEqual(m["set_v2_bytes"], 3 * self.PAGE_BYTES)
        # get_v2: one call, 3 pages requested, all hit
        self.assertEqual(m["get_v2_calls"], 1)
        self.assertEqual(m["get_v2_pages"], 3)
        self.assertEqual(m["get_v2_hit_pages"], 3)
        self.assertEqual(m["get_v2_bytes"], 3 * self.PAGE_BYTES)
        # exist_v2: probed 3 candidate keys; all 3 form the usable hit prefix.
        # This ratio (hit/probe) is the DSA L3 hit-rate signal.
        self.assertEqual(m["exist_v2_calls"], 1)
        self.assertEqual(m["exist_v2_probe_pages"], 3)
        self.assertEqual(m["exist_v2_hit_pages"], res.kv_hit_pages)
        self.assertEqual(m["exist_v2_hit_pages"], 3)
        self.assertEqual(m["exist_calls"], 1)
        self.assertEqual(m["exist_probe_pages"], 3)
        self.assertEqual(m["exist_present_pages"], 3)
        self.assertEqual(m["exist_contiguous_pages"], 3)
        self.assertEqual(m["exist_result_full_hit"], 1)
        self.assertEqual(m["exist_observations"], 1)
        # v2 latency histograms observed once per set_v2/get_v2 call
        self.assertEqual(m["set_v2_observations"], 1)
        self.assertEqual(m["get_v2_observations"], 1)
        # the v1 write counter never moved: batch_set_v1 hit the anchor branch
        # and returned before on_set — proving the *_v1 blindness these fix.
        self.assertEqual(m["set_calls"], 0)

    def test_v2_set_metrics_skip_on_mla_rank_nonzero(self):
        # MLA rank!=0 replicates the latent, so batch_set_v2 is a no-op skip
        # ([True] markers, no I/O). It must NOT inflate the write-ok metric.
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer
        members, _, _ = self._node("metricsv2skip")
        cfg = self._cfg(members, model="glm-5.2", tp_rank=3)  # rank!=0, MLA
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        side = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st.register_mem_host_pool_v2(side, "deepseek_v4_c4")
        keys = ["s0", "s1"]
        hi = list(range(2 * self.PAGE_SIZE))
        st.batch_set_v2([PoolTransfer(name="deepseek_v4_c4",
                                      host_indices=hi, keys=keys)])
        m = st._metrics.snapshot()
        self.assertEqual(m["set_v2_calls"], 0)
        self.assertEqual(m["set_v2_ok_pages"], 0)
        self.assertEqual(m["set_v2_bytes"], 0)

    def test_v2_follower_rank_writes_rank_sharded_pool(self):
        # Kimi-K3 hybrid recurrent state is rank-sharded, so every follower rank
        # must persist its own temporal/conv bytes. backup_skip applies only to
        # physically replicated pools. Component coordinates and sharding are
        # inferred from the registered host layout rather than model names; the
        # v2 metrics must therefore record real follower I/O instead of gating
        # every nonzero TP rank.
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer
        T, C = 4096, 512  # temporal / conv bytes per page (shape-agnostic)
        members, _, _ = self._node("v2ranksharded")
        cfg = self._cfg(members, model="kimi-k3", tp_rank=3, tp_size=16)
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        mamba = FakeHybridStatePool(2, T, C, self.PAGE_SIZE)
        st.register_mem_host_pool_v2(mamba, "mamba")
        self.assertEqual(
            st._pool_components("mamba"), ("temporal", "conv0"))
        self.assertFalse(st._pool_is_replicated("mamba"))
        for i in range(2):
            mamba.fill_page(i, 30 + i)
        keys = ["m0", "m1"]
        hi = list(range(2 * self.PAGE_SIZE))
        res = st.batch_set_v2([PoolTransfer(name="mamba",
                                            host_indices=hi, keys=keys)])
        self.assertEqual(res["mamba"], [True, True])
        # the follower's write is real I/O and must be visible in the metrics
        m = st._metrics.snapshot()
        self.assertEqual(m["set_v2_calls"], 1)
        self.assertEqual(m["set_v2_pages"], 2)
        self.assertEqual(m["set_v2_ok_pages"], 2)
        self.assertEqual(m["set_v2_bytes"], 2 * (T + C))
        # both components round-trip byte-exact on the same rank
        exp = [mamba.page_state_at(i) for i in range(2)]
        mamba.zero()
        g = st.batch_get_v2([PoolTransfer(name="mamba",
                                          host_indices=hi, keys=keys)])
        self.assertEqual(g["mamba"], [True, True])
        for i in range(2):
            self.assertEqual(mamba.page_state_at(i), exp[i])
        # keys are sharded per rank: rank 0 sees a MISS for the same page hashes
        cfg0 = self._cfg(members, model="kimi-k3", tp_rank=0, tp_size=16)
        st0 = dfkv_hicache.DfkvHiCache(cfg0, cfg0.extra_config)
        st0.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        mamba0 = FakeHybridStatePool(2, T, C, self.PAGE_SIZE)
        st0.register_mem_host_pool_v2(mamba0, "mamba")
        g0 = st0.batch_get_v2([PoolTransfer(name="mamba",
                                            host_indices=hi, keys=keys)])
        self.assertEqual(g0["mamba"], [False, False])

    # --- PP key-isolation tests (pure logic, no server/lib needed) ---
    # _keys()/_pool_keys() are pure functions of self.{model,tp_rank,tp_size,
    # is_mla,pp_rank,pp_size,enable_pp}. Bypass __init__ (which needs libdfkv.so
    # + a live server) via __new__ and set those attrs directly.
    def _keyonly(self, **kw):
        st = dfkv_hicache.DfkvHiCache.__new__(dfkv_hicache.DfkvHiCache)
        st.model = kw.get("model", "glm-5.1")
        st.tp_rank = kw.get("tp_rank", 0)
        st.tp_size = kw.get("tp_size", 8)
        st.is_mla = kw.get("is_mla", True)
        st.pp_rank = kw.get("pp_rank", 0)
        st.pp_size = kw.get("pp_size", 1)
        st.enable_pp = st.pp_size > 1
        st.pcp_size = kw.get("pcp_size", 1)
        st.pcp_rank = kw.get("pcp_rank", 0)
        st.dcp_size = kw.get("dcp_size", 1)
        st.dcp_rank = kw.get("dcp_rank", 0)
        st.prefill_cp_storage_layout = kw.get(
            "prefill_cp_storage_layout", None)
        st._storage_pcp_rank = (
            0 if st.prefill_cp_storage_layout == "replicated"
            else st.pcp_rank)
        st._pool_component_names = {"extra": ("all",)}
        st._pool_replicated = {"extra": True}
        return st

    def test_keys_use_canonical_binary_pool_schema(self):
        mla = self._keyonly(is_mla=True)
        self.assertEqual(
            mla._keys(b"abc\x00\xff"),
            [
                dfkv_hicache.pool_key(
                    b"abc\x00\xff", pool="kv", tp_size=8, tp_rank=-1,
                    component="all")
            ],
        )
        self.assertNotEqual(mla._keys(b"abc\x00\xff"), mla._keys(b"abc"))
        mha = self._keyonly(is_mla=False, tp_rank=2, tp_size=8)
        self.assertEqual(
            mha._keys("abc"),
            [
                dfkv_hicache.pool_key(
                    "abc", pool="kv", tp_size=8, tp_rank=2, component="k"),
                dfkv_hicache.pool_key(
                    "abc", pool="kv", tp_size=8, tp_rank=2, component="v"),
            ],
        )

    def test_replicated_prefill_cp_uses_writer_key(self):
        rank0 = self._keyonly(
            pcp_size=8,
            pcp_rank=0,
            prefill_cp_storage_layout="replicated",
        )
        rank7 = self._keyonly(
            pcp_size=8,
            pcp_rank=7,
            prefill_cp_storage_layout="replicated",
        )
        self.assertEqual(rank0._keys("same-page"), rank7._keys("same-page"))
        self.assertEqual(
            rank7._keys("same-page"),
            [dfkv_hicache.pool_key(
                "same-page", pool="kv", tp_size=8, tp_rank=-1,
                pcp_size=8, pcp_rank=0, component="all")],
        )

    def test_keys_separate_pcp_and_dcp_physical_ranks(self):
        page_hash = b"same-page\x00hash"
        for axis in ("pcp", "dcp"):
            rank0 = self._keyonly(**{
                f"{axis}_size": 2, f"{axis}_rank": 0})
            rank1 = self._keyonly(**{
                f"{axis}_size": 2, f"{axis}_rank": 1})
            key0 = rank0._keys(page_hash)[0]
            key1 = rank1._keys(page_hash)[0]
            self.assertNotEqual(key0, key1)
            expected = {
                "pcp_size": 2 if axis == "pcp" else 1,
                "pcp_rank": 1 if axis == "pcp" else 0,
                "dcp_size": 2 if axis == "dcp" else 1,
                "dcp_rank": 1 if axis == "dcp" else 0,
            }
            self.assertEqual(
                key1,
                dfkv_hicache.pool_key(
                    page_hash, pool="kv", tp_size=8, tp_rank=-1,
                    component="all", **expected))

    def test_multirank_pcp_dcp_require_explicit_in_range_rank(self):
        for axis in ("pcp", "dcp"):
            missing = self._cfg("n=127.0.0.1:1")
            missing.extra_config[f"{axis}_size"] = 2
            with self.assertRaisesRegex(
                    ValueError, f"{axis}_rank is required"):
                dfkv_hicache.DfkvHiCache(
                    missing, missing.extra_config)

            invalid = self._cfg("n=127.0.0.1:1")
            invalid.extra_config.update({
                f"{axis}_size": 2,
                f"{axis}_rank": 2,
            })
            with self.assertRaisesRegex(ValueError, f"{axis}_rank"):
                dfkv_hicache.DfkvHiCache(
                    invalid, invalid.extra_config)

    def test_multi_pool_probe_failures_do_not_register_replicated_fallback(self):
        class BrokenPool:
            page_size = 64

            def get_page_buffer_meta(self, _indices):
                raise RuntimeError("probe failed")

        class EmptyPool:
            page_size = 64

            def get_page_buffer_meta(self, _indices):
                return [], []

        class AmbiguousPool:
            page_size = 64

            def get_page_buffer_meta(self, _indices):
                return [1, 2], [4, 4]

        for pool in (BrokenPool(), EmptyPool(), AmbiguousPool()):
            st = self._keyonly()
            st.registered_pools = {}
            st._pool_component_names = {}
            st._pool_replicated = {}
            with self.assertRaisesRegex(
                    RuntimeError, "cannot discover physical layout"):
                st.register_mem_host_pool_v2(pool, "unsafe")
            self.assertNotIn("unsafe", st.registered_pools)
            self.assertNotIn("unsafe", st._pool_component_names)
            with self.assertRaisesRegex(RuntimeError, "no discovered layout"):
                st._pool_keys("unsafe", "page")

    def test_keys_mha_pp_on_encodes_pp_rank(self):
        pp0 = self._keyonly(
            is_mla=False, tp_rank=0, tp_size=8, pp_rank=0, pp_size=4)
        pp1 = self._keyonly(
            is_mla=False, tp_rank=0, tp_size=8, pp_rank=1, pp_size=4)
        self.assertEqual(
            pp0._keys("xyz"),
            [
                dfkv_hicache.pool_key(
                    "xyz", tp_size=8, tp_rank=0,
                    pp_size=4, pp_rank=0, component="k"),
                dfkv_hicache.pool_key(
                    "xyz", tp_size=8, tp_rank=0,
                    pp_size=4, pp_rank=0, component="v"),
            ],
        )
        self.assertEqual(
            pp1._keys("xyz"),
            [
                dfkv_hicache.pool_key(
                    "xyz", tp_size=8, tp_rank=0,
                    pp_size=4, pp_rank=1, component="k"),
                dfkv_hicache.pool_key(
                    "xyz", tp_size=8, tp_rank=0,
                    pp_size=4, pp_rank=1, component="v"),
            ],
        )
        self.assertNotEqual(set(pp0._keys("xyz")), set(pp1._keys("xyz")))

    def test_keys_mla_pp_on_encodes_pp_rank(self):
        pp0 = self._keyonly(is_mla=True, pp_rank=0, pp_size=2)
        pp1 = self._keyonly(is_mla=True, pp_rank=1, pp_size=2)
        self.assertEqual(
            pp0._keys("h"),
            [dfkv_hicache.pool_key(
                "h", tp_size=8, tp_rank=-1,
                pp_size=2, pp_rank=0, component="all")],
        )
        self.assertEqual(
            pp1._keys("h"),
            [dfkv_hicache.pool_key(
                "h", tp_size=8, tp_rank=-1,
                pp_size=2, pp_rank=1, component="all")],
        )
        self.assertNotEqual(pp0._keys("h"), pp1._keys("h"))

    def test_pool_keys_aux_pool_carries_pp_coordinate(self):
        pp0 = self._keyonly(is_mla=True, pp_rank=0, pp_size=2)
        pp1 = self._keyonly(is_mla=True, pp_rank=1, pp_size=2)
        self.assertEqual(
            pp0._pool_keys("extra", "p"),
            [dfkv_hicache.pool_key(
                "p", pool="extra", tp_size=8, tp_rank=-1,
                pp_size=2, pp_rank=0, component="all")],
        )
        self.assertEqual(
            pp1._pool_keys("extra", "p"),
            [dfkv_hicache.pool_key(
                "p", pool="extra", tp_size=8, tp_rank=-1,
                pp_size=2, pp_rank=1, component="all")],
        )
        self.assertEqual(pp0._pool_keys("kv", "p"), pp0._keys("p"))

    def test_keys_pp_isolates_across_full_tp_pp_matrix(self):
        # Exhaustive: no two (tp_rank, pp_rank) pairs may produce the same key
        # set on the same page_hash for MHA.
        seen = set()
        for tp_r in range(4):
            for pp_r in range(3):
                st = self._keyonly(is_mla=False, tp_rank=tp_r, tp_size=4,
                                   pp_rank=pp_r, pp_size=3)
                k = tuple(sorted(st._keys("page")))
                self.assertNotIn(k, seen, f"key collision at tp{tp_r}/pp{pp_r}: {k}")
                seen.add(k)

    def test_mla_writes_single_object_per_page(self):
        members, port, ndir = self._node("mla")
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        keys = ["q0", "q1"]
        st.batch_set_v1(keys, list(range(2 * self.PAGE_SIZE)))
        # one object per page (MLA), not two (no _k/_v split)
        self.assertEqual(_count_objects(ndir), 2)

    def test_backup_skip_nonzero_tp_rank_does_not_write(self):
        members, port, ndir = self._node("skip")
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members, tp_rank=3), pool)  # MLA + rank!=0
        keys = ["r0", "r1"]
        self.assertEqual(st.batch_set_v1(keys, list(range(2 * self.PAGE_SIZE))),
                         [True, True])  # reported success...
        self.assertEqual(_count_objects(ndir), 0)  # ...but nothing written (backup_skip)

    def test_batch_exists_longest_prefix(self):
        members, _, _ = self._node("exist")
        pool = FakeMlaPool(4, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(self._cfg(members), pool)
        st.batch_set_v1(["e0", "e1"], list(range(2 * self.PAGE_SIZE)))
        # first 2 exist, 3rd missing -> prefix length 2
        n = st.batch_exists(["e0", "e1", "e2"])
        self.assertEqual(n, 2)

    def test_namespace_mismatch_is_miss(self):
        members, _, _ = self._node("namespace-miss")
        pool_w = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        pool_w.fill_page(0, 7)
        writer_cfg = self._cfg(members, model="model/layout-a")
        writer = self._plugin(writer_cfg, pool_w)
        writer.batch_set_v1(["h0"], list(range(self.PAGE_SIZE)))
        pool_r = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        reader_cfg = self._cfg(members, model="model/layout-b")
        reader = self._plugin(reader_cfg, pool_r)
        self.assertEqual(
            reader.batch_get_v1(["h0"], list(range(self.PAGE_SIZE))), [False])

    def test_pp_stages_isolate_on_same_page_hash(self):
        # Two PP stages (pp_rank 0 and 1) writing the SAME page_hash must land
        # in separate storage objects — PP splits the model by layer, so each
        # stage holds a different KV slice for the same prefix. Before the fix,
        # _keys() ignored pp_rank and the stages overwrote each other.
        members, _, _ = self._node("pp")
        pool0 = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        pool0.fill_page(0, 0xAA)
        st0 = self._plugin(self._cfg(members, pp_rank=0, pp_size=2), pool0)
        st0.batch_set_v1(["shared"], list(range(self.PAGE_SIZE)))
        # pp_rank=1 writes a different payload for the same page_hash; with the
        # fix it must NOT clobber pp_rank=0's object.
        pool1 = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        pool1.fill_page(0, 0xBB)
        st1 = self._plugin(self._cfg(members, pp_rank=1, pp_size=2), pool1)
        st1.batch_set_v1(["shared"], list(range(self.PAGE_SIZE)))
        # pp_rank=0 reads back its OWN bytes, not pp_rank=1's.
        pool0_r = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        st0_r = self._plugin(self._cfg(members, pp_rank=0, pp_size=2), pool0_r)
        self.assertEqual(st0_r.batch_get_v1(["shared"], list(range(self.PAGE_SIZE))),
                         [True])
        self.assertTrue((pool0_r.buf[0:self.PAGE_BYTES] == 0xAA).all(),
                        "pp_rank=0 payload clobbered by pp_rank=1")

    def test_batch_v2_multi_pool_roundtrip(self):
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer
        members, _, _ = self._node("v2")
        cfg = self._cfg(members)
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        kv = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        ex = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        st.register_mem_host_pool_v2(kv, "kv")
        st.register_mem_host_pool_v2(ex, "extra")
        for i in range(3):
            kv.fill_page(i, 20 + i)
            ex.fill_page(i, 50 + i)
        keys = ["a0", "a1", "a2"]
        hi = list(range(3 * self.PAGE_SIZE))
        trs = [PoolTransfer(name="kv", host_indices=hi, keys=keys),
               PoolTransfer(name="extra", host_indices=hi, keys=keys)]
        res = st.batch_set_v2(trs)
        self.assertEqual(res["kv"], [True, True, True])
        self.assertEqual(res["extra"], [True, True, True])
        expk = [kv.page_bytes_at(i) for i in range(3)]
        expe = [ex.page_bytes_at(i) for i in range(3)]
        kv.zero(); ex.zero()
        g = st.batch_get_v2(trs)
        self.assertEqual(g["kv"], [True, True, True])
        self.assertEqual(g["extra"], [True, True, True])
        for i in range(3):
            self.assertEqual(kv.page_bytes_at(i), expk[i])
            self.assertEqual(ex.page_bytes_at(i), expe[i])
        r = st.batch_exists_v2(keys, [PoolTransfer(name="extra", host_indices=hi, keys=keys)])
        self.assertEqual(r.kv_hit_pages, 3)

    def test_v1_logical_anchor_writes_markers_no_crash(self):
        # V4/DSA models (e.g. GLM-5.2): the primary "kv" pool is a logical anchor
        # whose get_page_buffer_meta() returns None. batch_set_v1 must not crash
        # (was: TypeError unpacking None); it writes a one-byte "kv" marker per page
        # so batch_exists can find the primary prefix, and batch_get_v1 no-ops
        # (all pages "present" so the hybrid controller loads the real side pools).
        members, _, ndir = self._node("anchor")
        st = self._plugin(self._cfg(members, model="glm-5.2"),
                          FakeLogicalAnchorPool(self.PAGE_SIZE))
        keys = ["kv0", "kv1", "kv2"]
        hi = list(range(3 * self.PAGE_SIZE))
        self.assertEqual(st.batch_set_v1(keys, hi), [True, True, True])
        # markers were written (MLA: one object per page) and are discoverable
        self.assertEqual(_count_objects(ndir), 3)
        self.assertEqual(st.batch_exists(keys), 3)
        self.assertEqual(st.batch_exists(keys + ["kv3"]), 3)  # absent page stops prefix
        # anchor load is a no-op but reports all pages complete
        self.assertEqual(st.batch_get_v1(keys, hi), [True, True, True])
        self.assertTrue(st._anchor_noop_warned)

    def test_v1_logical_anchor_tuple_none_contract(self):
        # SGLang v0.5.17+ (DeepSeekV4PagedHostPool) declares the logical anchor
        # as a tuple containing None instead of None itself. Same marker
        # semantics as the None contract; must not crash or hard-miss.
        members, _, ndir = self._node("anchor17")
        st = self._plugin(self._cfg(members, model="glm-5.2"),
                          FakeTupleNoneAnchorPool(self.PAGE_SIZE))
        keys = ["t0", "t1"]
        hi = list(range(2 * self.PAGE_SIZE))
        self.assertEqual(st.batch_set_v1(keys, hi), [True, True])
        self.assertEqual(_count_objects(ndir), 2)
        self.assertEqual(st.batch_exists(keys), 2)
        self.assertEqual(st.batch_get_v1(keys, hi), [True, True])
        self.assertTrue(st._anchor_noop_warned)

    def test_register_v2_logical_anchor_marker_only_no_raise(self):
        # The v0.5.17 DSV4 hybrid stack registers the logical "kv" pool through
        # register_mem_host_pool_v2 at attach time. Both no-layout wire forms
        # must register marker-only instead of raising (was: RuntimeError
        # "pool component probe returned no layout" crashing the scheduler).
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer
        members, _, ndir = self._node("regv2anchor")
        cfg = self._cfg(members, model="glm-5.2")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_host_pool_v2(FakeLogicalAnchorPool(self.PAGE_SIZE), "kv")
        self.assertTrue(st._pool_logical["kv"])
        st.register_mem_host_pool_v2(
            FakeTupleNoneAnchorPool(self.PAGE_SIZE), "kv")
        self.assertTrue(st._pool_logical["kv"])
        # v2 writes on the logical pool land one-byte markers that gate the
        # prefix; v2 reads no-op complete so side pools still get loaded.
        keys = ["r0", "r1", "r2"]
        hi = list(range(3 * self.PAGE_SIZE))
        res = st.batch_set_v2(
            [PoolTransfer(name="kv", host_indices=hi, keys=keys)])
        self.assertEqual(res["kv"], [True, True, True])
        self.assertEqual(_count_objects(ndir), 3)
        self.assertEqual(st.batch_exists(keys), 3)
        res = st.batch_get_v2(
            [PoolTransfer(name="kv", host_indices=hi, keys=keys)])
        self.assertEqual(res["kv"], [True, True, True])

    def test_register_v2_physical_pool_clears_logical_flag(self):
        # Re-registering the same name with a real layout must drop the
        # marker-only state so the physical path takes over.
        members, _, _ = self._node("regv2flip")
        cfg = self._cfg(members, model="glm-5.2")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_host_pool_v2(
            FakeTupleNoneAnchorPool(self.PAGE_SIZE), "side")
        self.assertTrue(st._pool_logical["side"])
        st.register_mem_host_pool_v2(
            FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE), "side")
        self.assertNotIn("side", st._pool_logical)

    def test_v2_exists_with_logical_anchor_full_and_shrunk(self):
        # End-to-end multi-pool existence for a logical-anchor (DSA) model: the
        # "kv" markers (v1) + a compressed side pool (v2) together define the hit
        # prefix, and a missing side-pool page shrinks it.
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer
        members, _, _ = self._node("anchorv2")
        st = dfkv_hicache.DfkvHiCache(self._cfg(members, model="glm-5.2"),
                                      self._cfg(members, model="glm-5.2").extra_config)
        st.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        side = FakeMlaPool(3, self.PAGE_BYTES, self.PAGE_SIZE)
        st.register_mem_host_pool_v2(side, "deepseek_v4_c4")
        keys = ["a0", "a1", "a2"]
        hi = list(range(3 * self.PAGE_SIZE))
        # backup: v1 anchor markers (all 3) + v2 side pool for only the first 2
        st.batch_set_v1(keys, hi)
        st.batch_set_v2([PoolTransfer(name="deepseek_v4_c4",
                                      host_indices=hi[:2 * self.PAGE_SIZE],
                                      keys=keys[:2])])
        trs = [PoolTransfer(name="deepseek_v4_c4", host_indices=hi, keys=keys)]
        # side pool present for 2/3 -> usable prefix shrinks to 2
        self.assertEqual(st.batch_exists_v2(keys, trs).kv_hit_pages, 2)
        # fill the 3rd side-pool page -> full 3-page hit
        st.batch_set_v2([PoolTransfer(name="deepseek_v4_c4",
                                      host_indices=hi[2 * self.PAGE_SIZE:],
                                      keys=keys[2:])])
        self.assertEqual(st.batch_exists_v2(keys, trs).kv_hit_pages, 3)

    def test_v2_exists_trailing_pool_sliding_window(self):
        # TRAILING_PAGES (SWA / Mamba state): only the last N pages of a prefix
        # need this pool. A hit must NOT collapse just because earlier window
        # pages were evicted (the old `all(present)` logic wrongly did).
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer, PoolHitPolicy
        members, _, _ = self._node("swa")
        cfg = self._cfg(members, model="glm-5.2")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        swa = FakeMlaPool(4, self.PAGE_BYTES, self.PAGE_SIZE)
        st.register_mem_host_pool_v2(swa, "swa")
        keys = ["s0", "s1", "s2", "s3"]
        hi = list(range(4 * self.PAGE_SIZE))
        st.batch_set_v1(keys, hi)  # 4 "kv" markers -> kv prefix = 4
        # SWA present for only the trailing 2 pages (sliding window = 2)
        st.batch_set_v2([PoolTransfer(name="swa",
                                      host_indices=hi[2 * self.PAGE_SIZE:],
                                      keys=keys[2:])])
        trs = [PoolTransfer(name="swa", host_indices=hi, keys=keys[2:],
                            hit_policy=PoolHitPolicy.TRAILING_PAGES)]
        # trailing window (last 2) present -> full 4-page prefix stays usable
        self.assertEqual(st.batch_exists_v2(keys, trs).kv_hit_pages, 4)
        # but if the trailing window is broken (last page missing), it shrinks
        swa2 = FakeMlaPool(4, self.PAGE_BYTES, self.PAGE_SIZE)
        members2, _, _ = self._node("swa2")
        cfg2 = self._cfg(members2, model="glm-5.2")
        st2 = dfkv_hicache.DfkvHiCache(cfg2, cfg2.extra_config)
        st2.register_mem_pool_host(FakeLogicalAnchorPool(self.PAGE_SIZE))
        st2.register_mem_host_pool_v2(swa2, "swa")
        st2.batch_set_v1(keys, hi)
        # SWA present only for pages 0,1 (window covers a stale tail) -> for a
        # window of 2 the best prefix whose last 2 pages are present is len 2.
        st2.batch_set_v2([PoolTransfer(name="swa",
                                       host_indices=hi[:2 * self.PAGE_SIZE],
                                       keys=keys[:2])])
        self.assertEqual(st2.batch_exists_v2(keys, trs).kv_hit_pages, 2)

    def test_register_mem_host_pool_v2_registers_backing_buffer(self):
        members, _, _ = self._node("v2reg")
        cfg = self._cfg(members)
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        pool = RegistrablePool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        calls = []
        st.register_memory = lambda base, size: calls.append((base, size)) or True
        st.register_mem_host_pool_v2(pool, "extra")
        self.assertIs(st.registered_pools["extra"], pool)
        self.assertEqual(calls, [
            (pool.data_buffer.data_ptr(),
             pool.data_buffer.numel() * pool.data_buffer.element_size())
        ])

    def test_register_dsa_indexer_pool_via_hybrid_accessor(self):
        # DSAIndexerPoolHost keeps its buffer under `index_k_with_scale_buffer`,
        # not kv_buffer — reachable only via get_hybrid_pool_buffer(). The old
        # fixed-attr probe missed it ("no backing buffer found") and every DSA
        # indexer page fell back to per-op MR registration, which then failed in
        # RdmaTransport::CacheFrom and flunked the whole write batch.
        members, _, _ = self._node("dsaidx")
        cfg = self._cfg(members, model="glm-5.2")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        pool = FakeDSAIndexerPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        calls = []
        st.register_memory = lambda base, size: calls.append((base, size)) or True
        # Drive the real failing entrypoint, not just the helper.
        st.register_mem_host_pool_v2(pool, "deepseek_v4_c4_indexer")
        self.assertIs(st.registered_pools["deepseek_v4_c4_indexer"], pool)
        self.assertEqual(calls, [
            (pool.index_k_with_scale_buffer.data_ptr(),
             pool.index_k_with_scale_buffer.numel()
             * pool.index_k_with_scale_buffer.element_size())
        ])

    def test_register_dsa_paged_pool_registers_all_layer_buffers(self):
        # DeepSeekV4PagedHostPool exposes a LIST of per-layer tensors; every one
        # must be registered (the old "first attribute wins" break stopped early).
        members, _, _ = self._node("dsapaged")
        cfg = self._cfg(members, model="glm-5.2")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        pool = FakeDSAPagedPool(2, self.PAGE_BYTES, self.PAGE_SIZE, layers=3)
        calls = []
        st.register_memory = lambda base, size: calls.append((base, size)) or True
        self.assertEqual(st._register_pool_buffers(pool), 3)
        self.assertEqual(
            calls,
            [(b.data_ptr(), b.numel() * b.element_size()) for b in pool.kv_buffer],
        )

    def test_hybrid_accessor_preferred_over_attribute_probe(self):
        # When a pool offers get_hybrid_pool_buffer(), it is authoritative — a
        # stale/misleading kv_buffer attribute must NOT be probed instead.
        members, _, _ = self._node("dsaprec")
        cfg = self._cfg(members, model="glm-5.2")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        accessor_buf = FlatBuf(2 * self.PAGE_BYTES)
        decoy = FlatBuf(2 * self.PAGE_BYTES)
        pool.kv_buffer = decoy  # would be picked by the fixed-attr probe
        pool.get_hybrid_pool_buffer = lambda: [accessor_buf]
        calls = []
        st.register_memory = lambda base, size: calls.append((base, size)) or True
        st._register_pool_buffers(pool)
        self.assertEqual(calls, [
            (accessor_buf.data_ptr(),
             accessor_buf.numel() * accessor_buf.element_size())
        ])

    def test_register_dedups_region_shared_across_pools(self):
        # A DSA model registers the KV anchor plus several sidecar pools; if two
        # pools surface the same physical tensor it must be registered once.
        members, _, _ = self._node("dsadedup")
        cfg = self._cfg(members, model="glm-5.2")
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        shared = FlatBuf(2 * self.PAGE_BYTES)
        pool_a = FakeDSAIndexerPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        pool_a.index_k_with_scale_buffer = shared
        pool_b = FakeDSAIndexerPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        pool_b.index_k_with_scale_buffer = shared
        calls = []
        st.register_memory = lambda base, size: calls.append((base, size)) or True
        self.assertEqual(st._register_pool_buffers(pool_a), 1)
        self.assertEqual(st._register_pool_buffers(pool_b), 0)  # already registered
        self.assertEqual(calls, [
            (shared.data_ptr(), shared.numel() * shared.element_size())
        ])


def _reset_access_log():
    """Reset the process-global access-log state for test isolation.

    configure() is idempotent per process, so without this every test after the
    first would inherit the first config. Tests legitimately reach into the
    module internals here (there is no production reset API by design)."""
    alog._stop_listener(alog._listener)
    logging.getLogger("dfkv.access").handlers.clear()
    alog._ENABLED = False
    alog._THRESHOLD_US = 0
    alog._logger = None
    alog._listener = None
    alog._configured = False


class DfkvAccessLogTest(unittest.TestCase):
    """Access log on the inherited interface methods: toggle, path, format."""

    PAGE_SIZE = 64
    PAGE_BYTES = 4096

    @classmethod
    def setUpClass(cls):
        cls.procs = []
        cls.dirs = []

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
            if p.stdout is not None:
                p.stdout.close()
        for d in cls.dirs:
            shutil.rmtree(d, ignore_errors=True)

    def setUp(self):
        _reset_access_log()
        self.tmp = tempfile.mkdtemp(prefix="dfkv_alog_")

    def tearDown(self):
        _reset_access_log()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _node(self, tag):
        p, d, port = _spawn_node(tag)
        self.procs.append(p)
        self.dirs.append(d)
        return f"{tag}=127.0.0.1:{port}"

    def _cfg(self, members, alog_extra, tp_rank=0):
        ec = {
            "members": members,
            "dtype_tag": 0x46384534, "page_size": self.PAGE_SIZE,
            "layer_num": 78, "head_num": 1, "head_dim": 576,
            "interface_v1": 1,
        }
        ec.update(alog_extra)
        return HiCacheStorageConfig(
            tp_rank=tp_rank, tp_size=8, is_mla_model=True,
            is_page_first_layout=False, model_name="glm-5.1", extra_config=ec)

    def _plugin(self, cfg, pool=None):
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        if pool is not None:
            st.register_mem_pool_host(pool)
        return st

    def _flush(self):
        # Drain the async queue so emitted lines hit the file before we read it.
        if alog._listener is not None:
            alog._listener.stop()
            alog._listener = None

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_disabled_by_default_writes_no_file(self):
        members = self._node("ad")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        # access_log absent -> disabled even though a path is supplied.
        cfg = self._cfg(members, {"access_log_path": path})
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        st.batch_set_v1(["d0", "d1"], list(range(2 * self.PAGE_SIZE)))
        self._flush()
        self.assertFalse(alog.is_enabled())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "acc.0.log")))

    def test_enabled_writes_one_line_per_op(self):
        members = self._node("aw")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path})
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        self.assertTrue(alog.is_enabled())
        hi = list(range(2 * self.PAGE_SIZE))
        st.batch_set_v1(["w0", "w1"], hi)
        st.batch_get_v1(["w0", "w1"], hi)
        st.batch_exists(["w0", "w1", "w_miss"])  # 2 present, 1 missing
        self._flush()
        txt = self._read(os.path.join(self.tmp, "acc.0.log"))
        self.assertIn(
            "init(r0 glm-5.1 tp=0/8 pcp=0/1 storage_pcp=0/1 "
            "dcp=0/1 physical_rank=- mla=1) : ok static", txt
        )
        self.assertIn("batch_set_v1(r0 2 keys) : ok 2/2", txt)
        self.assertIn("batch_get_v1(r0 2 keys) : hits=2/2", txt)
        self.assertIn("batch_exists(r0 3 keys) : prefix=2/3", txt)
        self.assertIn("<0.", txt)  # duration token present

    def test_generic_get_set_exists_logged(self):
        members = self._node("ag")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path})
        st = self._plugin(cfg)
        payload = bytes((i * 3) & 0xFF for i in range(self.PAGE_BYTES))
        self.assertTrue(st.set("g0", payload))
        self.assertIs(st.get("g0", FlatBuf(self.PAGE_BYTES)).__class__, FlatBuf)
        self.assertTrue(st.exists("g0"))
        self.assertFalse(st.exists("g_missing"))
        self.assertIsNone(st.get("g_missing", FlatBuf(self.PAGE_BYTES)))
        self._flush()
        txt = self._read(os.path.join(self.tmp, "acc.0.log"))
        g0 = dfkv_hicache._key_label(st._keys("g0")[0])
        missing = dfkv_hicache._key_label(st._keys("g_missing")[0])
        self.assertIn(f"set(r0 {g0}, 4.00KiB) : ok", txt)
        self.assertIn(": hit", txt)        # get hit
        self.assertIn(": miss", txt)       # get miss
        self.assertIn(f"exists(r0 {g0}) : found", txt)
        self.assertIn(f"exists(r0 {missing}) : not_found", txt)

    def test_auto_rank_suffix_and_backup_skip(self):
        members = self._node("as")
        # no {rank} placeholder -> path is auto-suffixed .r{rank}
        path = os.path.join(self.tmp, "acc.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path},
                        tp_rank=3)  # MLA + rank!=0 -> backup_skip
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        self.assertEqual(st.batch_set_v1(["s0", "s1"], list(range(2 * self.PAGE_SIZE))),
                         [True, True])  # value unchanged by logging
        self._flush()
        suffixed = os.path.join(self.tmp, "acc.log.r3")
        self.assertTrue(os.path.exists(suffixed))
        self.assertIn("batch_set_v1(r3 2 keys) : backup_skip", self._read(suffixed))

    def test_path_placeholder_substituted(self):
        members = self._node("ap")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path},
                        tp_rank=5)
        pool = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        st.batch_set_v1(["x0"], list(range(self.PAGE_SIZE)))
        self._flush()
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "acc.5.log")))

    def test_threshold_filters_fast_ops(self):
        members = self._node("at")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path,
                                  "access_log_threshold_us": 10_000_000})  # 10s
        pool = FakeMlaPool(2, self.PAGE_BYTES, self.PAGE_SIZE)
        st = self._plugin(cfg, pool)
        hi = list(range(2 * self.PAGE_SIZE))
        st.batch_set_v1(["t0", "t1"], hi)
        st.batch_get_v1(["t0", "t1"], hi)
        self._flush()
        p = os.path.join(self.tmp, "acc.0.log")
        txt = self._read(p) if os.path.exists(p) else ""
        self.assertNotIn("batch_set_v1", txt)
        self.assertNotIn("batch_get_v1", txt)

    def test_env_var_fallback_enables(self):
        members = self._node("ae")
        envp = os.path.join(self.tmp, "env.{rank}.log")
        keys = ("DFKV_ACCESS_LOG_ENABLED", "DFKV_ACCESS_LOG_PATH")
        old = {k: os.environ.get(k) for k in keys}
        os.environ["DFKV_ACCESS_LOG_ENABLED"] = "1"
        os.environ["DFKV_ACCESS_LOG_PATH"] = envp
        try:
            cfg = self._cfg(members, {})  # no extra_config access_log keys
            pool = FakeMlaPool(1, self.PAGE_BYTES, self.PAGE_SIZE)
            st = self._plugin(cfg, pool)
            self.assertTrue(alog.is_enabled())
            st.batch_get_v1(["z0"], list(range(self.PAGE_SIZE)))
            self._flush()
            self.assertTrue(os.path.exists(os.path.join(self.tmp, "env.0.log")))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_failures_are_logged_and_reraised(self):
        members = self._node("af")
        path = os.path.join(self.tmp, "acc.{rank}.log")
        cfg = self._cfg(members, {"access_log": 1, "access_log_path": path})
        st = self._plugin(cfg)
        self.assertFalse(st.set("f0", None))  # value None -> "fail none"

        def _boom(*a, **k):
            raise RuntimeError("boom")
        st._lib.dfkv_put = _boom  # fresh per-plugin CDLL, isolated to this test
        with self.assertRaises(RuntimeError):
            st.set("f1", b"x" * 16)  # exception must propagate, not be swallowed
        self._flush()
        txt = self._read(os.path.join(self.tmp, "acc.0.log"))
        self.assertIn(
            f"set(r0 {dfkv_hicache._key_label(st._keys('f0')[0])}, 0B) : fail none",
            txt,
        )
        self.assertIn("FAIL RuntimeError: boom", txt)


class DfkvAccessLogRotationTest(unittest.TestCase):
    """Size-based rotation of the access-log file (RotatingFileHandler).

    Pure module-level: drives dfkv_access_log.configure + the logger directly, so
    it needs no cache node / libdfkv.so (unlike DfkvAccessLogTest). Codifies that
    an enabled access log no longer grows a single file unbounded."""

    def setUp(self):
        _reset_access_log()
        self.tmp = tempfile.mkdtemp(prefix="dfkv_alog_rot_")

    def tearDown(self):
        _reset_access_log()

    def _flush(self):
        # stop() drains the queue (writing + rolling over remaining records) and
        # joins the listener thread before we inspect the files.
        if alog._listener is not None:
            alog._listener.stop()
            alog._listener = None

    def _emit(self, n, nbytes=200):
        line = "x" * nbytes
        for _ in range(n):
            alog._logger.info("%s", line)

    def test_rotates_by_size_and_caps_backup_count(self):
        path = os.path.join(self.tmp, "acc.{rank}.log")
        alog.configure({"access_log": 1, "access_log_path": path,
                        "access_log_max_bytes": 4096,
                        "access_log_backup_count": 2}, tp_rank=0)
        self.assertTrue(alog.is_enabled())
        base = os.path.join(self.tmp, "acc.0.log")
        self._emit(400)  # ~400 * ~225B = ~90KB >> 4KiB*3 cap -> many rollovers
        self._flush()
        # base + .1 + .2 only; .3 must never appear (backup_count=2 caps it)
        self.assertTrue(os.path.exists(base))
        self.assertTrue(os.path.exists(base + ".1"))
        self.assertTrue(os.path.exists(base + ".2"))
        self.assertFalse(os.path.exists(base + ".3"))
        # disk is bounded: each retained file <= maxBytes (+ one line of slack)
        for suffix in ("", ".1", ".2"):
            self.assertLessEqual(os.path.getsize(base + suffix), 4096 + 512)

    def test_max_bytes_zero_keeps_single_unbounded_file(self):
        # Escape hatch: max_bytes=0 restores the legacy plain-FileHandler behavior
        # (one file, no rotation) for anyone who wants it.
        path = os.path.join(self.tmp, "acc.{rank}.log")
        alog.configure({"access_log": 1, "access_log_path": path,
                        "access_log_max_bytes": 0}, tp_rank=0)
        base = os.path.join(self.tmp, "acc.0.log")
        self._emit(200)
        self._flush()
        self.assertFalse(os.path.exists(base + ".1"))   # rotation disabled
        self.assertGreater(os.path.getsize(base), 4096)  # grew past a rotation size

    def test_defaults_enable_rotation(self):
        # No explicit knobs -> rotation ON by default so disk is always bounded.
        # The 128MiB default isn't reached here, so only the base file exists.
        path = os.path.join(self.tmp, "acc.{rank}.log")
        alog.configure({"access_log": 1, "access_log_path": path}, tp_rank=0)
        import logging.handlers as _h
        sink = alog._listener.handlers[0]  # capture before _flush() nulls listener
        base = os.path.join(self.tmp, "acc.0.log")
        self._emit(50)
        self._flush()
        self.assertIsInstance(sink, _h.RotatingFileHandler)
        self.assertTrue(os.path.exists(base))
        self.assertFalse(os.path.exists(base + ".1"))  # 128MiB default not reached


class ClientStatsPollerTest(unittest.TestCase):
    """Mirrors the C client's Prometheus snapshot onto delta counters. Pure
    module-level: feeds a fake snapshot provider, no node / libdfkv.so."""

    def _poller(self, texts):
        from dfkv_metrics import ClientStatsPoller
        # get_text returns successive snapshots from `texts`, repeating the last
        seq = {"i": 0}

        def get_text():
            i = min(seq["i"], len(texts) - 1)
            seq["i"] += 1
            return texts[i]

        return ClientStatsPoller(get_text, tp_rank=0, interval_s=0.01)

    def test_shared_parser_handles_bare_and_labeled_metrics(self):
        from dfkv_common.client_metrics import parse_snapshot
        text = ("# TYPE dfkv_client_ops_served_total counter\n"
                "dfkv_client_ops_served_total 7\n"
                "dfkv_client_peer_errors_total{peer=\"1.2.3.4:1\"} 3\n")
        samples = {sample.name: sample for sample in parse_snapshot(text)}
        self.assertEqual(samples["dfkv_client_ops_served_total"].value, 7)
        self.assertEqual(samples["dfkv_client_peer_errors_total"].value, 3)
        self.assertEqual(samples["dfkv_client_peer_errors_total"].labels,
                         ("1.2.3.4:1",))

    def test_shared_parser_keeps_scheduler_and_resource_budget_metrics(self):
        from dfkv_common.client_metrics import parse_snapshot
        text = (
            "dfkv_read_scheduler_pending_batches 2\n"
            "dfkv_read_scheduler_pending_shards 11\n"
            "dfkv_read_scheduler_active_shards 7\n"
            "dfkv_read_scheduler_queue_delay_us_total 1234\n"
            "dfkv_read_scheduler_fairness_yields_total 9\n"
            'dfkv_rdma_client_endpoint_budget{kind="used"} 16\n'
            'dfkv_rdma_client_endpoint_budget{kind="limit"} 256\n'
            'dfkv_rdma_client_qp_budget{kind="used"} 32\n'
            'dfkv_rdma_client_wr_slot_budget{kind="limit"} 8192\n'
            'dfkv_rdma_client_registered_slot_bytes_budget{kind="used"} 4096\n'
            "dfkv_rdma_client_resource_budget_timeouts_total 3\n"
            "dfkv_rdma_endpoint_cache_hits_total 20\n"
            "dfkv_rdma_endpoint_cache_misses_total 4\n"
            "dfkv_rdma_endpoint_cache_evictions_total 1\n"
            "dfkv_unknown_future_metric 99\n"
        )
        samples = {
            (sample.name, sample.labels): sample
            for sample in parse_snapshot(text)
        }
        self.assertEqual(
            samples[("dfkv_read_scheduler_pending_shards", ())].kind,
            "gauge",
        )
        self.assertEqual(
            samples[("dfkv_read_scheduler_queue_delay_us_total", ())].kind,
            "counter",
        )
        self.assertEqual(
            samples[
                ("dfkv_rdma_client_endpoint_budget", ("used",))
            ].value,
            16,
        )
        self.assertEqual(
            samples[
                ("dfkv_rdma_client_endpoint_budget", ("limit",))
            ].value,
            256,
        )
        self.assertEqual(
            samples[
                ("dfkv_rdma_client_resource_budget_timeouts_total", ())
            ].value,
            3,
        )
        self.assertEqual(
            samples[("dfkv_rdma_endpoint_cache_evictions_total", ())].value,
            1,
        )
        self.assertNotIn(
            ("dfkv_unknown_future_metric", ()),
            samples,
        )

    def test_poll_once_mirrors_scheduler_and_resource_budget_metrics(self):
        p = self._poller([
            (
                "dfkv_read_scheduler_pending_shards 11\n"
                "dfkv_read_scheduler_active_shards 7\n"
                "dfkv_read_scheduler_queue_delay_us_total 1200\n"
                'dfkv_rdma_client_qp_budget{kind="used"} 32\n'
                'dfkv_rdma_client_qp_budget{kind="limit"} 256\n'
                "dfkv_rdma_client_resource_budget_timeouts_total 2\n"
            ),
            (
                "dfkv_read_scheduler_pending_shards 3\n"
                "dfkv_read_scheduler_active_shards 1\n"
                "dfkv_read_scheduler_queue_delay_us_total 1500\n"
                'dfkv_rdma_client_qp_budget{kind="used"} 16\n'
                'dfkv_rdma_client_qp_budget{kind="limit"} 256\n'
                "dfkv_rdma_client_resource_budget_timeouts_total 3\n"
            ),
        ])
        p.poll_once()
        p.poll_once()
        self.assertEqual(
            p.gauges()["dfkv_read_scheduler_pending_shards"],
            3,
        )
        self.assertEqual(
            p.gauges()['dfkv_rdma_client_qp_budget{kind="used"}'],
            16,
        )
        self.assertEqual(
            p.gauges()['dfkv_rdma_client_qp_budget{kind="limit"}'],
            256,
        )
        self.assertEqual(
            p.totals()["dfkv_read_scheduler_queue_delay_us_total"],
            1500,
        )
        self.assertEqual(
            p.totals()[
                "dfkv_rdma_client_resource_budget_timeouts_total"
            ],
            3,
        )

    def test_poll_once_accumulates_deltas(self):
        p = self._poller([
            "dfkv_client_ops_served_total 5\ndfkv_client_io_errors_total 1\n",
            "dfkv_client_ops_served_total 8\ndfkv_client_io_errors_total 1\n",
        ])
        p.poll_once()  # first read: delta 5 (from 0)
        t1 = p.totals()
        self.assertEqual(t1["dfkv_client_ops_served_total"], 5)
        self.assertEqual(t1["dfkv_client_io_errors_total"], 1)
        p.poll_once()  # cumulative 8 -> delta +3
        t2 = p.totals()
        self.assertEqual(t2["dfkv_client_ops_served_total"], 8)
        self.assertEqual(t2["dfkv_client_io_errors_total"], 1)  # unchanged

    def test_mirrors_mds_reachability_counter_and_gauges(self):
        # MDS-reachability metrics: the unreachable-polls COUNTER mirrors by
        # delta; ring_members / mds_reachable are GAUGES that track the current
        # value (0 during an outage, then the recovered value).
        p = self._poller([
            ("dfkv_client_mds_unreachable_polls_total 2\n"
             "dfkv_client_ring_members 0\n"
             "dfkv_client_mds_reachable 0\n"),
            ("dfkv_client_mds_unreachable_polls_total 2\n"
             "dfkv_client_ring_members 5\n"
             "dfkv_client_mds_reachable 1\n"),
        ])
        p.poll_once()  # outage: 2 failed polls, empty ring, unreachable
        self.assertEqual(p.totals()["dfkv_client_mds_unreachable_polls_total"], 2)
        self.assertEqual(p.gauges()["dfkv_client_ring_members"], 0)
        self.assertEqual(p.gauges()["dfkv_client_mds_reachable"], 0)
        p.poll_once()  # recovered: ring of 5, reachable; counter unchanged
        self.assertEqual(p.totals()["dfkv_client_mds_unreachable_polls_total"], 2)
        self.assertEqual(p.gauges()["dfkv_client_ring_members"], 5)
        self.assertEqual(p.gauges()["dfkv_client_mds_reachable"], 1)

    def test_disabled_interval_starts_no_thread(self):
        from dfkv_metrics import ClientStatsPoller
        p = ClientStatsPoller(lambda: "", tp_rank=0, interval_s=0)
        p.start()
        self.assertIsNone(p._thread)
        p.stop()  # must not crash

    def test_start_stop_lifecycle(self):
        p = self._poller(["dfkv_client_ops_served_total 1\n"])
        p.start()
        self.assertIsNotNone(p._thread)
        p.stop()
        self.assertIsNone(p._thread)


class DfkvClientRegistrationTest(unittest.TestCase):
    """SGLang HiCache ABI-v2 construction and MDS registration.

    A fake CDLL decodes the versioned options passed to ``dfkv_open_v2`` so the
    test proves discovery, registration identity, and opt-out policy are
    configured atomically without a real cache node or MDS.
    """
    PAGE_SIZE = 64
    PAGE_BYTES = 4096

    def _fake_lib(self):
        """A ctypes-shaped stand-in recording decoded ABI-v2 open options."""
        calls = {"discovery": [], "registration": [], "open": []}

        class _FakeLib:
            def __init__(self):
                self.dfkv_open_v2 = self._open_v2
                self.dfkv_transport_mode = lambda *a, **k: b"tcp"
                self.dfkv_close = lambda *a, **k: None
                self.dfkv_version = lambda *a, **k: b"2.0.0"

            def _open_v2(self, options_ptr):
                options = ctypes.cast(
                    options_ptr,
                    ctypes.POINTER(dfkv_hicache.DfkvClientOptionsV2),
                ).contents
                mds = (
                    options.mds_endpoints.decode()
                    if options.mds_endpoints
                    else ""
                )
                group = (
                    options.mds_group.decode()
                    if options.mds_group
                    else "default"
                )
                calls["open"].append(options.abi_version)
                if mds:
                    calls["discovery"].append(
                        (mds, group, options.mds_poll_ms))
                if options.flags:
                    calls["registration"].append(
                        (
                            mds,
                            group,
                            options.client_id.decode(),
                            (
                                options.client_info.decode()
                                if options.client_info
                                else ""
                            ),
                            options.client_heartbeat_ms,
                        )
                    )
                return ctypes.c_void_p(0xDEADBEEF)

        return _FakeLib(), calls

    def _mds_cfg(self, mds="127.0.0.1:9999", tp_rank=0, tp_size=8,
                 model="glm-5.1", extra=None):
        ec = {
            "mds_endpoints": mds, "mds_group": "t-grp",
            "dtype_tag": 0x46384534, "page_size": self.PAGE_SIZE,
            "layer_num": 78, "head_num": 1, "head_dim": 576,
            "interface_v1": 1,
        }
        if extra:
            ec.update(extra)
        return HiCacheStorageConfig(
            tp_rank=tp_rank, tp_size=tp_size, is_mla_model=True,
            is_page_first_layout=False, model_name=model, extra_config=ec)

    def _make_plugin(self, cfg, fake_lib):
        with patch.object(dfkv_hicache, "_load_lib", return_value=fake_lib):
            return dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)

    def test_registers_on_mds_path_with_hicache_info(self):
        # Default (DFKV_CLIENT_REGISTER unset, client_register unset) => registers.
        # Info must be type=hicache + model + tp fields, mirroring vLLM/LMCache so
        # `dfkvctl clients` populates TYPE/MODEL/TP columns.
        os.environ.pop("DFKV_CLIENT_REGISTER", None)
        fake, calls = self._fake_lib()
        cfg = self._mds_cfg(tp_rank=3, tp_size=8, model="glm-5.2")
        self._make_plugin(cfg, fake)
        self.assertEqual(len(calls["discovery"]), 1)
        self.assertEqual(calls["discovery"][0][1], "t-grp")
        self.assertEqual(len(calls["registration"]), 1)
        mds, grp, cid, info, hb = calls["registration"][0]
        self.assertEqual(mds, "127.0.0.1:9999")
        self.assertEqual(grp, "t-grp")
        self.assertEqual(hb, 10000)
        self.assertIn("type=hicache", info)
        self.assertIn("model=glm-5.2", info)
        self.assertIn("tp_size=8", info)
        self.assertIn("tp_rank=3", info)
        self.assertTrue(cid)  # client_id resolves to host_pid_rank
        # The id is the etcd key tail /clients/<id>, so it must pass the MDS
        # IsValidGroupOrId alphabet [A-Za-z0-9._-] (no ":" — see resolve_connector_id).
        self.assertTrue(all(
            (c.isalnum() and c.isascii()) or c in "._-" for c in cid),
            f"cid {cid!r} has chars outside IsValidGroupOrId alphabet")

    def test_opt_out_via_extra_config(self):
        # client_register=0 in extra_config disables registration; discovery still runs.
        os.environ.pop("DFKV_CLIENT_REGISTER", None)
        fake, calls = self._fake_lib()
        cfg = self._mds_cfg(extra={"client_register": 0})
        self._make_plugin(cfg, fake)
        self.assertEqual(len(calls["discovery"]), 1)
        self.assertEqual(calls["registration"], [])

    def test_opt_out_via_env(self):
        # DFKV_CLIENT_REGISTER=0 disables registration too.
        with _env("DFKV_CLIENT_REGISTER", "0"):
            fake, calls = self._fake_lib()
            cfg = self._mds_cfg()
            self._make_plugin(cfg, fake)
        self.assertEqual(len(calls["discovery"]), 1)
        self.assertEqual(calls["registration"], [])

    def test_extra_config_wins_over_env(self):
        # explicit client_register=1 wins over DFKV_CLIENT_REGISTER=0.
        with _env("DFKV_CLIENT_REGISTER", "0"):
            fake, calls = self._fake_lib()
            cfg = self._mds_cfg(extra={"client_register": 1})
            self._make_plugin(cfg, fake)
        self.assertEqual(len(calls["registration"]), 1)


    def test_no_registration_on_static_members_path(self):
        # Static-members deployments (no mds_endpoints) never register: there's
        # no MDS to register with, and discovery doesn't run either.
        os.environ.pop("DFKV_CLIENT_REGISTER", None)
        fake, calls = self._fake_lib()
        ec = {
            "members": "n1=127.0.0.1:1",
            "dtype_tag": 0x46384534, "page_size": self.PAGE_SIZE,
            "layer_num": 78, "head_num": 1, "head_dim": 576,
            "interface_v1": 1,
        }
        cfg = HiCacheStorageConfig(
            tp_rank=0, tp_size=8, is_mla_model=True, is_page_first_layout=False,
            model_name="glm-5.1", extra_config=ec)
        self._make_plugin(cfg, fake)
        self.assertEqual(calls["discovery"], [])
        self.assertEqual(calls["registration"], [])


class TestNodeDedupDefaults(unittest.TestCase):
    """resolve_node_dedup policy (R2, 2026-07-17): default OFF for every
    topology — the phase-9 MLA+TP>1 auto-on was refuted by the full-workload
    A/B (hot rounds +19-36% TTFT worse with rendezvous vs -43% TTFT better
    without). Explicit extra-config > env > default precedence unchanged."""

    def _r(self, cfg, env, mla, tp):
        from dfkv_hicache import resolve_node_dedup
        return resolve_node_dedup(cfg, env, mla, tp)

    def test_no_auto_for_mla_tp_gt1(self):
        # The phase-9 auto-on case: now stays off unless explicitly enabled.
        self.assertEqual(self._r(None, None, True, 8), (None, False))

    def test_no_auto_for_mha(self):
        self.assertEqual(self._r(None, None, False, 8), (None, False))

    def test_no_auto_for_tp1(self):
        self.assertEqual(self._r(None, None, True, 1), (None, False))

    def test_explicit_optin_via_config(self):
        self.assertEqual(self._r("1", None, True, 8), ("1", False))

    def test_env_stands(self):
        # operator already set it (either way): leave untouched, no auto log
        self.assertEqual(self._r(None, "0", True, 8), (None, False))
        self.assertEqual(self._r(None, "1", False, 1), (None, False))

    def test_config_beats_env(self):
        self.assertEqual(self._r("0", "1", True, 8), ("0", False))
        self.assertEqual(self._r("1", "0", False, 1), ("1", False))

    def test_config_truthy_forms(self):
        self.assertEqual(self._r("off", None, True, 8), ("0", False))
        self.assertEqual(self._r(1, None, False, 1), ("1", False))


class PutRetryTest(unittest.TestCase):
    """_put_flat single-retry of transiently failed puts (R1 finding,
    2026-07-17): under a cold-round write burst ~0.1% of batch_put keys fail;
    SGLang's backup bookkeeping ignores per-page False, so an unretried
    failure becomes a phantom "backed" page the hot round cannot retrieve.
    One delayed retry recovers the transient portion; keys that keep failing
    stay False (the honest result is unchanged)."""
    PAGE_SIZE = 64
    PAGE_BYTES = 4096

    def _fake_lib(self, fail_first_call_last_n=0, fail_always_keys=()):
        calls = {"put": [], "exist": 0}

        class _FakeLib:
            def __init__(self):
                self.dfkv_open_v2 = (
                    lambda *a, **k: ctypes.c_void_p(0xBEEF))
                self.dfkv_transport_mode = lambda *a, **k: b"tcp"
                self.dfkv_close = lambda *a, **k: None
                self.dfkv_version = lambda *a, **k: b"1.27.0"
                self.dfkv_batch_exist = self._exist
                self.dfkv_batch_put = self._put

            def _exist(self, h, karr, klens, n, out):
                calls["exist"] += 1
                keys = [
                    ctypes.string_at(karr[i], klens[i]) for i in range(n)]
                self.assert_lengths = [len(key) for key in keys]
                self.assert_lens = list(klens)
                assert self.assert_lengths == self.assert_lens
                for i in range(n):
                    out[i] = 0  # nothing in L3 yet: all keys proceed to put
                return 0

            def _put(self, h, karr, klens, parr, sarr, n, out):
                keys = [
                    ctypes.string_at(karr[i], klens[i]) for i in range(n)]
                assert [len(key) for key in keys] == list(klens)
                calls["put"].append(keys)
                first = len(calls["put"]) == 1
                for i, k in enumerate(keys):
                    if k in fail_always_keys:
                        out[i] = 0
                    elif first and i >= n - fail_first_call_last_n:
                        out[i] = 0  # transient burst failure
                    else:
                        out[i] = 1
                return 0

        return _FakeLib(), calls

    def _plugin(self, fake):
        os.environ["DFKV_CLIENT_NODE_DEDUP"] = "0"
        cfg = HiCacheStorageConfig(
            tp_rank=0, tp_size=8, is_mla_model=True,
            is_page_first_layout=False, model_name="glm-retry",
            extra_config={
                "members": "n0=127.0.0.1:1",
                "interface_v1": 1,
            })
        with patch.object(dfkv_hicache, "_load_lib", return_value=fake):
            st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        st.register_mem_pool_host(FakeMlaPool(8, self.PAGE_BYTES,
                                              self.PAGE_SIZE))
        return st

    def _set(self, st, npages):
        keys = [f"k{i}" for i in range(npages)]
        idx = list(range(npages * self.PAGE_SIZE))
        return st.batch_set_v1(keys, idx)

    def test_transient_failures_recovered_by_retry(self):
        fake, calls = self._fake_lib(fail_first_call_last_n=3)
        st = self._plugin(fake)
        res = self._set(st, 8)
        self.assertEqual(res, [True] * 8)
        self.assertEqual(len(calls["put"]), 2)          # initial + one retry
        self.assertEqual(len(calls["put"][1]), 3)       # only failed keys retried
        self.assertEqual(st._put_retry_recovered, 3)

    def test_persistent_failures_stay_false(self):
        key = dfkv_hicache.pool_key(
            "k0", pool="kv", tp_size=8, tp_rank=-1, component="all")
        fake, calls = self._fake_lib(fail_always_keys={key})
        st = self._plugin(fake)
        res = self._set(st, 4)
        self.assertEqual(res, [False, True, True, True])
        self.assertEqual(len(calls["put"]), 2)
        self.assertEqual(calls["put"][1], [key])
        self.assertEqual(st._put_retry_recovered, 0)

    def test_no_retry_when_all_succeed(self):
        fake, calls = self._fake_lib()
        st = self._plugin(fake)
        res = self._set(st, 8)
        self.assertEqual(res, [True] * 8)
        self.assertEqual(len(calls["put"]), 1)          # no second call
        self.assertEqual(st._put_retry_recovered, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
