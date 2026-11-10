"""HiCache L3 under SGLang DCP: admission flag and logical-page probes.

Pure Python: no libdfkv, server, torch or GPU. The local ``sglang`` shim has no
``l3_page_span``, which also proves the plugin loads on SGLang builds without
the DCP storage helpers.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "integration", "hicache"))

import dfkv_hicache  # noqa: E402
from sglang.srt.mem_cache import hicache_storage  # noqa: E402


class _Pool:
    """Host pool whose page-buffer probe insists on whole logical pages, as
    SGLang's get_page_buffer_meta does under DCP."""

    def __init__(self, page_size, dcp_size=None, components=1, nbytes=4096):
        self.page_size = page_size
        if dcp_size is not None:
            self.dcp_size = dcp_size
        self.components = components
        self.nbytes = nbytes
        self.probed = []

    def get_page_buffer_meta(self, indices):
        n = len(indices)
        span = self.page_size * getattr(self, "dcp_size", 1)
        assert n % span == 0, f"probe of {n} slots is not whole logical pages of {span}"
        self.probed.append(n)
        pages = n // span
        return ([0x1000] * (pages * self.components),
                [self.nbytes] * (pages * self.components))


def _backend():
    st = dfkv_hicache.DfkvHiCache.__new__(dfkv_hicache.DfkvHiCache)
    st._alog_tag = "r0"
    st._validate_pool_object_sizes = lambda sizes, context: None
    st._register_pool_buffers = lambda pool: 0
    return st


class DcpStorageAdmission(unittest.TestCase):
    def test_class_declares_dcp_storage(self):
        # Read by DCP-gating SGLang builds before any instance exists.
        self.assertIs(dfkv_hicache.DfkvHiCache.supports_dcp_storage, True)

    def test_plugin_loads_without_sglang_l3_page_span(self):
        self.assertFalse(hasattr(hicache_storage, "l3_page_span"))


class LogicalPageSpan(unittest.TestCase):
    def test_span(self):
        span = dfkv_hicache._l3_page_span
        self.assertEqual(span(_Pool(64)), 64)               # no DCP attribute
        self.assertEqual(span(_Pool(64, dcp_size=1)), 64)
        self.assertEqual(span(_Pool(64, dcp_size=8)), 512)
        self.assertEqual(span(_Pool(1)), 1)                 # KDA state pool
        self.assertEqual(span(_Pool(64, dcp_size=0)), 64)   # unset treated as 1

    def test_v2_registration_probes_one_logical_page(self):
        for dcp, expected in ((None, 64), (1, 64), (8, 512)):
            st = _backend()
            pool = _Pool(64, dcp_size=dcp)
            st.register_mem_host_pool_v2(pool, "indexer")
            self.assertEqual(pool.probed, [expected])
            self.assertEqual(st._pool_component_names["indexer"], ("all",))
            self.assertTrue(st._pool_is_replicated("indexer"))

    def test_v2_state_pool_probe_is_unchanged(self):
        st = _backend()
        pool = _Pool(1, components=2)
        pool.conv_state_elem_sizes = [1]
        pool.temporal_state_elem_size = 4
        st.register_mem_host_pool_v2(pool, "mamba")
        self.assertEqual(pool.probed, [1])
        self.assertEqual(st._pool_component_names["mamba"], ("temporal", "conv0"))

    def test_primary_pool_probes_one_logical_page(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("register_mem_pool_host's rdma probe needs torch")
        for dcp, expected in ((None, 64), (8, 512)):
            st = _backend()
            st.transport_mode = "rdma"
            pool = _Pool(64, dcp_size=dcp)
            st.register_mem_pool_host(pool)
            self.assertEqual(pool.probed, [expected])


if __name__ == "__main__":
    unittest.main()
