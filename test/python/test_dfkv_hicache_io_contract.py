"""HiCache connector I/O contracts that must hold without SGLang or a GPU.

Pure Python: no libdfkv, server, torch or numpy. A fake native client stands
in for libdfkv and small stubs for the SGLang pools the connector is handed.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "integration", "hicache"))

import dfkv_hicache  # noqa: E402


class _HostPool:
    """A host pool with a physical layout: one buffer per page object."""

    def __init__(self, components=1, nbytes=64):
        self.components = components
        self.nbytes = nbytes

    def get_page_buffer_meta(self, indices):
        return ([0x1000] * self.components, [self.nbytes] * self.components)


class _FakeLib:
    """What these tests need of libdfkv: the SG width and a host batch GET."""

    def __init__(self, hit=True):
        self.hit = hit
        self.gets = 0

    def dfkv_max_sg_segs(self, _handle):
        return 4

    def dfkv_batch_get(self, _handle, _keys, _lens, _ptrs, _sizes, n, out):
        self.gets += 1
        for i in range(n):
            out[i] = 1 if self.hit else 0
        return 0


def _plugin(**attrs):
    """A DfkvHiCache with only what a unit needs (no client opened)."""
    st = object.__new__(dfkv_hicache.DfkvHiCache)
    st.model = "m"
    st.tp_rank = 0
    st.tp_size = 1
    st.is_mla = True
    st.pp_rank = 0
    st.pp_size = 1
    st.enable_pp = False
    st.pcp_size = 1
    st.pcp_rank = 0
    st.dcp_size = 1
    st.dcp_rank = 0
    st._device_direct_requested = False
    st._device_registration_failed = False
    st._pool_component_names = {}
    st._pool_replicated = {}
    st._has_retain = False
    st._alog_tag = "r0"
    st._metrics = dfkv_hicache._Metrics(0)
    st._lib = _FakeLib()
    st._h = 1
    st.mem_pool_host = _HostPool()
    st._log_exist_probe = lambda *a, **k: None
    for name, value in attrs.items():
        setattr(st, name, value)
    return st


class _Tensor:
    def __init__(self, ptr):
        self.ptr = ptr

    def data_ptr(self):
        return self.ptr

    def element_size(self):
        return 1


class _MlaDevicePool:
    """A layer-first MLA device pool, as the test device_page_meta reads it."""

    def __init__(self, layers, page_size=2, kv_cache_dim=8):
        self.page_size = page_size
        self.kv_cache_dim = kv_cache_dim
        self.kv_buffer = [_Tensor(0x100000 * (i + 1)) for i in range(layers)]


class _Store:
    """The keys that exist in the cache, as the existence probe sees them."""

    def __init__(self, plugin):
        self.present = set()
        self.probes = []
        plugin._batch_exist_flat = self.exist

    def exist(self, subkeys):
        self.probes.append(list(subkeys))
        return [sk in self.present for sk in subkeys]


class DeviceDirectHostGet(unittest.TestCase):
    """batch_get_v1 while primary KV is stored device-direct.

    It cannot load device-direct objects into a host buffer, so it loads
    nothing and answers for presence. It used to answer "retrieved" for every
    page, whether or not the page was in the store.
    """

    def _device_direct(self, **attrs):
        st = _plugin(mem_pool_device=object(), _device_direct_requested=True,
                     **attrs)
        return st, _Store(st)

    def test_page_missing_from_the_store_is_not_reported_retrieved(self):
        st, store = self._device_direct()
        store.present.update(st._stored_keys("kv", "present"))
        self.assertEqual(
            st.batch_get_v1(["present", "missing"], [0, 1]), [True, False])
        self.assertEqual(st._lib.gets, 0, "nothing is read into the host pool")

    def test_every_stored_group_of_a_page_must_exist(self):
        st, store = self._device_direct(_sg_chunks={"kv": 2})
        groups = st._stored_keys("kv", "page")
        self.assertEqual(len(groups), 2)
        store.present.add(groups[0])
        self.assertEqual(st.batch_get_v1(["page"], [0]), [False])
        store.present.add(groups[1])
        self.assertEqual(st.batch_get_v1(["page"], [0]), [True])

    def test_probe_failure_is_a_miss(self):
        st, store = self._device_direct()
        st._batch_exist_flat = lambda subkeys: [False] * len(subkeys)
        self.assertEqual(st.batch_get_v1(["a", "b"], [0, 1]), [False, False])

    def test_empty_request(self):
        st, _store = self._device_direct()
        self.assertEqual(st.batch_get_v1([], []), [])


class DeviceDirectState(unittest.TestCase):
    """Which key form is in use is explicit, not "a device pool exists"."""

    def test_registered_pool_without_bypass_keeps_the_host_form(self):
        # SGLang may register the device pool without using the bypass. The
        # host path then writes bare keys; probes and reads must use them.
        st = _plugin(mem_pool_device=object())
        store = _Store(st)
        self.assertFalse(st._device_direct_active())
        bare = st._keys("page")
        self.assertEqual(st._stored_keys("kv", "page"), bare)
        store.present.update(bare)
        self.assertEqual(st.batch_exists(["page"]), 1)
        self.assertEqual(store.probes[-1], bare)
        self.assertEqual(st.batch_get_v1(["page"], [0]), [True])
        self.assertEqual(st._lib.gets, 1, "served by the host read")

    def test_host_read_failure_is_a_miss(self):
        st = _plugin(mem_pool_device=object(), _lib=_FakeLib(hit=False))
        self.assertEqual(st.batch_get_v1(["page"], [0]), [False])

    def test_requested_bypass_selects_the_device_form(self):
        st = _plugin(mem_pool_device=object(), _device_direct_requested=True)
        store = _Store(st)
        self.assertTrue(st._device_direct_active())
        group0 = [st._sg_group_key(sk, 0) for sk in st._keys("page")]
        self.assertEqual(st._stored_keys("kv", "page"), group0)
        st.batch_exists(["page"])
        self.assertEqual(store.probes[-1], group0)

    def test_a_device_transfer_selects_the_device_form(self):
        # Bypass not requested, but SGLang moves primary KV device-direct:
        # from then on the stored form is the SG one, with every group.
        st = _plugin(mem_pool_device=_MlaDevicePool(layers=6))
        store = _Store(st)
        st._batch_get_sg = lambda sks, sp, sc: (
            [1] * len(sks), [sum(c) for c in sc])
        self.assertFalse(st._device_direct_active())
        self.assertEqual(st.batch_get_v1_device(["page"], [0, 1]), [True])
        self.assertTrue(st._device_direct_active())
        groups = st._stored_keys("kv", "page")
        self.assertEqual(len(groups), 2, "6 layers in groups of 4")
        store.present.add(groups[0])
        self.assertEqual(st.batch_get_v1(["page"], [0]), [False])
        store.present.add(groups[1])
        self.assertEqual(st.batch_get_v1(["page"], [0]), [True])

    def test_no_device_form_without_a_pool_or_after_failed_registration(self):
        self.assertFalse(
            _plugin(_device_direct_requested=True)._device_direct_active())
        st = _plugin(mem_pool_device=object(), _device_direct_requested=True,
                     _device_registration_failed=True)
        self.assertFalse(st._device_direct_active())


if __name__ == "__main__":
    unittest.main()
