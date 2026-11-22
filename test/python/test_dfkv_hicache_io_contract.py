"""HiCache connector I/O contracts that must hold without SGLang or a GPU.

Pure Python: no libdfkv, server, torch or numpy. A fake native client stands
in for libdfkv and small stubs for the SGLang pools the connector is handed.
"""

import contextlib
import os
import sys
import threading
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "integration", "hicache"))

import dfkv_hicache  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import (  # noqa: E402
    HiCacheStorageConfig, PoolTransfer)


class _HostPool:
    """A host pool with a physical layout: one buffer per page object."""

    def __init__(self, components=1, nbytes=64):
        self.components = components
        self.nbytes = nbytes

    def get_page_buffer_meta(self, indices):
        n = len(indices) * self.components  # one index per page here
        return ([0x1000] * n, [self.nbytes] * n)


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
    st._mla_replica_writer = True
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


class _RecordingMetrics:
    """Collects the v2 metric calls with their arguments."""

    def __init__(self):
        self.calls = []

    def on_set_v2(self, **kwargs):
        self.calls.append(("set_v2", kwargs))

    def on_get_v2(self, **kwargs):
        self.calls.append(("get_v2", kwargs))


class ConcurrentV2Transfers(unittest.TestCase):
    """SGLang backs up (batch_set_v2) and prefetches (batch_get_v2) on
    different threads. Each call must report its own pools, bytes and time.

    The transfer used to hand these to its caller through instance
    attributes. A set that had finished its transfer but not yet read them,
    while a get ran to completion, read the get's pool names: its metrics took
    the get's bytes, and with different pools it failed with a KeyError.
    """

    def test_set_and_get_each_see_their_own_result(self):
        metrics = _RecordingMetrics()
        st = _plugin(
            registered_pools={"state": _HostPool(nbytes=100),
                              "index": _HostPool(nbytes=7)},
            _pool_component_names={"state": ("all",), "index": ("all",)},
            _pool_replicated={"state": False, "index": False},
            _put_retry_recovered=0,
            _metrics=metrics,
        )
        st._put_flat = lambda sks, _ptrs, _sizes: [True] * len(sks)
        outcome = {}

        def backup():
            try:
                outcome["set"] = st.batch_set_v2(
                    [PoolTransfer("state", host_indices=[0, 1],
                                  keys=["k1", "k2"])])
            except Exception as exc:  # reported below, in the test thread
                outcome["error"] = exc

        # Hold the set between the end of its transfer and its use of the
        # transfer's volume: formatting the result is the first thing it does
        # in between.
        writer = threading.Thread(target=backup)
        transferred, release = threading.Event(), threading.Event()
        fmt_pool_results = dfkv_hicache._fmt_pool_results

        def held(results):
            if threading.current_thread() is writer:
                transferred.set()
                release.wait(5)
            return fmt_pool_results(results)

        dfkv_hicache._fmt_pool_results = held
        try:
            writer.start()
            self.assertTrue(transferred.wait(5), outcome.get("error"))
            got = st.batch_get_v2(
                [PoolTransfer("index", host_indices=[0], keys=["k3"])])
            release.set()
            writer.join(5)
        finally:
            release.set()
            dfkv_hicache._fmt_pool_results = fmt_pool_results
        self.assertFalse(writer.is_alive())

        self.assertNotIn("error", outcome, repr(outcome.get("error")))
        self.assertEqual(got, {"index": [True]})
        self.assertEqual(outcome["set"], {"state": [True, True]})
        by_op = dict(metrics.calls)
        self.assertEqual(len(metrics.calls), 2)
        self.assertEqual(by_op["get_v2"]["pages"], 1)
        self.assertEqual(by_op["get_v2"]["nbytes"], 7)
        self.assertEqual(by_op["set_v2"]["pages"], 2)
        self.assertEqual(by_op["set_v2"]["ok_pages"], 2)
        self.assertEqual(by_op["set_v2"]["nbytes"], 200)

    def test_transfer_leaves_no_result_on_the_instance(self):
        st = _plugin(
            registered_pools={"state": _HostPool()},
            _pool_component_names={"state": ("all",)},
            _pool_replicated={"state": False},
        )
        before = set(vars(st))
        results, io = st._v2_io(
            [PoolTransfer("state", host_indices=[0], keys=["k"])],
            putting=False)
        self.assertEqual(results, {"state": [True]})
        self.assertEqual((io.nbytes, io.pools), (64, ("state",)))
        self.assertEqual(set(vars(st)), before)


@contextlib.contextmanager
def _parallel_api(get_parallel):
    """An SGLang build whose parallel-context API is ``get_parallel``."""
    name = "sglang.srt.runtime_context"
    previous = sys.modules.get(name)
    module = types.ModuleType(name)
    module.get_parallel = get_parallel
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _uninitialized():
    raise AssertionError("parallel state is not initialized")


def _storage_config(**kwargs):
    return HiCacheStorageConfig(tp_rank=0, tp_size=8, **kwargs)


class ParallelCoordinates(unittest.TestCase):
    """Storage coordinates when SGLang's parallel context cannot be read.

    They used to fall back to pcp=0/1 and dcp=0/1 without a word. A rank of
    a sharded deployment then stored and read under another rank's keys.
    """

    resolve = staticmethod(dfkv_hicache._resolve_parallel_coordinates)

    def test_api_present_but_failing_refuses_unknown_coordinates(self):
        with _parallel_api(_uninitialized):
            # Without the storage config, PCP is unknown as well. This used
            # to resolve to pcp=0/1 and dcp=0/1.
            with self.assertRaisesRegex(RuntimeError, "pcp coordinate"):
                self.resolve({})
            with self.assertRaisesRegex(RuntimeError, "dcp coordinate"):
                self.resolve({}, storage_config=_storage_config())

    def test_dcp_carried_by_the_storage_config_is_used(self):
        # Current SGLang passes dcp_size/dcp_rank in HiCacheStorageConfig.
        config = types.SimpleNamespace(
            tp_rank=1, tp_size=8, attn_cp_size=1, attn_cp_rank=0,
            dcp_size=2, dcp_rank=1)
        for api in (_uninitialized, None):
            sources = {}
            with (_parallel_api(api) if api else contextlib.nullcontext()):
                got = self.resolve({}, storage_config=config, sources=sources)
            self.assertEqual(got, ((1, 0), (2, 1), None))
            self.assertEqual(
                sources, {"pcp": "storage_config", "dcp": "storage_config"})

    def test_api_present_but_failing_accepts_explicit_coordinates(self):
        sources = {}
        with _parallel_api(_uninitialized):
            pcp, dcp, attn_tp_rank = self.resolve(
                {"dcp_size": 2, "dcp_rank": 1},
                storage_config=_storage_config(attn_cp_size=4, attn_cp_rank=2),
                sources=sources)
        self.assertEqual((pcp, dcp, attn_tp_rank), ((4, 2), (2, 1), None))
        self.assertEqual(
            sources, {"pcp": "storage_config", "dcp": "extra_config"})

    def test_build_without_the_api_takes_pcp_from_the_storage_config(self):
        # The test shim has no parallel-context module at all.
        sources = {}
        pcp, dcp, attn_tp_rank = self.resolve(
            {}, storage_config=_storage_config(attn_cp_size=4, attn_cp_rank=3),
            sources=sources)
        self.assertEqual((pcp, dcp, attn_tp_rank), ((4, 3), (1, 0), None))
        self.assertEqual(sources, {"pcp": "storage_config", "dcp": "default"})

    def test_build_without_the_api_and_no_cp_is_unsharded(self):
        sources = {}
        self.assertEqual(
            self.resolve({}, storage_config=_storage_config(),
                         sources=sources),
            ((1, 0), (1, 0), None))
        self.assertEqual(self.resolve({}), ((1, 0), (1, 0), None))

    def test_invalid_storage_config_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            self.resolve(
                {}, storage_config=_storage_config(attn_cp_size=4,
                                                   attn_cp_rank=4))

    def test_available_context_is_still_the_source(self):
        context = types.SimpleNamespace(
            attn_cp_size=8, attn_cp_rank=3, attn_dcp_size=2, attn_dcp_rank=1,
            attn_tp_rank=0)
        sources = {}
        with _parallel_api(lambda: context):
            got = self.resolve(
                {}, storage_config=_storage_config(), sources=sources)
        self.assertEqual(got, ((8, 3), (2, 1), 0))
        self.assertEqual(
            sources, {"pcp": "parallel_context", "dcp": "parallel_context"})

    def test_probe_tells_a_missing_api_from_a_failing_one(self):
        self.assertEqual(dfkv_hicache._probe_parallel_context(), (None, None))
        with _parallel_api(_uninitialized):
            context, failure = dfkv_hicache._probe_parallel_context()
        self.assertIsNone(context)
        self.assertIsInstance(failure, AssertionError)

    def test_fallback_is_counted(self):
        metrics = dfkv_hicache._Metrics(0)
        metrics.on_parallel_context_fallback()
        self.assertEqual(metrics._c["parallel_context_fallbacks"], 1)


class _DcpPool:
    def __init__(self, dcp_size):
        self.page_size = 4
        self.dcp_size = dcp_size


class PoolDcpCrossCheck(unittest.TestCase):
    """A pool laid out for decode CP contradicts coordinates without it."""

    def test_pool_with_another_dcp_width_is_refused(self):
        st = _plugin(dcp_size=1, _coordinate_sources={"dcp": "default"})
        with self.assertRaisesRegex(RuntimeError, "dcp_size=2"):
            st.register_mem_host_pool_v2(_DcpPool(2), "state")
        st.transport_mode = "tcp"
        with self.assertRaisesRegex(RuntimeError, "dcp_size=2"):
            st.register_mem_pool_host(_DcpPool(2))

    def test_matching_or_absent_width_passes(self):
        st = _plugin(dcp_size=2, dcp_rank=1)
        st._check_pool_dcp(_DcpPool(2), "pool")
        st._check_pool_dcp(object(), "pool")   # e.g. a state pool, page 1
        st._check_pool_dcp(_DcpPool(1), "pool")


if __name__ == "__main__":
    unittest.main()
