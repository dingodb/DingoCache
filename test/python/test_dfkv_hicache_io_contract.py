"""HiCache connector I/O contracts that must hold without SGLang or a GPU.

Pure Python: no libdfkv, server, torch or numpy. A fake native client stands
in for libdfkv and small stubs for the SGLang pools the connector is handed.
"""

import contextlib
import ctypes
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

    def numel(self):
        return 4096


class _MlaDevicePool:
    """A layer-first MLA device pool, as the test device_page_meta reads it."""

    def __init__(self, layers, page_size=2, kv_cache_dim=8,
                 dtype="torch.float8_e4m3fn"):
        self.page_size = page_size
        self.kv_cache_dim = kv_cache_dim
        self.dtype = dtype
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


class DeviceDirectGroupsBeforeTheFirstTransfer(unittest.TestCase):
    """A process that has not moved a page yet (it reads what other
    instances wrote) must still address every SG group of a page."""

    def _registered(self, pool):
        st = _plugin(_device_direct_requested=True, _registered_regions=set())
        st.register_memory = lambda base, size: True
        st.register_mem_pool_device(pool)
        return st, _Store(st)

    def test_registration_records_the_group_count(self):
        st, store = self._registered(_MlaDevicePool(layers=6))
        groups = st._stored_keys("kv", "page")
        self.assertEqual(len(groups), 2, "6 layers in groups of 4")
        store.present.add(groups[0])
        self.assertEqual(st.batch_exists(["page"]), 0)
        self.assertEqual(st.batch_get_v1(["page"], [0]), [False])
        store.present.add(groups[1])
        self.assertEqual(st.batch_exists(["page"]), 1)
        self.assertEqual(st.batch_get_v1(["page"], [0]), [True])
        self.assertEqual(store.probes[-1], groups)

    def test_unknown_count_keeps_group_zero(self):
        pool = _MlaDevicePool(layers=6)
        pool.page_size = "not a size"   # the metadata call cannot be made
        with self.assertLogs(dfkv_hicache._log, "WARNING"):
            st, _store = self._registered(pool)
        self.assertEqual(st._stored_keys("kv", "page"),
                         [st._sg_group_key(sk, 0) for sk in st._keys("page")])


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
            _metrics=metrics,
        )
        st._put_flat = lambda sks, _ptrs, _sizes: ([True] * len(sks), 0)
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


class _Metrics2(_RecordingMetrics):
    def on_set(self, **kwargs):
        self.calls.append(("set", kwargs))


class _ObjectStore:
    """A native client that stores object sizes by key: an exact-size GET of
    a stored key hits, as in libdfkv. No bytes move (pointers are fake)."""

    def __init__(self):
        self.objects = {}

    def dfkv_max_sg_segs(self, _handle):
        return 4

    @staticmethod
    def _key(karr, klens, i):
        return ctypes.string_at(karr[i], klens[i])

    def dfkv_batch_put(self, _h, karr, klens, _parr, sarr, n, out):
        for i in range(n):
            self.objects[self._key(karr, klens, i)] = sarr[i]
            out[i] = 1
        return 0

    def dfkv_batch_get(self, _h, karr, klens, _parr, sarr, n, out):
        for i in range(n):
            out[i] = int(self.objects.get(self._key(karr, klens, i)) == sarr[i])
        return 0

    def dfkv_batch_exist(self, _h, karr, klens, n, out):
        for i in range(n):
            out[i] = int(self._key(karr, klens, i) in self.objects)
        return 0


class _LaidOutHostPool(_HostPool):
    """An MHA host pool: two objects (k, v) per page of the same size in
    every layout, but bytes in an order that depends on the layout."""

    def __init__(self, layout, dtype="torch.bfloat16"):
        super().__init__(components=2, nbytes=4096)
        self.layout = layout
        self.dtype = dtype
        self.page_size = 64


class RuntimeLayoutBinding(unittest.TestCase):
    """bind_runtime_layout: objects of a pool with another byte format (same
    size, so the exact-size read accepts them) are not served as this
    pool's. Off by default: then the keys are exactly today's."""

    def _plugin(self, store, pool, bind):
        st = _plugin(is_mla=False, _lib=store, transport_mode="tcp",
                     _backup_exist_gate=False, _put_retry_recovered=0,
                     _registered_regions=set(), _bind_runtime_layout=bind)
        st.register_mem_pool_host(pool)
        return st

    def _write_then_read(self, writer_pool, reader_pool, bind):
        store = _ObjectStore()
        writer = self._plugin(store, writer_pool, bind)
        reader = self._plugin(store, reader_pool, bind)
        self.assertEqual(writer.batch_set_v1(["page"], [0]), [True])
        return reader.batch_get_v1(["page"], [0]), reader.batch_exists(["page"])

    def test_default_keys_are_unchanged(self):
        st = self._plugin(_ObjectStore(), _LaidOutHostPool("page_first"),
                          bind=False)
        self.assertEqual(st._keys("page"), [
            dfkv_hicache.pool_key("page", pool="kv", tp_size=1, tp_rank=0,
                                  component=c) for c in ("k", "v")])

    def test_without_binding_another_layout_is_served(self):
        # The hazard the switch closes: same size, other byte order, a hit.
        got = self._write_then_read(_LaidOutHostPool("page_first"),
                                    _LaidOutHostPool("page_first_direct"),
                                    bind=False)
        self.assertEqual(got, ([True], 1))

    def test_another_layout_or_dtype_misses(self):
        for reader in (_LaidOutHostPool("page_first_direct"),
                       _LaidOutHostPool("page_head"),
                       _LaidOutHostPool("page_first", dtype="torch.float16")):
            got = self._write_then_read(_LaidOutHostPool("page_first"),
                                        reader, bind=True)
            self.assertEqual(got, ([False], 0), reader.layout)

    def test_same_layout_still_hits(self):
        got = self._write_then_read(_LaidOutHostPool("page_first"),
                                    _LaidOutHostPool("page_first"), bind=True)
        self.assertEqual(got, ([True], 1))

    def test_side_pools_are_bound_by_their_own_pool(self):
        store = _ObjectStore()
        keys = []
        for layout in ("layer_first", "page_first"):
            st = self._plugin(store, _LaidOutHostPool("page_first"), bind=True)
            st._pool_component_names = {"state": ("all",)}
            st._pool_replicated = {"state": False}
            st._note_pool_layout("state", _LaidOutHostPool(layout))
            keys.append(st._pool_keys("state", "page"))
        self.assertNotEqual(keys[0], keys[1])

    def test_another_pool_class_with_the_same_strings_misses(self):
        class PackedHostPool(_LaidOutHostPool):
            """Same layout and dtype strings, another packing."""

        got = self._write_then_read(_LaidOutHostPool("page_first"),
                                    PackedHostPool("page_first"), bind=True)
        self.assertEqual(got, ([False], 0))

    def test_objects_per_page_are_part_of_the_tag(self):
        st = self._plugin(_ObjectStore(), _LaidOutHostPool("page_first"),
                          bind=True)
        self.assertNotEqual(st._pool_id("kv", 1), st._pool_id("kv", 2))

    def _device_form(self, device_pool, host_layout="page_first"):
        st = self._plugin(_ObjectStore(), _LaidOutHostPool(host_layout),
                          bind=True)
        st.is_mla = True
        st._device_direct_requested = True
        st.register_memory = lambda base, size: True
        st.register_mem_pool_device(device_pool)
        return st

    def test_device_form_keys_are_tagged_by_the_device_pool(self):
        # Equal host pools, device pools that segment a page differently:
        # the device-form objects differ, so must their keys; the host-form
        # keys stay tagged by the (equal) host pool.
        a = self._device_form(_MlaDevicePool(layers=6, kv_cache_dim=8))
        b = self._device_form(_MlaDevicePool(layers=6, kv_cache_dim=4))
        self.assertTrue(a._device_direct_active())
        self.assertNotEqual(a._stored_keys("kv", "p"), b._stored_keys("kv", "p"))
        self.assertEqual(a._keys("p"), b._keys("p"))
        self.assertNotEqual(a._keys("p", device=True), a._keys("p"))
        dtype_b = _MlaDevicePool(layers=6, kv_cache_dim=8)
        dtype_b.dtype = "torch.float8_e5m2"
        b = self._device_form(dtype_b)
        self.assertNotEqual(a._stored_keys("kv", "p"), b._stored_keys("kv", "p"))

    def test_device_form_is_shared_across_host_layouts(self):
        # The device-form objects do not depend on the host layout: no miss
        # between instances that differ only there. Host-form keys differ.
        a = self._device_form(_MlaDevicePool(layers=6), "page_first")
        b = self._device_form(_MlaDevicePool(layers=6), "page_first_direct")
        self.assertEqual(a._stored_keys("kv", "p"), b._stored_keys("kv", "p"))
        self.assertNotEqual(a._keys("p"), b._keys("p"))
        # Another device pool class, or another layer count, does not share.
        class OtherDevicePool(_MlaDevicePool):
            pass

        for other in (OtherDevicePool(layers=6), _MlaDevicePool(layers=5)):
            c = self._device_form(other)
            self.assertNotEqual(a._keys("p", device=True),
                                c._keys("p", device=True), type(other))

    def test_sidecar_draft_and_device_form_keys_carry_the_tag(self):
        st = self._device_form(_MlaDevicePool(layers=6))
        st._pool_component_names = {"indexer": ("all",)}
        st._pool_replicated = {"indexer": True}
        st._sidecar_device_pools = {"indexer": object()}
        st._note_pool_layout("indexer", _MlaDevicePool(layers=6), device=True)
        st._note_pool_layout("draft", _MlaDevicePool(layers=2), device=True)
        tagged = {
            "sidecar": st._pool_keys("indexer", "p", True),
            "sidecar stored": st._stored_keys("indexer", "p"),
            "draft": st._draft_keys("p", 1),
            "kv stored": st._stored_keys("kv", "p"),
        }
        st._bind_runtime_layout = False
        untagged = {
            "sidecar": st._pool_keys("indexer", "p", True),
            "sidecar stored": st._stored_keys("indexer", "p"),
            "draft": st._draft_keys("p", 1),
            "kv stored": st._stored_keys("kv", "p"),
        }
        for form, keys in tagged.items():
            self.assertNotEqual(keys, untagged[form], form)
            self.assertTrue(all(b"@" in k for k in keys), form)

    def test_missing_fields_are_warned_once_per_pool(self):
        class Bare:
            page_size = 64

            def get_page_buffer_meta(self, indices):
                return [0x1000] * 2, [4096] * 2

        st = _plugin(is_mla=False, _lib=_ObjectStore(), transport_mode="tcp",
                     _registered_regions=set(), _bind_runtime_layout=True)
        with self.assertLogs(dfkv_hicache._log, "WARNING") as logs:
            st.register_mem_pool_host(Bare())
            st.register_mem_pool_host(Bare())
        self.assertEqual(len(logs.output), 2, logs.output)
        self.assertIn("'layout'", logs.output[0])
        self.assertIn("'dtype'", logs.output[1])

    def test_device_pools_are_not_warned_for_layout(self):
        # Device pools have no layout choice and no `layout` attribute:
        # only a missing dtype is reported.
        st = self._plugin(_ObjectStore(), _LaidOutHostPool("page_first"),
                          bind=True)
        with self.assertNoLogs(dfkv_hicache._log, "WARNING"):
            st._note_pool_layout("indexer", _MlaDevicePool(layers=2),
                                 device=True)
        no_dtype = _MlaDevicePool(layers=2)
        del no_dtype.dtype
        with self.assertLogs(dfkv_hicache._log, "WARNING") as logs:
            st._note_pool_layout("draft", no_dtype, device=True)
        self.assertEqual(len(logs.output), 1, logs.output)
        self.assertIn("'dtype'", logs.output[0])

    def test_keys_of_an_unregistered_pool_are_refused(self):
        st = _plugin(is_mla=False, _bind_runtime_layout=True)
        with self.assertRaisesRegex(RuntimeError, "before the pool"):
            st._keys("page")

    def test_reregistering_another_format_is_refused(self):
        st = self._plugin(_ObjectStore(), _LaidOutHostPool("page_first"),
                          bind=True)
        with self.assertRaisesRegex(RuntimeError, "another layout"):
            st.register_mem_pool_host(_LaidOutHostPool("page_head"))


class DeviceReadsReturnTheirIo(unittest.TestCase):
    """The device-direct read helpers return a _V2Io, like their write
    twins, instead of a bare (results, bytes, seconds) tuple."""

    def _plugin(self):
        metrics = _Metrics2()
        metrics.on_get = lambda **kwargs: metrics.calls.append(("get", kwargs))
        st = _plugin(mem_pool_device=_MlaDevicePool(layers=6), _metrics=metrics)
        st._batch_get_sg = lambda sks, sp, sc: (
            [1] * len(sks), [sum(c) for c in sc])
        return st, metrics

    def test_kv_read_returns_its_io(self):
        st, _metrics = self._plugin()
        res, io = st._kv_device_get(["p"], [0, 1])
        self.assertEqual(res, [True])
        # 6 layers x one page of 2 tokens x 8 bytes.
        self.assertEqual((io.nbytes, io.pools, io.recovered), (96, ("kv",), 0))

    def test_v2_device_read_reports_the_kv_volume(self):
        st, metrics = self._plugin()
        self.assertEqual(st.batch_get_v2_device(["p"], [0, 1], []),
                         {"kv": [True]})
        self.assertEqual(metrics.calls[0][0], "get")
        self.assertEqual(metrics.calls[0][1]["nbytes"], 96)


class DeviceV2SetMetrics(unittest.TestCase):
    """batch_set_v2_device: the sidecar metric counts the pools that did I/O,
    with the bytes and time of the host sidecars too."""

    def _plugin(self, writer):
        metrics = _Metrics2()
        st = _plugin(
            _mla_replica_writer=writer,
            registered_pools={"state": _HostPool(nbytes=100)},
            _pool_component_names={"state": ("all",), "indexer": ("all",)},
            _pool_replicated={"state": False, "indexer": True},
            _sidecar_device_pools={"indexer": object()},
            _metrics=metrics,
        )
        st._put_flat = lambda sks, _ptrs, _sizes: ([True] * len(sks), 1)
        return st, metrics

    def _set(self, st):
        return st.batch_set_v2_device(
            ["k1", "k2"], [0, 1],
            [PoolTransfer("state", host_indices=[0, 1], keys=["k1", "k2"]),
             types.SimpleNamespace(name="indexer", keys=["k1", "k2"],
                                   host_indices=None, device_indices=[0, 1])])

    def test_follower_counts_its_rank_sharded_host_sidecar(self):
        # The latent and the replicated indexer are skipped on a follower,
        # but the rank-sharded state is written there and must be counted.
        st, metrics = self._plugin(writer=False)
        res = self._set(st)
        self.assertEqual(res["state"], [True, True])
        self.assertEqual(len(metrics.calls), 1, "the sidecar write is counted")
        self.assertEqual(metrics.calls, [("set_v2", {
            "pages": 2, "ok_pages": 2, "nbytes": 200,
            "seconds": metrics.calls[0][1]["seconds"]})])

    def test_retry_recoveries_come_back_with_the_result(self):
        st, _metrics = self._plugin(writer=False)
        results, io = st._v2_io(
            [PoolTransfer("state", host_indices=[0], keys=["k"])],
            putting=True)
        self.assertEqual(results, {"state": [True]})
        self.assertEqual(io.recovered, 1)
        self.assertNotIn("_put_retry_recovered", vars(st))


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
        for api, context in ((_uninitialized, "failed"), (None, "missing")):
            sources = {}
            with (_parallel_api(api) if api else contextlib.nullcontext()):
                got = self.resolve({}, storage_config=config, sources=sources)
            self.assertEqual(got, ((1, 0), (2, 1), None))
            self.assertEqual(
                sources, {"pcp": "storage_config", "dcp": "storage_config",
                          "context": context})

    def test_context_whose_fields_raise_is_a_failed_context(self):
        class Uninitialized:
            attn_cp_size = 4
            attn_cp_rank = 1

            @property
            def attn_dcp_size(self):
                raise AssertionError("decode CP group is not initialized")

            attn_dcp_rank = attn_dcp_size
            attn_tp_rank = 0

        with _parallel_api(Uninitialized):
            # The DCP coordinate is unknown: refuse, as for get_parallel
            # itself raising.
            with self.assertRaisesRegex(RuntimeError, "dcp coordinate"):
                self.resolve({}, storage_config=_storage_config())
            # With the coordinates given elsewhere it resolves, and nothing
            # is taken from the failed context (not even attn_tp_rank).
            sources = {}
            got = self.resolve(
                {"dcp_size": 2, "dcp_rank": 1},
                storage_config=_storage_config(attn_cp_size=2, attn_cp_rank=1),
                sources=sources)
        self.assertEqual(got, ((2, 1), (2, 1), None))
        self.assertEqual(sources, {"pcp": "storage_config",
                                   "dcp": "extra_config", "context": "failed"})

    def test_storage_config_axis_needs_both_size_and_rank(self):
        # A size without a rank is not a coordinate; never rank 0 by default.
        config = types.SimpleNamespace(
            tp_rank=1, tp_size=8, attn_cp_size=1, attn_cp_rank=0, dcp_size=2)
        with _parallel_api(_uninitialized):
            with self.assertRaisesRegex(RuntimeError, "dcp coordinate"):
                self.resolve({}, storage_config=config)

    def test_a_given_probe_is_used_and_not_repeated(self):
        def probed_again():
            raise AssertionError("probed twice")

        with _parallel_api(probed_again):
            got = self.resolve({}, probe=(None, None),
                               storage_config=_storage_config())
        self.assertEqual(got, ((1, 0), (1, 0), None))

    def test_mla_cp_from_the_storage_config_needs_an_explicit_layout(self):
        # Without the API, attn_cp_size>1 now reaches the CP layout check.
        pcp, dcp, _rank = self.resolve(
            {}, storage_config=_storage_config(attn_cp_size=2, attn_cp_rank=1))
        with self.assertRaisesRegex(Exception, "prefill_cp_storage_layout"):
            dfkv_hicache._resolve_prefill_cp_storage_layout(
                {}, is_mla=True, pcp_size=pcp[0], dcp_size=dcp[0],
                parallel=None)

    def test_api_present_but_failing_accepts_explicit_coordinates(self):
        sources = {}
        with _parallel_api(_uninitialized):
            pcp, dcp, attn_tp_rank = self.resolve(
                {"dcp_size": 2, "dcp_rank": 1},
                storage_config=_storage_config(attn_cp_size=4, attn_cp_rank=2),
                sources=sources)
        self.assertEqual((pcp, dcp, attn_tp_rank), ((4, 2), (2, 1), None))
        self.assertEqual(
            sources, {"pcp": "storage_config", "dcp": "extra_config",
                      "context": "failed"})

    def test_build_without_the_api_takes_pcp_from_the_storage_config(self):
        # The test shim has no parallel-context module at all.
        sources = {}
        pcp, dcp, attn_tp_rank = self.resolve(
            {}, storage_config=_storage_config(attn_cp_size=4, attn_cp_rank=3),
            sources=sources)
        self.assertEqual((pcp, dcp, attn_tp_rank), ((4, 3), (1, 0), None))
        self.assertEqual(sources, {"pcp": "storage_config", "dcp": "default",
                                   "context": "missing"})

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
            sources, {"pcp": "parallel_context", "dcp": "parallel_context",
                      "context": "available"})

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
