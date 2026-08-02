"""GPU/torch-free unit tests for the HiCache L2-bypass (device-direct write) path
in dfkv_hicache.py.

Covers the pieces that do NOT need a live cache node or libdfkv: sub-key
expansion of per-layer device segments (_flatten_device), the exist-gate +
transient-retry semantics of the scatter-gather put (_put_sg_flat, mirroring the
contiguous _put_flat), and the supports_device_transfer() capability gate
(SG symbols present AND max_sg_segs >= layer_num). Uses object.__new__ +
monkeypatched primitives so no client is opened.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
# shim 'sglang' (no torch) first, then the real plugin source dir.
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "integration", "hicache"))

os.environ.setdefault("DFKV_CLIENT_NODE_DEDUP", "0")

import ctypes  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # noqa: E402
import dfkv_hicache  # noqa: E402

BUILD = os.environ.get("DFKV_BUILD", os.path.join(HERE, "..", "..", "build"))
SERVER_BIN = os.path.join(BUILD, "dfkv_server")


def _bare(**attrs):
    """A DfkvHiCache with only the attributes a unit needs (no client opened)."""
    obj = object.__new__(dfkv_hicache.DfkvHiCache)
    obj.model = "m"
    obj.tp_rank = 0
    obj.tp_size = 1
    obj.is_mla = True
    obj.pp_rank = 0
    obj.pp_size = 1
    obj.enable_pp = False
    obj.pcp_size = 1
    obj.pcp_rank = 0
    obj.dcp_size = 1
    obj.dcp_rank = 0
    obj._device_direct_requested = False
    obj._device_registration_failed = False
    obj._registered_regions = set()
    obj._pool_component_names = {
        "indexer": ("all",),
        "draft_indexer": ("all",),
    }
    obj._pool_replicated = {
        "indexer": True,
        "draft_indexer": True,
    }
    obj._put_retry_recovered = 0
    obj.transport_mode = "rdma"
    obj._backup_exist_gate = True
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


class FakeLib:
    def __init__(self, has_sg=True, has_get_sg=True, has_width=True, max_segs=32):
        self._max_segs = max_segs
        if has_sg:
            self.dfkv_batch_put_sg = lambda *a: 0
        if has_get_sg:
            self.dfkv_batch_get_auto_sg = lambda *a: 0
        if has_width:
            self.dfkv_max_sg_segs = lambda h: self._max_segs


class TestFlattenDevice(unittest.TestCase):
    def test_mla_single_subkey_carries_layer_segments(self):
        obj = _bare(is_mla=True)
        keys = ["h0", "h1"]
        # sub=1; per-page one entry, each a 3-layer segment list.
        seg_ptrs = [[10, 11, 12], [20, 21, 22]]
        seg_sizes = [[100, 100, 100], [100, 100, 100]]
        sub, sks, sp, ss = obj._flatten_device(keys, seg_ptrs, seg_sizes)
        self.assertEqual(sub, 1)
        self.assertEqual(
            sks,
            [obj._sg_group_key(obj._keys("h0")[0], 0),
             obj._sg_group_key(obj._keys("h1")[0], 0)],
        )
        self.assertEqual(sp, [[10, 11, 12], [20, 21, 22]])
        self.assertEqual(ss, [[100, 100, 100], [100, 100, 100]])

    def test_mha_expands_k_and_v_subkeys(self):
        obj = _bare(is_mla=False, tp_size=2, tp_rank=1)
        keys = ["h0"]
        seg_ptrs = [[1, 2], [3, 4]]  # entry0=k, entry1=v (sub=2)
        seg_sizes = [[8, 8], [8, 8]]
        sub, sks, sp, ss = obj._flatten_device(keys, seg_ptrs, seg_sizes)
        self.assertEqual(sub, 2)
        self.assertEqual(
            sks,
            [obj._sg_group_key(obj._keys("h0")[0], 0),
             obj._sg_group_key(obj._keys("h0")[1], 0)],
        )
        self.assertEqual(sp, [[1, 2], [3, 4]])

    def test_chunk_split_when_width_below_layers(self):
        # 5 layers on a width-2 lib -> 3 chunks per sub-object, consecutive
        # slices, deterministic order (the read side derives the same split).
        obj = _bare(is_mla=True, _lib=FakeLib(max_segs=2), _h=1)
        seg_ptrs = [[10, 11, 12, 13, 14]]
        seg_sizes = [[1, 2, 3, 4, 5]]
        sub, sks, sp, ss = obj._flatten_device(["h0"], seg_ptrs, seg_sizes)
        self.assertEqual(sub, 3)
        self.assertEqual(
            sks,
            [obj._sg_group_key(obj._keys("h0")[0], i) for i in range(3)],
        )
        self.assertEqual(sp, [[10, 11], [12, 13], [14]])
        self.assertEqual(ss, [[1, 2], [3, 4], [5]])

    def test_shape_mismatch_asserts(self):
        obj = _bare(is_mla=False)  # sub=2
        with self.assertRaises(AssertionError):
            obj._flatten_device(["h0"], [[1, 2]], [[8, 8]])  # only 1 entry, needs 2


class TestPutSgFlat(unittest.TestCase):
    def test_exist_gate_skips_present_subkeys(self):
        obj = _bare()
        obj._batch_exist_flat = lambda sks: [True, True]  # both already in L3
        calls = []
        obj._batch_put_sg = lambda sks, sp, ss: calls.append(sks) or [True] * len(sks)
        res = obj._put_sg_flat(["a", "b"], [[1]], [[8]])
        self.assertEqual(res, [True, True])
        self.assertEqual(calls, [])  # nothing written — gate short-circuited

    def test_transient_failure_recovered_on_retry(self):
        obj = _bare()
        obj._batch_exist_flat = lambda sks: [False]  # not present -> must write
        seq = [[False], [True]]  # first put fails, retry succeeds
        obj._batch_put_sg = lambda sks, sp, ss: seq.pop(0)
        res = obj._put_sg_flat(["a"], [[1]], [[8]])
        self.assertEqual(res, [True])
        self.assertEqual(obj._put_retry_recovered, 1)

    def test_persistent_failure_stays_false(self):
        obj = _bare()
        obj._batch_exist_flat = lambda sks: [False]
        obj._batch_put_sg = lambda sks, sp, ss: [False]  # never succeeds
        res = obj._put_sg_flat(["a"], [[1]], [[8]])
        self.assertEqual(res, [False])
        self.assertEqual(obj._put_retry_recovered, 0)

    def test_gate_disabled_writes_all(self):
        obj = _bare()
        obj._backup_exist_gate = False
        obj._batch_exist_flat = lambda sks: (_ for _ in ()).throw(
            AssertionError("exist must not be called when gate disabled"))
        obj._batch_put_sg = lambda sks, sp, ss: [True] * len(sks)
        self.assertEqual(obj._put_sg_flat(["a", "b"], [[1], [2]], [[8], [8]]),
                         [True, True])


class TestSupportsDeviceTransfer(unittest.TestCase):
    def test_enabled_when_symbols_present_and_width_ok(self):
        obj = _bare(cfg={"layer_num": 32}, _lib=FakeLib(max_segs=32), _h=1)
        self.assertTrue(obj.supports_device_transfer())

    def test_declined_on_tcp_even_with_sg_symbols(self):
        obj = _bare(
            cfg={"layer_num": 32},
            _lib=FakeLib(max_segs=32),
            _h=1,
            transport_mode="tcp",
        )
        self.assertFalse(obj.supports_device_transfer())

    def test_chunked_when_width_too_small(self):
        # Narrow HCA no longer declines: pages chunk into @sg{n} sub-keys.
        obj = _bare(cfg={"layer_num": 61}, _lib=FakeLib(max_segs=29), _h=1)
        self.assertTrue(obj.supports_device_transfer())

    def test_declined_when_width_zero(self):
        obj = _bare(cfg={"layer_num": 61}, _lib=FakeLib(max_segs=0), _h=1)
        self.assertFalse(obj.supports_device_transfer())

    def test_declined_without_sg_symbol(self):
        obj = _bare(cfg={"layer_num": 4}, _lib=FakeLib(has_sg=False), _h=1)
        self.assertFalse(obj.supports_device_transfer())

    def test_declined_without_get_sg_symbol(self):
        # Read-side symbol missing: a page could be written but never read back,
        # so the bypass path is declined (increment 2 requires the SG GET twin).
        obj = _bare(cfg={"layer_num": 4}, _lib=FakeLib(has_get_sg=False), _h=1)
        self.assertFalse(obj.supports_device_transfer())

    def test_declined_without_layer_num(self):
        obj = _bare(cfg={}, _lib=FakeLib(), _h=1)
        self.assertFalse(obj.supports_device_transfer())

    def test_required_mr_failure_raises_and_revokes_capability(self):
        obj = _bare(
            cfg={"layer_num": 4},
            _lib=FakeLib(max_segs=4),
            _h=1,
            _device_direct_requested=True,
        )
        obj.register_memory = lambda _base, _size: False
        self.assertTrue(obj.supports_device_transfer())
        with self.assertRaisesRegex(
                RuntimeError, "required primary GPU MR registration failed"):
            obj._register_device_regions(lambda: [(0x1000, 4096)], "primary")
        self.assertTrue(obj._device_registration_failed)
        self.assertFalse(obj.supports_device_transfer())

    def test_optional_registration_failure_stays_best_effort(self):
        obj = _bare(
            cfg={"layer_num": 4},
            _lib=FakeLib(max_segs=4),
            _h=1,
        )
        obj.register_memory = lambda _base, _size: False
        self.assertEqual(
            obj._register_device_regions(lambda: [(0x1000, 4096)], "optional"),
            0,
        )
        self.assertFalse(obj._device_registration_failed)
        self.assertTrue(obj.supports_device_transfer())


class TestBatchGetV1DeviceFold(unittest.TestCase):
    """Pure-python fold semantics of the device-direct READ (no client/GPU): a
    page is a hit only if every sub-object hit AND returned its full capacity;
    a short read (truncated blob) is a per-page failure so the caller recomputes."""

    def _meta_pool(self):
        # Minimal MLA layer-first pool (1 page @ slot0, 2 layers, dim 4).
        return FakeLayerFirstMlaDevicePool(
            layer_num=2, size=8, page_size=4, kv_cache_dim=4)

    def _obj(self, pool):
        return _bare(is_mla=True, mem_pool_device=pool, _alog_tag="r0",
                     _metrics=dfkv_hicache._Metrics(0))

    def test_full_hit_is_page_success(self):
        obj = self._obj(self._meta_pool())
        # each MLA page sub-object wants page_size*kv_dim per layer * 2 layers.
        want = 4 * 4 * 2
        obj._batch_get_sg = lambda sks, sp, sc: ([1], [want])
        res = obj.batch_get_v1_device(["h0"], list(range(4)))
        self.assertEqual(res, [True])

    def test_miss_is_page_failure(self):
        obj = self._obj(self._meta_pool())
        obj._batch_get_sg = lambda sks, sp, sc: ([0], [0])
        self.assertEqual(obj.batch_get_v1_device(["h0"], list(range(4))), [False])

    def test_short_read_is_page_failure(self):
        obj = self._obj(self._meta_pool())
        want = 4 * 4 * 2
        # hit==1 but only half the bytes came back -> corrupt page -> failure.
        obj._batch_get_sg = lambda sks, sp, sc: ([1], [want // 2])
        self.assertEqual(obj.batch_get_v1_device(["h0"], list(range(4))), [False])


class _NpTensor:
    """numpy-backed stand-in for a torch device tensor (data_ptr/numel/elt size)."""

    def __init__(self, arr):
        self._arr = arr

    def data_ptr(self):
        return self._arr.ctypes.data

    def numel(self):
        return int(self._arr.size)

    def element_size(self):
        return int(self._arr.itemsize)


class _Transfer:
    """Minimal PoolTransfer stand-in for the HOST v2 sidecar path
    (name/keys/host_indices; device_indices None)."""

    def __init__(self, name, keys, host_indices):
        self.name = name
        self.keys = keys
        self.host_indices = host_indices
        self.device_indices = None


class _DeviceTransfer:
    """Minimal PoolTransfer stand-in for the DEVICE-direct sidecar path (task 4):
    device_indices set (the main-KV slots the indexer rides), host_indices None."""

    def __init__(self, name, keys, device_indices):
        self.name = name
        self.keys = keys
        self.device_indices = device_indices
        self.host_indices = None


class _NpTensor2D:
    """numpy-backed 2D stand-in for a torch device tensor, exposing the shape/stride
    surface get_device_sidecar_page_buffer_meta needs (plus data_ptr/numel/elt)."""

    def __init__(self, arr):
        self._arr = arr

    def data_ptr(self):
        return self._arr.ctypes.data

    def numel(self):
        return int(self._arr.size)

    def element_size(self):
        return int(self._arr.itemsize)

    @property
    def shape(self):
        return self._arr.shape

    def stride(self, dim=None):
        strides = tuple(s // self._arr.itemsize for s in self._arr.strides)
        return strides if dim is None else strides[dim]


class FakeLayerFirstIndexerDevicePool:
    """Layer-first, PAGE-indexed DSA indexer GPU pool emulated with host numpy: a
    list of per-layer 2D buffers (num_pages, page_bytes), so the device-direct sidecar
    SG path (task 4) can be exercised end-to-end without a GPU. Row (slot // page_size)
    of layer L holds the whole page's index+scale for that layer."""

    def __init__(self, layer_num, num_pages, page_size, page_bytes):
        self.page_size = page_size
        self.page_bytes = page_bytes
        self._np = []
        self.index_k_with_scale_buffer = []
        for L in range(layer_num):
            arr = np.zeros((num_pages, page_bytes), dtype=np.uint8)
            for pg in range(num_pages):
                for b in range(page_bytes):
                    arr[pg, b] = (L * 90 + pg * 13 + b) % 256
            self._np.append(arr)
            self.index_k_with_scale_buffer.append(_NpTensor2D(arr))

    def layer_page_bytes(self, layer, page_idx):
        return bytes(self._np[layer][page_idx].tobytes())


class FakePageFirstHostSidecarPool:
    """Page-first host pool for the DSA indexer sidecar, emulated with a single
    numpy buffer of shape (num_pages, page_bytes). Mirrors the real host pool
    surface the v2 path uses: get_page_buffer_meta(host_indices) -> (ptrs, sizes),
    one contiguous (ptr, size) per page. Unlike the layer-first device pool, the
    sidecar rides the *host* v2 path (D2H already done), so its page is one
    contiguous blob — exactly what the stock host read reconstructs."""

    def __init__(self, num_pages, page_size, page_bytes, fill_base=0):
        self.page_size = page_size
        self.page_bytes = page_bytes
        self._arr = np.zeros(num_pages * page_bytes, dtype=np.uint8)
        for pg in range(num_pages):
            for b in range(page_bytes):
                self._arr[pg * page_bytes + b] = (fill_base + pg * 7 + b) % 256

    def get_page_buffer_meta(self, indices):
        # indices are page-aligned token slots; one (ptr, size) per page (sub=1).
        idx = list(indices)
        assert len(idx) % self.page_size == 0
        npages = len(idx) // self.page_size
        base = self._arr.ctypes.data
        ptrs, sizes = [], []
        for p in range(npages):
            page_idx = idx[p * self.page_size] // self.page_size
            ptrs.append(base + page_idx * self.page_bytes)
            sizes.append(self.page_bytes)
        return ptrs, sizes

    def page_bytes_at(self, page_idx):
        off = page_idx * self.page_bytes
        return bytes(self._arr[off:off + self.page_bytes])


class FakeLayerFirstMlaDevicePool:
    """Layer-first MLA GPU pool emulated with host numpy per-layer buffers, so the
    device-direct SG write path can be exercised end-to-end without a GPU."""

    def __init__(self, layer_num, size, page_size, kv_cache_dim):
        self.page_size = page_size
        self.kv_cache_dim = kv_cache_dim
        self.use_dsa = False
        slots = size + page_size
        self._np = []
        self.kv_buffer = []
        for L in range(layer_num):
            arr = np.zeros(slots * kv_cache_dim, dtype=np.uint8)
            # Recognizable per-(layer, slot, dim) byte pattern for ordering checks.
            for s in range(slots):
                for d in range(kv_cache_dim):
                    arr[s * kv_cache_dim + d] = (L * 100 + s * 10 + d) % 256
            self._np.append(arr)
            self.kv_buffer.append(_NpTensor(arr))

    def layer_page_bytes(self, layer, slot0, npages_slots):
        off = slot0 * self.kv_cache_dim
        return bytes(self._np[layer][off:off + npages_slots * self.kv_cache_dim])


class FakeDsaDraftDevicePool(FakeLayerFirstMlaDevicePool):
    """A DSA draft GPU pool (GLM-5.2 MTP draft = DSATokenToKVPool): the main MLA
    latent (kv_buffer, token-indexed) AND the DSA indexer sidecar
    (index_k_with_scale_buffer, layer-first page-indexed), sharing the page slots.
    Exercises the device-direct draft latent + indexer path off-GPU."""

    def __init__(self, layer_num, size, page_size, kv_cache_dim, num_pages, side_bytes):
        super().__init__(layer_num, size, page_size, kv_cache_dim)
        self.use_dsa = True
        self._side_np = []
        self.index_k_with_scale_buffer = []
        for L in range(layer_num):
            arr = np.zeros((num_pages, side_bytes), dtype=np.uint8)
            for pg in range(num_pages):
                for b in range(side_bytes):
                    arr[pg, b] = (L * 77 + pg * 19 + b) % 256
            self._side_np.append(arr)
            self.index_k_with_scale_buffer.append(_NpTensor2D(arr))

    def side_layer_page_bytes(self, layer, page_idx):
        return bytes(self._side_np[layer][page_idx].tobytes())


@unittest.skipUnless(
    os.path.exists(SERVER_BIN)
    and hasattr(dfkv_hicache._load_lib(os.environ.get("DFKV_LIB")), "dfkv_batch_put_sg"),
    "needs dfkv_server + a libdfkv.so with dfkv_batch_put_sg",
)
class TestDeviceDirectEndToEnd(unittest.TestCase):
    """Drives batch_set_v1_device against a real cache node and reads the stored
    blob back, proving (a) the write path works and (b) the stored blob is
    LAYER-major — the transpose of a page-first host read (the load-bearing
    correctness limitation for increment 1)."""

    LAYER_NUM = 3
    PAGE_SIZE = 4
    KV_DIM = 5  # bytes/token/layer (dtype itemsize 1)

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
            except Exception:
                p.kill()
        for d in cls.dirs:
            shutil.rmtree(d, ignore_errors=True)

    def _node(self, tag):
        d = tempfile.mkdtemp(prefix=f"dfkv_dd_{tag}_")
        p = subprocess.Popen(
            [SERVER_BIN, "--dir", d, "--port", "0", "--cap", str(1 << 30)],
            stdout=subprocess.PIPE, text=True)
        line = p.stdout.readline().strip()
        assert line.startswith("PORT "), f"bad server greeting: {line!r}"
        self.procs.append(p)
        self.dirs.append(d)
        return f"{tag}=127.0.0.1:{int(line.split()[1])}"

    def _plugin(self, members):
        cfg = HiCacheStorageConfig(
            tp_rank=0, tp_size=1, is_mla_model=True, is_page_first_layout=False,
            model_name="dd-mla", pp_rank=0, pp_size=1,
            extra_config={
                "members": members, "dtype_tag": 0,
                "page_size": self.PAGE_SIZE, "layer_num": self.LAYER_NUM,
                "head_num": 1, "head_dim": self.KV_DIM, "interface_v1": 1,
            })
        st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        if not st.supports_device_transfer():
            self.skipTest(
                f"device-direct requires RDMA; transport={st.transport_mode}")
        return st

    def test_sg_write_stores_layer_major_blob(self):
        st = self._plugin(self._node("dd"))
        pool = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        st.register_mem_pool_device(pool)
        self.assertTrue(st.supports_device_transfer())

        page_hash = "deadbeef"
        device_indices = list(range(0, self.PAGE_SIZE))  # one page at slot 0
        res = st.batch_set_v1_device([page_hash], device_indices)
        self.assertEqual(res, [True])

        # Read the stored blob back contiguously using the exact binary SG key.
        sk = st._sg_group_key(st._keys(page_hash)[0], 0)
        cap = self.LAYER_NUM * self.PAGE_SIZE * self.KV_DIM
        buf = ctypes.create_string_buffer(cap)
        key_ptr, key_owner = dfkv_hicache.make_key_buffer(sk)
        rc = st._lib.dfkv_get(
            st._h, key_ptr, ctypes.c_uint64(len(sk)),
            ctypes.cast(buf, ctypes.c_void_p), ctypes.c_uint64(cap))
        del key_owner
        self.assertEqual(rc, 1, "device-direct page must be retrievable")
        blob = bytes(buf.raw[:cap])

        # The SG-stored blob is the in-order concat of per-layer segments:
        # layer0(page), layer1(page), layer2(page) — LAYER-major.
        layer_major = b"".join(
            pool.layer_page_bytes(L, 0, self.PAGE_SIZE)
            for L in range(self.LAYER_NUM))
        self.assertEqual(blob, layer_major)

        # A page-first host pool would lay the same page out TOKEN-major
        # (tok0[L0..L2], tok1[...], ...). It differs — so an unchanged page-first
        # host read of this device-written page would transpose the bytes. This
        # is the documented increment-1 limitation.
        token_major = bytearray()
        for tok in range(self.PAGE_SIZE):
            for L in range(self.LAYER_NUM):
                base = tok * self.KV_DIM
                token_major += pool._np[L][base:base + self.KV_DIM].tobytes()
        self.assertNotEqual(blob, bytes(token_major),
                            "layer-major vs page-first must differ for layer_num>1")

    def test_device_direct_write_then_read_roundtrip(self):
        """Increment 2: write a page device-direct, then read it back device-direct
        into a FRESH destination pool via SG GET, and assert every per-layer page
        segment is byte-identical to the source. This proves the layer-major SG
        write + layer-major SG-GET scatter reassemble a byte-consistent device page
        (the pairing the increment-1 transpose note requires)."""
        st = self._plugin(self._node("rt"))
        src = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        st.register_mem_pool_device(src)
        self.assertTrue(st.supports_device_transfer())

        page_hash = "cafef00d"
        device_indices = list(range(0, self.PAGE_SIZE))  # one page at slot 0
        self.assertEqual(st.batch_set_v1_device([page_hash], device_indices), [True])

        # Fresh, zeroed destination pool. Re-point the plugin's device pool to it
        # (register_mem_pool_device sets self.mem_pool_device), so the SG GET
        # scatters the stored blob back into the destination's per-layer segments.
        dst = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        for L in range(self.LAYER_NUM):
            dst._np[L][:] = 0
        st.register_mem_pool_device(dst)

        res = st.batch_get_v1_device([page_hash], device_indices)
        self.assertEqual(res, [True], "device-direct page must read back")

        # Every layer's page bytes in the destination must match the source: the
        # write stored layer0(page)|layer1(page)|... and the SG GET scattered that
        # same order back into dst's per-layer segments -> byte-consistent.
        for L in range(self.LAYER_NUM):
            self.assertEqual(
                dst.layer_page_bytes(L, 0, self.PAGE_SIZE),
                src.layer_page_bytes(L, 0, self.PAGE_SIZE),
                f"layer {L} page mismatch after device-direct roundtrip")

    def test_device_direct_read_miss_returns_false(self):
        """A key never written must read back as a miss (not a phantom hit) so the
        scheduler recomputes instead of serving garbage device slots."""
        st = self._plugin(self._node("miss"))
        dst = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        st.register_mem_pool_device(dst)
        self.assertEqual(
            st.batch_get_v1_device(["never_written"], list(range(self.PAGE_SIZE))),
            [False])

    # Increment 2.5: DSA split-value (main KV device-direct + indexer sidecar host).
    SIDE_BYTES = 4 * 3  # indexer page bytes (page_size * indexer_size_per_token-ish)

    def test_dsa_split_value_write_then_read_roundtrip(self):
        """DSA (GLM-5.2) L2-bypass: the main MLA latent rides the device-direct SG
        path (layer-major, straight from GPU slots) while the small DSA indexer
        sidecar rides the host v2 path. batch_set_v2_device writes the split value
        under two key namespaces (kv @sg keys + indexer host keys); batch_get_v2_device
        reads BOTH back byte-identical into fresh destination pools. Proves the
        composite is split honestly without any C-server change: main KV and sidecar
        never collide, and each component round-trips byte-exact."""
        st = self._plugin(self._node("dsa"))
        # Source pools: main latent (layer-first device) + indexer sidecar (page-first host).
        src_kv = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        src_side = FakePageFirstHostSidecarPool(
            num_pages=4, page_size=self.PAGE_SIZE, page_bytes=self.SIDE_BYTES,
            fill_base=17)
        st.register_mem_pool_device(src_kv)
        st.register_mem_host_pool_v2(src_side, "indexer")
        self.assertTrue(st.supports_device_transfer())

        page_hash = "d5a10001"
        kv_device_indices = list(range(0, self.PAGE_SIZE))  # one page at slot 0
        side_host_indices = list(range(0, self.PAGE_SIZE))  # indexer shares slot 0
        set_res = st.batch_set_v2_device(
            [page_hash], kv_device_indices,
            [_Transfer("indexer", [page_hash], side_host_indices)])
        self.assertEqual(set_res["kv"], [True], "main KV device-direct write failed")
        self.assertEqual(set_res["indexer"], [True], "indexer host write failed")

        # Fresh, zeroed destination pools; re-point the plugin at them for the read.
        dst_kv = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        for L in range(self.LAYER_NUM):
            dst_kv._np[L][:] = 0
        dst_side = FakePageFirstHostSidecarPool(
            num_pages=4, page_size=self.PAGE_SIZE, page_bytes=self.SIDE_BYTES,
            fill_base=0)
        dst_side._arr[:] = 0
        st.register_mem_pool_device(dst_kv)
        st.register_mem_host_pool_v2(dst_side, "indexer")

        get_res = st.batch_get_v2_device(
            [page_hash], kv_device_indices,
            [_Transfer("indexer", [page_hash], side_host_indices)])
        self.assertEqual(get_res["kv"], [True], "main KV device-direct read failed")
        self.assertEqual(get_res["indexer"], [True], "indexer host read failed")

        # Main KV: every layer's page must match source (layer-major roundtrip).
        for L in range(self.LAYER_NUM):
            self.assertEqual(
                dst_kv.layer_page_bytes(L, 0, self.PAGE_SIZE),
                src_kv.layer_page_bytes(L, 0, self.PAGE_SIZE),
                f"main KV layer {L} mismatch after DSA split roundtrip")
        # Sidecar: the indexer page must match source (host v2 roundtrip).
        self.assertEqual(
            dst_side.page_bytes_at(0), src_side.page_bytes_at(0),
            "indexer sidecar page mismatch after DSA split roundtrip")

    def test_dsa_split_value_kv_and_sidecar_use_distinct_keys(self):
        """The split value's two components must live under DISTINCT key namespaces
        so they never overwrite each other: main KV under the @sg-chunked v1-style
        'kv' keys, the indexer under its own '_indexer_k' key. A cross-namespace
        collision would corrupt one component."""
        st = _bare(is_mla=True, tp_size=1, tp_rank=0, pp_rank=0, pp_size=1,
                   enable_pp=False, model="m")
        h = "feedface"
        kv_sub = st._sg_group_key(st._keys(h)[0], 0)
        side_sub = st._pool_keys("indexer", h)[0]  # host v2 indexer sub-key
        self.assertNotEqual(kv_sub, side_sub)
        self.assertNotIn(b"indexer", kv_sub)
        self.assertIn(b"indexer", side_sub)
        # Task 4: the device-direct sidecar stores an "@sg0"-chunked indexer key —
        # still its own namespace, distinct from the main KV.
        side_sub_dev = st._sg_group_key(st._pool_keys("indexer", h)[0], 0)
        self.assertNotEqual(kv_sub, side_sub_dev)
        self.assertIn(b"indexer", side_sub_dev)

    # Task 4: DSA split-value with the indexer sidecar ALSO device-direct.
    SIDE_DEV_BYTES = 6  # indexer page-row bytes (page_size * indexer bytes/token-ish)

    def test_dsa_split_value_device_sidecar_roundtrip(self):
        """DSA (GLM-5.2) L2-bypass, task 4: BOTH the main MLA latent AND the DSA
        indexer sidecar ride the device-direct SG path — no host staging. The indexer
        is layer-first & PAGE-indexed, so it @sg-chunks exactly like the main latent.
        batch_set_v2_device writes both from GPU slots; batch_get_v2_device reads both
        back into FRESH device pools, byte-identical per layer. Proves the sidecar's
        layer-major device write/read reassemble byte-exact under distinct keys."""
        st = self._plugin(self._node("dsadev"))
        src_kv = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        src_side = FakeLayerFirstIndexerDevicePool(
            self.LAYER_NUM, num_pages=4, page_size=self.PAGE_SIZE,
            page_bytes=self.SIDE_DEV_BYTES)
        st.register_mem_pool_device(src_kv)
        st.register_mem_pool_device_sidecar("indexer", src_side)
        self.assertTrue(st.supports_device_transfer())

        page_hash = "d5adev01"
        kv_dev = list(range(0, self.PAGE_SIZE))  # one page at slot 0
        set_res = st.batch_set_v2_device(
            [page_hash], kv_dev,
            [_DeviceTransfer("indexer", [page_hash], kv_dev)])
        self.assertEqual(set_res["kv"], [True], "main KV device-direct write failed")
        self.assertEqual(set_res["indexer"], [True],
                         "indexer device-direct write failed")

        # Fresh, zeroed destination pools; re-point the plugin at them for the read.
        dst_kv = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        for L in range(self.LAYER_NUM):
            dst_kv._np[L][:] = 0
        dst_side = FakeLayerFirstIndexerDevicePool(
            self.LAYER_NUM, num_pages=4, page_size=self.PAGE_SIZE,
            page_bytes=self.SIDE_DEV_BYTES)
        for L in range(self.LAYER_NUM):
            dst_side._np[L][:] = 0
        st.register_mem_pool_device(dst_kv)
        st.register_mem_pool_device_sidecar("indexer", dst_side)

        get_res = st.batch_get_v2_device(
            [page_hash], kv_dev,
            [_DeviceTransfer("indexer", [page_hash], kv_dev)])
        self.assertEqual(get_res["kv"], [True], "main KV device-direct read failed")
        self.assertEqual(get_res["indexer"], [True],
                         "indexer device-direct read failed")

        for L in range(self.LAYER_NUM):
            self.assertEqual(
                dst_kv.layer_page_bytes(L, 0, self.PAGE_SIZE),
                src_kv.layer_page_bytes(L, 0, self.PAGE_SIZE),
                f"main KV layer {L} mismatch after device-sidecar roundtrip")
        # Indexer page-row 0 (slot 0 // page_size) must match source per layer.
        for L in range(self.LAYER_NUM):
            self.assertEqual(
                dst_side.layer_page_bytes(L, 0),
                src_side.layer_page_bytes(L, 0),
                f"indexer layer {L} mismatch after device-sidecar roundtrip")

    def test_device_sidecar_read_miss_returns_false(self):
        """A device-direct sidecar page never written must read back as a miss so the
        controller recomputes rather than serving garbage GPU index slots."""
        st = self._plugin(self._node("dsadevmiss"))
        dst_kv = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        dst_side = FakeLayerFirstIndexerDevicePool(
            self.LAYER_NUM, num_pages=4, page_size=self.PAGE_SIZE,
            page_bytes=self.SIDE_DEV_BYTES)
        st.register_mem_pool_device(dst_kv)
        st.register_mem_pool_device_sidecar("indexer", dst_side)
        kv_dev = list(range(0, self.PAGE_SIZE))
        res = st.batch_get_v2_device(
            ["never_dev"], kv_dev, [_DeviceTransfer("indexer", ["never_dev"], kv_dev)])
        self.assertEqual(res["kv"], [False])
        self.assertEqual(res["indexer"], [False])

    # Task 6: EAGLE draft KV device-direct L3.
    def test_draft_device_direct_roundtrip(self):
        """Task 6: the EAGLE draft KV rides the device-direct SG path under its own
        `.draft` namespace (distinct from the target). Write the draft from GPU slots,
        read it back into a FRESH draft pool, byte-identical per layer."""
        st = self._plugin(self._node("draft"))
        src = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        st.register_mem_pool_device_draft(src)
        page_hash = "draf7001"
        dev = list(range(0, self.PAGE_SIZE))
        self.assertEqual(st.batch_set_v1_device_draft([page_hash], dev), [True])

        dst = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        for L in range(self.LAYER_NUM):
            dst._np[L][:] = 0
        st.register_mem_pool_device_draft(dst)
        self.assertEqual(st.batch_get_v1_device_draft([page_hash], dev), [True])
        for L in range(self.LAYER_NUM):
            self.assertEqual(
                dst.layer_page_bytes(L, 0, self.PAGE_SIZE),
                src.layer_page_bytes(L, 0, self.PAGE_SIZE),
                f"draft layer {L} mismatch after device-direct roundtrip")

    def test_draft_keys_distinct_and_tp_aware(self):
        """Draft keys are a distinct namespace from the target and follow the DRAFT
        pool shape: MLA draft (sub=1) has no tp suffix (replicated); MHA draft (sub=2)
        carries tp_size/tp_rank so TP ranks do not collide."""
        st = _bare(is_mla=True, tp_size=2, tp_rank=1, pp_rank=0, pp_size=1,
                   enable_pp=False, model="m")
        h = "abc123"
        self.assertNotEqual(st._keys(h)[0], st._draft_keys(h, 1)[0])
        mla_draft = st._draft_keys(h, 1)
        self.assertEqual(
            mla_draft,
            [dfkv_hicache.pool_key(
                h, pool="draft", tp_size=2, tp_rank=-1,
                pp_size=1, pp_rank=0, component="all")],
        )
        mha_draft = st._draft_keys(h, 2)
        self.assertEqual(
            mha_draft,
            [dfkv_hicache.pool_key(
                 h, pool="draft", tp_size=2, tp_rank=1,
                 pp_size=1, pp_rank=0, component=component)
             for component in ("k", "v")],
        )

    # DSA draft (GLM-5.2 MTP): draft main latent + draft indexer sidecar, both device.
    def test_dsa_draft_device_direct_latent_and_indexer_roundtrip(self):
        """A GLM-5.2 MTP draft is a DSATokenToKVPool: its main MLA latent AND its DSA
        indexer sidecar (index_k_with_scale_buffer) both ride the device-direct path
        under distinct namespaces (`.draft_k` / `draft_indexer`). Without the indexer
        the draft's sparse attention would select on garbage; this proves BOTH
        components reassemble byte-exact so the loaded draft page is coherent."""
        st = self._plugin(self._node("dsadraft"))
        src = FakeDsaDraftDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM, num_pages=4, side_bytes=self.SIDE_DEV_BYTES)
        st.register_mem_pool_device_draft(src)
        st.register_mem_pool_device_draft_sidecar(src)
        page_hash = "dsadr701"
        dev = list(range(0, self.PAGE_SIZE))
        self.assertEqual(
            st.batch_set_v1_device_draft([page_hash], dev), [True],
            "DSA draft latent+indexer device-direct write failed")

        dst = FakeDsaDraftDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM, num_pages=4, side_bytes=self.SIDE_DEV_BYTES)
        for L in range(self.LAYER_NUM):
            dst._np[L][:] = 0
            dst._side_np[L][:] = 0
        st.register_mem_pool_device_draft(dst)
        st.register_mem_pool_device_draft_sidecar(dst)
        self.assertEqual(
            st.batch_get_v1_device_draft([page_hash], dev), [True],
            "DSA draft latent+indexer device-direct read failed")
        for L in range(self.LAYER_NUM):
            self.assertEqual(
                dst.layer_page_bytes(L, 0, self.PAGE_SIZE),
                src.layer_page_bytes(L, 0, self.PAGE_SIZE),
                f"draft latent layer {L} mismatch after DSA-draft roundtrip")
            self.assertEqual(
                dst.side_layer_page_bytes(L, 0),
                src.side_layer_page_bytes(L, 0),
                f"draft indexer layer {L} mismatch after DSA-draft roundtrip")

    def test_dsa_draft_indexer_read_miss_is_page_failure(self):
        """Coherence AND on read: a DSA draft page whose indexer was never stored
        reads back as a miss (even if the latent existed), so the draft recomputes
        rather than attending over a stale/garbage indexer."""
        st = self._plugin(self._node("dsadraftmiss"))
        dst = FakeDsaDraftDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM, num_pages=4, side_bytes=self.SIDE_DEV_BYTES)
        st.register_mem_pool_device_draft(dst)
        st.register_mem_pool_device_draft_sidecar(dst)
        dev = list(range(0, self.PAGE_SIZE))
        self.assertEqual(st.batch_get_v1_device_draft(["neverdsa"], dev), [False])

    def test_dense_draft_no_sidecar_is_latent_only(self):
        """A dense (non-DSA) draft never registers a sidecar, so the draft path is
        latent-only and byte-identical to task 6 — the sidecar hooks return None and
        do not touch the result."""
        st = self._plugin(self._node("densedraft"))
        self.assertIsNone(st._draft_sidecar_device_set(["h"], list(range(self.PAGE_SIZE))))
        self.assertIsNone(st._draft_sidecar_device_get(["h"], list(range(self.PAGE_SIZE))))
        src = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        st.register_mem_pool_device_draft(src)
        page_hash = "dense701"
        dev = list(range(0, self.PAGE_SIZE))
        self.assertEqual(st.batch_set_v1_device_draft([page_hash], dev), [True])
        dst = FakeLayerFirstMlaDevicePool(
            self.LAYER_NUM, size=self.PAGE_SIZE * 2, page_size=self.PAGE_SIZE,
            kv_cache_dim=self.KV_DIM)
        for L in range(self.LAYER_NUM):
            dst._np[L][:] = 0
        st.register_mem_pool_device_draft(dst)
        self.assertEqual(st.batch_get_v1_device_draft([page_hash], dev), [True])

    def test_draft_indexer_keys_distinct_namespace(self):
        """The DSA draft indexer rides its own `draft_indexer` key namespace, distinct
        from the draft latent (`.draft_k`) and the target indexer (`_indexer_k`)."""
        st = _bare(is_mla=True, tp_size=1, tp_rank=0, pp_rank=0, pp_size=1,
                   enable_pp=False, model="m")
        h = "abc123"
        self.assertEqual(
            st._pool_keys("draft_indexer", h),
            [dfkv_hicache.pool_key(
                h, pool="draft_indexer", tp_size=1, tp_rank=-1,
                pp_size=1, pp_rank=0, component="all")],
        )
        self.assertNotEqual(st._pool_keys("draft_indexer", h), st._draft_keys(h, 1))
        self.assertNotEqual(st._pool_keys("draft_indexer", h), st._pool_keys("indexer", h))


if __name__ == "__main__":
    unittest.main(verbosity=2)
