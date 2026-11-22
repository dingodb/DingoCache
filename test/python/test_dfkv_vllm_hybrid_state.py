"""Hybrid (attention + recurrent state) models: what the vLLM connector saves.

Runs on CPU without vLLM or torch: the stand-ins of test_dfkv_vllm_worker are
installed first, and the KV caches are described by tensor stand-ins that
carry exactly what register_kv_caches reads (storage, pointer, shape, strides,
element size).
"""

from __future__ import annotations

import os
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_vllm import worker as worker_module  # noqa: E402
from dfkv_vllm.data import (  # noqa: E402
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from dfkv_vllm.worker import (  # noqa: E402
    DfkvStoreWorker,
    KVCacheStoreRecvingThread,
)
from vllm.v1.core.kv_cache_utils import BlockHash  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integration" / "vllm" / "tests"))
from _coordinator_fake import fake_coordinator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class _Storage:
    def __init__(self, base: int, nbytes: int) -> None:
        self._base = base
        self._nbytes = nbytes

    def data_ptr(self) -> int:
        return self._base

    def nbytes(self) -> int:
        return self._nbytes


class _View:
    """A strided view into ``storage``, as torch.as_strided makes one."""

    def __init__(self, storage, offset_bytes, shape, strides, element_size):
        self._storage = storage
        self._offset = offset_bytes
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self._el = element_size
        self.ndim = len(self.shape)

    def untyped_storage(self):
        return self._storage

    def data_ptr(self) -> int:
        return self._storage.data_ptr() + self._offset

    def element_size(self) -> int:
        return self._el

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


NUM_BLOCKS = 4
BASE = 0x100000
# One recurrent-state layer with two state tensors in one page, as vLLM up
# to 0.26.x builds them for Mamba2 / GDN / KDA layers (gpu_model_runner,
# _reshape_kv_cache_tensors): tensor i is a view of the layer's raw page
# buffer at the sum of the earlier tensors' sizes, block stride = page.
CONV_SHAPE, CONV_EL = (3, 8), 2          # 48 bytes (fp16)
SSM_SHAPE, SSM_EL = (2, 4, 4), 4         # 128 bytes (fp32)
CONV_BYTES = _prod(CONV_SHAPE) * CONV_EL
SSM_BYTES = _prod(SSM_SHAPE) * SSM_EL
PAGE = CONV_BYTES + SSM_BYTES


def _contiguous_strides(shape):
    strides, acc = [], 1
    for d in reversed(shape):
        strides.append(acc)
        acc *= d
    return list(reversed(strides))


def _state_views(storage, page=PAGE):
    """vLLM <= 0.26.x: a list with one view per state tensor."""
    views, offset = [], 0
    for shape, el in ((CONV_SHAPE, CONV_EL), (SSM_SHAPE, SSM_EL)):
        strides = [page // el, *_contiguous_strides(shape)]
        views.append(_View(storage, offset, (NUM_BLOCKS, *shape), strides, el))
        offset += _prod(shape) * el
    return views


def _page_view(storage, content=PAGE, page=PAGE):
    """vLLM >= 0.27: one int8 view over the page (0.28: [B, 1, 1, page])."""
    return _View(storage, 0, (NUM_BLOCKS, 1, 1, content), (page, content, content, 1), 1)


def _metadata(pool_name: str) -> KeyMetadata:
    return KeyMetadata(
        model_name="model", dp_size=1, dp_rank=-1, tp_size=1, tp_rank=0,
        pcp_size=1, pcp_rank=0, dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0,
        group_id=0, pool_name=pool_name,
    )


class _ReadyThread:
    def __init__(self, *args, **_kwargs) -> None:
        # ready_event is positional argument 5 (recv) / 8 (send); find it.
        for arg in args:
            if isinstance(arg, threading.Event):
                arg.set()

    def start(self) -> None:
        pass


def _registered_worker(kv_caches):
    worker = object.__new__(DfkvStoreWorker)
    worker._closed = False
    worker._kv_cache_config = SimpleNamespace(num_blocks=NUM_BLOCKS)
    worker._kv_cache_groups = [SimpleNamespace(layer_names=sorted(kv_caches))]
    worker.token_dbs = [ChunkedTokenDatabase(_metadata("mamba"), block_size=16)]
    worker._kv_pool_regions = []
    worker.client = None
    worker.kv_role = "kv_consumer"
    worker.block_size = 16
    worker.tp_rank = 0
    worker._record_kv_connector_operation = None
    worker._record_kv_connector_observation = None
    worker._ensure_client_for_load = None
    worker._record_kv_connector_pool_sample = None
    worker.request_level_loads = False
    worker.coord = None
    # The views stand in for torch tensors (no torch on a CPU test runner).
    fake_torch = SimpleNamespace(Tensor=_View)
    with patch.object(worker_module, "KVCacheStoreRecvingThread", _ReadyThread), \
            patch.object(worker_module, "torch", fake_torch):
        worker.register_kv_caches(kv_caches)
    return worker


class RecurrentStateLayersSaveEveryStateTensor(unittest.TestCase):
    def test_every_state_tensor_of_a_layer_is_in_the_object(self) -> None:
        storage = _Storage(BASE, NUM_BLOCKS * PAGE)
        db = _registered_worker({"layer.0": _state_views(storage)}).token_dbs[0]
        self.assertEqual(
            db.geometry.logical_bytes_per_block, PAGE,
            "only the first state tensor of the layer is saved and loaded",
        )
        self.assertEqual(
            sorted(db._seg_layout),
            [(BASE, PAGE, CONV_BYTES), (BASE + CONV_BYTES, PAGE, SSM_BYTES)],
        )

    def test_several_layers_keep_layer_then_tensor_order(self) -> None:
        a = _Storage(BASE, NUM_BLOCKS * PAGE)
        b = _Storage(BASE + 0x10000, NUM_BLOCKS * PAGE)
        db = _registered_worker(
            {"layer.1": _state_views(b), "layer.0": _state_views(a)}
        ).token_dbs[0]
        self.assertEqual(db._seg_layout, [
            (BASE, PAGE, CONV_BYTES),
            (BASE + CONV_BYTES, PAGE, SSM_BYTES),
            (BASE + 0x10000, PAGE, CONV_BYTES),
            (BASE + 0x10000 + CONV_BYTES, PAGE, SSM_BYTES),
        ])

    def test_objects_of_the_first_tensor_only_are_not_found_again(self) -> None:
        # Objects saved before the fix hold the first state tensor only, under
        # the same chunk hashes. They must not be taken for the whole state:
        # the state pool of such a layer gets a new name.
        storage = _Storage(BASE, NUM_BLOCKS * PAGE)
        db = _registered_worker({"layer.0": _state_views(storage)}).token_dbs[0]
        self.assertEqual(db.metadata.pool_name, "mamba-states")

    def test_one_view_per_layer_is_unchanged(self) -> None:
        # vLLM >= 0.27 hands one view over the whole page: that was saved in
        # full before, so its objects and their keys stay as they were.
        storage = _Storage(BASE, NUM_BLOCKS * PAGE)
        db = _registered_worker({"layer.0": _page_view(storage)}).token_dbs[0]
        self.assertEqual(db._seg_layout, [(BASE, PAGE, PAGE)])
        self.assertEqual(db.metadata.pool_name, "mamba")
        single = _registered_worker(
            {"layer.0": [_state_views(storage)[0]]}
        ).token_dbs[0]
        self.assertEqual(single.metadata.pool_name, "mamba")
        self.assertEqual(single._seg_layout, [(BASE, PAGE, CONV_BYTES)])

    def test_a_padded_page_saves_the_states_not_the_padding(self) -> None:
        padded = PAGE + 32
        storage = _Storage(BASE, NUM_BLOCKS * padded)
        db = _registered_worker(
            {"layer.0": _state_views(storage, page=padded)}
        ).token_dbs[0]
        self.assertEqual(db._seg_layout, [
            (BASE, padded, CONV_BYTES),
            (BASE + CONV_BYTES, padded, SSM_BYTES),
        ])
        self.assertEqual(db.geometry.logical_bytes_per_block, PAGE)


class _StaleClient:
    """A ring answering every GET with ``(hit, stored_length)``: by default
    an object of the first state tensor only, what the connector saved
    before."""

    def __init__(self, stored_length: int, removable: bool = True, hit: int = 1,
                 get_error: bool = False) -> None:
        self.stored_length = stored_length
        self.removable = removable
        self.hit = hit
        self.get_error = get_error
        self.removed: list[list[bytes]] = []
        self.remove_release = threading.Event()
        self.remove_release.set()
        self.remove_entered = threading.Event()

    def batch_get_auto_sg(self, keys, _ptrs, _caps):
        if self.get_error:
            raise RuntimeError("injected transport failure")
        return [self.hit] * len(keys), [self.stored_length] * len(keys)

    def supports_remove(self) -> bool:
        return self.removable

    def batch_remove(self, keys):
        self.remove_entered.set()
        assert self.remove_release.wait(10)
        self.removed.append(list(keys))
        return [1] * len(keys)


class AnObjectOfTheOldLayoutIsNotLoaded(unittest.TestCase):
    def _receiver(self, client):
        storage = _Storage(BASE, NUM_BLOCKS * PAGE)
        db = _registered_worker({"layer.0": _state_views(storage)}).token_dbs[0]
        self.operations = []
        receiver = KVCacheStoreRecvingThread(
            client,
            fake_coordinator(16, load_mask=lambda _h, _n: [[True]]),
            [db],
            block_size=16,
            tp_rank=0,
            ready_event=threading.Event(),
            record_operation=lambda operation, status="ok", **kw:
                self.operations.append(
                    (operation, status, kw.get("num_failed_keys"))),
        )
        block_hash = BlockHash(b"s" * 32)
        request = ReqMeta(
            req_id="old-object", token_len_chunk=16, block_ids=([2],),
            block_hashes=[block_hash],
            load_spec=LoadSpec(0, 16, True, token_len=16),
        )
        key = db._make_key_by_hash(bytes(block_hash).hex()).to_bytes()
        return receiver, request, key

    def _load(self, client):
        receiver, request, key = self._receiver(client)
        receiver._handle_request(request)
        errors = receiver.get_and_clear_block_ids_with_load_errors()
        receiver.stop()  # waits for a background remove in flight
        return errors, key

    def _removes(self):
        return [op for op in self.operations if op[0] == "load_remove_stale"]

    def test_a_short_object_is_a_load_error_and_is_removed(self) -> None:
        client = _StaleClient(stored_length=CONV_BYTES)
        errors, key = self._load(client)
        self.assertEqual(errors, {2})
        self.assertEqual(client.removed, [[key]])
        self.assertEqual(self._removes(), [("load_remove_stale", "ok", 0)])

    def test_without_the_remove_rpc_it_is_only_counted(self) -> None:
        client = _StaleClient(stored_length=CONV_BYTES, removable=False)
        errors, _key = self._load(client)
        self.assertEqual(errors, {2})
        self.assertEqual(client.removed, [])
        self.assertEqual(self._removes(), [("load_remove_stale", "unsupported", 1)])

    def test_an_object_of_the_right_length_is_kept(self) -> None:
        client = _StaleClient(stored_length=PAGE)
        errors, _key = self._load(client)
        self.assertEqual(errors, set())
        self.assertEqual(client.removed, [])
        self.assertEqual(self._removes(), [])

    def test_a_miss_or_a_failed_transfer_removes_nothing(self) -> None:
        # The ring reports a failed or short transfer as no hit, length 0.
        for client in (
            _StaleClient(stored_length=0, hit=0),
            _StaleClient(stored_length=CONV_BYTES, get_error=True),
        ):
            with self.subTest(hit=client.hit, get_error=client.get_error):
                errors, _key = self._load(client)
                self.assertEqual(errors, {2})
                self.assertEqual(client.removed, [])
                self.assertEqual(self._removes(), [])

    def test_a_longer_object_is_not_removed(self) -> None:
        # Not a hit on a real ring (stored length above the capacity); even
        # if reported as one, only a shorter object counts as stale.
        for hit in (0, 1):
            with self.subTest(hit=hit):
                client = _StaleClient(stored_length=PAGE + 8, hit=hit)
                errors, _key = self._load(client)
                self.assertEqual(errors, {2})
                self.assertEqual(client.removed, [])
                self.assertEqual(self._removes(), [])

    def test_a_failing_remove_is_counted_not_lost(self) -> None:
        class _NoCapability(_StaleClient):
            supports_remove = None  # calling it raises TypeError

        class _RaisingRemove(_StaleClient):
            def batch_remove(self, keys):
                raise RuntimeError("injected remove failure")

        for client in (_NoCapability(CONV_BYTES), _RaisingRemove(CONV_BYTES)):
            with self.subTest(client=type(client).__name__):
                errors, _key = self._load(client)
                self.assertEqual(errors, {2})
                self.assertEqual(self._removes(), [("load_remove_stale", "error", 1)])

    def test_a_slow_remove_does_not_hold_up_the_load(self) -> None:
        client = _StaleClient(stored_length=CONV_BYTES)
        client.remove_release.clear()
        receiver, request, key = self._receiver(client)
        receiver._handle_request(request)  # would hang if it waited
        self.assertEqual(receiver.get_and_clear_block_ids_with_load_errors(), {2})
        self.assertTrue(client.remove_entered.wait(5))
        self.assertEqual(client.removed, [])
        client.remove_release.set()
        receiver.stop()
        self.assertEqual(client.removed, [[key]])

if __name__ == "__main__":
    unittest.main(verbosity=2)
