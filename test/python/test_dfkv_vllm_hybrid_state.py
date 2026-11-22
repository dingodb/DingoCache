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
    PoolKey,
    ReqMeta,
)
from dfkv_vllm.worker import (  # noqa: E402
    DfkvStoreWorker,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
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


# --------------------------------------------------------------------------
# SAVE of a hybrid model: what the model thread waits for
# --------------------------------------------------------------------------

ATTENTION, STATE = 0, 1
ADDR = {ATTENTION: 0x1000, STATE: 0x9000}


class _GroupDb:
    """One cache group of one block: a single segment at ADDR[group]."""

    block_size = 64

    def __init__(self, group: int) -> None:
        self.group = group
        self.metadata = _metadata("kv" if group == ATTENTION else "mamba")
        self.metadata = KeyMetadata(**{**self.metadata.__dict__, "group_id": group})
        self.cacheable = True
        self._seg_layout = [(ADDR[group], 64, 64)]

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        if token_len < 64 or mask_num >= 64:
            return []
        return [(0, 64, PoolKey(self.metadata, bytes(block_hashes[0]).hex()))]

    def chunk_hash(self, chunk_id, block_hashes):
        return block_hashes[chunk_id]

    def descriptor_shape(self, _start, _end, block_ids):
        return 1, 64, block_ids[0]

    def fill_descriptors(self, _start, _end, _block_ids, pointers, sizes, offset=0):
        pointers[offset] = ADDR[self.group]
        sizes[offset] = 64
        return offset + 1


class _UncachedDb(_GroupDb):
    def __init__(self) -> None:
        super().__init__(ATTENTION)
        self.metadata = KeyMetadata(**{**self.metadata.__dict__, "group_id": 2})
        self.cacheable = False

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        return []


class _Client:
    """PUTs of a group can be held until released, or fail."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stored: dict[bytes, int] = {}
        self.hold = {ATTENTION: threading.Event(), STATE: threading.Event()}
        self.hold[ATTENTION].set()
        self.hold[STATE].set()
        self.entered = {ATTENTION: threading.Event(), STATE: threading.Event()}
        self.fail_group: int | None = None

    def batch_exist(self, keys):
        with self.lock:
            return [1 if k in self.stored else 0 for k in keys]

    def batch_put_sg(self, keys, ptrs, _sizes):
        groups = {
            g for g, addr in ADDR.items() for p in ptrs if list(p)[0] == addr
        }
        for g in groups:
            self.entered[g].set()
        for g in groups:
            assert self.hold[g].wait(10), "a held PUT was never released"
        if self.fail_group in groups:
            raise RuntimeError("injected PUT failure")
        with self.lock:
            for key, p in zip(keys, ptrs):
                self.stored[key] = list(p)[0]
        return [0] * len(keys)


class _Event:
    def record(self) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _hash(seed: str) -> BlockHash:
    import hashlib
    return BlockHash(hashlib.sha256(seed.encode()).digest())


class _SaveRig:
    """The SAVE half of a worker for a model with one attention group and
    one recurrent-state group, over a real sending thread."""

    def __init__(
        self,
        test: unittest.TestCase,
        *,
        fenced=frozenset({STATE}),
        uncached_group: bool = False,
        queue_capacity: int = 64,
    ) -> None:
        self.client = _Client()
        self.operations: list[tuple[str, str]] = []
        self.dbs = [_GroupDb(ATTENTION), _GroupDb(STATE)]
        masks = [[True], [True]]
        if uncached_group:
            # Never cached (a scratch / circular-buffer pool): no objects.
            self.dbs.append(_UncachedDb())
            masks.append([False])
        coord = fake_coordinator(64, store_mask=lambda _n: masks)
        self.sender = KVCacheStoreSendingThread(
            self.client, coord, self.dbs, block_size=64, tp_rank=0,
            stripe_idx=0, stripe_step=1, kv_role="kv_both",
            ready_event=threading.Event(),
            record_operation=lambda operation, status="ok", **_kw:
                self.operations.append((operation, status)),
            queue_capacity=queue_capacity,
        )
        worker = object.__new__(DfkvStoreWorker)
        worker.load_async = False
        worker.kv_role = "kv_both"
        worker.kv_send_thread = self.sender
        worker.kv_recv_thread = None
        worker._fenced_store_groups = frozenset(fenced)
        worker.token_dbs = self.dbs
        worker.finished_store_req = set()
        self.worker = worker
        self.sender.start()
        assert self.sender.ready_event.wait(5)
        test.addCleanup(self.close)
        patcher = patch.object(worker_module.torch.cuda, "Event", _Event, create=True)
        patcher.start()
        test.addCleanup(patcher.stop)

    def close(self) -> None:
        for event in self.client.hold.values():
            event.set()
        self.sender.stop(cancel_pending=False)

    def request(self, name: str) -> ReqMeta:
        return ReqMeta(
            req_id=name, token_len_chunk=64,
            block_ids=tuple([g + 1] for g in range(len(self.dbs))),
            block_hashes=[_hash(name)], can_save=True,
        )

    def key(self, group: int, name: str) -> bytes:
        return PoolKey(self.dbs[group].metadata, bytes(_hash(name)).hex()).to_bytes()

    def submit_in_background(self, *requests: ReqMeta):
        done = threading.Event()
        errors: list[BaseException] = []

        def step() -> None:
            try:
                self.worker._submit_transfers(SimpleNamespace(requests=list(requests)))
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)
            finally:
                done.set()

        threading.Thread(target=step, daemon=True).start()
        return done, errors

    def finished(self, names) -> set:
        return self.worker._get_and_clear_finished_sending(set(names))


class HybridSaveFencesOnlyTheMutableState(unittest.TestCase):
    def test_the_model_step_does_not_wait_for_the_attention_put(self) -> None:
        rig = _SaveRig(self)
        rig.client.hold[ATTENTION].clear()
        done, errors = rig.submit_in_background(rig.request("r"))
        self.assertTrue(rig.client.entered[ATTENTION].wait(5))
        self.assertTrue(
            done.wait(2),
            "the model step waited for the PUT of immutable attention blocks",
        )
        self.assertEqual(errors, [])
        # The state is stored before the step returns.
        self.assertIn(rig.key(STATE, "r"), rig.client.stored)
        self.assertNotIn(rig.key(ATTENTION, "r"), rig.client.stored)
        # The request is not reported finished while its attention PUT runs:
        # vLLM keeps those blocks until then.
        self.assertEqual(rig.finished({"r"}), set())
        rig.client.hold[ATTENTION].set()
        rig.sender.request_queue.join()
        self.assertIn(rig.key(ATTENTION, "r"), rig.client.stored)
        self.assertEqual(rig.finished(set()), {"r"})

    def test_the_model_step_waits_for_the_state_put(self) -> None:
        rig = _SaveRig(self)
        rig.client.hold[STATE].clear()
        done, errors = rig.submit_in_background(rig.request("r"))
        self.assertTrue(rig.client.entered[STATE].wait(5))
        self.assertFalse(
            done.wait(0.5),
            "the next step could overwrite the state before its PUT is done",
        )
        rig.client.hold[STATE].set()
        self.assertTrue(done.wait(5))
        self.assertEqual(errors, [])
        self.assertIn(rig.key(STATE, "r"), rig.client.stored)

    def test_a_running_attention_put_of_an_earlier_step_is_not_waited_for(self) -> None:
        rig = _SaveRig(self)
        rig.client.hold[ATTENTION].clear()
        first, _ = rig.submit_in_background(rig.request("a"))
        self.assertTrue(first.wait(2))
        self.assertTrue(rig.client.entered[ATTENTION].wait(5))
        second, errors = rig.submit_in_background(rig.request("b"))
        self.assertTrue(
            second.wait(2),
            "the state fence waited behind another request's attention PUT",
        )
        self.assertEqual(errors, [])
        self.assertIn(rig.key(STATE, "b"), rig.client.stored)

    def test_a_failed_state_put_releases_the_step_and_is_accounted(self) -> None:
        rig = _SaveRig(self)
        rig.client.fail_group = STATE
        done, errors = rig.submit_in_background(rig.request("r"))
        self.assertTrue(done.wait(5))
        self.assertEqual(errors, [])
        rig.sender.request_queue.join()
        self.assertNotIn(rig.key(STATE, "r"), rig.client.stored)
        self.assertIn(rig.key(ATTENTION, "r"), rig.client.stored)
        self.assertIn(("save_put", "error"), rig.operations)
        self.assertEqual(rig.finished({"r"}), {"r"})

    def test_preemption_waits_for_either_part_in_flight(self) -> None:
        rig = _SaveRig(self)
        rig.client.hold[ATTENTION].clear()
        done, _ = rig.submit_in_background(rig.request("r"))
        self.assertTrue(done.wait(2))
        self.assertTrue(rig.client.entered[ATTENTION].wait(5))
        self.assertFalse(rig.sender.wait_for_inflight_put("r", timeout_s=0.2))
        rig.client.hold[ATTENTION].set()
        self.assertTrue(rig.sender.wait_for_inflight_put("r", timeout_s=5))

    def test_no_attention_part_when_every_cached_group_is_fenced(self) -> None:
        rig = _SaveRig(
            self, fenced=frozenset({ATTENTION, STATE}), uncached_group=True,
        )
        queued = []
        add_request = rig.sender.add_request
        rig.sender.add_request = lambda request: (
            queued.append(request), add_request(request))[1]
        done, errors = rig.submit_in_background(rig.request("r"))
        self.assertTrue(done.wait(5))
        self.assertEqual(errors, [])
        self.assertEqual(queued, [], "an empty attention part was queued")
        self.assertEqual(rig.finished({"r"}), {"r"})

    def test_preempted_while_the_attention_part_is_still_queued(self) -> None:
        rig = _SaveRig(self)
        rig.client.hold[ATTENTION].clear()
        first, _ = rig.submit_in_background(rig.request("a"))
        self.assertTrue(first.wait(2))
        self.assertTrue(rig.client.entered[ATTENTION].wait(5))
        # "b": its state is stored in the step, its attention part waits
        # behind "a"'s PUT on the regular worker.
        second, errors = rig.submit_in_background(rig.request("b"))
        self.assertTrue(second.wait(2))
        self.assertEqual(errors, [])
        rig.worker.handle_preemptions(
            SimpleNamespace(preempted_req_ids={"b"}, requests=[])
        )
        rig.client.hold[ATTENTION].set()
        rig.sender.request_queue.join()
        # The queued part was dropped at the generation gate: no PUT read
        # the preempted request's blocks, and no counter went negative.
        self.assertNotIn(rig.key(ATTENTION, "b"), rig.client.stored)
        self.assertNotIn("b", rig.sender.stored_requests)
        self.assertTrue(all(n >= 0 for n in rig.sender.stored_requests.values()))
        # Resumed under the same id, it saves and finishes normally.
        third, errors = rig.submit_in_background(rig.request("b"))
        self.assertTrue(third.wait(5))
        rig.sender.request_queue.join()
        self.assertIn(rig.key(ATTENTION, "b"), rig.client.stored)
        self.assertEqual(rig.finished({"a", "b"}), {"a", "b"})

    def test_stop_releases_a_step_waiting_for_its_state_puts(self) -> None:
        rig = _SaveRig(self)
        rig.client.hold[STATE].clear()
        done, errors = rig.submit_in_background(rig.request("a"), rig.request("b"))
        self.assertTrue(rig.client.entered[STATE].wait(5))
        # "a"'s state PUT is running, "b"'s waits in the fence queue.
        stopped = threading.Event()

        def stop() -> None:
            rig.sender.stop(cancel_pending=True)
            stopped.set()

        threading.Thread(target=stop, daemon=True).start()
        self.assertFalse(done.wait(0.3), "returned while a state PUT runs")
        rig.client.hold[STATE].set()
        self.assertTrue(done.wait(5), "the step stayed in fence_queue.join()")
        self.assertTrue(stopped.wait(5))
        self.assertEqual(errors, [])
        self.assertIn(rig.key(STATE, "a"), rig.client.stored)
        self.assertNotIn(rig.key(STATE, "b"), rig.client.stored)

    def test_a_rejected_attention_part_still_finishes_the_request(self) -> None:
        rig = _SaveRig(self, queue_capacity=1)
        rig.client.hold[ATTENTION].clear()
        first, _ = rig.submit_in_background(rig.request("a"))
        self.assertTrue(first.wait(2))
        self.assertTrue(rig.client.entered[ATTENTION].wait(5))
        second, _ = rig.submit_in_background(rig.request("b"))  # fills it
        self.assertTrue(second.wait(2))
        third, errors = rig.submit_in_background(rig.request("c"))
        self.assertTrue(third.wait(2))
        self.assertEqual(errors, [])
        self.assertIn(("save_rejected", "error"), rig.operations)
        self.assertIn(rig.key(STATE, "c"), rig.client.stored)
        self.assertNotIn(rig.key(ATTENTION, "c"), rig.client.stored)
        self.assertEqual(rig.finished({"c"}), {"c"})
        rig.client.hold[ATTENTION].set()
        rig.sender.request_queue.join()
        self.assertEqual(rig.finished({"a", "b"}), {"a", "b"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
