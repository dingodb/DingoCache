"""CPU-only contracts for the vLLM lookup IPC and lookup accounting."""

from __future__ import annotations

import hashlib
import logging
from importlib.util import find_spec
import queue
import sys
import threading
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _install_runtime_dependency_stubs() -> None:
    """Keep imports CPU-only when torch and pyzmq are not installed."""
    if "torch" not in sys.modules and find_spec("torch") is None:
        class Tensor:
            pass

        class Event:
            pass

        torch_module = ModuleType("torch")
        torch_cuda_module = ModuleType("torch.cuda")
        torch_cuda_module.Event = Event
        torch_module.Tensor = Tensor
        torch_module.cuda = torch_cuda_module
        sys.modules["torch"] = torch_module
        sys.modules["torch.cuda"] = torch_cuda_module

    if "zmq" not in sys.modules and find_spec("zmq") is None:
        class Again(Exception):
            pass

        class ZMQError(Exception):
            pass

        class Context:
            pass

        zmq_module = ModuleType("zmq")
        zmq_module.Again = Again
        zmq_module.ZMQError = ZMQError
        zmq_module.Context = Context
        zmq_module.REQ = 1
        zmq_module.REP = 2
        zmq_module.RCVTIMEO = 3
        sys.modules["zmq"] = zmq_module


def _install_vllm_stubs() -> None:
    """Install the import surface used by the connector under standalone Python."""
    if "vllm" in sys.modules or find_spec("vllm") is not None:
        return

    def stub(name: str, **attributes):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        sys.modules[name] = module
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(sys.modules[parent_name], child_name, module)
        return module

    class Placeholder:
        pass

    class BlockHash(bytes):
        pass

    @dataclass
    class KVCacheBlock:
        block_id: int

    @dataclass
    class KVConnectorStats:
        data: dict = field(default_factory=dict)

        def is_empty(self) -> bool:
            return not self.data

    class KVConnectorPromMetrics:
        pass

    class PromMetric:
        pass

    class BlockStored:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)

    class KVCacheSpecRegistry:
        @classmethod
        def get_manager_class(cls, _spec):
            return Placeholder

    stub("vllm")
    stub("vllm.envs", VLLM_RPC_BASE_PATH="/tmp")
    stub("vllm.config", VllmConfig=Placeholder, ParallelConfig=Placeholder)
    stub(
        "vllm.distributed",
        get_dcp_group=lambda: Placeholder(),
        get_pcp_group=lambda: Placeholder(),
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    stub("vllm.distributed.kv_events", BlockStored=BlockStored)
    stub("vllm.distributed.kv_transfer")
    stub("vllm.distributed.kv_transfer.kv_connector")
    stub("vllm.distributed.kv_transfer.kv_connector.v1")
    stub(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorMetadata=Placeholder,
    )
    stub(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        KVConnectorPromMetrics=KVConnectorPromMetrics,
        KVConnectorStats=KVConnectorStats,
        PromMetric=PromMetric,
        PromMetricT=PromMetric,
    )
    stub("vllm.logger", init_logger=logging.getLogger)
    stub("vllm.utils")
    stub(
        "vllm.utils.math_utils",
        cdiv=lambda numerator, denominator: (
            numerator + denominator - 1
        )
        // denominator,
    )
    stub(
        "vllm.utils.network_utils",
        make_zmq_socket=lambda *_args, **_kwargs: None,
    )
    stub("vllm.v1")
    stub("vllm.v1.core")
    stub("vllm.v1.core.block_pool", BlockPool=Placeholder)
    stub("vllm.v1.core.kv_cache_manager", KVCacheBlocks=Placeholder)
    stub(
        "vllm.v1.core.kv_cache_utils",
        BlockHash=BlockHash,
        BlockHashList=list[BlockHash],
        BlockHashListWithBlockSize=tuple[list[BlockHash], int],
        KVCacheBlock=KVCacheBlock,
        maybe_convert_block_hash=lambda value: value,
        resolve_kv_cache_block_sizes=lambda *_args, **_kwargs: (64, 64),
    )
    stub(
        "vllm.v1.core.single_type_kv_cache_manager",
        SingleTypeKVCacheManager=Placeholder,
    )
    stub("vllm.v1.core.sched")
    stub(
        "vllm.v1.core.sched.output",
        NewRequestData=Placeholder,
        SchedulerOutput=Placeholder,
    )
    stub(
        "vllm.v1.kv_cache_interface",
        FullAttentionSpec=Placeholder,
        KVCacheConfig=Placeholder,
        KVCacheGroupSpec=Placeholder,
        KVCacheSpec=Placeholder,
        UniformTypeKVCacheSpecs=Placeholder,
    )
    stub(
        "vllm.v1.kv_cache_spec_registry",
        KVCacheSpecRegistry=KVCacheSpecRegistry,
    )
    stub("vllm.v1.request", Request=Placeholder)


_install_runtime_dependency_stubs()
_install_vllm_stubs()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration" / "common" / "src"))
sys.path.insert(0, str(ROOT / "integration" / "vllm" / "src"))

from vllm.v1.core.kv_cache_utils import BlockHash  # noqa: E402

from dfkv_vllm.data import (  # noqa: E402
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
)
from dfkv_vllm.protocol import LOOKUP_MSG  # noqa: E402
from dfkv_vllm.worker import (  # noqa: E402
    DfkvStoreWorker,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    LookupKeyClient,
    LookupKeyServer,
)


class _Frame:
    def __init__(self, value: bytes) -> None:
        self.buffer = value

    def __bytes__(self) -> bytes:
        return self.buffer


class _MemoryLookupSocket:
    """In-memory REQ/REP endpoint with the subset used by the lookup classes."""

    def __init__(self) -> None:
        self.requests: queue.Queue[list[bytes]] = queue.Queue()
        self.responses: queue.Queue[bytes] = queue.Queue()
        self.last_request: list[bytes] | None = None
        self.closed = False

    def setsockopt(self, _option, _value) -> None:
        pass

    def send_multipart(self, frames, copy=False) -> None:
        del copy
        request = [bytes(frame) for frame in frames]
        self.last_request = request
        self.requests.put(request)

    def recv_multipart(self, copy=False):
        del copy
        try:
            return [_Frame(value) for value in self.requests.get(timeout=0.02)]
        except queue.Empty as exc:
            import zmq

            raise zmq.Again() from exc

    def send(self, response) -> None:
        self.responses.put(bytes(response))

    def recv(self) -> bytes:
        return self.responses.get(timeout=1)

    def close(self, linger=0) -> None:
        del linger
        self.closed = True


class _ControlledLookupSocket:
    """Thread-safe fake whose replies are released explicitly by token length."""

    def __init__(self, token_lengths: tuple[int, ...]) -> None:
        self.started = {value: threading.Event() for value in token_lengths}
        self.release = {value: threading.Event() for value in token_lengths}
        self.send_count = {value: 0 for value in token_lengths}
        self._local = threading.local()
        self.closed = False

    def send_multipart(self, frames, copy=False) -> None:
        del copy
        token_len = int.from_bytes(bytes(frames[1]), "big")
        self._local.token_len = token_len
        self.send_count[token_len] += 1
        self.started[token_len].set()

    def recv(self) -> bytes:
        token_len = self._local.token_len
        if not self.release[token_len].wait(timeout=1):
            raise TimeoutError(f"lookup {token_len} was not released")
        return token_len.to_bytes(4, "big")

    def send(self, _frame) -> None:
        raise AssertionError("reset is not used by these tests")

    def close(self, linger=0) -> None:
        del linger
        self.closed = True


class _MemoryContext:
    def __init__(self) -> None:
        self.terminated = False

    def term(self) -> None:
        self.terminated = True


class _StoreWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[int, list[bytes]]] = []
        self.kv_send_thread = None
        self.observations: list[tuple[str, float]] = []

    def lookup(self, token_len: int, block_hashes: list[BlockHash]) -> int:
        self.calls.append((token_len, [bytes(value) for value in block_hashes]))
        return min(token_len, len(block_hashes) * 64)
    def _record_kv_connector_observation(
        self, name: str, duration_seconds: float
    ) -> None:
        self.observations.append((name, duration_seconds))



class LookupFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = _MemoryLookupSocket()
        self.context = _MemoryContext()
        self.store = _StoreWorker()
        patches = (
            patch("dfkv_vllm.worker.zmq.Context", return_value=self.context),
            patch("dfkv_vllm.worker.get_zmq_rpc_path_lookup", return_value="ipc:///tmp/dfkv-test-lookup"),
            patch("dfkv_vllm.worker.make_zmq_socket", return_value=self.endpoint),
            patch("dfkv_vllm.worker.os.path.exists", return_value=False),
        )
        self._patches = patches
        for active_patch in patches:
            active_patch.start()
        self.server = LookupKeyServer(self.store, SimpleNamespace())
        self.client = LookupKeyClient(SimpleNamespace())

    def tearDown(self) -> None:
        self.client.close()
        self.server.close()
        for active_patch in reversed(self._patches):
            active_patch.stop()

    @staticmethod
    def _hash(seed: str) -> BlockHash:
        return BlockHash(hashlib.sha256(seed.encode()).digest())

    def test_binary_lookup_frame_round_trip(self) -> None:
        hashes = [self._hash("alpha"), self._hash("beta")]

        self.assertEqual(self.client.lookup("round-trip", 160, hashes), 128)

        frames = self.endpoint.last_request
        self.assertIsNotNone(frames)
        assert frames is not None
        self.assertEqual(frames[0], LOOKUP_MSG)
        self.assertEqual(frames[1], (160).to_bytes(4, "big"))
        self.assertEqual(frames[2], (32).to_bytes(2, "big"))
        self.assertEqual(frames[3], b"".join(bytes(value) for value in hashes))
        self.assertEqual(self.store.calls, [(160, [bytes(value) for value in hashes])])

    def test_empty_hash_list_has_canonical_binary_frame(self) -> None:
        self.assertEqual(self.client.lookup("empty", 64, []), 0)
        self.assertEqual(
            self.endpoint.last_request,
            [LOOKUP_MSG, (64).to_bytes(4, "big"), (0).to_bytes(2, "big"), b""],
        )
        self.assertEqual(self.store.calls, [(64, [])])

    def test_malformed_lookup_frames_are_rejected_and_server_survives(self) -> None:
        malformed = (
            [LOOKUP_MSG],
            [LOOKUP_MSG, b"\x00", (32).to_bytes(2, "big"), b"x" * 32],
            [LOOKUP_MSG, (64).to_bytes(4, "big"), b"\x00", b"x" * 32],
            [LOOKUP_MSG, (64).to_bytes(4, "big"), (0).to_bytes(2, "big"), b"x"],
            [LOOKUP_MSG, (64).to_bytes(4, "big"), (32).to_bytes(2, "big"), b"x" * 33],
        )
        for frames in malformed:
            with self.subTest(frames=frames):
                self.endpoint.send_multipart(frames)
                self.assertEqual(self.endpoint.recv(), (0).to_bytes(4, "big"))
                self.assertEqual(self.store.calls, [])

        valid_hash = self._hash("still-alive")
        self.assertEqual(self.client.lookup("valid-after-errors", 64, [valid_hash]), 64)

class LookupFutureTest(unittest.TestCase):
    @staticmethod
    def _hash(seed: str) -> BlockHash:
        return BlockHash(hashlib.sha256(seed.encode()).digest())

    def _client(self, socket: _ControlledLookupSocket) -> LookupKeyClient:
        context = _MemoryContext()
        with (
            patch("dfkv_vllm.worker.zmq.Context", return_value=context),
            patch(
                "dfkv_vllm.worker.get_zmq_rpc_path_lookup",
                return_value="ipc:///tmp/dfkv-test-futures",
            ),
            patch("dfkv_vllm.worker.make_zmq_socket", return_value=socket),
        ):
            return LookupKeyClient(SimpleNamespace())

    def test_only_one_future_is_created_per_request(self) -> None:
        socket = _ControlledLookupSocket((64,))
        client = self._client(socket)
        hashes = [self._hash("same")]
        try:
            self.assertIsNone(client.lookup("same", 64, hashes, non_block=True))
            self.assertTrue(socket.started[64].wait(timeout=1))
            pending = client.futures["same"]

            self.assertIsNone(client.lookup("same", 64, hashes, non_block=True))
            self.assertIs(client.futures["same"], pending)
            self.assertEqual(socket.send_count[64], 1)

            socket.release[64].set()
            self.assertEqual(pending.result(timeout=1), 64)
            self.assertEqual(client.lookup("same", 64, hashes, non_block=True), 64)
            self.assertNotIn("same", client.futures)
        finally:
            socket.release[64].set()
            client.close()

    def test_concurrent_request_ids_have_distinct_futures(self) -> None:
        socket = _ControlledLookupSocket((64, 128))
        client = self._client(socket)
        hashes = [self._hash("concurrent")]
        try:
            self.assertIsNone(client.lookup("left", 64, hashes, non_block=True))
            self.assertIsNone(client.lookup("right", 128, hashes, non_block=True))
            self.assertEqual(set(client.futures), {"left", "right"})
            self.assertIsNot(client.futures["left"], client.futures["right"])

            socket.release[64].set()
            socket.release[128].set()
            self.assertEqual(client.futures["left"].result(timeout=1), 64)
            self.assertEqual(client.futures["right"].result(timeout=1), 128)
            self.assertEqual(client.lookup("left", 64, hashes, non_block=True), 64)
            self.assertEqual(client.lookup("right", 128, hashes, non_block=True), 128)
        finally:
            socket.release[64].set()
            socket.release[128].set()
            client.close()

    def test_discard_and_close_remove_and_cancel_pending_futures(self) -> None:
        socket = _ControlledLookupSocket((32, 64, 128))
        client = self._client(socket)
        hashes = [self._hash("cancel")]
        try:
            self.assertIsNone(client.lookup("discarded", 32, hashes, non_block=True))
            self.assertTrue(socket.started[32].wait(timeout=1))
            client.discard("discarded")
            self.assertNotIn("discarded", client.futures)
            socket.release[32].set()

            self.assertIsNone(
                client.lookup("running-at-close", 64, hashes, non_block=True)
            )
            self.assertTrue(socket.started[64].wait(timeout=1))
            self.assertIsNone(
                client.lookup("queued-at-close", 128, hashes, non_block=True)
            )
            queued = client.futures["queued-at-close"]
            client.close()

            self.assertEqual(client.futures, {})
            self.assertTrue(queued.cancelled())
            self.assertTrue(socket.closed)
        finally:
            socket.release[32].set()
            socket.release[64].set()
            socket.release[128].set()
            client.close()



class LogicalChunkTransferTest(unittest.TestCase):
    @staticmethod
    def _metadata() -> KeyMetadata:
        return KeyMetadata(
            model_name="model",
            dp_size=1,
            dp_rank=-1,
            tp_size=1,
            tp_rank=-1,
            pcp_size=1,
            pcp_rank=0,
            dcp_size=1,
            dcp_rank=0,
            pp_size=1,
            pp_rank=0,
            group_id=0,
        )

    @staticmethod
    def _hash(seed: str) -> BlockHash:
        return BlockHash(hashlib.sha256(seed.encode()).digest())

    def test_save_uses_one_key_and_preserves_more_than_29_descriptors(self) -> None:
        metadata = self._metadata()
        logical_key = PoolKey(metadata, self._hash("save").hex())
        addresses = [0x1000 + i * 0x100 for i in range(35)]
        sizes = [i + 1 for i in range(35)]

        class _Database:
            block_size = 64
            _seg_layout = [(0, 1, 1)] * len(addresses)

            def process_tokens(self, *_args):
                return [(0, 64, logical_key)]

            def descriptor_shape(self, _start, _end, block_ids):
                return len(addresses), sum(sizes), block_ids[0]

            def fill_descriptors(
                self, _start, _end, _block_ids, pointers, out_sizes, offset=0
            ):
                for index, (address, size) in enumerate(
                    zip(addresses, sizes, strict=True), start=offset
                ):
                    pointers[index] = address
                    out_sizes[index] = size
                return offset + len(addresses)

            def prepare_value(self, *_args):
                return list(addresses), list(sizes), 7

        class _Client:
            def __init__(self) -> None:
                self.exist_calls: list[list[bytes]] = []
                self.put_calls: list[
                    tuple[list[bytes], list[list[int]], list[list[int]]]
                ] = []
                self.remove_calls: list[list[bytes]] = []

            def max_sg_segs(self) -> int:
                return 29

            def batch_exist(self, keys):
                self.exist_calls.append(list(keys))
                return [0] * len(keys)

            def batch_put_sg(self, keys, ptrs, seg_sizes):
                self.put_calls.append(
                    (list(keys), [list(v) for v in ptrs],
                     [list(v) for v in seg_sizes])
                )
                return [0] * len(keys)

            def supports_remove(self) -> bool:
                return True

            def batch_remove(self, keys):
                self.remove_calls.append(list(keys))
                return [0] * len(keys)

        client = _Client()
        coord = SimpleNamespace(
            lcm_block_size=64,
            store_mask=lambda _token_len: [[True]],
        )
        sender = KVCacheStoreSendingThread(
            client,
            coord,
            [_Database()],
            block_size=64,
            tp_rank=0,
            stripe_idx=0,
            stripe_step=1,
            kv_role="kv_both",
            ready_event=threading.Event(),
        )
        sender.add_stored_request("save")
        sender._handle_request(
            ReqMeta(
                req_id="save",
                token_len_chunk=64,
                block_ids=([7],),
                block_hashes=[self._hash("save")],
            )
        )

        expected_key = logical_key.to_bytes()
        self.assertEqual(client.exist_calls, [[expected_key]])
        self.assertEqual(len(client.put_calls), 1)
        put_keys, put_ptrs, put_sizes = client.put_calls[0]
        self.assertEqual(put_keys, [expected_key])
        self.assertEqual(put_ptrs, [addresses])
        self.assertEqual(put_sizes, [sizes])
        self.assertEqual(client.remove_calls, [])

    def test_put_timeout_stays_fenced_until_exception_and_close_complete(
        self,
    ) -> None:
        metadata = self._metadata()
        logical_key = PoolKey(metadata, self._hash("blocked-save").hex())
        native_started = threading.Event()
        native_release = threading.Event()

        class _Database:
            block_size = 64
            _seg_layout = [(0, 1, 1)]

            @staticmethod
            def process_tokens(*_args):
                return [(0, 64, logical_key)]

            @staticmethod
            def descriptor_shape(_start, _end, block_ids):
                return 1, 64, block_ids[0]

            @staticmethod
            def fill_descriptors(
                _start, _end, block_ids, pointers, sizes, offset=0
            ):
                pointers[offset] = 0x10000 + block_ids[0] * 0x100
                sizes[offset] = 64
                return offset + 1

        client_closed = threading.Event()
        class _Client:
            @staticmethod
            def batch_exist(keys):
                return [0] * len(keys)

            @staticmethod
            def batch_put_sg(_keys, _ptrs, _sizes):
                native_started.set()
                if not native_release.wait(timeout=2):
                    raise AssertionError("native PUT release was not signalled")
                raise RuntimeError("controlled native PUT failure")

            @staticmethod
            def close():
                client_closed.set()

        client = _Client()
        sender = KVCacheStoreSendingThread(
            client,
            SimpleNamespace(
                lcm_block_size=64,
                store_mask=lambda _token_len: [[True]],
            ),
            [_Database()],
            block_size=64,
            tp_rank=0,
            stripe_idx=0,
            stripe_step=1,
            kv_role="kv_both",
            ready_event=threading.Event(),
        )
        request = ReqMeta(
            req_id="blocked-save",
            token_len_chunk=64,
            block_ids=([7],),
            block_hashes=[self._hash("blocked-save")],
        )
        sender.add_stored_request(request.req_id)
        sender.start()
        self.assertTrue(sender.ready_event.wait(timeout=1))
        self.assertTrue(sender.add_request(request))
        self.assertTrue(native_started.wait(timeout=1))

        diagnostic_expired = threading.Event()
        wait_outcomes: list[bool] = []
        original_wait = sender.wait_for_inflight_put

        def short_diagnostic_wait(req_id: str) -> bool:
            result = original_wait(req_id, timeout_s=0.01)
            wait_outcomes.append(result)
            if not result:
                diagnostic_expired.set()
            return result

        sender.wait_for_inflight_put = short_diagnostic_wait
        worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
        worker.kv_recv_thread = None
        worker.kv_send_thread = sender
        worker.lookup_server = None
        worker._close_lock = threading.Lock()
        worker._closed = False
        worker._close_done = threading.Event()
        worker._lazy_client_lock = threading.Lock()
        worker._lazy_client_kwargs = None
        worker.client = client
        preemption_returned = threading.Event()
        preemption_errors: list[BaseException] = []

        def preempt() -> None:
            try:
                worker.start_load_kv(
                    SimpleNamespace(preempted_req_ids={request.req_id})
                )
            except BaseException as exc:
                preemption_errors.append(exc)
            finally:
                preemption_returned.set()

        close_entered = threading.Event()
        close_returned = threading.Event()
        close_errors: list[BaseException] = []

        def close() -> None:
            close_entered.set()
            try:
                worker.close()
            except BaseException as exc:
                close_errors.append(exc)
            finally:
                close_returned.set()

        preemption_thread = threading.Thread(target=preempt)
        close_thread = threading.Thread(target=close)
        preemption_thread.start()
        try:
            self.assertTrue(diagnostic_expired.wait(timeout=1))
            self.assertFalse(preemption_returned.is_set())

            close_thread.start()
            self.assertTrue(close_entered.wait(timeout=1))
            self.assertFalse(close_returned.is_set())

            native_release.set()
            self.assertTrue(preemption_returned.wait(timeout=1))
            self.assertTrue(close_returned.wait(timeout=1))
            preemption_thread.join(timeout=1)
            close_thread.join(timeout=1)

            self.assertEqual(preemption_errors, [])
            self.assertEqual(close_errors, [])
            self.assertGreaterEqual(wait_outcomes.count(False), 1)
            self.assertEqual(wait_outcomes.count(True), 1)
            self.assertIsNone(sender._active_req_id)
            self.assertFalse(sender.is_alive())
            self.assertTrue(client_closed.is_set())
            self.assertIsNone(worker.kv_send_thread)
            self.assertIsNone(worker.client)
        finally:
            native_release.set()
            preemption_thread.join(timeout=1)
            if close_thread.ident is not None:
                close_thread.join(timeout=1)
            sender.stop(cancel_pending=True)

    def test_partial_chunk_load_preserves_descriptor_order(self) -> None:
        metadata = self._metadata()
        hashes = [self._hash("load-hit"), self._hash("load-miss")]
        keys = [PoolKey(metadata, value.hex()) for value in hashes]
        addresses = [
            [0x2000 + i * 0x80 for i in range(35)],
            [0x8000 + i * 0x40 for i in range(33)],
        ]
        sizes = [
            [2 + i for i in range(35)],
            [7 + (i % 5) for i in range(33)],
        ]

        class _Database:
            block_size = 64
            _seg_layout = [(0, 1, 1)] * 35

            def process_tokens(self, *_args):
                return [(0, 64, keys[0]), (64, 128, keys[1])]

            def descriptor_shape(self, start, _end, block_ids):
                index = start // 64
                return (
                    len(addresses[index]),
                    sum(sizes[index]),
                    block_ids[index],
                )

            def fill_descriptors(
                self, start, _end, _block_ids, pointers, out_sizes, offset=0
            ):
                index = start // 64
                for cursor, (address, size) in enumerate(
                    zip(addresses[index], sizes[index], strict=True),
                    start=offset,
                ):
                    pointers[cursor] = address
                    out_sizes[cursor] = size
                return offset + len(addresses[index])

            def prepare_value(self, start, *_args):
                index = start // 64
                return list(addresses[index]), list(sizes[index]), (7, 9)[index]

        class _Client:
            def __init__(self) -> None:
                self.get_call = None
                self.truncate = False

            def max_sg_segs(self) -> int:
                return 29

            def batch_get_auto_sg(self, get_keys, ptrs, caps):
                key_order = list(get_keys)
                self.get_call = (
                    key_order,
                    [list(value) for value in ptrs],
                    [list(value) for value in caps],
                )
                hits = [int(key == keys[0].to_bytes()) for key in key_order]
                lengths = [
                    sum(sizes[0]) if key == keys[0].to_bytes() else 0
                    for key in key_order
                ]
                if self.truncate:
                    return hits[:1], lengths[:1]
                return hits, lengths

        client = _Client()
        coord = SimpleNamespace(
            load_mask=lambda _hashes, _token_len: [[True, True]],
        )
        receiver = KVCacheStoreRecvingThread(
            client,
            coord,
            [_Database()],
            block_size=64,
            tp_rank=0,
            ready_event=threading.Event(),
        )
        request = ReqMeta(
            req_id="load",
            token_len_chunk=128,
            block_ids=([7, 9],),
            block_hashes=hashes,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=128,
                can_load=True,
                token_len=128,
            ),
        )
        receiver._handle_request(request)

        get_keys, get_addrs, get_sizes = client.get_call
        observed = {
            key: (chunk_addrs, chunk_sizes)
            for key, chunk_addrs, chunk_sizes in zip(
                get_keys, get_addrs, get_sizes, strict=True
            )
        }
        self.assertEqual(
            observed,
            {
                keys[0].to_bytes(): (addresses[0], sizes[0]),
                keys[1].to_bytes(): (addresses[1], sizes[1]),
            },
        )
        load_errors = receiver.get_and_clear_block_ids_with_load_errors()
        self.assertNotIn(7, load_errors)
        self.assertIn(9, load_errors)
        self.assertEqual(load_errors, {9})

        # A truncated native result must be rejected before per-key indexing.
        # Fail the whole batch closed rather than silently treating the omitted
        # logical chunk as a hit.
        client.truncate = True
        receiver._handle_request(request)
        self.assertEqual(
            receiver.get_and_clear_block_ids_with_load_errors(), {7, 9}
        )


class GeometryReuseTest(unittest.TestCase):
    @staticmethod
    def _metadata() -> KeyMetadata:
        return KeyMetadata(
            model_name="geometry",
            dp_size=1,
            dp_rank=-1,
            tp_size=1,
            tp_rank=-1,
            pcp_size=1,
            pcp_rank=0,
            dcp_size=1,
            dcp_rank=0,
            pp_size=1,
            pp_rank=0,
        )

    def test_cached_layout_uses_current_arbitrary_ids_and_returns_owned_arrays(
        self,
    ) -> None:
        database = ChunkedTokenDatabase(self._metadata(), block_size=64)
        database.set_seg_layout(
            [
                (0x1000, 0x80, 0x60),
                (0x8000, 0x100, 0xC0),
            ]
        )
        geometry = database.geometry
        with self.assertRaises(AttributeError):
            geometry.segment_bases = ()  # type: ignore[misc]

        first_ids = [17, 2, 999]
        first_addrs, first_sizes, first_block_id = database.prepare_value(
            0, 192, first_ids
        )
        expected_first_addrs = [
            0x1000 + block_id * 0x80 for block_id in first_ids
        ] + [
            0x8000 + block_id * 0x100 for block_id in first_ids
        ]
        self.assertEqual(list(first_addrs), expected_first_addrs)
        self.assertEqual(list(first_sizes), [0x60] * 3 + [0xC0] * 3)
        self.assertEqual(first_block_id, 17)

        first_snapshot = (list(first_addrs), list(first_sizes))
        second_addrs, second_sizes, second_block_id = database.prepare_value(
            0, 192, [4, 81, 6]
        )
        self.assertEqual(second_block_id, 4)
        self.assertEqual(
            list(second_addrs),
            [
                0x1000 + 4 * 0x80,
                0x1000 + 81 * 0x80,
                0x1000 + 6 * 0x80,
                0x8000 + 4 * 0x100,
                0x8000 + 81 * 0x100,
                0x8000 + 6 * 0x100,
            ],
        )
        self.assertEqual(list(second_sizes), [0x60] * 3 + [0xC0] * 3)
        self.assertIsNot(first_addrs, second_addrs)
        self.assertIsNot(first_sizes, second_sizes)
        self.assertEqual((list(first_addrs), list(first_sizes)), first_snapshot)
        self.assertIs(database.geometry, geometry)

    def test_sequential_requests_do_not_share_mutable_native_vectors(self) -> None:
        database = ChunkedTokenDatabase(self._metadata(), block_size=64)
        database.set_seg_layout([(0x5000, 0x200, 0x180)])

        class _Client:
            def __init__(self) -> None:
                self.calls = []

            def batch_get_auto_sg(self, keys, pointers, capacities):
                self.calls.append((keys, pointers, capacities))
                return [1], [0x180]

        client = _Client()
        receiver = KVCacheStoreRecvingThread(
            client,
            SimpleNamespace(load_mask=lambda _hashes, _token_len: [[True]]),
            [database],
            block_size=64,
            tp_rank=0,
            ready_event=threading.Event(),
        )

        def request(request_id: str, block_id: int) -> ReqMeta:
            block_hash = BlockHash(hashlib.sha256(request_id.encode()).digest())
            return ReqMeta(
                req_id=request_id,
                token_len_chunk=64,
                block_ids=([block_id],),
                block_hashes=[block_hash],
                load_spec=LoadSpec(
                    vllm_cached_tokens=0,
                    kvpool_cached_tokens=64,
                    can_load=True,
                    token_len=64,
                ),
            )

        receiver._handle_request(request("first", 37))
        first_pointers = client.calls[0][1]
        first_capacities = client.calls[0][2]
        first_snapshot = (
            [list(chunk) for chunk in first_pointers],
            [list(chunk) for chunk in first_capacities],
        )
        receiver._handle_request(request("second", 901))

        self.assertIsNot(first_pointers, client.calls[1][1])
        self.assertIsNot(first_capacities, client.calls[1][2])
        self.assertEqual(
            (
                [list(chunk) for chunk in first_pointers],
                [list(chunk) for chunk in first_capacities],
            ),
            first_snapshot,
        )
        self.assertEqual(first_snapshot, ([[0x5000 + 37 * 0x200]], [[0x180]]))
        self.assertEqual(
            [list(chunk) for chunk in client.calls[1][1]],
            [[0x5000 + 901 * 0x200]],
        )


class ReceiveRotationTest(unittest.TestCase):
    @staticmethod
    def _hash(seed: str) -> BlockHash:
        return BlockHash(hashlib.sha256(seed.encode()).digest())

    @staticmethod
    def _metadata() -> KeyMetadata:
        return KeyMetadata(
            model_name="rotation",
            dp_size=1,
            dp_rank=-1,
            tp_size=8,
            tp_rank=-1,
            pcp_size=1,
            pcp_rank=0,
            dcp_size=1,
            dcp_rank=0,
            pp_size=1,
            pp_rank=0,
        )

    def _exercise(
        self, req_id: str, tp_rank: int
    ) -> tuple[list[bytes], set[int]]:
        hashes = [self._hash(f"chunk-{index}") for index in range(4)]
        keys = [PoolKey(self._metadata(), value.hex()).to_bytes() for value in hashes]
        block_ids = [101, 7, 4095, 23]
        expected_by_key = {
            key: (0x2000 + block_id * 0x100, 64, block_id)
            for key, block_id in zip(keys, block_ids, strict=True)
        }

        class _Database:
            block_size = 64
            _seg_layout = [(0x2000, 0x100, 64)]

            def process_tokens(self, *_args):
                return [
                    (index * 64, (index + 1) * 64, PoolKey(
                        ReceiveRotationTest._metadata(), hashes[index].hex()
                    ))
                    for index in range(4)
                ]

            def descriptor_shape(self, start, _end, current_block_ids):
                index = start // 64
                return 1, 64, current_block_ids[index]

            def fill_descriptors(
                self, start, _end, current_block_ids, pointers, sizes, offset=0
            ):
                index = start // 64
                pointers[offset] = 0x2000 + current_block_ids[index] * 0x100
                sizes[offset] = 64
                return offset + 1

            def prepare_value(self, start, _end, current_block_ids):
                index = start // 64
                block_id = current_block_ids[index]
                return [0x2000 + block_id * 0x100], [64], block_id

        class _Client:
            def __init__(self) -> None:
                self.orders: list[list[bytes]] = []

            def batch_get_auto_sg(self, get_keys, ptrs, caps):
                key_order = list(get_keys)
                self.orders.append(key_order)
                hits: list[int] = []
                lengths: list[int] = []
                for key, addresses, sizes in zip(
                    key_order, ptrs, caps, strict=True
                ):
                    expected_addr, expected_size, expected_block_id = expected_by_key[key]
                    if list(addresses) != [expected_addr]:
                        raise AssertionError(
                            f"key {key!r} used stale block {expected_block_id}"
                        )
                    if list(sizes) != [expected_size]:
                        raise AssertionError(f"key {key!r} used wrong geometry")
                    hit = key != keys[2]
                    hits.append(int(hit))
                    lengths.append(expected_size if hit else 0)
                return hits, lengths

        client = _Client()
        receiver = KVCacheStoreRecvingThread(
            client,
            SimpleNamespace(load_mask=lambda _hashes, _token_len: [[True] * 4]),
            [_Database()],
            block_size=64,
            tp_rank=tp_rank,
            ready_event=threading.Event(),
        )
        request = ReqMeta(
            req_id=req_id,
            token_len_chunk=256,
            block_ids=(block_ids,),
            block_hashes=hashes,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=256,
                can_load=True,
                token_len=256,
            ),
        )
        receiver._handle_request(request)
        return client.orders[0], receiver.get_and_clear_block_ids_with_load_errors()

    def test_request_seed_rotates_initial_chunk_stably_without_reordering_results(
        self,
    ) -> None:
        canonical = [
            PoolKey(self._metadata(), self._hash(f"chunk-{index}").hex()).to_bytes()
            for index in range(4)
        ]
        orders: list[list[bytes]] = []
        for index in range(12):
            order, failed_blocks = self._exercise(f"request-{index}", tp_rank=0)
            orders.append(order)
            self.assertEqual(failed_blocks, {4095})
            doubled = canonical + canonical
            start = doubled.index(order[0])
            self.assertEqual(order, doubled[start:start + len(canonical)])

        self.assertGreater(
            len({tuple(order) for order in orders}),
            1,
            "request identity must spread the first logical chunk at fixed TP rank",
        )
        repeated, failed_blocks = self._exercise("request-3", tp_rank=0)
        self.assertEqual(repeated, orders[3])
        self.assertEqual(failed_blocks, {4095})

    def test_tp_ranks_rotate_only_the_native_issue_order(self) -> None:
        orders = []
        for tp_rank in range(4):
            order, failed_blocks = self._exercise("same-request", tp_rank)
            orders.append(tuple(order))
            self.assertEqual(failed_blocks, {4095})
        self.assertEqual(len(set(orders)), 4)


class LongestCompletePrefixLookupTest(unittest.TestCase):
    @staticmethod
    def _hash(seed: str) -> BlockHash:
        return BlockHash(hashlib.sha256(seed.encode()).digest())

    @staticmethod
    def _metadata(group_id: int) -> KeyMetadata:
        return KeyMetadata(
            model_name="lookup-prefix",
            dp_size=1,
            dp_rank=-1,
            tp_size=1,
            tp_rank=-1,
            pcp_size=1,
            pcp_rank=0,
            dcp_size=1,
            dcp_rank=0,
            pp_size=1,
            pp_rank=0,
            group_id=group_id,
        )

    def _worker(self, response, *, repeated_hashes: bool = False):
        if repeated_hashes:
            wide_hashes = [self._hash("wide-repeat")] * 3
            fine_hashes = [self._hash("fine-repeat")] * 6
        else:
            wide_hashes = [self._hash(f"wide-{index}") for index in range(3)]
            fine_hashes = [self._hash(f"fine-{index}") for index in range(6)]
        wide_spec, fine_spec = object(), object()

        class _Coordinator:
            lcm_block_size = 128

            @staticmethod
            def store_mask(_token_len):
                return [[True] * 3, [True] * 6]

            @staticmethod
            def block_hashes_for_spec(_block_hashes, spec):
                return wide_hashes if spec is wide_spec else fine_hashes

            @staticmethod
            def find_longest_cache_hit(_block_hashes, _token_len, pool):
                hit = 0
                for chunk in range(3):
                    required = (
                        (0, wide_hashes[chunk]),
                        (1, fine_hashes[chunk * 2]),
                        (1, fine_hashes[chunk * 2 + 1]),
                    )
                    if not all(
                        pool.get_cached_block(block_hash, [group_id])
                        for group_id, block_hash in required
                    ):
                        break
                    hit += 128
                return [[], []], hit

        class _Client:
            def __init__(self):
                self.calls: list[list[bytes]] = []

            def batch_exist(self, keys):
                self.calls.append(list(keys))
                if isinstance(response, Exception):
                    raise response
                return response

        client = _Client()
        worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
        worker.client = client
        worker.coord = _Coordinator()
        worker.token_dbs = [
            SimpleNamespace(block_size=128, metadata=self._metadata(0)),
            SimpleNamespace(block_size=64, metadata=self._metadata(1)),
        ]
        worker._kv_cache_groups = [
            SimpleNamespace(kv_cache_spec=wide_spec),
            SimpleNamespace(kv_cache_spec=fine_spec),
        ]
        worker.tp_size = 1
        worker.num_kv_head = 1
        worker.pp_size = 1
        worker._record_kv_connector_operation = lambda *_args, **_kwargs: None
        return worker, client

    def test_stops_at_first_incomplete_cross_group_chunk_and_ignores_later_hits(
        self,
    ) -> None:
        # Candidate order is three wide-group chunks followed by six fine-group
        # subchunks. Logical chunk 1 is incomplete only in the fine group;
        # logical chunk 2 is an isolated later hit and must not be loaded.
        worker, client = self._worker(
            [1, 1, 1, 1, 1, 1, 0, 1, 1]
        )
        block_hashes = [self._hash(f"scheduler-{index}") for index in range(6)]
        self.assertEqual(worker.lookup(384, block_hashes), 128)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0]), 9)

        first_missing_worker, _ = self._worker(
            [1, 1, 1, 1, 0, 1, 1, 1, 1]
        )
        self.assertEqual(first_missing_worker.lookup(384, block_hashes), 0)

    def test_repeated_equal_hashes_do_not_mask_an_incomplete_occurrence(self) -> None:
        worker, _client = self._worker(
            # Fine-group occurrence 0 is absent; identical hashes in later
            # chunks are present and must not inflate this occurrence's count.
            [1, 1, 1, 0, 1, 1, 1, 1, 1],
            repeated_hashes=True,
        )
        block_hashes = [self._hash("scheduler-repeat")] * 6
        self.assertEqual(worker.lookup(384, block_hashes), 0)

    def test_malformed_exist_result_lengths_fail_closed(self) -> None:
        block_hashes = [self._hash(f"scheduler-{index}") for index in range(6)]
        for malformed in (
            [1] * 8,
            [1] * 10,
            None,
        ):
            with self.subTest(result=malformed):
                worker, _client = self._worker(malformed)
                self.assertEqual(worker.lookup(384, block_hashes), 0)


class _BlockingReceiveClient:
    def __init__(self, request_keys: dict[bytes, str]) -> None:
        self.request_keys = request_keys
        self.started = {
            request_id: threading.Event() for request_id in request_keys.values()
        }
        self.release = {
            request_id: threading.Event() for request_id in request_keys.values()
        }
        self.returned = {
            request_id: threading.Event() for request_id in request_keys.values()
        }
        self.raise_for: set[str] = set()
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def batch_get_auto_sg(self, keys, addresses, sizes):
        self.assert_one_aligned_object(keys, addresses, sizes)
        request_id = self.request_keys[bytes(keys[0])]
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started[request_id].set()
        try:
            if request_id in self.raise_for:
                raise RuntimeError(f"injected receive failure for {request_id}")
            if not self.release[request_id].wait(timeout=5):
                raise TimeoutError(f"test did not release {request_id}")
            return [1], [64]
        finally:
            with self._lock:
                self.active -= 1
            self.returned[request_id].set()

    @staticmethod
    def assert_one_aligned_object(keys, addresses, sizes) -> None:
        if len(keys) != 1 or len(addresses) != 1 or len(sizes) != 1:
            raise AssertionError("expected one logical object")
        if len(addresses[0]) != 1 or list(sizes[0]) != [64]:
            raise AssertionError("key/descriptor geometry was not aligned")


class ReceiveConcurrencyTest(unittest.TestCase):
    @staticmethod
    def _metadata() -> KeyMetadata:
        return KeyMetadata(
            model_name="receive-concurrency",
            dp_size=1,
            dp_rank=-1,
            tp_size=1,
            tp_rank=-1,
            pcp_size=1,
            pcp_rank=0,
            dcp_size=1,
            dcp_rank=0,
            pp_size=1,
            pp_rank=0,
        )

    @staticmethod
    def _hash(request_id: str) -> BlockHash:
        return BlockHash(hashlib.sha256(request_id.encode()).digest())

    def test_recv_worker_configuration_is_positive_and_bounded(self) -> None:
        common = (
            object(),
            SimpleNamespace(),
            [],
            64,
            0,
            threading.Event(),
        )
        for invalid in (0, 33, True, "four"):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    KVCacheStoreRecvingThread(
                        *common, recv_workers=invalid  # type: ignore[arg-type]
                    )

        receiver = KVCacheStoreRecvingThread(
            *common, recv_workers="4"  # type: ignore[arg-type]
        )
        self.assertEqual(receiver.recv_workers, 4)
        self.assertEqual(len(receiver.worker_threads), 4)
        receiver.stop()

    def _requests(self, request_ids: tuple[str, ...]) -> tuple[
        dict[str, ReqMeta], dict[bytes, str]
    ]:
        requests: dict[str, ReqMeta] = {}
        request_keys: dict[bytes, str] = {}
        metadata = self._metadata()
        for index, request_id in enumerate(request_ids):
            block_hash = self._hash(request_id)
            request_keys[PoolKey(metadata, block_hash.hex()).to_bytes()] = request_id
            requests[request_id] = ReqMeta(
                req_id=request_id,
                token_len_chunk=64,
                block_ids=([1000 + index * 17],),
                block_hashes=[block_hash],
                load_spec=LoadSpec(
                    vllm_cached_tokens=0,
                    kvpool_cached_tokens=64,
                    can_load=True,
                    token_len=64,
                ),
            )
        return requests, request_keys

    def _receiver(
        self,
        request_ids: tuple[str, ...],
        *,
        recv_workers: int = 1,
        queue_capacity: int = 8,
    ) -> tuple[
        KVCacheStoreRecvingThread,
        _BlockingReceiveClient,
        dict[str, ReqMeta],
        dict[str, threading.Event],
        dict[str, int],
        list[str],
    ]:
        requests, request_keys = self._requests(request_ids)
        metadata = self._metadata()

        class _Database:
            block_size = 64
            _seg_layout = [(0x10000, 0x100, 64)]

            @staticmethod
            def process_tokens(_token_len, block_hashes, _mask_num):
                return [(0, 64, PoolKey(metadata, block_hashes[0].hex()))]

            @staticmethod
            def descriptor_shape(_start, _end, block_ids):
                return 1, 64, block_ids[0]

            @staticmethod
            def fill_descriptors(
                _start, _end, block_ids, pointers, sizes, offset=0
            ):
                pointers[offset] = 0x10000 + block_ids[0] * 0x100
                sizes[offset] = 64
                return offset + 1

            @staticmethod
            def prepare_value(_start, _end, block_ids):
                return [0x10000 + block_ids[0] * 0x100], [64], block_ids[0]

        client = _BlockingReceiveClient(request_keys)
        receiver = KVCacheStoreRecvingThread(
            client,
            SimpleNamespace(load_mask=lambda _hashes, _token_len: [[True]]),
            [_Database()],
            block_size=64,
            tp_rank=0,
            ready_event=threading.Event(),
            queue_capacity=queue_capacity,
            recv_workers=recv_workers,
        )
        done_events = {
            request_id: threading.Event() for request_id in request_ids
        }
        transition_counts = {request_id: 0 for request_id in request_ids}
        transition_order: list[str] = []
        transition_lock = threading.Lock()
        terminalize = receiver._terminalize_locked

        def record_terminal(state, *, failed):
            request = state.request
            request_id = request.req_id if request is not None else None
            transitioned = terminalize(state, failed=failed)
            if transitioned and request_id is not None:
                with transition_lock:
                    transition_counts[request_id] += 1
                    transition_order.append(request_id)
                    done_events[request_id].set()
            return transitioned

        receiver._terminalize_locked = record_terminal
        receiver.start()
        self.assertTrue(receiver.ready_event.wait(timeout=1))
        return (
            receiver,
            client,
            requests,
            done_events,
            transition_counts,
            transition_order,
        )

    @staticmethod
    def _wait(event: threading.Event, label: str) -> None:
        if not event.wait(timeout=2):
            raise AssertionError(f"timed out waiting for {label}")

    def test_recv_workers_overlap_complete_out_of_order_and_bound_active_calls(
        self,
    ) -> None:
        request_ids = ("slow", "fast", "queued", "rejected")
        (
            receiver,
            client,
            requests,
            done,
            counts,
            order,
        ) = self._receiver(request_ids, recv_workers=2, queue_capacity=1)
        try:
            self.assertTrue(receiver.add_request(requests["slow"]))
            self._wait(client.started["slow"], "slow native GET")
            self.assertTrue(receiver.add_request(requests["fast"]))
            self._wait(client.started["fast"], "fast native GET")

            self.assertTrue(receiver.add_request(requests["queued"]))
            self.assertFalse(receiver.add_request(requests["rejected"]))
            self._wait(done["rejected"], "saturated request terminalization")
            self.assertFalse(client.started["queued"].is_set())
            self.assertLessEqual(client.max_active, 2)

            client.release["fast"].set()
            self._wait(done["fast"], "fast completion")
            self._wait(client.started["queued"], "queued request start")
            self.assertFalse(done["slow"].is_set())
            client.release["queued"].set()
            self._wait(done["queued"], "queued completion")
            client.release["slow"].set()
            self._wait(done["slow"], "slow completion")

            self.assertLess(order.index("fast"), order.index("slow"))
            self.assertEqual(counts, {request_id: 1 for request_id in request_ids})
            self.assertEqual(
                receiver.get_and_clear_block_ids_with_load_errors(),
                {requests["rejected"].block_ids[0][0]},
            )
            self.assertEqual(
                receiver.get_and_clear_finished_requests(), set(request_ids)
            )
            self.assertEqual(receiver.get_and_clear_finished_requests(), set())
        finally:
            for event in client.release.values():
                event.set()
            receiver.stop(cancel_pending=True)

    def test_default_single_worker_preserves_compatibility_without_overlap(
        self,
    ) -> None:
        receiver, client, requests, done, counts, _order = self._receiver(
            ("first", "second")
        )
        try:
            self.assertTrue(receiver.add_request(requests["first"]))
            self._wait(client.started["first"], "first native GET")
            self.assertTrue(receiver.add_request(requests["second"]))
            self.assertFalse(client.started["second"].is_set())
            self.assertEqual(client.max_active, 1)

            client.release["first"].set()
            self._wait(done["first"], "first completion")
            self._wait(client.started["second"], "second native GET")
            client.release["second"].set()
            self._wait(done["second"], "second completion")
            self.assertEqual(counts, {"first": 1, "second": 1})
            self.assertEqual(client.max_active, 1)
        finally:
            for event in client.release.values():
                event.set()
            receiver.stop(cancel_pending=True)

    def test_cancel_active_and_queued_requests_fences_each_block_once(self) -> None:
        receiver, client, requests, done, counts, _order = self._receiver(
            ("active", "queued"), recv_workers=1
        )
        try:
            self.assertTrue(receiver.add_request(requests["active"]))
            self._wait(client.started["active"], "active native GET")
            self.assertTrue(receiver.add_request(requests["queued"]))

            receiver.cancel_requests({"active", "queued"}, wait=False)
            self._wait(done["queued"], "queued cancellation")
            self.assertFalse(done["active"].is_set())
            self.assertFalse(client.started["queued"].is_set())

            client.release["active"].set()
            self._wait(done["active"], "active cancellation fence")
            self.assertEqual(counts, {"active": 1, "queued": 1})
            self.assertEqual(
                receiver.get_and_clear_block_ids_with_load_errors(),
                {
                    requests["active"].block_ids[0][0],
                    requests["queued"].block_ids[0][0],
                },
            )
            receiver.cancel_requests({"active", "queued"}, wait=True)
            self.assertEqual(counts, {"active": 1, "queued": 1})
        finally:
            for event in client.release.values():
                event.set()
            receiver.stop(cancel_pending=True)

    def test_preemption_hook_waits_for_only_its_inflight_receive(self) -> None:
        receiver, client, requests, done, counts, _order = self._receiver(
            ("preempted",), recv_workers=1
        )
        self.assertTrue(receiver.add_request(requests["preempted"]))
        self._wait(client.started["preempted"], "preempted native GET")

        cancel_entered = threading.Event()
        original_cancel = receiver.cancel_requests

        def observed_cancel(req_ids, *, wait=True, **kwargs):
            cancel_entered.set()
            return original_cancel(req_ids, wait=wait, **kwargs)

        receiver.cancel_requests = observed_cancel
        send_thread = SimpleNamespace(
            delete_finished_stored_request=lambda _req_id: None,
            wait_for_inflight_put=lambda _req_id: True,
        )
        worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
        worker.kv_send_thread = send_thread
        worker.kv_recv_thread = receiver
        returned = threading.Event()

        def preempt() -> None:
            worker.start_load_kv(SimpleNamespace(preempted_req_ids={"preempted"}))
            returned.set()

        thread = threading.Thread(target=preempt)
        thread.start()
        try:
            self._wait(cancel_entered, "receive preemption hook")
            self.assertFalse(returned.is_set())
            self.assertFalse(done["preempted"].is_set())
            client.release["preempted"].set()
            self._wait(returned, "preemption fence return")
            thread.join(timeout=1)
            self._wait(done["preempted"], "preempted terminalization")
            self.assertEqual(counts["preempted"], 1)
            self.assertEqual(
                receiver.get_and_clear_block_ids_with_load_errors(), set()
            )
        finally:
            client.release["preempted"].set()
            thread.join(timeout=1)
            receiver.stop(cancel_pending=True)

    def test_finished_abort_waits_for_inflight_receive_before_acknowledging(
        self,
    ) -> None:
        receiver, client, requests, done, counts, _order = self._receiver(
            ("aborted",), recv_workers=1
        )
        self.assertTrue(receiver.add_request(requests["aborted"]))
        self._wait(client.started["aborted"], "aborted native GET")

        cancel_entered = threading.Event()
        original_cancel = receiver.cancel_requests

        def observed_cancel(req_ids, *, wait=True, **kwargs):
            cancel_entered.set()
            return original_cancel(req_ids, wait=wait, **kwargs)

        receiver.cancel_requests = observed_cancel
        worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
        worker.kv_recv_thread = receiver
        worker.kv_send_thread = None
        worker.kv_role = "kv_consumer"
        worker.load_async = True
        worker.tp_rank = 0
        returned = threading.Event()
        result: list[tuple[set[str], set[str]]] = []

        def finish_abort() -> None:
            result.append(
                worker.get_finished(
                    {"aborted"},
                    SimpleNamespace(requests=[], preempted_req_ids=set()),
                )
            )
            returned.set()

        thread = threading.Thread(target=finish_abort)
        thread.start()
        try:
            self._wait(cancel_entered, "finished-request receive cancellation")
            self.assertFalse(returned.is_set())
            self.assertFalse(done["aborted"].is_set())
            client.release["aborted"].set()
            self._wait(returned, "finished-request receive fence")
            thread.join(timeout=1)
            self.assertEqual(result, [(set(), {"aborted"})])
            self.assertEqual(counts["aborted"], 1)
            self.assertEqual(
                receiver.get_and_clear_block_ids_with_load_errors(), set()
            )
        finally:
            client.release["aborted"].set()
            thread.join(timeout=1)
            receiver.stop(cancel_pending=True)

    def test_stop_drains_when_requested_and_native_exception_is_fail_closed(
        self,
    ) -> None:
        receiver, client, requests, done, counts, order = self._receiver(
            ("failing", "drained"), recv_workers=1
        )
        thread: threading.Thread | None = None
        try:
            client.raise_for.add("failing")
            self.assertTrue(receiver.add_request(requests["failing"]))
            self._wait(done["failing"], "exception terminalization")
            self.assertTrue(receiver.add_request(requests["drained"]))
            self._wait(client.started["drained"], "drained native GET")

            stopped = threading.Event()

            def drain() -> None:
                receiver.stop(cancel_pending=False)
                stopped.set()

            thread = threading.Thread(target=drain)
            thread.start()
            self.assertFalse(stopped.is_set())
            client.release["drained"].set()
            self._wait(stopped, "draining stop")
            thread.join(timeout=1)
            self._wait(done["drained"], "drained request completion")

            self.assertEqual(counts, {"failing": 1, "drained": 1})
            self.assertEqual(order, ["failing", "drained"])
            self.assertEqual(
                receiver.get_and_clear_block_ids_with_load_errors(),
                {requests["failing"].block_ids[0][0]},
            )
        finally:
            for event in client.release.values():
                event.set()
            if thread is not None:
                thread.join(timeout=1)
            receiver.stop(cancel_pending=True)

    def test_close_cancels_queued_and_fences_active_before_returning(self) -> None:
        receiver, client, requests, done, counts, _order = self._receiver(
            ("active", "queued", "after-close"), recv_workers=1
        )
        thread: threading.Thread | None = None
        try:
            self.assertTrue(receiver.add_request(requests["active"]))
            self._wait(client.started["active"], "active native GET")
            self.assertTrue(receiver.add_request(requests["queued"]))
            stopped = threading.Event()

            def close() -> None:
                receiver.close(cancel_pending=True)
                stopped.set()

            thread = threading.Thread(target=close)
            thread.start()
            self._wait(done["queued"], "queued close cancellation")
            self.assertFalse(stopped.is_set())
            self.assertFalse(done["active"].is_set())
            client.release["active"].set()
            self._wait(stopped, "close after active fence")
            thread.join(timeout=1)
            self._wait(done["active"], "active close cancellation")

            self.assertFalse(receiver.add_request(requests["after-close"]))
            self._wait(done["after-close"], "post-close rejection")
            self.assertEqual(
                counts, {"active": 1, "queued": 1, "after-close": 1}
            )
            self.assertEqual(
                receiver.get_and_clear_block_ids_with_load_errors(),
                {requests["after-close"].block_ids[0][0]},
            )
        finally:
            client.release["active"].set()
            if thread is not None:
                thread.join(timeout=1)
            receiver.stop(cancel_pending=True)


class LookupAccountingTest(unittest.TestCase):
    @staticmethod
    def _hash(seed: str) -> BlockHash:
        return BlockHash(hashlib.sha256(seed.encode()).digest())

    def test_lookup_probes_one_v2_key_per_rank_and_keeps_partial_prefix(self) -> None:
        hashes = [self._hash("one"), self._hash("two")]
        metadata = KeyMetadata(
            model_name="model",
            dp_size=1,
            dp_rank=-1,
            tp_size=2,
            tp_rank=0,
            pcp_size=1,
            pcp_rank=0,
            dcp_size=1,
            dcp_rank=0,
            pp_size=2,
            pp_rank=0,
            group_id=0,
        )

        class _Client:
            _sg_segs_cache = 2

            def __init__(self) -> None:
                self.keys: list[bytes] = []

            def batch_exist(self, keys):
                self.keys = list(keys)
                # Every rank replica of chunk 0 is present. Chunk 1 is only
                # partially replicated and therefore cannot extend the prefix.
                return [1, 1, 1, 1, 1, 1, 1, 0]

        client = _Client()
        records: list[dict[str, object]] = []
        db = SimpleNamespace(
            block_size=64,
            metadata=metadata,
            _seg_layout=[1, 2, 3],
        )

        def _find_longest(values, _token_len, pool):
            hit = 0
            for value in values:
                if pool.get_cached_block(value, [0]) is None:
                    break
                hit += 64
            return None, hit

        coord = SimpleNamespace(
            lcm_block_size=64,
            store_mask=lambda _token_len: [[True, True]],
            block_hashes_for_spec=lambda values, _spec: values,
            find_longest_cache_hit=_find_longest,
        )
        worker = SimpleNamespace(
            coord=coord,
            token_dbs=[db],
            client=client,
            tp_size=2,
            num_kv_head=2,
            pp_size=2,
            _kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())],
            _record_kv_connector_operation=lambda operation, duration, num_keys, **kwargs: records.append(
                {
                    "operation": operation,
                    "duration": duration,
                    "num_keys": num_keys,
                    **kwargs,
                }
            ),
        )

        self.assertEqual(DfkvStoreWorker.lookup(worker, 128, hashes), 64)
        expected_keys = [
            PoolKey(
                replace(metadata, tp_rank=tp, pp_rank=pp),
                value.hex(),
            ).to_bytes()
            for value in hashes
            for tp in range(2)
            for pp in range(2)
        ]
        self.assertEqual(client.keys, expected_keys)
        self.assertEqual(len(set(client.keys)), 8)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["operation"], "lookup_exists")
        self.assertEqual(records[0]["num_keys"], 8)
        self.assertEqual(records[0]["num_logical_keys"], 2)


if __name__ == "__main__":
    unittest.main()
