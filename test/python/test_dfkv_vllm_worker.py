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
            put_step=1,
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

            def prepare_value(self, start, *_args):
                index = start // 64
                return list(addresses[index]), list(sizes[index]), (7, 9)[index]

        class _Client:
            def __init__(self) -> None:
                self.get_call = None
                self.result = ([1, 0], [sum(sizes[0]), 0])

            def max_sg_segs(self) -> int:
                return 29

            def batch_get_auto_sg(self, get_keys, ptrs, caps):
                self.get_call = (
                    list(get_keys),
                    [list(value) for value in ptrs],
                    [list(value) for value in caps],
                )
                return self.result

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

        self.assertEqual(
            client.get_call,
            ([key.to_bytes() for key in keys], addresses, sizes),
        )
        load_errors = receiver.get_and_clear_block_ids_with_load_errors()
        self.assertNotIn(7, load_errors)
        self.assertIn(9, load_errors)
        self.assertEqual(load_errors, {9})

        # A truncated native result must be rejected before per-key indexing.
        # Fail the whole batch closed rather than silently treating the omitted
        # logical chunk as a hit.
        client.result = ([1], [sum(sizes[0])])
        receiver._handle_request(request)
        self.assertEqual(
            receiver.get_and_clear_block_ids_with_load_errors(), {7, 9}
        )


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
