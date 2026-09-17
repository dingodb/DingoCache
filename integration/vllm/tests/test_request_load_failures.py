"""Request-level receives publish one fenced completion/failure snapshot."""

import ctypes
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.v1.core.kv_cache_utils import BlockHash

from dfkv_vllm.connector import DfkvStoreConnector
from dfkv_vllm.data import (
    ChunkedTokenDatabase,
    DfkvStoreConnectorMetadata,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from dfkv_vllm.worker import DfkvStoreWorker, KVCacheStoreRecvingThread


BLOCK = 16


def request(req_id="load", block_id=1):
    return ReqMeta(
        req_id=req_id, token_len_chunk=BLOCK, block_ids=([block_id],),
        block_hashes=[BlockHash(b"x" * 32)],
        load_spec=LoadSpec(0, BLOCK, True, token_len=BLOCK),
    )


class MemoryClient:
    def __init__(self, outcome="ok"):
        self.outcome = outcome
        self.calls = 0
        self.thread_ids = []

    def batch_get_auto_sg(self, keys, pointers, capacities):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        for ptrs, caps in zip(pointers, capacities, strict=True):
            for ptr, cap in zip(ptrs, caps, strict=True):
                ctypes.memset(ptr, 17, cap)
        if self.outcome == "native":
            raise RuntimeError("native GET failed after a partial write")
        if self.outcome == "incomplete":
            return [], []
        if self.outcome == "miss":
            return [False] * len(keys), [0] * len(keys)
        length = {"short": BLOCK - 1, "oversized": BLOCK + 1}.get(
            self.outcome, BLOCK,
        )
        return [True] * len(keys), [length] * len(keys)


@pytest.fixture
def make_receiver(monkeypatch):
    # The fixture tests native-call ownership with CPU pointers. GPU ordering
    # is exercised separately by explicitly controlled owner-device fences.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    receivers = []
    pools = []

    def make(client=None, *, request_level=True, workers=1, capacity=4):
        pool = ctypes.create_string_buffer(4 * BLOCK)
        pools.append(pool)
        metadata = KeyMetadata(
            model_name="request-recovery", dp_size=1, dp_rank=-1,
            tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0,
            dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0,
        )
        database = ChunkedTokenDatabase(metadata, BLOCK, hash_block_size=BLOCK)
        database.set_seg_layout([(ctypes.addressof(pool), BLOCK, BLOCK)])
        receiver = KVCacheStoreRecvingThread(
            client, SimpleNamespace(load_mask=lambda hashes, length: [[True]]),
            [database], BLOCK, tp_rank=0, ready_event=threading.Event(),
            request_level_loads=request_level, recv_workers=workers,
            queue_capacity=capacity,
        )
        receivers.append(receiver)
        return receiver, pool

    yield make
    for receiver in receivers:
        receiver.stop(cancel_pending=True)


@pytest.mark.parametrize(
    "outcome", ["miss", "short", "oversized", "incomplete", "native", "no-client", "geometry"],
)
@pytest.mark.parametrize("inline", [False, True])
def test_all_failures_are_terminal_request_outcomes(make_receiver, outcome, inline):
    receiver, _ = make_receiver(None if outcome == "no-client" else MemoryClient(outcome))
    load = request(block_id=0 if outcome == "geometry" else 1)
    if inline:
        receiver.load_request_sync(load)
    else:
        receiver.start()
        assert receiver.add_request(load)
        receiver.request_queue.join()
    assert receiver.get_and_clear_block_ids_with_load_errors() == set()
    assert receiver.get_and_clear_receive_results() == ({"load"}, {"load"})
    assert receiver.get_and_clear_receive_results() == (set(), set())


def test_full_attention_sync_keeps_block_errors_without_async_completion(make_receiver):
    receiver, _ = make_receiver(MemoryClient("miss"), request_level=False)
    receiver.load_request_sync(request())
    assert receiver.get_and_clear_block_ids_with_load_errors() == {1}
    assert receiver.get_and_clear_receive_results() == (set(), set())


def test_rejection_and_shutdown_cleanup_preserve_paired_outcomes(make_receiver):
    receiver, _ = make_receiver(capacity=1)
    assert receiver.add_request(request("queued"))
    assert not receiver.add_request(request("rejected"))
    receiver.stop(cancel_pending=True)
    # Shutdown clears ownership states, not the pending failure snapshot.
    assert receiver.get_and_clear_receive_results() == (
        {"queued", "rejected"}, {"rejected"},
    )
    assert not receiver.add_request(request("closed"))
    assert not receiver.add_request(request("closed"))
    assert receiver.get_and_clear_receive_results() == ({"closed"}, {"closed"})
    assert receiver.get_and_clear_receive_results() == (set(), set())
    assert receiver.get_and_clear_block_ids_with_load_errors() == set()


def test_native_exception_waits_for_owner_gpu_before_atomic_drain(make_receiver, monkeypatch):
    monkeypatch.setenv("DFKV_GPU_LOAD_FENCE", "0")
    receiver, _ = make_receiver(MemoryClient("native"))
    receiver._cuda_device = 7
    fence_entered = threading.Event()
    release_fence = threading.Event()
    devices = []

    def synchronize(device):
        devices.append(device)
        fence_entered.set()
        if not release_fence.wait(5):
            raise TimeoutError("test did not release the native completion fence")

    monkeypatch.setattr("torch.cuda.synchronize", synchronize)
    receiver.start()
    try:
        receiver.add_request(request())
        assert fence_entered.wait(5)
        # Native failure has been recorded, but its GPU ownership is not done.
        assert receiver.get_and_clear_receive_results() == (set(), set())
        assert receiver.get_and_clear_block_ids_with_load_errors() == set()
    finally:
        release_fence.set()
        receiver.request_queue.join()
    assert devices == [7]
    assert receiver.get_and_clear_receive_results() == ({"load"}, {"load"})


@pytest.mark.parametrize("native_failure", [False, True])
@pytest.mark.parametrize("fail_closed", [False, True])
@pytest.mark.parametrize("inline", [False, True])
def test_cancellation_never_releases_active_native_ownership(
    make_receiver, native_failure, fail_closed, inline,
):
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    class DelayedClient(MemoryClient):
        def batch_get_auto_sg(self, *args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release native GET")
            return super().batch_get_auto_sg(*args)

    receiver, _ = make_receiver(DelayedClient("native" if native_failure else "ok"))
    load_thread = None
    if inline:
        load_thread = threading.Thread(
            target=receiver.load_request_sync, args=(request(),),
        )
        load_thread.start()
    else:
        receiver.start()
        receiver.add_request(request())
    assert entered.wait(5)

    def cancel():
        receiver.cancel_requests({"load"}, wait=True, fail_closed=fail_closed)
        cancelled.set()

    cancel_thread = threading.Thread(target=cancel)
    cancel_thread.start()
    try:
        assert not cancelled.wait(0.05)
        assert receiver.get_and_clear_receive_results() == (set(), set())
    finally:
        release.set()
        cancel_thread.join(5)
        receiver.request_queue.join()
        if load_thread is not None:
            load_thread.join(5)
            assert not load_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert cancelled.is_set()
    assert receiver.get_and_clear_receive_results() == (
        {"load"}, {"load"} if native_failure or fail_closed else set(),
    )
    assert receiver.get_and_clear_block_ids_with_load_errors() == set()


def test_independent_receives_finish_out_of_order(make_receiver):
    slow_entered = threading.Event()
    release_slow = threading.Event()
    fast_done = threading.Event()

    class OutOfOrderClient(MemoryClient):
        def batch_get_auto_sg(self, keys, pointers, capacities):
            if not slow_entered.is_set():
                slow_entered.set()
                if not release_slow.wait(5):
                    raise TimeoutError("test did not release the slow GET")
                raise RuntimeError("slow receive failed")
            result = super().batch_get_auto_sg(keys, pointers, capacities)
            fast_done.set()
            return result

    receiver, _ = make_receiver(OutOfOrderClient(), workers=2)
    receiver.start()
    receiver.add_request(request("slow", 1))
    assert slow_entered.wait(5)
    receiver.add_request(request("fast", 2))
    try:
        assert fast_done.wait(5)
        # Join just the fast request's ownership fence, not the shared queue.
        receiver.cancel_requests({"fast"}, wait=True, fail_closed=False)
        assert receiver.get_and_clear_receive_results() == ({"fast"}, set())
    finally:
        release_slow.set()
        receiver.request_queue.join()
    assert receiver.get_and_clear_receive_results() == ({"slow"}, {"slow"})


def connector_for(receiver, *, inline):
    worker = object.__new__(DfkvStoreWorker)
    worker.load_async = not inline
    worker.request_level_loads = True
    worker.kv_recv_thread = receiver
    worker.kv_send_thread = None
    worker.kv_role = "kv_consumer"
    connector = object.__new__(DfkvStoreConnector)
    connector.connector_worker = worker
    connector._shutdown_condition = threading.Condition()
    connector._shutdown = False
    connector._inflight_calls = 0
    metadata = DfkvStoreConnectorMetadata({"load"}, set())
    metadata.add_request(request())
    connector.bind_connector_metadata(metadata)
    return connector


def test_inline_parked_load_fences_both_sides_and_reports_once(make_receiver, monkeypatch):
    client = MemoryClient("miss")
    receiver, _ = make_receiver(client)
    receiver._cuda_device = 7
    calls_seen_at_fence = []

    def synchronize(device):
        assert device == 7
        calls_seen_at_fence.append(client.calls)

    monkeypatch.setattr("torch.cuda.synchronize", synchronize)
    connector = connector_for(receiver, inline=True)
    connector.start_load_kv(None)
    assert client.thread_ids == [threading.get_ident()]
    assert calls_seen_at_fence == [0, 1]
    result = connector.get_transfer_results(set())
    assert result.finished_recving == result.failed_recving == {"load"}
    assert result.finished_sending == set()
    # Repeated polling/hooks with the same metadata must not resubmit writes.
    connector.start_load_kv(None)
    result = connector.get_transfer_results(set())
    assert result.finished_recving == result.failed_recving == set()
    assert client.calls == 1
    assert connector._inflight_calls == 0
    assert connector.get_block_ids_with_load_errors() == set()


def test_pool_workers_emit_one_completion_and_never_resubmit_polled_metadata(make_receiver):
    client = MemoryClient()
    receiver, pool = make_receiver(client, workers=3)
    connector = connector_for(receiver, inline=False)
    receiver.start()
    connector.start_load_kv(None)
    assert client.calls == 0
    first = connector.get_transfer_results(set())
    receiver.request_queue.join()
    second = connector.get_transfer_results(set())
    assert first.finished_recving.isdisjoint(second.finished_recving)
    assert first.finished_recving | second.finished_recving == {"load"}
    assert first.failed_recving == second.failed_recving == set()
    assert ctypes.string_at(ctypes.addressof(pool) + BLOCK, BLOCK) == bytes([17]) * BLOCK
    assert client.calls == 1
    assert connector.get_transfer_results(set()).finished_recving == set()


def test_connector_failure_update_is_not_skipped_without_kv_events():
    failed = set()
    connector = object.__new__(DfkvStoreConnector)
    connector.connector_scheduler = SimpleNamespace(
        update_connector_output=lambda output: failed.update(output.failed_recving),
    )
    connector._kv_cache_events = None
    connector.update_connector_output(SimpleNamespace(
        failed_recving={"load"}, kv_cache_events=None,
    ))
    assert failed == {"load"}
