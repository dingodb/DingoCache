"""Bounded transfer queue and deterministic worker-shutdown contracts."""

import threading
import types
import unittest

try:
    from dfkv_vllm.worker import (
        DfkvStoreWorker,
        KVCacheStoreSendingThread,
        KVTransferThread,
        _parse_transfer_queue_capacity,
    )

    HAVE_VLLM = True
except ImportError:  # pragma: no cover - vllm not installed
    HAVE_VLLM = False
    KVTransferThread = object  # type: ignore[assignment,misc]


def _request(req_id: str):
    return types.SimpleNamespace(req_id=req_id, block_ids=([1, 2],))


@unittest.skipUnless(HAVE_VLLM, "requires vllm (dfkv_vllm.worker imports it)")
class TransferQueueLifecycleTest(unittest.TestCase):
    def setUp(self):
        self._transfers = []

    def tearDown(self):
        for transfer in self._transfers:
            if hasattr(transfer, "release"):
                transfer.release.set()
            transfer.stop(cancel_pending=True)

    class BlockingThread(KVTransferThread):
        def __init__(self, capacity: int = 1):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.cancelled = threading.Event()
            self.handled: list[str] = []
            super().__init__(
                client=None,
                token_databases=[],
                block_size=16,
                tp_rank=0,
                ready_event=threading.Event(),
                name="BlockingTransferThread",
                queue_capacity=capacity,
            )

        def _handle_request(self, req_meta):
            self.entered.set()
            self.release.wait(5)
            self.handled.append(req_meta.req_id)

        def _cancel_request(self, req_meta):  # noqa: ANN001
            super()._cancel_request(req_meta)
            self.cancelled.set()

    def test_saturation_rejects_without_exceeding_bound(self):
        transfer = self.BlockingThread(capacity=1)
        self._transfers.append(transfer)
        transfer.start()
        self.assertTrue(transfer.add_request(_request("active")))
        self.assertTrue(transfer.entered.wait(1))
        self.assertTrue(transfer.add_request(_request("queued")))
        self.assertFalse(transfer.add_request(_request("rejected")))
        self.assertEqual(transfer.request_queue.qsize(), 1)
        self.assertEqual(transfer.get_and_clear_finished_requests(), {"rejected"})
        transfer.release.set()
        transfer.stop(cancel_pending=True)

    def test_stop_cancels_queued_work_and_joins_active_worker(self):
        transfer = self.BlockingThread(capacity=2)
        transfer.start()
        self._transfers.append(transfer)
        self.assertTrue(transfer.add_request(_request("active")))
        self.assertTrue(transfer.entered.wait(1))
        self.assertTrue(transfer.add_request(_request("queued")))

        stopped = threading.Event()

        def stop():
            transfer.stop(cancel_pending=True)
            stopped.set()

        stopper = threading.Thread(target=stop)
        stopper.start()
        self.assertTrue(transfer.cancelled.wait(1))
        self.assertFalse(stopped.wait(0.1), "stop must wait for active native work")
        self.assertEqual(transfer.get_and_clear_finished_requests(), {"queued"})
        transfer.release.set()
        stopper.join(2)
        self.assertTrue(stopped.is_set())
        self.assertFalse(transfer.is_alive())
        self.assertEqual(transfer.handled, ["active"])

    def test_stop_can_drain_all_accepted_work(self):
        transfer = self.BlockingThread(capacity=2)
        self._transfers.append(transfer)
        transfer.start()
        self.assertTrue(transfer.add_request(_request("active")))
        self.assertTrue(transfer.entered.wait(1))
        self.assertTrue(transfer.add_request(_request("queued")))

        stopper = threading.Thread(
            target=transfer.stop, kwargs={"cancel_pending": False}
        )
        stopper.start()
        transfer.release.set()
        stopper.join(2)

        self.assertFalse(stopper.is_alive())
        self.assertFalse(transfer.is_alive())
        self.assertEqual(transfer.handled, ["active", "queued"])
        self.assertEqual(transfer.get_and_clear_finished_requests(), set())

    def test_no_work_is_accepted_after_close(self):
        transfer = self.BlockingThread()
        transfer.start()
        self._transfers.append(transfer)
        transfer.stop(cancel_pending=True)
        self.assertFalse(transfer.add_request(_request("late")))
        self.assertEqual(transfer.get_and_clear_finished_requests(), {"late"})

    def test_saturated_save_releases_finish_fence(self):
        coord = types.SimpleNamespace(lcm_block_size=16)
        transfer = KVCacheStoreSendingThread(
            client=None,
            coord=coord,
            token_databases=[],
            block_size=16,
            tp_rank=0,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            queue_capacity=1,
        )
        self._transfers.append(transfer)
        first = _request("first")
        second = _request("second")
        transfer.add_stored_request(first.req_id)
        self.assertTrue(transfer.add_request(first))
        transfer.add_stored_request(second.req_id)
        self.assertFalse(transfer.add_request(second))
        self.assertEqual(transfer.stored_requests[second.req_id], 0)
        transfer.stop(cancel_pending=True)
        self.assertEqual(transfer.stored_requests[first.req_id], 0)

    def test_queue_capacity_is_bounded_and_fail_closed(self):
        self.assertEqual(_parse_transfer_queue_capacity("256"), 256)
        for value in (0, 65537, True, 1.5, "unbounded"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _parse_transfer_queue_capacity(value)


@unittest.skipUnless(HAVE_VLLM, "requires vllm (dfkv_vllm.worker imports it)")
class WorkerCloseTest(unittest.TestCase):
    class FakeTransfer:
        def __init__(self):
            self.stops = 0

        def stop(self, *, cancel_pending):
            assert cancel_pending
            self.stops += 1

    class FakeClose:
        def __init__(self):
            self.closes = 0

        def close(self):
            self.closes += 1

    def test_repeated_close_joins_resources_and_closes_native_once(self):
        worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
        worker._close_lock = threading.Lock()
        worker._lazy_client_lock = threading.Lock()
        worker._closed = False
        worker._lazy_client_kwargs = {"members": "unused"}
        worker.kv_recv_thread = self.FakeTransfer()
        worker.kv_send_thread = self.FakeTransfer()
        worker.lookup_server = self.FakeClose()
        worker.client = self.FakeClose()

        recv = worker.kv_recv_thread
        send = worker.kv_send_thread
        lookup = worker.lookup_server
        native = worker.client
        worker.close()
        worker.close()

        self.assertEqual(recv.stops, 1)
        self.assertEqual(send.stops, 1)
        self.assertEqual(lookup.closes, 1)
        self.assertEqual(native.closes, 1)
        self.assertIsNone(worker.client)
        self.assertIsNone(worker._lazy_client_kwargs)
        with self.assertRaisesRegex(RuntimeError, "worker is closed"):
            worker.register_kv_caches({"layer": object()})


if __name__ == "__main__":
    unittest.main()
