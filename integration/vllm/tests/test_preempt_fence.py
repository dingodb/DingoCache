"""Preemption-fence unit tests for KVCacheStoreSendingThread.

No GPU or dfkv server needed, but importing dfkv_vllm.worker pulls in vllm
(and torch), so the whole module self-skips where vllm is absent -- same
convention as the other tests in this directory.

Run: python3 -m unittest test_preempt_fence  (from this directory, with
dfkv_vllm on PYTHONPATH, e.g. pip install -e integration/vllm)
"""

import threading
import types
import unittest

try:
    from dfkv_vllm.worker import KVCacheStoreSendingThread

    HAVE_VLLM = True
except ImportError:  # pragma: no cover - vllm not installed
    HAVE_VLLM = False


def _req_meta(req_id: str) -> "types.SimpleNamespace":
    # The fields _handle_request touches before it bails out on empty
    # token_databases (masks come from coord.store_mask; the db loop is empty).
    return types.SimpleNamespace(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=(),
        block_hashes=[],
        current_event=None,
        token_ids=None,
    )


@unittest.skipUnless(HAVE_VLLM, "requires vllm (dfkv_vllm.worker imports it)")
class PreemptFenceTest(unittest.TestCase):
    def setUp(self):
        self._threads: list["KVCacheStoreSendingThread"] = []

    def tearDown(self):
        for thread in self._threads:
            thread.stop(cancel_pending=True)

    def _mk_thread(self, coord) -> "KVCacheStoreSendingThread":
        ready = threading.Event()
        t = KVCacheStoreSendingThread(
            client=None,  # never reached: empty token_databases => no keys
            coord=coord,
            token_databases=[],
            block_size=16,
            tp_rank=0,
            put_step=1,
            kv_role="kv_producer",
            ready_event=ready,
        )
        t.start()
        self.assertTrue(ready.wait(5))
        self._threads.append(t)
        return t

    def test_wait_blocks_while_put_executes_and_returns_after(self):
        # A store that is already executing cannot be cancelled: the fence
        # must report it in flight until it completes, then release.
        entered = threading.Event()
        release = threading.Event()

        class Coord:
            lcm_block_size = 16

            def store_mask(self, token_len):
                entered.set()
                release.wait(10)
                return []

        t = self._mk_thread(Coord())
        t.add_stored_request("r1")
        t.add_request(_req_meta("r1"))
        self.assertTrue(entered.wait(5), "store should be executing")

        # Preemption path: drop queued entries, then join the executing one.
        t.delete_finished_stored_request("r1")
        self.assertFalse(
            t.wait_for_inflight_put("r1", timeout_s=0.3),
            "fence must NOT pass while the store still reads the blocks",
        )
        release.set()
        self.assertTrue(
            t.wait_for_inflight_put("r1", timeout_s=5),
            "fence must release once the store completed",
        )

    def test_deleted_request_is_dropped_at_the_dequeue_gate(self):
        # Entries still queued when the request is preempted must be dropped
        # before any work (store_mask is the first touch inside the body).
        calls: list[int] = []

        class Coord:
            lcm_block_size = 16

            def store_mask(self, token_len):
                calls.append(token_len)
                return []

        t = self._mk_thread(Coord())
        t.add_stored_request("r2")
        t.delete_finished_stored_request("r2")  # preempt before dequeue
        t.add_request(_req_meta("r2"))
        t.request_queue.join()  # waits for task_done of the dropped entry
        self.assertEqual(calls, [], "dropped entry must never start a store")
        self.assertTrue(t.wait_for_inflight_put("r2", timeout_s=1))

    def test_fence_is_noop_for_unknown_request(self):
        class Coord:
            lcm_block_size = 16

            def store_mask(self, token_len):
                return []

        t = self._mk_thread(Coord())
        self.assertTrue(t.wait_for_inflight_put("never-seen", timeout_s=1))


if __name__ == "__main__":
    unittest.main()
