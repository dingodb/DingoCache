"""Preemption-fence unit tests for KVCacheStoreSendingThread.

No GPU or dfkv server needed, but importing dfkv_vllm.worker pulls in vllm
(and torch), so the whole module self-skips where vllm is absent -- same
convention as the other tests in this directory.

Run: python3 -m unittest test_preempt_fence  (from this directory, with
dfkv_vllm on PYTHONPATH, e.g. pip install -e integration/vllm)
"""

import ctypes
import threading
import types
import unittest
from unittest.mock import patch

try:
    import torch
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    from dfkv_vllm.connector import DfkvStoreConnector
    from dfkv_vllm.coordinator import DfkvStoreCoordinator
    from dfkv_vllm.data import (
        ChunkedTokenDatabase,
        DfkvStoreConnectorMetadata,
        KeyMetadata,
        PoolKey,
        ReqMeta,
    )
    from dfkv_vllm.worker import DfkvStoreWorker, KVCacheStoreSendingThread

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
            stripe_idx=0,
            stripe_step=1,
            kv_role="kv_producer",
            ready_event=ready,
        )
        t.start()
        self.assertTrue(ready.wait(5))
        self._threads.append(t)
        return t

    def test_pre_forward_preserves_preempted_save_source_until_put_finishes(self):
        # Exercise the installed engine hook when available, without importing
        # or constructing GPUModelRunner. Older engines can still exercise the
        # connector's pre-forward contract, but that is NOT engine-order proof.
        try:
            from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
        except ImportError:
            ActiveKVConnector = None

        block_size = 16
        original = bytes([17]) * block_size
        replacement = bytes([99]) * block_size
        pool = ctypes.create_string_buffer(2 * block_size)
        source = ctypes.addressof(pool) + block_size
        ctypes.memmove(source, original, block_size)
        key_metadata = KeyMetadata(
            model_name="preempt-source-lifetime", dp_size=1, dp_rank=-1,
            tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0,
            dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0,
        )
        database = ChunkedTokenDatabase(key_metadata, block_size=block_size)
        database.set_seg_layout([
            (ctypes.addressof(pool), block_size, block_size),
        ])
        spec = FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=1,
            head_size_v=0, dtype=torch.uint8,
        )
        coordinator = DfkvStoreCoordinator(
            [KVCacheGroupSpec(["kv"], spec)], block_size, block_size,
        )
        put_entered = threading.Event()
        release_put = threading.Event()
        fence_or_step_finished = threading.Event()
        overwritten = threading.Event()
        saved: dict[bytes, bytes] = {}
        step_errors: list[BaseException] = []

        class DelayedMemoryClient:
            # Substitute only the native boundary: descriptor construction,
            # queue/dequeue, cancellation and in-flight tracking stay real.
            def batch_retain(self, keys, ttl_s=0, mode=1, **kw):
                return [1 if e == 1 else 0 for e in self.batch_exist(keys)]

            def batch_exist(self, keys):
                return [0] * len(keys)

            def batch_put_sg(self, keys, pointers, sizes):
                put_entered.set()
                if not release_put.wait(10):
                    raise TimeoutError("test did not release the pending SAVE")
                for key, ptrs, lengths in zip(keys, pointers, sizes, strict=True):
                    saved[key] = b"".join(
                        ctypes.string_at(ptr, length)
                        for ptr, length in zip(ptrs, lengths, strict=True)
                    )
                return [0] * len(keys)

        sender = KVCacheStoreSendingThread(
            client=DelayedMemoryClient(), coord=coordinator,
            token_databases=[database], block_size=block_size,
            tp_rank=0, stripe_idx=0, stripe_step=1, kv_role="kv_producer",
            ready_event=threading.Event(),
        )
        worker = object.__new__(DfkvStoreWorker)
        worker.kv_send_thread = sender
        worker.kv_recv_thread = None
        connector = object.__new__(DfkvStoreConnector)
        connector.connector_worker = worker
        connector._shutdown_condition = threading.Condition()
        connector._shutdown = False
        connector._inflight_calls = 0
        metadata = DfkvStoreConnectorMetadata(set(), {"preempted"})
        if ActiveKVConnector is not None:
            runner_hook = object.__new__(ActiveKVConnector)
            runner_hook.kv_connector = connector
            runner_hook._disabled = False
            runner_hook._pending_load_start = False
            scheduler_output = types.SimpleNamespace(
                kv_connector_metadata=metadata, has_sync_kv_loads=False,
            )

        def next_model_step():
            try:
                if ActiveKVConnector is not None:
                    runner_hook.pre_forward(scheduler_output)
                else:
                    connector.handle_preemptions(metadata)
                # Model reuse of the same physical block immediately after
                # the real pre-forward hook admits the next model step.
                ctypes.memmove(source, replacement, block_size)
                overwritten.set()
            except BaseException as exc:
                step_errors.append(exc)
            finally:
                fence_or_step_finished.set()

        # Observe the real condition wait, not an arbitrary sleep. Acquiring
        # its lock below guarantees the hook has actually blocked, or finished
        # early (the v2.26.1 missing-hook bug), before inspecting source bytes.
        condition_wait = sender._active_cv.wait

        def observed_wait(timeout=None):
            fence_or_step_finished.set()
            return condition_wait(timeout)

        step = threading.Thread(target=next_model_step, daemon=True)
        request = ReqMeta(
            req_id="preempted", token_len_chunk=block_size,
            block_ids=([1],), block_hashes=[BlockHash(b"x" * 32)],
        )
        with patch.object(sender._active_cv, "wait", observed_wait):
            try:
                sender.start()
                self.assertTrue(sender.ready_event.wait(5))
                sender.add_stored_request(request)
                self.assertTrue(sender.add_request(request))
                self.assertTrue(put_entered.wait(5), "SAVE never reached native PUT")
                step.start()
                self.assertTrue(fence_or_step_finished.wait(5))
                with sender._active_cv:
                    self.assertEqual(step_errors, [])
                    self.assertFalse(
                        overwritten.is_set(),
                        "pre-forward admitted block reuse while SAVE was reading",
                    )
                    self.assertEqual(ctypes.string_at(source, block_size), original)
                release_put.set()
                step.join(5)
                self.assertFalse(step.is_alive(), "preemption fence did not release")
                self.assertEqual(step_errors, [])
                self.assertTrue(overwritten.is_set())
                key = PoolKey(key_metadata, request.block_hashes[0].hex()).to_bytes()
                self.assertEqual(saved, {key: original})
                self.assertEqual(ctypes.string_at(source, block_size), replacement)
            finally:
                release_put.set()
                sender.stop(cancel_pending=True)
                if step.ident is not None:
                    step.join(5)


    def test_wait_blocks_while_put_executes_and_returns_after(self):
        # A store that is already executing cannot be cancelled: the fence
        # must report it in flight until it completes, then release.
        entered = threading.Event()
        release = threading.Event()

        class Coord:
            lcm_block_size = 16
            enable_partial_hash_hits = False

            def store_mask(self, token_len):
                entered.set()
                release.wait(10)
                return []

        t = self._mk_thread(Coord())
        request = _req_meta("r1")
        t.add_stored_request(request)
        t.add_request(request)
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
            enable_partial_hash_hits = False

            def store_mask(self, token_len):
                calls.append(token_len)
                return []

        t = self._mk_thread(Coord())
        request = _req_meta("r2")
        t.add_stored_request(request)
        t.delete_finished_stored_request("r2")  # preempt before dequeue
        t.add_request(request)
        t.request_queue.join()  # waits for task_done of the dropped entry
        self.assertEqual(calls, [], "dropped entry must never start a store")
        self.assertTrue(t.wait_for_inflight_put("r2", timeout_s=1))

    def test_fence_is_noop_for_unknown_request(self):
        class Coord:
            lcm_block_size = 16
            enable_partial_hash_hits = False

            def store_mask(self, token_len):
                return []

        t = self._mk_thread(Coord())
        self.assertTrue(t.wait_for_inflight_put("never-seen", timeout_s=1))


if __name__ == "__main__":
    unittest.main()
