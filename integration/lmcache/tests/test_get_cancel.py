# SPDX-License-Identifier: Apache-2.0
"""A GET that is cancelled must not give its buffer back while the transport
is still writing into it.

The native GET runs on a worker thread and fills LMCache ``MemoryObj`` buffers
taken from the L1 arena. LMCache cancels a ``get`` that exceeds its blocking
timeout; the thread does not stop. If the buffer went back to the arena at
that moment, the next allocation would get memory that the late write then
overwrites: corruption in the host process. On the batch paths a cancellation
used to skip the release altogether.

The tests drive the real connector and the real client's ``batch_get`` over a
native call that blocks until told to return, and an arena that hands a freed
slot straight to the next allocation.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import _lmcache_stub  # noqa: E402

_lmcache_stub.install()

from dfkv_connector.exists_cache import ExistsLRU  # noqa: E402
from dfkv_connector.native_client import DfkvNativeClient  # noqa: E402
from dfkv_connector.remote_connector import DfkvConnector  # noqa: E402

CHUNK = 64


class _Obj:
    def __init__(self, arena: "_Arena", slot: int) -> None:
        self._arena = arena
        self.slot = slot

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._arena.slots[self.slot])

    def ref_count_down(self) -> None:
        self._arena.free(self.slot)


class _Arena:
    """A fixed pool. The slot freed last is the one allocated next, as in a
    busy L1 arena."""

    def __init__(self, n: int) -> None:
        self.slots = [bytearray(CHUNK) for _ in range(n)]
        self._free = list(range(n))
        self.double_frees = 0
        self._lock = threading.Lock()

    def allocate(self, *_shape_dtype_fmt):
        with self._lock:
            if not self._free:
                return None
            return _Obj(self, self._free.pop())

    def free(self, slot: int) -> None:
        with self._lock:
            if slot in self._free:
                self.double_frees += 1
                return
            self._free.append(slot)

    @property
    def free_slots(self) -> int:
        with self._lock:
            return len(self._free)


class _Native:
    """Stands in for the blocking libdfkv call: it writes ``payload`` into
    every buffer, but only once ``gate`` is opened."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.entered = threading.Semaphore(0)
        self.calls = 0
        self.fail = False
        self.miss = False
        self.payload = b"L" * CHUNK  # the late write
        self._lock = threading.Lock()

    def __call__(self, keys, bufs):
        with self._lock:
            self.calls += 1
        self.entered.release()
        self.gate.wait(10)
        if self.fail:
            raise RuntimeError("injected native failure")
        if self.miss:
            return True, [False] * len(keys), [0] * len(keys)
        for buf in bufs:
            buf[: len(self.payload)] = self.payload
        return True, [True] * len(keys), [len(self.payload)] * len(keys)


def _key(i: int):
    return types.SimpleNamespace(
        model_name="unit-test", world_size=1, worker_id=0, chunk_hash=i
    )


class _Rig:
    def __init__(self, loop, slots: int, workers: int = 1,
                 batch_max_keys: int = 512, parallelism: int = 1) -> None:
        self.arena = _Arena(slots)
        self.native = _Native()
        self.executor = ThreadPoolExecutor(max_workers=workers)
        client = object.__new__(DfkvNativeClient)
        client._loop = loop
        client._executor = self.executor
        client._batch_get_blocking = self.native
        self.client = client
        connector = object.__new__(DfkvConnector)
        connector._client = client
        connector._exists_lru = ExistsLRU(64)
        connector._batch_max_keys = batch_max_keys
        connector._get_parallelism = parallelism
        connector._local_cpu_backend = self.arena
        connector.meta_shapes = [(CHUNK,)]
        connector.meta_dtypes = ["uint8"]
        connector.meta_fmt = "raw"
        connector.full_chunk_size_bytes = CHUNK
        connector.single_token_size = 1
        self.connector = connector

    async def native_entered(self, times: int = 1) -> None:
        loop = asyncio.get_running_loop()
        for _ in range(times):
            ok = await loop.run_in_executor(None, self.native.entered.acquire, True, 10)
            assert ok, "the native call was never entered"

    async def until(self, predicate, what: str) -> None:
        for _ in range(2000):
            if predicate():
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"timed out waiting for {what}")

    def close(self) -> None:
        self.native.gate.set()
        self.executor.shutdown(wait=True)


def _run(coro_fn):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro_fn(loop))
    finally:
        loop.close()


class CancelledGetKeepsItsBufferUntilTheTransportIsDone(unittest.TestCase):
    def test_scalar_get_late_write_does_not_land_in_a_reused_buffer(self):
        async def scenario(loop):
            rig = _Rig(loop, slots=2)
            try:
                task = asyncio.ensure_future(rig.connector.get(_key(1)))
                await rig.native_entered()
                # LMCache gives up on the get: this is what its blocking
                # timeout does.
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0.02)
                self.assertEqual(
                    rig.arena.free_slots, 1,
                    "the buffer went back to the arena under a running GET",
                )
                # The next request allocates and fills its own chunk.
                other = rig.arena.allocate()
                other.byte_array[:] = b"N" * CHUNK
                # Now the transport finishes the cancelled GET.
                rig.native.gate.set()
                await rig.until(
                    lambda: rig.arena.free_slots == 1 and
                    bytes(rig.arena.slots[1 - other.slot][:1]) == b"L",
                    "the cancelled GET's buffer to be released",
                )
                self.assertEqual(bytes(other.byte_array), b"N" * CHUNK,
                                 "the late write landed in a reused buffer")
                self.assertEqual(rig.client.deferred_buffer_releases, 1)
                self.assertEqual(rig.arena.double_frees, 0)
            finally:
                rig.close()

        _run(scenario)

    def test_batched_get_cancel_releases_every_buffer_and_none_too_early(self):
        async def scenario(loop):
            # Two chunks of two keys, one after the other: the first is in
            # flight when the cancellation arrives, the second never starts.
            rig = _Rig(loop, slots=4, batch_max_keys=2)
            try:
                task = asyncio.ensure_future(
                    rig.connector.batched_get([_key(i) for i in range(4)])
                )
                await rig.native_entered()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0.02)
                self.assertEqual(
                    rig.arena.free_slots, 2,
                    "only the chunk that was never sent may be released now",
                )
                rig.native.gate.set()
                await rig.until(lambda: rig.arena.free_slots == 4,
                                "all four buffers to be released (no leak)")
                self.assertEqual(rig.native.calls, 1)
                self.assertEqual(rig.arena.double_frees, 0)
            finally:
                rig.close()

        _run(scenario)

    def test_batched_get_non_blocking_cancel_releases_every_buffer(self):
        async def scenario(loop):
            # Two chunks at once: both are in flight when it is cancelled.
            rig = _Rig(loop, slots=4, workers=2, batch_max_keys=2, parallelism=2)
            try:
                task = asyncio.ensure_future(
                    rig.connector.batched_get_non_blocking(
                        "lookup", [_key(i) for i in range(4)]
                    )
                )
                await rig.native_entered(2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0.02)
                self.assertEqual(rig.arena.free_slots, 0,
                                 "released under two running GETs")
                rig.native.gate.set()
                await rig.until(lambda: rig.arena.free_slots == 4,
                                "all four buffers to be released (no leak)")
                self.assertEqual(rig.client.deferred_buffer_releases, 2)
                self.assertEqual(rig.arena.double_frees, 0)
            finally:
                rig.close()

        _run(scenario)

    def test_get_cancelled_before_the_transport_took_it_is_released_at_once(self):
        async def scenario(loop):
            rig = _Rig(loop, slots=2)  # one worker
            try:
                first = asyncio.ensure_future(rig.connector.get(_key(1)))
                await rig.native_entered()
                queued = asyncio.ensure_future(rig.connector.get(_key(2)))
                await asyncio.sleep(0.02)  # queued behind the first
                self.assertEqual(rig.arena.free_slots, 0)
                queued.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await queued
                await rig.until(lambda: rig.arena.free_slots == 1,
                                "the never-sent GET's buffer")
                rig.native.gate.set()
                self.assertIsNotNone(await first)
                self.assertEqual(rig.native.calls, 1, "the cancelled GET was sent")
                self.assertEqual(rig.client.deferred_buffer_releases, 0)
            finally:
                rig.close()

        _run(scenario)

    def test_failed_chunk_waits_for_its_sibling_and_everything_is_released(self):
        async def scenario(loop):
            rig = _Rig(loop, slots=4, workers=2, batch_max_keys=2, parallelism=2)
            rig.native.fail = True
            try:
                task = asyncio.ensure_future(
                    rig.connector.batched_get([_key(i) for i in range(4)])
                )
                await rig.native_entered(2)
                rig.native.gate.set()
                with self.assertRaises(RuntimeError):
                    await task
                await rig.until(lambda: rig.arena.free_slots == 4,
                                "all four buffers after a native failure")
                self.assertEqual(rig.arena.double_frees, 0)
            finally:
                rig.close()

        _run(scenario)

    def test_loop_closed_under_a_running_get_still_releases_afterwards(self):
        loop = asyncio.new_event_loop()
        rig = _Rig(loop, slots=1)
        try:
            async def cancel_one():
                task = asyncio.ensure_future(rig.connector.get(_key(1)))
                await rig.native_entered()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            loop.run_until_complete(cancel_one())
            loop.close()
            self.assertEqual(rig.arena.free_slots, 0)
            rig.native.gate.set()
            rig.executor.shutdown(wait=True)
            self.assertEqual(rig.arena.free_slots, 1)
            self.assertEqual(rig.arena.double_frees, 0)
        finally:
            rig.close()


    def test_get_racing_close_releases_its_buffer(self):
        async def scenario(loop):
            # close() has shut the executor down: the submit itself raises,
            # no native call exists, the buffer goes back at once.
            rig = _Rig(loop, slots=1)
            try:
                rig.executor.shutdown(wait=True)
                with self.assertRaises(RuntimeError):
                    await rig.connector.get(_key(1))
                self.assertEqual(rig.arena.free_slots, 1,
                                 "the buffer of a GET that was never sent leaked")
                with self.assertRaises(RuntimeError):
                    await rig.connector.batched_get([_key(2)])
                self.assertEqual(rig.arena.free_slots, 1)
                self.assertEqual(rig.arena.double_frees, 0)
            finally:
                rig.close()

        _run(scenario)

    def test_a_failure_after_the_cancel_is_not_reported_as_unretrieved(self):
        async def scenario(loop):
            reported = []
            loop.set_exception_handler(
                lambda _loop, ctx: reported.append(ctx.get("message")))
            rig = _Rig(loop, slots=1)
            rig.native.fail = True
            try:
                task = asyncio.ensure_future(rig.connector.get(_key(1)))
                await rig.native_entered()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                rig.native.gate.set()
                await rig.until(lambda: rig.arena.free_slots == 1,
                                "the cancelled GET's buffer")
                await asyncio.sleep(0.02)
                gc.collect()
                await asyncio.sleep(0)
                self.assertEqual(reported, [])
            finally:
                rig.close()

        _run(scenario)

class UncancelledGetsBehaveAsBefore(unittest.TestCase):
    def test_hit_hands_the_buffer_on_and_miss_releases_it(self):
        async def scenario(loop):
            rig = _Rig(loop, slots=2)
            rig.native.gate.set()
            try:
                hit = await rig.connector.get(_key(1))
                self.assertIsNotNone(hit)
                self.assertEqual(bytes(hit.byte_array), b"L" * CHUNK)
                self.assertEqual(rig.arena.free_slots, 1)
                rig.native.miss = True
                self.assertIsNone(await rig.connector.get(_key(2)))
                self.assertEqual(rig.arena.free_slots, 1)
                self.assertEqual(rig.client.deferred_buffer_releases, 0)
                self.assertEqual(rig.arena.double_frees, 0)
            finally:
                rig.close()

        _run(scenario)

    def test_batched_paths_return_hits_and_release_the_rest(self):
        async def scenario(loop):
            rig = _Rig(loop, slots=6, batch_max_keys=2)
            rig.native.gate.set()
            try:
                got = await rig.connector.batched_get([_key(i) for i in range(3)])
                self.assertEqual([g is not None for g in got], [True] * 3)
                self.assertEqual(rig.arena.free_slots, 3)
                prefix = await rig.connector.batched_get_non_blocking(
                    "lookup", [_key(i) for i in range(3)]
                )
                self.assertEqual(len(prefix), 3)
                self.assertEqual(rig.arena.free_slots, 0)
                self.assertEqual(rig.arena.double_frees, 0)
            finally:
                rig.close()

        _run(scenario)


if __name__ == "__main__":
    unittest.main(verbosity=2)
