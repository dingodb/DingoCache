"""A failed or rejected LOAD reports its own destination blocks only.

The request's local prefix-cache hit (the first ``vllm_cached_tokens``) is in
blocks vLLM shares with other running requests. vLLM rolls back every
request that holds a block reported invalid, and drops those blocks from its
prefix cache: reporting the whole block table made one failed LOAD recompute
unrelated requests. Runs on the stand-ins of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from pathlib import Path  # noqa: E402

from dfkv_vllm.data import (  # noqa: E402
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from dfkv_vllm.worker import KVCacheStoreRecvingThread  # noqa: E402
from vllm.v1.core.kv_cache_utils import BlockHash  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integration" / "vllm" / "tests"))
from _coordinator_fake import fake_coordinator  # noqa: E402

BLOCK = 16
LOCAL_HIT, LOADED, COMPUTED = 5, 7, 9  # block ids, in request order


def _db() -> ChunkedTokenDatabase:
    db = ChunkedTokenDatabase(
        KeyMetadata(
            model_name="m", dp_size=1, dp_rank=-1, tp_size=1, tp_rank=0,
            pcp_size=1, pcp_rank=0, dcp_size=1, dcp_rank=0, pp_size=1,
            pp_rank=0,
        ),
        block_size=BLOCK,
    )
    db.set_seg_layout([(0x1000, 64, 64)])
    return db


def _request(req_id: str = "r") -> ReqMeta:
    # One block hit in vLLM's own prefix cache, one to LOAD, one to compute.
    hashes = [BlockHash(hashlib.sha256(f"{req_id}{i}".encode()).digest())
              for i in range(3)]
    return ReqMeta(
        req_id=req_id, token_len_chunk=2 * BLOCK,
        block_ids=([LOCAL_HIT, LOADED, COMPUTED],), block_hashes=hashes,
        load_spec=LoadSpec(BLOCK, 2 * BLOCK, True, token_len=2 * BLOCK),
    )


class _FailingClient:
    def batch_get_auto_sg(self, keys, _ptrs, _caps):
        return [0] * len(keys), [0] * len(keys)


def _receiver(capacity: int = 8, client=None) -> KVCacheStoreRecvingThread:
    return KVCacheStoreRecvingThread(
        client or _FailingClient(),
        fake_coordinator(BLOCK, load_mask=lambda _h, _n: [[True, True, True]]),
        [_db()],
        block_size=BLOCK,
        tp_rank=0,
        ready_event=threading.Event(),
        queue_capacity=capacity,
    )


class LoadFailuresReportTheirOwnBlocks(unittest.TestCase):
    def test_a_missed_load(self) -> None:
        receiver = _receiver()
        receiver._handle_request(_request())
        self.assertEqual(receiver.get_and_clear_block_ids_with_load_errors(), {LOADED})

    def test_a_load_rejected_by_a_full_queue(self) -> None:
        receiver = _receiver(capacity=1)
        self.assertTrue(receiver.add_request(_request("queued")))
        self.assertFalse(receiver.add_request(_request("rejected")))
        self.assertEqual(
            receiver.get_and_clear_block_ids_with_load_errors(), {LOADED},
            "the rejected load reported its local prefix-cache hit too",
        )

    def test_a_load_after_close(self) -> None:
        receiver = _receiver()
        receiver.stop()
        self.assertFalse(receiver.add_request(_request()))
        self.assertEqual(receiver.get_and_clear_block_ids_with_load_errors(), {LOADED})

    def test_a_failure_in_the_load_path_never_reports_the_null_block(self) -> None:
        # An unallocated destination (block 0, vLLM's shared null block)
        # fails the request in the outer handler. Block 0 is never
        # reported: every request that pads with it would be rolled back.
        request = _request()
        request.block_ids = ([LOCAL_HIT, 0, COMPUTED],)
        request.load_spec = LoadSpec(BLOCK, 3 * BLOCK, True, token_len=3 * BLOCK)
        receiver = _receiver()
        receiver._handle_request(request)
        self.assertEqual(receiver.get_and_clear_block_ids_with_load_errors(), {COMPUTED})

    def test_no_usable_destination_falls_back_to_the_table_without_null(self) -> None:
        request = _request()
        request.block_ids = ([LOCAL_HIT, 0, COMPUTED],)
        receiver = _receiver()
        receiver._handle_request(request)
        self.assertEqual(
            receiver.get_and_clear_block_ids_with_load_errors(),
            {LOCAL_HIT, COMPUTED},
        )

    def test_a_cancelled_queued_load(self) -> None:
        receiver = _receiver()
        self.assertTrue(receiver.add_request(_request()))
        receiver.cancel_requests({"r"}, wait=False, fail_closed=True)
        self.assertEqual(receiver.get_and_clear_block_ids_with_load_errors(), {LOADED})

    def test_unresolvable_geometry_reports_the_whole_table(self) -> None:
        receiver = _receiver()
        receiver.coord = fake_coordinator(
            BLOCK, load_mask=lambda _h, _n: (_ for _ in ()).throw(ValueError("x")))
        receiver.stop()
        receiver.add_request(_request())
        self.assertEqual(
            receiver.get_and_clear_block_ids_with_load_errors(),
            {LOCAL_HIT, LOADED, COMPUTED},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
