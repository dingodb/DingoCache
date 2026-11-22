"""A request's SAVEs resume where its earlier steps left off.

Each step's SAVE enumerated the request's prefix from chunk 0 and probed the
existence of every chunk already computed: over a chunked prefill, quadratic
in the prompt length. A per-request mark per group now records how far this
rank's chunks are known stored (PUT or found); a SAVE starts there. A failed
PUT holds the mark back, so a later step retries the chunk; stripe positions
and KV-event parents are those of a full enumeration. Runs on the stand-ins
of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integration" / "vllm" / "tests"))
from _coordinator_fake import fake_coordinator  # noqa: E402

from dfkv_vllm.data import ChunkedTokenDatabase, KeyMetadata, PoolKey, ReqMeta  # noqa: E402
from dfkv_vllm.worker import KVCacheStoreSendingThread  # noqa: E402
from vllm.v1.core.kv_cache_utils import BlockHash  # noqa: E402

BLOCK = 16


def _db() -> ChunkedTokenDatabase:
    db = ChunkedTokenDatabase(
        KeyMetadata(
            model_name="m", dp_size=1, dp_rank=-1, tp_size=1, tp_rank=0,
            pcp_size=1, pcp_rank=0, dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0,
        ),
        block_size=BLOCK,
    )
    db.set_seg_layout([(0x1000, 64, 64)])
    return db


class _Ring:
    def __init__(self) -> None:
        self.stored: set[bytes] = set()
        self.probed: list[list[bytes]] = []
        self.put: list[list[bytes]] = []
        self.fail_once: set[bytes] = set()
        self.retained: list[list[bytes]] = []

    def batch_exist(self, keys):
        self.probed.append(list(keys))
        return [1 if k in self.stored else 0 for k in keys]

    def batch_retain(self, keys, ttl_s, mode, **_kw):
        # SET: protects present keys; the codes double as the exists probe.
        self.retained.append(list(keys))
        return [1 if k in self.stored else 0 for k in keys]

    def batch_put_sg(self, keys, _ptrs, _sizes):
        self.put.append(list(keys))
        out = []
        for k in keys:
            if k in self.fail_once:
                self.fail_once.discard(k)
                out.append(1)
            else:
                self.stored.add(k)
                out.append(0)
        return out


HASHES = [BlockHash(hashlib.sha256(b"%d" % i).digest()) for i in range(64)]


def _key(db, i):
    return PoolKey(db.metadata, HASHES[i].hex()).to_bytes()


def _sender(ring, db, *, stripe_idx=0, stripe_step=1, mask=None, events=False):
    coord = fake_coordinator(
        BLOCK, store_mask=mask or (lambda n: [[True] * (n // BLOCK)]))
    return KVCacheStoreSendingThread(
        ring, coord, [db], block_size=BLOCK, tp_rank=0, stripe_idx=stripe_idx,
        stripe_step=stripe_step, kv_role="kv_both", ready_event=threading.Event(),
        enable_kv_event=events,
    )


def _save(sender, blocks, req_id="r", retain_ttl_s=0):
    request = ReqMeta(
        req_id=req_id, token_len_chunk=blocks * BLOCK,
        block_ids=(list(range(1, 65)),), block_hashes=HASHES, can_save=True,
        token_ids=list(range(64 * BLOCK)), retain_ttl_s=retain_ttl_s,
    )
    sender.add_stored_request(request)
    sender._handle_request(request)


class SaveMarks(unittest.TestCase):
    def test_each_step_probes_only_its_new_chunks(self) -> None:
        ring, db = _Ring(), _db()
        sender = _sender(ring, db)
        for blocks in (8, 16, 24):
            _save(sender, blocks)
        self.assertEqual([len(p) for p in ring.probed], [8, 8, 8])
        self.assertEqual(ring.stored, {_key(db, i) for i in range(24)})

    def test_a_failed_put_is_retried_by_a_later_step(self) -> None:
        ring, db = _Ring(), _db()
        ring.fail_once = {_key(db, 5)}
        sender = _sender(ring, db)
        _save(sender, 8)
        self.assertNotIn(_key(db, 5), ring.stored)
        _save(sender, 16)
        # The mark stayed at chunk 5: probed again from there, and stored.
        self.assertEqual(ring.probed[1], [_key(db, i) for i in range(5, 16)])
        self.assertIn(_key(db, 5), ring.stored)

    def test_stripes_are_those_of_a_full_enumeration(self) -> None:
        for rank in (0, 1):
            with self.subTest(rank=rank):
                ring, db = _Ring(), _db()
                sender = _sender(ring, db, stripe_idx=rank, stripe_step=2)
                for blocks in (7, 13, 20):
                    _save(sender, blocks)
                self.assertEqual(
                    ring.stored, {_key(db, i) for i in range(20) if i % 2 == rank})

    def test_a_chunk_masked_out_holds_the_mark(self) -> None:
        # The last block of each step is left out (a volatile tail): the next
        # step must still save it.
        ring, db = _Ring(), _db()
        sender = _sender(ring, db, mask=lambda n: [[True] * (n // BLOCK - 1) + [False]])
        _save(sender, 8)
        _save(sender, 16)
        self.assertIn(_key(db, 7), ring.stored)
        self.assertEqual(ring.probed[1][0], _key(db, 7))

    def test_a_new_save_generation_starts_over(self) -> None:
        ring, db = _Ring(), _db()
        sender = _sender(ring, db)
        _save(sender, 8)
        sender.delete_finished_stored_request("r")  # preempted
        _save(sender, 8)
        self.assertEqual([len(p) for p in ring.probed], [8, 8])

    def test_with_a_ttl_every_step_re_protects_the_whole_prefix(self) -> None:
        # Retention SET must cover chunks saved by earlier steps too, or they
        # expire before the later ones (prefix closure).
        ring, db = _Ring(), _db()
        sender = _sender(ring, db)
        with unittest.mock.patch.object(
            sys.modules["dfkv_vllm.worker"].dfkv_cabi, "RETAIN_HIT_CODES", (1,),
        ):
            _save(sender, 8, retain_ttl_s=60)
            first = len(ring.retained)
            _save(sender, 16, retain_ttl_s=60)
        second = {k for call in ring.retained[first:] for k in call}
        self.assertLessEqual({_key(db, i) for i in range(16)}, second)
        self.assertEqual(sum(len(p) for p in ring.put), 16, "nothing put twice")

    def test_event_parents_are_those_of_a_full_enumeration(self) -> None:
        ring, db = _Ring(), _db()
        sender = _sender(ring, db, events=True)
        _save(sender, 4)
        _save(sender, 8)
        events = sender.get_kv_events()
        parents = [bytes(e.parent_block_hash) if e.parent_block_hash else None
                   for e in events]
        self.assertEqual(parents, [None] + [bytes(HASHES[i]) for i in range(7)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
