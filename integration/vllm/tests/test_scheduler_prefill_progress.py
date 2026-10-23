"""Prefill progress must follow computed tokens, not block allocation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm")

from dfkv_vllm.data import RequestTracker
from dfkv_vllm.scheduler import DfkvStoreScheduler


BLOCK = 4
PREFILL_END = 12


def _scheduler(
    *,
    block=BLOCK,
    prefill_end=PREFILL_END,
    token_len=BLOCK,
    total_tokens=16,
):
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._failed_load_req_ids = set()
    scheduler.kv_role = "kv_both"
    scheduler.client = MagicMock()
    scheduler.load_specs = {}
    scheduler._request_trackers = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = {"req"}
    scheduler._allocated_req_ids = {"req"}
    scheduler._block_size = block
    request = SimpleNamespace(
        request_id="req",
        block_hashes=[],
        all_token_ids=list(range(total_tokens)),
    )
    scheduler._unfinished_requests = {"req": (request, ([10],))}
    scheduler._request_trackers["req"] = RequestTracker(
        req_id="req",
        token_len=token_len,
        allocated_block_ids=([10],),
        token_ids=list(range(token_len)),
        prefill_end_tokens=prefill_end,
    )
    return scheduler, request


def _step(scheduler, *, computed, scheduled, new_block_ids):
    return scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["req"],
                new_block_ids=[new_block_ids],
                num_computed_tokens=[computed],
            ),
            num_scheduled_tokens={"req": scheduled},
        )
    )


def test_prefill_chunk_without_new_block_advances_tracker():
    scheduler, _ = _scheduler()

    metadata = _step(
        scheduler, computed=BLOCK, scheduled=BLOCK, new_block_ids=None
    )

    tracker = scheduler._request_trackers["req"]
    assert tracker.token_len == 2 * BLOCK
    assert tracker.token_ids == list(range(2 * BLOCK))
    assert len(metadata.requests) == 1
    assert metadata.requests[0].token_len_chunk == 2 * BLOCK
    assert metadata.requests[0].token_ids == list(range(2 * BLOCK))
    assert metadata.requests[0].can_save is True


def test_two_allocation_free_chunks_reach_last_prefill_boundary():
    scheduler, _ = _scheduler()

    _step(scheduler, computed=BLOCK, scheduled=BLOCK, new_block_ids=None)
    metadata = _step(
        scheduler, computed=2 * BLOCK, scheduled=BLOCK, new_block_ids=None
    )

    tracker = scheduler._request_trackers["req"]
    assert tracker.token_len == PREFILL_END
    assert tracker.token_ids == list(range(PREFILL_END))
    assert metadata.requests[-1].token_len_chunk == PREFILL_END
    assert metadata.requests[-1].is_last_chunk is True


def test_tracker_resynchronizes_from_authoritative_computed_position():
    scheduler, _ = _scheduler()
    tracker = scheduler._request_trackers["req"]
    # Simulate state left behind by a skipped prefill step.
    tracker.token_len = 0
    tracker.token_ids = []

    metadata = _step(
        scheduler, computed=BLOCK, scheduled=BLOCK, new_block_ids=None
    )

    assert tracker.token_len == 2 * BLOCK
    assert tracker.token_ids == list(range(2 * BLOCK))
    assert metadata.requests[0].token_ids == list(range(2 * BLOCK))


def test_decode_step_emits_no_prefill_metadata():
    scheduler, _ = _scheduler()
    tracker = scheduler._request_trackers["req"]
    tracker.token_len = PREFILL_END
    tracker.token_ids = list(range(PREFILL_END))

    metadata = _step(
        scheduler, computed=PREFILL_END, scheduled=1, new_block_ids=None
    )

    assert metadata.requests == []
    assert tracker.token_len == PREFILL_END


def test_new_blocks_are_appended_when_allocated():
    scheduler, _ = _scheduler()

    _step(
        scheduler, computed=BLOCK, scheduled=BLOCK, new_block_ids=([11],)
    )

    tracker = scheduler._request_trackers["req"]
    assert tracker.allocated_block_ids == ([10, 11],)
    assert tracker.token_len == 2 * BLOCK


def test_encoder_only_step_does_not_advance_tracker():
    scheduler, _ = _scheduler()

    metadata = _step(scheduler, computed=BLOCK, scheduled=0, new_block_ids=None)

    tracker = scheduler._request_trackers["req"]
    assert tracker.token_len == BLOCK
    assert metadata.requests == []


def test_sub_block_chunks_do_not_strand_640_boundary():
    scheduler, _ = _scheduler(
        block=640, prefill_end=1280, token_len=256, total_tokens=2048
    )

    first = _step(scheduler, computed=256, scheduled=256, new_block_ids=None)
    assert first.requests == []
    assert scheduler._request_trackers["req"].token_len == 512

    second = _step(scheduler, computed=512, scheduled=128, new_block_ids=None)
    assert scheduler._request_trackers["req"].token_len == 640
    assert second.requests[-1].token_len_chunk == 640

    third = _step(
        scheduler, computed=640, scheduled=640, new_block_ids=([11],)
    )
    assert scheduler._request_trackers["req"].token_len == 1280
    assert third.requests[-1].token_len_chunk == 1280
    assert third.requests[-1].is_last_chunk is True


def test_empty_block_delta_is_not_treated_as_an_allocation():
    scheduler, _ = _scheduler()

    metadata = _step(
        scheduler,
        computed=BLOCK,
        scheduled=BLOCK,
        new_block_ids=([], []),
    )

    tracker = scheduler._request_trackers["req"]
    assert tracker.token_len == 2 * BLOCK
    assert tracker.token_ids == list(range(2 * BLOCK))
    assert tracker.allocated_block_ids == ([10],)
    assert metadata.requests[0].token_ids == list(range(2 * BLOCK))
