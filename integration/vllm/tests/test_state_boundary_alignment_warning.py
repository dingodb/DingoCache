"""State hand-offs off the scheduler block grid are reported, not silent."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dfkv_vllm import scheduler as scheduler_module
from dfkv_vllm.scheduler import DfkvStoreScheduler

LCM = 2048


def _scheduler():
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._block_size = LCM
    return scheduler


def _meta(*req_ids):
    return SimpleNamespace(requests=[
        SimpleNamespace(req_id=r, can_save=True, state_blocks=None) for r in req_ids
    ])


def _output(offloads):
    return SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(boundary_state_offloads=offloads))


@pytest.fixture
def warn(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(scheduler_module, "logger", logger)
    clock = [1000.0]
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: clock[0])
    return logger.warning, clock


def test_lcm_aligned_hand_off_is_silent(warn):
    warning, _ = warn
    meta = _meta("r")
    _scheduler()._attach_state_blocks(
        meta, _output({"r": [(1, 9, 19 * LCM), (2, 10, 19 * LCM)]}))
    assert meta.requests[0].state_blocks == {1: {19 * LCM: 9}, 2: {19 * LCM: 10}}
    warning.assert_not_called()


def test_misaligned_hand_off_warns_and_still_attaches(warn):
    """DCP: a 256-aligned prompt end (39936) is off the 2048 grid."""
    warning, _ = warn
    meta = _meta("r")
    _scheduler()._attach_state_blocks(
        meta, _output({"r": [(1, 9, 39936), (2, 10, 39936), (3, 11, 39936)]}))
    assert meta.requests[0].state_blocks == {1: {39936: 9}, 2: {39936: 10}, 3: {39936: 11}}
    warning.assert_called_once()
    assert warning.call_args.args[1:] == (3, LCM)


def test_warning_is_rate_limited_with_a_running_count(warn):
    warning, clock = warn
    scheduler = _scheduler()
    for _ in range(3):
        scheduler._attach_state_blocks(_meta("r"), _output({"r": [(1, 9, 256)]}))
        clock[0] += 10
    assert warning.call_count == 1
    clock[0] += scheduler_module._MISALIGNED_STATE_WARN_INTERVAL_S
    scheduler._attach_state_blocks(_meta("r"), _output({"r": [(1, 9, 256)]}))
    assert warning.call_count == 2
    assert warning.call_args.args[1] == 4


def test_requests_that_cannot_save_are_not_counted(warn):
    warning, _ = warn
    meta = _meta("r")
    meta.requests[0].can_save = False
    _scheduler()._attach_state_blocks(meta, _output({"r": [(1, 9, 256)]}))
    assert meta.requests[0].state_blocks is None
    warning.assert_not_called()
