"""A failed block-level LOAD drops the request's cached lookup.

vLLM rolls requests holding an invalid block back and asks the connector
for them again. The scheduler reused a completed lookup for
``lookup_cache_ttl_ms``, so the retry was answered with the hit whose load
had just failed (and loaded again) until the entry expired; request-level
layouts already dropped it. Runs on the stand-ins of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_vllm.scheduler import DfkvStoreScheduler  # noqa: E402


class _Lookups:
    def __init__(self) -> None:
        self.discarded: list[str] = []

    def discard(self, req_id: str) -> None:
        self.discarded.append(req_id)


def _scheduler():
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler.request_level_loads = False
    scheduler.client = _Lookups()
    scheduler._request_trackers = {
        "failed": SimpleNamespace(allocated_block_ids=([3, 7, 9],)),
        "other": SimpleNamespace(allocated_block_ids=([4, 5],)),
    }
    return scheduler


class FailedBlockLoad(unittest.TestCase):
    def test_the_failed_requests_lookup_is_dropped(self) -> None:
        scheduler = _scheduler()
        scheduler.update_connector_output(SimpleNamespace(invalid_block_ids={7}))
        self.assertEqual(scheduler.client.discarded, ["failed"])

    def test_no_invalid_blocks_drops_nothing(self) -> None:
        scheduler = _scheduler()
        scheduler.update_connector_output(SimpleNamespace(invalid_block_ids=set()))
        scheduler.update_connector_output(SimpleNamespace())
        self.assertEqual(scheduler.client.discarded, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
