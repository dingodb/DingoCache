"""Per-step scheduler metadata stays small.

* A request's token ids are only read by KV events; without them they are
  no longer pickled to every worker every step (a 32k-token request's
  ReqMeta went from about 115 KiB to 19 KiB).
* The pending-load scan of build_connector_meta checks scheduled request ids
  in a set, not in lists (quadratic in the number of requests).

Runs on the stand-ins of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_vllm import scheduler as scheduler_module  # noqa: E402
from dfkv_vllm.data import ReqMeta, RequestTracker  # noqa: E402


def _tracker():
    return RequestTracker(
        req_id="r", token_len=256, allocated_block_ids=([1, 2, 3, 4],),
        token_ids=list(range(256)),
    )


class StepMetadata(unittest.TestCase):
    def test_token_ids_only_when_asked_for(self) -> None:
        self.assertEqual(
            ReqMeta.from_request_tracker(_tracker(), 64).token_ids, list(range(256)))
        self.assertIsNone(
            ReqMeta.from_request_tracker(_tracker(), 64, with_token_ids=False).token_ids)

    def test_the_scheduler_asks_for_them_with_kv_events_only(self) -> None:
        source = inspect.getsource(scheduler_module.DfkvStoreScheduler)
        calls = source.count("ReqMeta.from_request_tracker(")
        self.assertEqual(source.count("with_token_ids=self._kv_events"), calls)
        # The flag itself: test_dfkv_vllm_connector.SchedulerLookupTest.


if __name__ == "__main__":
    unittest.main(verbosity=2)
