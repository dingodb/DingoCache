"""With PP > 1 the scheduler's lookup probes one pipeline stage only.

The lookup runs on rank 0 and checks the objects of its stage; the other
stages' objects live in namespaces of their own (each stage's cache groups
list only its layers). A prefix another stage has lost is admitted, and that
stage's load fails: with vLLM's default ``kv_load_failure_policy="fail"`` the
request fails. The connector warns at startup in that configuration. Runs on
the stand-ins of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import logging
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_vllm import worker as worker_module  # noqa: E402


class PipelineLookupScope(unittest.TestCase):
    def warn(self, pp_size, policy):
        return worker_module._warn_pipeline_lookup_scope(
            pp_size, SimpleNamespace(kv_load_failure_policy=policy))

    def test_pp_with_the_fail_policy_warns(self) -> None:
        with self.assertLogs(worker_module.logger, level=logging.WARNING) as logs:
            self.assertTrue(self.warn(2, "fail"))
        self.assertIn('kv_load_failure_policy="recompute"', logs.output[0])

    def test_no_warning_otherwise(self) -> None:
        with self.assertNoLogs(worker_module.logger, level=logging.WARNING):
            self.assertFalse(self.warn(1, "fail"))
            self.assertFalse(self.warn(2, "recompute"))
            self.assertFalse(worker_module._warn_pipeline_lookup_scope(
                4, SimpleNamespace()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
