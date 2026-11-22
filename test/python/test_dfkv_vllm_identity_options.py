"""What the vLLM object identity binds beyond the bytes' layout.

* The DP size does not change an object (every DP replica writes the same
  bytes; ``dp_rank`` is -1 in every key), yet it was always bound, so
  deployments of different DP sizes missed each other. ``share_across_dp_sizes``
  drops it; off by default, which keeps today's keys.
* The weights version fell back to the model path: for a local directory
  that does not change when other weights are put there. Now a warning, or an
  error with ``require_model_revision``.

Runs on the stand-ins of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_vllm import worker as worker_module  # noqa: E402


class DpSize(unittest.TestCase):
    def test_bound_by_default(self) -> None:
        self.assertEqual(worker_module._identity_dp_size({}, 4), 4)
        self.assertEqual(
            worker_module._identity_dp_size({"share_across_dp_sizes": "false"}, 4), 4)

    def test_dropped_when_shared(self) -> None:
        for value in (True, "true", "1"):
            with self.subTest(value=value):
                self.assertEqual(
                    worker_module._identity_dp_size(
                        {"share_across_dp_sizes": value}, 4),
                    1,
                )


class ModelRevision(unittest.TestCase):
    def _config(self, model, revision=None, commit=None):
        return SimpleNamespace(
            model=model, revision=revision,
            hf_config=SimpleNamespace(_commit_hash=commit),
        )

    def test_a_known_version_is_used(self) -> None:
        resolve = worker_module._resolve_model_revision
        self.assertEqual(resolve({"model_revision": "v3"}, self._config("org/m", "r")), "v3")
        self.assertEqual(resolve({}, self._config("org/m", "r", "c")), "r")
        self.assertEqual(resolve({}, self._config("org/m", None, "c")), "c")

    def test_a_hub_id_without_version_falls_back_quietly(self) -> None:
        with self.assertNoLogs(worker_module.logger, level=logging.WARNING):
            self.assertEqual(
                worker_module._resolve_model_revision({}, self._config("org/m")),
                "org/m",
            )

    def test_a_local_path_without_version_warns_or_fails(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            with self.assertLogs(worker_module.logger, level=logging.WARNING) as logs:
                self.assertEqual(
                    worker_module._resolve_model_revision({}, self._config(path)),
                    path,
                )
            self.assertIn("model_revision", logs.output[0])
            with self.assertRaises(ValueError):
                worker_module._resolve_model_revision(
                    {"require_model_revision": "true"}, self._config(path))
            # An explicit version settles it.
            self.assertEqual(
                worker_module._resolve_model_revision(
                    {"require_model_revision": "true", "model_revision": "w1"},
                    self._config(path)),
                "w1",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
