"""Small fixes in the vLLM connector. Runs on the stand-ins of
test_dfkv_vllm_worker.

* The TP rank the worker knows labels the client's metrics and names its
  access log; it was read from the environment (vLLM sets none of the
  variables), so every rank was rank 0 and shared one access-log file.
* The save-retention SET is timed from its own start (its recorded duration
  was always 0).
* The hot-config watcher keeps its stop event out of threading.Thread's own
  ``_stop``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_vllm import _hot_config  # noqa: E402
from dfkv_vllm import dfkv_client as client_module  # noqa: E402
from dfkv_vllm.worker import KVCacheStoreSendingThread  # noqa: E402


class _Lib:
    def dfkv_open_v2(self, _options):
        return 0xBEEF

    def dfkv_transport_mode(self, _handle):
        return b"rdma"

    def dfkv_version(self):
        return b"2.0.0"

    def dfkv_close(self, _handle):
        pass


class ClientRank(unittest.TestCase):
    def test_the_given_tp_rank_is_used(self) -> None:
        seen = {}
        with patch.object(client_module, "load_lib", lambda _p: _Lib()), \
                patch.object(client_module.dfkv_cabi, "bind_retain", lambda _l: False), \
                patch.object(client_module._push_metrics, "configure",
                             lambda *a, **k: seen.setdefault("metrics", k["tp_rank"])), \
                patch.object(client_module._push_tracing, "configure",
                             lambda *a, **k: seen.setdefault("tracing", k["tp_rank"])), \
                patch.object(client_module._alog, "configure",
                             lambda *a, **k: seen.setdefault("access_log", k["tp_rank"])), \
                patch.object(client_module, "_acquire_hot_config",
                             lambda rank: seen.setdefault("hot_config", rank)), \
                patch.dict(os.environ, {"DFKV_CLIENT_STATS_POLL_S": "0"}):
            for name in ("DFKV_TP_RANK", "LOCAL_RANK", "RANK"):
                os.environ.pop(name, None)
            client = client_module.DfkvDeviceClient(
                key_namespace=b"ns", members="n=127.0.0.1:1", client_register=False,
                tp_rank=5)
            client._hot_config_acquired = False
            client.close()
        self.assertEqual(seen, {"metrics": 5, "tracing": 5, "access_log": 5,
                                "hot_config": 5})


class RetainTiming(unittest.TestCase):
    def test_the_retention_set_is_timed_from_its_start(self) -> None:
        recorded = []
        sender = object.__new__(KVCacheStoreSendingThread)
        sender._record_operation_cb = (
            lambda operation, duration_seconds, *a, **k:
                recorded.append((operation, duration_seconds)))
        start = time.perf_counter()
        time.sleep(0.02)
        sender._note_retain_codes("save_retain_set", [1, 1], start)
        self.assertEqual(recorded[0][0], "save_retain_set")
        self.assertGreaterEqual(recorded[0][1], 0.015)


class HotConfigWatcher(unittest.TestCase):
    def test_threads_own_stop_is_not_shadowed(self) -> None:
        watcher = _hot_config._Watcher("/nonexistent/dfkv-hot", 0.01, 0)
        self.assertNotIsInstance(watcher.__dict__.get("_stop"), threading.Event)
        watcher.start()
        watcher.stop()
        watcher.join(2)
        self.assertFalse(watcher.is_alive())
        watcher.join(0.1)  # joining a finished thread must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
