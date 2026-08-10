# SPDX-License-Identifier: Apache-2.0
"""Runtime hot-reload of the access-log knob (dfkv_hot_config + dfkv_access_log).

Pure-Python: no native lib, no cache nodes. Drives dfkv_access_log.apply_hot()
directly and the dfkv_hot_config file watcher end-to-end.
"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "integration", "hicache"))

import dfkv_access_log as alog  # noqa: E402
import dfkv_hot_config as hot   # noqa: E402


def _reset():
    """Reset both modules' process-global state for test isolation."""
    alog._stop_listener(alog._listener)
    import logging
    logging.getLogger("dfkv.access").handlers.clear()
    alog._ENABLED = False
    alog._THRESHOLD_US = 0
    alog._logger = None
    alog._listener = None
    alog._configured = False
    alog._sink_want = None
    alog._launch_cfg = {}
    hot.stop()
    with hot._lock:
        hot._appliers.clear()
        hot._watcher = None
    os.environ.pop("DFKV_HOT_CONFIG", None)
    os.environ.pop("DFKV_HOT_CONFIG_POLL_S", None)


class AccessLogHotToggleTest(unittest.TestCase):
    def setUp(self):
        _reset()
        self.tmp = tempfile.mkdtemp(prefix="dfkv_hot_")
        self.logpath = os.path.join(self.tmp, "acc.log")

    def tearDown(self):
        _reset()

    def _emit(self, op):
        with alog.access_log(op, lambda: "x"):
            pass

    def _flush(self):
        if alog._listener is not None:
            alog._listener.stop()
            alog._listener = None

    def _read(self):
        p = self.logpath + ".r0"
        if not os.path.exists(p):
            return ""
        with open(p) as f:
            return f.read()

    def test_hot_enable_then_revert(self):
        # launch disabled
        alog.configure({"access_log": False}, tp_rank=0, model="m")
        self.assertFalse(alog.is_enabled())
        self._emit("get")
        self.assertEqual(self._read(), "")

        # hot enable
        alog.apply_hot({"access_log": True, "access_log_path": self.logpath})
        self.assertTrue(alog.is_enabled())
        self._emit("batch_get")
        self._flush()
        self.assertIn("batch_get", self._read())

        # empty overrides -> revert to launch (disabled)
        alog.apply_hot({})
        self.assertFalse(alog.is_enabled())

    def test_threshold_hot_change(self):
        alog.configure({"access_log": True, "access_log_path": self.logpath},
                       tp_rank=0, model="m")
        # 10s threshold suppresses a sub-ms op
        alog.apply_hot({"access_log": True, "access_log_path": self.logpath,
                        "access_log_threshold_us": 10_000_000})
        n0 = self._read().count("\n")
        self._emit("fast")
        self.assertEqual(self._read().count("\n"), n0)
        # threshold 0 logs again
        alog.apply_hot({"access_log": True, "access_log_path": self.logpath,
                        "access_log_threshold_us": 0})
        self._emit("logged")
        self._flush()
        self.assertIn("logged", self._read())

    def test_apply_hot_before_configure_is_noop(self):
        # no configure() yet -> apply_hot must not enable
        alog.apply_hot({"access_log": True, "access_log_path": self.logpath})
        self.assertFalse(alog.is_enabled())


class WatcherOptInTest(unittest.TestCase):
    def setUp(self):
        _reset()
        self.tmp = tempfile.mkdtemp(prefix="dfkv_hotw_")
        self.logpath = os.path.join(self.tmp, "acc.log")
        self.ctl = os.path.join(self.tmp, "hot.json")

    def tearDown(self):
        _reset()

    def test_watcher_disabled_without_explicit_path(self):
        hot.register("access_log", alog.apply_hot)
        self.assertIsNone(hot.start({}, tp_rank=0))

    def test_watcher_file_lifecycle(self):
        alog.configure({"access_log": False}, tp_rank=0, model="m")
        hot.register("access_log", alog.apply_hot)
        os.environ["DFKV_HOT_CONFIG"] = self.ctl
        os.environ["DFKV_HOT_CONFIG_POLL_S"] = "0.2"
        w = hot.start({}, tp_rank=0)
        self.assertIsNotNone(w)
        time.sleep(0.4)  # first tick, file absent -> disabled
        self.assertFalse(alog.is_enabled())

        with open(self.ctl, "w") as f:
            json.dump({"access_log": True, "access_log_path": self.logpath}, f)
        self._wait(lambda: alog.is_enabled(), 3.0)
        self.assertTrue(alog.is_enabled())

        os.remove(self.ctl)
        self._wait(lambda: not alog.is_enabled(), 3.0)
        self.assertFalse(alog.is_enabled())

    def _wait(self, pred, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return
            time.sleep(0.1)


# Garbage/malformed control-file payloads that must never crash or kill the
# watcher. (name, raw-bytes-or-text) pairs.
_GARBAGE = [
    ("binary", b"\x00\x01\x02\xff\xfe garbage \x80\x81"),
    ("truncated_json", b'{"access_log": tru'),
    ("not_json", b"this is not json at all !!!"),
    ("json_list", b"[1, 2, 3]"),
    ("json_number", b"42"),
    ("json_string", b'"hello"'),
    ("json_null", b"null"),
    ("empty", b""),
    ("deeply_nested", b"[" * 5000),  # RecursionError from json
    ("bad_threshold_type", b'{"access_log_threshold_us": "abc"}'),
    ("bad_enable_type", b'{"access_log": {"nested": 1}}'),
    ("bad_path_type", b'{"access_log_path": 12345}'),
    ("stray_placeholder", b'{"access_log": true, "access_log_path": "/tmp/x/{foo}.log"}'),
]


class WatcherRobustnessTest(unittest.TestCase):
    """A malformed/garbage control file must never crash the program or kill the
    watcher thread; it must keep the last good config and recover afterwards."""

    def setUp(self):
        _reset()
        self.tmp = tempfile.mkdtemp(prefix="dfkv_hotr_")
        self.ctl = os.path.join(self.tmp, "hot.json")
        self.logpath = os.path.join(self.tmp, "acc.log")

    def tearDown(self):
        _reset()

    def test_read_never_raises_on_garbage(self):
        w = hot._Watcher(self.ctl, 0.1, 0)
        for name, payload in _GARBAGE:
            with open(self.ctl, "wb") as f:
                f.write(payload)
            try:
                out = w._read()
            except Exception as exc:  # noqa: BLE001
                self.fail(f"_read raised on {name!r}: {type(exc).__name__}: {exc}")
            # never a crash; either ignored (None) or coerced to a dict
            self.assertTrue(out is None or isinstance(out, dict),
                            f"{name}: unexpected {out!r}")

    def test_apply_hot_survives_bad_value_types(self):
        alog.configure({"access_log": False}, tp_rank=0, model="m")
        for bad in ({"access_log_threshold_us": "abc"},
                    {"access_log": {"nested": 1}},
                    {"access_log_path": 12345},
                    {"access_log": True, "access_log_path": "/tmp/x/{foo}.log"}):
            try:
                alog.apply_hot(bad)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"apply_hot raised on {bad!r}: {exc}")
        # still responsive to a good config afterwards
        alog.apply_hot({"access_log": True, "access_log_path": self.logpath})
        self.assertTrue(alog.is_enabled())

    def test_watcher_survives_garbage_then_recovers(self):
        alog.configure({"access_log": False}, tp_rank=0, model="m")
        hot.register("access_log", alog.apply_hot)
        os.environ["DFKV_HOT_CONFIG"] = self.ctl
        os.environ["DFKV_HOT_CONFIG_POLL_S"] = "0.15"
        w = hot.start({}, tp_rank=0)
        self.assertIsNotNone(w)
        # fire every garbage payload through the live watcher loop
        for _, payload in _GARBAGE:
            with open(self.ctl, "wb") as f:
                f.write(payload)
            time.sleep(0.2)
            self.assertTrue(w.is_alive(), "watcher thread died on garbage")
        # recovery: a good file must still take effect
        with open(self.ctl, "w") as f:
            json.dump({"access_log": True, "access_log_path": self.logpath}, f)
        self._wait(lambda: alog.is_enabled(), 3.0)
        self.assertTrue(alog.is_enabled(), "watcher did not recover after garbage")
        self.assertTrue(w.is_alive())

    def _wait(self, pred, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return
            time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
