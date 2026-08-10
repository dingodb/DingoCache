# SPDX-License-Identifier: Apache-2.0
"""Hot-reload of access_log in the vLLM + LMCache connectors (Phase 1.5).

Loads each connector's access_log.py and its vendored _hot_config.py directly
by file path (stdlib-only modules, so no native lib / package __init__ needed)
and exercises the same toggle/threshold/watcher behavior proven for HiCache.
"""
import importlib.util
import json
import os
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

CONNECTORS = {
    "vllm": os.path.join(ROOT, "integration", "vllm", "src", "dfkv_vllm"),
    "lmcache": os.path.join(ROOT, "integration", "lmcache", "src", "dfkv_connector"),
}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ConnectorAccessLogHotTest(unittest.TestCase):
    def _mods(self, connector):
        d = CONNECTORS[connector]
        # unique module names per connector so the two don't collide in sys
        alog = _load(os.path.join(d, "access_log.py"), f"_alog_{connector}")
        hot = _load(os.path.join(d, "_hot_config.py"), f"_hot_{connector}")
        return alog, hot

    def _run_for(self, connector):
        saved_env = {
            key: value for key, value in os.environ.items()
            if key.startswith("DFKV_")
        }
        self.addCleanup(self._restore_dfkv_env, saved_env)
        self._restore_dfkv_env({})
        alog, hot = self._mods(connector)
        tmp = tempfile.mkdtemp(prefix=f"dfkv_{connector}_")
        logpath = os.path.join(tmp, "acc.log")
        ctl = os.path.join(tmp, "hot.json")

        def emit(op):
            with alog.access_log(op, lambda: "x"):
                pass

        def flush():
            if alog._listener is not None:
                alog._listener.stop()
                alog._listener = None

        def read():
            p = logpath + ".r0"
            if not os.path.exists(p):
                return ""
            with open(p) as f:
                return f.read()

        # launch disabled
        alog.configure({}, tp_rank=0)
        self.assertFalse(alog.is_enabled(), f"{connector}: launch disabled")
        emit("get")
        self.assertEqual(read(), "", f"{connector}: nothing written when off")

        # hot enable
        alog.apply_hot({"access_log": True, "access_log_path": logpath})
        self.assertTrue(alog.is_enabled(), f"{connector}: hot enabled")
        emit("batch_get")
        flush()
        self.assertIn("batch_get", read(), f"{connector}: op written")

        # revert
        alog.apply_hot({})
        self.assertFalse(alog.is_enabled(), f"{connector}: reverted")

        # garbage control file must not crash the vendored watcher
        w = hot._Watcher(ctl, 0.1, 0)
        for payload in (b"\x00\xffnot json", b"[1,2,3]", b"", b"[" * 4000,
                        b'{"access_log_threshold_us":"x"}'):
            with open(ctl, "wb") as f:
                f.write(payload)
            try:
                out = w._read()
            except Exception as exc:  # noqa: BLE001
                self.fail(f"{connector}: _read raised on garbage: {exc}")
            self.assertTrue(out is None or isinstance(out, dict))

        # end-to-end watcher: opt-in + file-driven enable
        alog._configured = False  # reset launch snapshot for a clean baseline
        alog.configure({}, tp_rank=0)
        hot.register("access_log", alog.apply_hot)
        os.environ["DFKV_HOT_CONFIG"] = ctl
        os.environ["DFKV_HOT_CONFIG_POLL_S"] = "0.15"
        os.remove(ctl)
        watcher = hot.start({}, tp_rank=0)
        self.assertIsNotNone(watcher, f"{connector}: watcher started")
        with open(ctl, "w") as f:
            json.dump({"access_log": True, "access_log_path": logpath}, f)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not alog.is_enabled():
            time.sleep(0.1)
        self.assertTrue(alog.is_enabled(), f"{connector}: watcher file-driven enable")
        self.assertTrue(watcher.is_alive())
        hot.stop()

    @staticmethod
    def _restore_dfkv_env(saved):
        for key in list(os.environ):
            if key.startswith("DFKV_"):
                del os.environ[key]
        os.environ.update(saved)

    def test_vllm(self):
        self._run_for("vllm")

    def test_lmcache(self):
        self._run_for("lmcache")


if __name__ == "__main__":
    unittest.main()
