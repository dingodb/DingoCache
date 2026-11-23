"""Drift guard: the vendored telemetry copies inside the vLLM / LMCache pip
connectors must stay byte-identical to the canonical integration/hicache/dfkv_telemetry/.

If this fails, run ``deploy/sync_telemetry.sh`` (you edited the canonical files
but not the vendored copies, or vice-versa).
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

CANON = os.path.join(ROOT, "integration", "hicache", "dfkv_telemetry")
VENDORED = [
    os.path.join(ROOT, "integration", "vllm", "src", "dfkv_vllm", "_telemetry"),
    os.path.join(ROOT, "integration", "lmcache", "src", "dfkv_connector", "_telemetry"),
]
FILES = ["__init__.py", "config.py", "metrics_push.py", "otlp_json.py",
         "tracing.py", "otlp_traces.py"]


def _modules(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".py"))


class VendorSyncTest(unittest.TestCase):
    def test_every_telemetry_module_is_checked(self):
        # FILES is the list compared below: a module added to the canonical
        # package, or left behind in a vendored copy, escaped the guard.
        self.assertEqual(_modules(CANON), sorted(FILES),
                         "update FILES and deploy/sync_telemetry.sh")
        for vdir in VENDORED:
            self.assertEqual(_modules(vdir), sorted(FILES),
                             vdir + ": run deploy/sync_telemetry.sh")
        with open(os.path.join(ROOT, "deploy", "sync_telemetry.sh")) as fh:
            script = fh.read()
        synced = re.search(r'^FILES="([^"]*)"', script, re.M).group(1)
        self.assertEqual(sorted(synced.split()), sorted(FILES),
                         "deploy/sync_telemetry.sh copies another file list")
        # The directories it copies to are the ones checked here.
        for array, checked in (("DSTS", VENDORED), ("HOT_DSTS", HOT_VENDORED)):
            body = re.search(r"^" + array + r"=\(\n(.*?)^\)", script,
                             re.M | re.S).group(1)
            targets = [os.path.join(ROOT, t) for t in re.findall(r'"([^"]+)"', body)]
            self.assertEqual(sorted(targets), sorted(checked),
                             "deploy/sync_telemetry.sh " + array)

    def test_vendored_copies_are_byte_identical(self):
        for f in FILES:
            with open(os.path.join(CANON, f), "rb") as fh:
                canon = fh.read()
            for vdir in VENDORED:
                path = os.path.join(vdir, f)
                self.assertTrue(os.path.exists(path), "missing vendored file: " + path)
                with open(path, "rb") as fh:
                    got = fh.read()
                self.assertEqual(
                    got, canon,
                    "{} drifted from canonical; run deploy/sync_telemetry.sh".format(path))


# Standalone hot-config watcher: canonical dfkv_hot_config.py -> _hot_config.py
# vendored byte-identical into each connector package root.
HOT_CANON = os.path.join(ROOT, "integration", "hicache", "dfkv_hot_config.py")
HOT_VENDORED = [
    os.path.join(ROOT, "integration", "vllm", "src", "dfkv_vllm", "_hot_config.py"),
    os.path.join(ROOT, "integration", "lmcache", "src", "dfkv_connector", "_hot_config.py"),
]


class HotConfigVendorSyncTest(unittest.TestCase):
    def test_vendored_hot_config_is_byte_identical(self):
        with open(HOT_CANON, "rb") as fh:
            canon = fh.read()
        for path in HOT_VENDORED:
            self.assertTrue(os.path.exists(path), "missing vendored file: " + path)
            with open(path, "rb") as fh:
                got = fh.read()
            self.assertEqual(
                got, canon,
                "{} drifted from canonical; run deploy/sync_telemetry.sh".format(path))


# Connector access log: the vLLM copy is canonical, LMCache vendors it.
ACCESS_CANON = os.path.join(ROOT, "integration", "vllm", "src", "dfkv_vllm", "access_log.py")
ACCESS_VENDORED = os.path.join(
    ROOT, "integration", "lmcache", "src", "dfkv_connector", "access_log.py")


class AccessLogVendorSyncTest(unittest.TestCase):
    def test_lmcache_access_log_is_byte_identical(self):
        with open(ACCESS_CANON, "rb") as fh:
            canon = fh.read()
        with open(ACCESS_VENDORED, "rb") as fh:
            got = fh.read()
        self.assertEqual(
            got, canon,
            "{} drifted from canonical; run deploy/sync_telemetry.sh".format(ACCESS_VENDORED))


if __name__ == "__main__":
    unittest.main()
