# SPDX-License-Identifier: Apache-2.0
"""Construction of the in-process LMCache connector.

* In LMCache's scheduler role the ConnectorContext's local_cpu_backend is a
  SafeLocalCPUBackend stand-in without ``config`` or ``metadata``; the
  connector read them from it and failed with AttributeError. It now takes
  them from the context.
* A failure after the native client is open closes it, and is logged as an
  error (LMCache turns a construction error into a warning and runs on
  without the backend).
* DfkvNativeClient closes its handle when an RDMA pool registration fails.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import _lmcache_stub  # noqa: E402

_lmcache_stub.install()

import dfkv_connector.native_client as nc  # noqa: E402
import dfkv_connector.remote_connector as rc  # noqa: E402
from dfkv_connector.adapter import DfkvConnectorAdapter  # noqa: E402


def _metadata():
    dtype = SimpleNamespace(itemsize=2)
    return SimpleNamespace(
        get_shapes=lambda: [(2, 4, 16, 8)], get_dtypes=lambda: [dtype],
        use_mla=False, chunk_size=16, world_size=1, worker_id=0,
        model_name="m", local_world_size=1,
    )


def _config():
    return SimpleNamespace(use_layerwise=False, remote_serde="naive")


class _Client:
    instances: list = []
    fail_validate = False

    def __init__(self, **_kwargs) -> None:
        self.closed = False
        self.transport_mode = "fake"
        _Client.instances.append(self)

    def validate_block_sizes(self, sizes, *, context):
        if _Client.fail_validate:
            raise ValueError("chunk over the bound")

    def close(self) -> None:
        self.closed = True


class _SafeLocalCPUBackend:
    """LMCache's scheduler-role stand-in: no config, no metadata."""


class Construction(unittest.TestCase):
    def setUp(self) -> None:
        _Client.instances = []
        _Client.fail_validate = False
        p = patch.object(rc, "DfkvNativeClient", _Client)
        p.start()
        self.addCleanup(p.stop)
        self.loop = asyncio.new_event_loop()
        self.addCleanup(self.loop.close)

    def _create(self):
        context = SimpleNamespace(
            url="dfkv://127.0.0.1:1/g", loop=self.loop,
            local_cpu_backend=_SafeLocalCPUBackend(),
            config=_config(), metadata=_metadata(),
        )
        return DfkvConnectorAdapter().create_connector(context)

    def test_the_scheduler_role_constructs(self) -> None:
        connector = self._create()
        self.assertEqual(connector.full_chunk_size_bytes, 2 * 4 * 16 * 8 * 2)
        self.assertEqual(len(_Client.instances), 1)

    def test_a_failure_after_open_closes_the_client_and_is_an_error(self) -> None:
        _Client.fail_validate = True
        with self.assertLogs(rc.logger, level=logging.ERROR):
            with self.assertRaises(ValueError):
                self._create()
        self.assertTrue(_Client.instances[0].closed, "the open client leaked")


class HashSeedWarning(Construction):
    def test_said_once_per_process(self) -> None:
        with patch.dict(os.environ), patch.object(rc, "_warned_hash_seed", False):
            os.environ.pop("PYTHONHASHSEED", None)
            with self.assertLogs(rc.logger, level=logging.WARNING) as logs:
                self._create()
                self._create()
        said = [m for m in logs.output if "PYTHONHASHSEED" in m]
        self.assertEqual(len(said), 1, logs.output)


class _Lib:
    def __init__(self) -> None:
        self.closed = []

    def dfkv_open_v2(self, _options):
        return 0x1234

    def dfkv_transport_mode(self, _h):
        return b"rdma"

    def dfkv_register_memory(self, _h, _base, _size):
        return -1  # the registration fails

    def dfkv_close(self, h):
        self.closed.append(h)


class NativeHandle(unittest.TestCase):
    def test_a_failed_registration_closes_the_handle(self) -> None:
        lib = _Lib()
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        with patch.object(nc, "load_lib", lambda _path: lib):
            with self.assertRaises(RuntimeError):
                nc.DfkvNativeClient(
                    raw_endpoint="127.0.0.1:1", group="g", membership="static",
                    key_namespace=b"ns", rdma_pools=[(0x1000, 4096)], loop=loop,
                )
        self.assertEqual(lib.closed, [0x1234])


if __name__ == "__main__":
    unittest.main(verbosity=2)
