"""DfkvNativeClient constructs one fully configured ABI-v2 handle."""

import asyncio
import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dfkv_common import (  # noqa: E402
    DFKV_CLIENT_OPT_REGISTER_WITH_MDS,
    DfkvClientOptionsV2,
)
from dfkv_connector import native_client as client_module  # noqa: E402
from dfkv_connector.native_client import DfkvNativeClient  # noqa: E402


def test_mds_discovery_and_registration_are_atomic_open_options(monkeypatch):
    captured = {}

    class FakeLib:
        def dfkv_open_v2(self, ptr):
            options = ctypes.cast(
                ptr, ctypes.POINTER(DfkvClientOptionsV2)).contents
            captured.update(
                members=options.members,
                namespace=ctypes.string_at(
                    options.key_namespace, options.key_namespace_len),
                batch=options.batch_concurrency,
                mds=options.mds_endpoints,
                group=options.mds_group,
                poll_ms=options.mds_poll_ms,
                flags=options.flags,
                client_id=options.client_id,
                client_info=options.client_info,
                heartbeat=options.client_heartbeat_ms,
            )
            return 0xCAFE

        def dfkv_transport_mode(self, _handle):
            return b"rdma"

        def dfkv_version(self):
            return b"2.0.0"

        def dfkv_close(self, _handle):
            captured["closed"] = True

    monkeypatch.setattr(client_module, "load_lib", lambda _path: FakeLib())
    monkeypatch.setattr(
        client_module._tcfg, "resolve_connector_id",
        lambda *_a, **_k: "host:77:tp3")
    monkeypatch.setattr(
        client_module._push_metrics, "configure", lambda *a, **k: None)
    monkeypatch.setattr(
        client_module._push_tracing, "configure", lambda *a, **k: None)
    monkeypatch.setattr(client_module._alog, "configure", lambda *a, **k: None)
    monkeypatch.setattr(client_module._hot_config, "register", lambda *a, **k: None)
    monkeypatch.setattr(client_module._hot_config, "start", lambda *a, **k: None)
    monkeypatch.setattr(client_module._hot_config, "stop", lambda *a, **k: None)
    monkeypatch.setenv("DFKV_CLIENT_REGISTER", "1")

    async def construct_and_close():
        client = DfkvNativeClient(
            raw_endpoint="10.0.0.1:9400,10.0.0.2:9400",
            group="prod",
            membership="mds",
            key_namespace=b"lmcache\x00layout",
            tp_size=8,
            tp_rank=3,
            mds_poll_ms=1700,
            get_parallelism=4,
        )
        assert client.transport_mode == "rdma"
        client.close()

    asyncio.run(construct_and_close())

    assert captured["members"] is None
    assert captured["namespace"] == b"lmcache\x00layout"
    assert captured["batch"] == 0
    assert captured["mds"] == b"10.0.0.1:9400,10.0.0.2:9400"
    assert captured["group"] == b"prod"
    assert captured["poll_ms"] == 1700
    assert captured["flags"] == DFKV_CLIENT_OPT_REGISTER_WITH_MDS
    assert captured["client_id"] == b"host:77:tp3"
    assert b"type=lmcache" in captured["client_info"]
    assert b"tp_size=8" in captured["client_info"]
    assert b"tp_rank=3" in captured["client_info"]
    assert captured["heartbeat"] == 10000
    assert captured["closed"]

def test_binary_keys_and_lengths_reach_every_native_operation():
    keys = [b"a\x00\xff", b"a"]
    seen = []

    def assert_keys(key_ptrs, key_lens):
        assert list(key_lens) == [3, 1]
        assert [
            ctypes.string_at(key_ptrs[i], key_lens[i])
            for i in range(2)
        ] == keys

    class FakeLib:
        def dfkv_batch_put(
            self, _h, key_ptrs, key_lens, ptrs, sizes, n, out,
        ):
            assert n == 2
            assert_keys(key_ptrs, key_lens)
            assert list(sizes) == [3, 4]
            assert out._type_ is ctypes.c_int
            out[0], out[1] = 1, 1
            seen.append("put")
            return 0

        def dfkv_batch_get_auto(
            self, _h, key_ptrs, key_lens, ptrs, caps, n,
            out_hit, out_len,
        ):
            assert n == 2
            assert_keys(key_ptrs, key_lens)
            assert list(caps) == [3, 4]
            assert out_hit._type_ is ctypes.c_int
            out_hit[0], out_hit[1] = 1, 0
            out_len[0], out_len[1] = 3, 0
            seen.append("get")
            return 0

        def dfkv_batch_exist(self, _h, key_ptrs, key_lens, n, out):
            assert n == 2
            assert_keys(key_ptrs, key_lens)
            assert out._type_ is ctypes.c_int
            out[0], out[1] = 1, 0
            seen.append("exist")
            return 0

        def dfkv_batch_remove(self, _h, key_ptrs, key_lens, n, out):
            assert n == 2
            assert_keys(key_ptrs, key_lens)
            assert out._type_ is ctypes.c_int
            out[0], out[1] = 1, 1
            seen.append("remove")
            return 0

        def dfkv_exist(self, _h, key_ptr, key_len):
            assert ctypes.string_at(key_ptr, key_len.value) == keys[0]
            seen.append("exist-one")
            return 1

        def dfkv_remove(self, _h, key_ptr, key_len):
            assert ctypes.string_at(key_ptr, key_len.value) == keys[0]
            seen.append("remove-one")
            return 1

    client = object.__new__(DfkvNativeClient)
    client._lib = FakeLib()
    client._h = 0xBEEF
    buffers = [memoryview(bytearray(3)), memoryview(bytearray(4))]

    assert client._batch_set_blocking(keys, buffers) == (True, [True, True])
    assert client._batch_get_blocking(keys, buffers) == (
        False, [True, False], [3, 0])
    assert client._batch_exists_blocking(keys) == [True, False]
    assert client._batch_remove_blocking(keys) == [True, True]
    assert client.exists_sync(keys[0])
    assert client.remove_sync(keys[0])
    assert seen == [
        "put", "get", "exist", "remove", "exist-one", "remove-one"]


def test_binary_key_log_label_is_digest_only():
    label = client_module._key_label(b"\xff\x00raw-secret")
    assert label.startswith("len=12 sha256=")
    assert len(label.removeprefix("len=12 sha256=")) == 16
    assert "raw-secret" not in label


def test_register_memory_surfaces_native_mr_failure():
    class FakeLib:
        def dfkv_register_memory(self, _h, _base, _size):
            return -1

    client = object.__new__(DfkvNativeClient)
    client._lib = FakeLib()
    client._h = 0xBEEF
    try:
        client._register_memory(0x1000, 4096)
    except RuntimeError as exc:
        assert "dfkv_register_memory(base=0x1000, size=4096) rc=-1" == str(exc)
    else:
        raise AssertionError("MR registration failure was not surfaced")
