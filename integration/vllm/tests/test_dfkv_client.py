"""DfkvDeviceClient ABI-v2 construction and opt-in live GPU round trip."""
import ctypes
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dfkv_common import (  # noqa: E402
    DFKV_CLIENT_OPT_REGISTER_WITH_MDS,
    DfkvClientOptionsV2,
)
from dfkv_vllm import dfkv_client as client_module  # noqa: E402
from dfkv_vllm.dfkv_client import DfkvDeviceClient  # noqa: E402


def test_constructs_fully_configured_v2_client(monkeypatch):
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
                flags=options.flags,
                client_id=options.client_id,
                client_info=options.client_info,
                heartbeat=options.client_heartbeat_ms,
            )
            return 0xBEEF

        def dfkv_transport_mode(self, _handle):
            return b"rdma"

        def dfkv_version(self):
            return b"2.0.0"

        def dfkv_close(self, _handle):
            captured["close_calls"] = captured.get("close_calls", 0) + 1

    monkeypatch.setattr(client_module, "load_lib", lambda _path: FakeLib())
    monkeypatch.setattr(
        client_module._push_metrics, "configure", lambda *a, **k: None)
    monkeypatch.setattr(
        client_module._push_tracing, "configure", lambda *a, **k: None)
    monkeypatch.setattr(client_module._alog, "configure", lambda *a, **k: None)
    monkeypatch.setattr(client_module._hot_config, "register", lambda *a, **k: None)
    monkeypatch.setattr(client_module._hot_config, "start", lambda *a, **k: None)
    monkeypatch.setattr(client_module._hot_config, "stop", lambda *a, **k: None)
    monkeypatch.setenv("DFKV_CLIENT_STATS_POLL_S", "0")

    client = DfkvDeviceClient(
        key_namespace=b"vllm\x00layout",
        batch_concurrency=11,
        mds_endpoints="10.0.0.1:9400,10.0.0.2:9400",
        mds_group="prod",
        client_register=True,
        client_id="host:42:tp0",
        client_info="type=vllm,tp_size=8,tp_rank=0",
        client_heartbeat_ms=7000,
    )
    assert client.transport_mode == "rdma"
    assert captured == {
        "members": None,
        "namespace": b"vllm\x00layout",
        "batch": 11,
        "mds": b"10.0.0.1:9400,10.0.0.2:9400",
        "group": b"prod",
        "flags": DFKV_CLIENT_OPT_REGISTER_WITH_MDS,
        "client_id": b"host:42:tp0",
        "client_info": b"type=vllm,tp_size=8,tp_rank=0",
        "heartbeat": 7000,
    }
    client.close()
    client.close()
    assert captured["close_calls"] == 1


def test_telemetry_setup_failure_releases_handle_and_lifecycle(monkeypatch):
    calls = []

    class FakeLib:
        def dfkv_open_v2(self, _ptr):
            return 0xCAFE

        def dfkv_transport_mode(self, _handle):
            return b"rdma"

        def dfkv_version(self):
            return b"2.0.0"

        def dfkv_close(self, handle):
            calls.append(("close", handle))

    def fail_metrics(*_args, **_kwargs):
        calls.append(("metrics-configure",))
        raise ValueError("bad telemetry identity")

    monkeypatch.setattr(client_module, "load_lib", lambda _path: FakeLib())
    monkeypatch.setattr(client_module._push_metrics, "configure", fail_metrics)
    monkeypatch.setattr(
        client_module._push_metrics, "release",
        lambda: calls.append(("metrics-release",)))
    monkeypatch.setattr(
        client_module._push_tracing, "release",
        lambda: calls.append(("tracing-release",)))
    monkeypatch.setattr(client_module._hot_config, "stop", lambda: None)

    with pytest.raises(ValueError, match="bad telemetry identity"):
        DfkvDeviceClient(
            members="n1=127.0.0.1:28001",
            key_namespace=b"dfkv/model/v1/telemetry-failure",
        )

    assert calls == [
        ("metrics-configure",),
        ("close", 0xCAFE),
        ("metrics-release",),
        ("tracing-release",),
    ]


def test_rejects_and_closes_tcp_handle_before_connector_startup(monkeypatch):
    calls = []

    class FakeLib:
        def dfkv_open_v2(self, _ptr):
            return 0xCAFE

        def dfkv_transport_mode(self, handle):
            calls.append(("transport", handle))
            return b"tcp"

        def dfkv_close(self, handle):
            calls.append(("close", handle))

    startup_calls = []
    monkeypatch.setattr(client_module, "load_lib", lambda _path: FakeLib())
    monkeypatch.setattr(
        client_module._push_metrics,
        "configure",
        lambda *a, **k: startup_calls.append("metrics"),
    )
    monkeypatch.setattr(
        client_module._hot_config,
        "start",
        lambda *a, **k: startup_calls.append("hot_config"),
    )

    with pytest.raises(
        RuntimeError, match="requires GPUDirect RDMA transport"
    ):
        DfkvDeviceClient(
            members="n1=127.0.0.1:28001",
            key_namespace=b"dfkv/model/v1/test/tcp-reject",
        )

    assert calls == [("transport", 0xCAFE), ("close", 0xCAFE)]
    assert startup_calls == []


def test_register_memory_surfaces_native_mr_failure():
    class FakeLib:
        def dfkv_register_memory(self, _h, base, size):
            assert base.value == 0x1000
            assert size.value == 4096
            return -1

    client = object.__new__(DfkvDeviceClient)
    client._lib = FakeLib()
    client._h = 0xBEEF
    with pytest.raises(
        RuntimeError,
        match=r"dfkv_register_memory\(base=0x1000, size=4096\) rc=-1",
    ):
        client.register_memory(0x1000, 4096)


def test_fake_lib_batch_arrays_reach_every_active_abi(monkeypatch):
    calls = []

    class FakeLib:
        def dfkv_open_v2(self, _ptr):
            return 0xD00D

        def dfkv_transport_mode(self, _handle):
            return b"rdma"

        def dfkv_version(self):
            return b"test"

        def dfkv_close(self, _handle):
            calls.append("close")

        def dfkv_batch_put(
            self, _h, keys, key_lens, ptrs, sizes, n, out
        ):
            assert n == 1
            assert ctypes.string_at(keys[0], key_lens[0]) == b"put\x00\xff"
            assert list(key_lens) == [5]
            assert ptrs[0] == 0x1000 and sizes[0] == 64
            assert out._type_ is ctypes.c_int
            out[0] = 1
            calls.append("put")
            return 0

        def dfkv_batch_get_auto(
            self, _h, keys, key_lens, ptrs, caps, n, out_hit, out_len
        ):
            assert n == 1
            assert ctypes.string_at(keys[0], key_lens[0]) == b"get\x00\xfe-long"
            assert list(key_lens) == [10]
            assert ptrs[0] == 0x2000 and caps[0] == 64
            assert out_hit._type_ is ctypes.c_int
            out_hit[0] = 1
            out_len[0] = 48
            calls.append("get")
            return 0

        def dfkv_batch_put_sg(
            self, _h, keys, key_lens, ptrs, sizes, num_segs, n, out
        ):
            assert n == 1
            assert ctypes.string_at(
                keys[0], key_lens[0]) == b"put-sg\x00\xfd"
            assert list(key_lens) == [8]
            assert num_segs._type_ is ctypes.c_int
            assert list(num_segs) == [2]
            assert [ptrs[0][i] for i in range(2)] == [0x3000, 0x4000]
            assert [sizes[0][i] for i in range(2)] == [16, 32]
            assert out._type_ is ctypes.c_int
            out[0] = 1
            calls.append("put-sg")
            return 0

        def dfkv_batch_get_auto_sg(
            self, _h, keys, key_lens, ptrs, caps, num_segs, n,
            out_hit, out_len,
        ):
            assert n == 1
            assert ctypes.string_at(
                keys[0], key_lens[0]) == b"get-sg\x00\xfc-tail"
            assert list(key_lens) == [13]
            assert num_segs._type_ is ctypes.c_int
            assert list(num_segs) == [2]
            assert [ptrs[0][i] for i in range(2)] == [0x5000, 0x6000]
            assert [caps[0][i] for i in range(2)] == [24, 40]
            assert out_hit._type_ is ctypes.c_int
            out_hit[0] = 1
            out_len[0] = 64
            calls.append("get-sg")
            return 0

        def dfkv_batch_exist(self, _h, keys, key_lens, n, out):
            assert n == 2
            assert list(key_lens) == [3, 1]
            assert [
                ctypes.string_at(keys[i], key_lens[i]) for i in range(n)
            ] == [b"a\x00b", b"a"]
            assert out._type_ is ctypes.c_int
            out[0], out[1] = 1, 0
            calls.append("exist")
            return 0

    monkeypatch.setattr(client_module, "load_lib", lambda _path: FakeLib())
    monkeypatch.setattr(
        client_module._push_metrics, "configure", lambda *a, **k: None)
    monkeypatch.setattr(
        client_module._push_tracing, "configure", lambda *a, **k: None)
    monkeypatch.setattr(client_module._alog, "configure", lambda *a, **k: None)
    monkeypatch.setattr(client_module._hot_config, "register", lambda *a, **k: None)
    monkeypatch.setattr(client_module._hot_config, "start", lambda *a, **k: None)
    monkeypatch.setattr(client_module._hot_config, "stop", lambda *a, **k: None)
    monkeypatch.setenv("DFKV_CLIENT_STATS_POLL_S", "0")

    client = DfkvDeviceClient(
        members="n1=127.0.0.1:28001",
        key_namespace=b"dfkv/model/v1/test/fake-abi",
    )
    assert client.batch_put([b"put\x00\xff"], [0x1000], [64]) == [0]
    assert client.batch_get(
        [b"get\x00\xfe-long"], [0x2000], [64]) == ([1], [48])
    assert client.batch_put_sg(
        [b"put-sg\x00\xfd"], [[0x3000, 0x4000]], [[16, 32]]
    ) == [0]
    assert client.batch_get_auto_sg(
        [b"get-sg\x00\xfc-tail"], [[0x5000, 0x6000]], [[24, 40]]
    ) == ([1], [64])
    assert client.batch_exist([b"a\x00b", b"a"]) == [1, 0]
    client.close()

    assert calls == ["put", "get", "put-sg", "get-sg", "exist", "close"]


def test_roundtrip_gpu_pointers():
    torch = pytest.importorskip("torch")
    if not os.environ.get("DFKV_MEMBERS") or not torch.cuda.is_available():
        pytest.skip("needs DFKV_MEMBERS + CUDA + a running dfkv_server")
    c = DfkvDeviceClient(
        members=os.environ["DFKV_MEMBERS"],
        key_namespace=b"dfkv/model/v1/test/gpu-roundtrip",
        lib_path=os.environ.get("DFKV_LIB"),
    )
    n = 1 << 20
    a = torch.arange(n, dtype=torch.uint8, device="cuda")
    b = torch.zeros(n, dtype=torch.uint8, device="cuda")
    c.register_memory(a.data_ptr(), n)
    c.register_memory(b.data_ptr(), n)

    assert c.batch_put([b"k0"], [a.data_ptr()], [n]) == [0]
    hits, lens = c.batch_get([b"k0"], [b.data_ptr()], [n])
    torch.cuda.synchronize()
    assert hits == [1] and lens == [n]
    assert torch.equal(a, b)
    assert c.batch_exist([b"k0", b"missing-key"]) == [1, 0]
    c.close()
