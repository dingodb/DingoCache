#!/usr/bin/env python3
"""Standalone smoke test for the dfkv LMCache connector's native client.

Exercises the full ctypes datapath (DfkvNativeClient -> libdfkv.so -> a real
dfkv_server over TCP loopback), including the new variable-size get C ABI:
batch_set / batch_exists / batch_get with a mix of full and partial chunks,
verifying per-key hits, the reported true lengths, and byte-for-byte contents.

No torch / lmcache needed — native_client only depends on ctypes + access_log.

Usage:
    DFKV_LIB=build/libdfkv.so DFKV_SERVER=build/dfkv_server \
        python3 test/python/dfkv_lmcache_native_smoke.py
"""
import asyncio
import importlib
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "integration", "lmcache", "src"))

# Import native_client WITHOUT triggering dfkv_connector/__init__.py (which pulls
# in lmcache via adapter.py — lmcache needs Python >= 3.10; the connector's
# native client itself only needs ctypes + access_log). We install a stub
# package into sys.modules so the relative imports resolve against the real
# source dir but the package __init__ never runs.
_pkg = types.ModuleType("dfkv_connector")
_pkg.__path__ = [os.path.join(SRC, "dfkv_connector")]
sys.modules["dfkv_connector"] = _pkg
DfkvNativeClient = importlib.import_module(
    "dfkv_connector.native_client"
).DfkvNativeClient

PAGE_SIZE = 16
KEY_NAMESPACE = b"dfkv/model/v1/test/lmcache-native-smoke"


def _start_server(server_bin, cache_dir):
    proc = subprocess.Popen(
        [server_bin, "--dir", cache_dir, "--port", "0", "--cap", str(1 << 30)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # First stdout line is "PORT <n>".
    line = proc.stdout.readline().strip()
    if not line.startswith("PORT "):
        out = proc.stdout.read()
        proc.kill()
        raise RuntimeError(f"server did not report a port; got {line!r}\n{out}")
    return proc, int(line.split()[1])


async def _run(client):
    full = PAGE_SIZE * 4096  # arbitrary "full chunk" byte size
    part = full // 2                     # an "unfull" chunk (variable size)

    keys = [f"smoke@1@0@{i:x}" for i in range(6)]
    payloads = []
    for i, k in enumerate(keys):
        sz = full if i % 2 == 0 else part   # even=full, odd=partial
        payloads.append(bytearray((bytes([0x41 + (i % 26)]) * sz)))

    # PUT (source buffers = the payload bytearrays, exact sizes).
    ok, per_key = await client.batch_set(keys, [memoryview(p) for p in payloads])
    assert ok and all(per_key), f"batch_set failed: {per_key}"
    print(f"[ok] batch_set: {len(keys)} keys "
          f"(full={full}B, partial={part}B)")

    # EXISTS.
    ex = await client.batch_exists(keys)
    assert all(ex), f"batch_exists missed: {ex}"
    print(f"[ok] batch_exists: {sum(ex)}/{len(keys)} present")

    # GET into FULL-sized buffers; the variable-size get must report the true
    # stored length per key and fill exactly that many bytes.
    bufs = [bytearray(full) for _ in keys]
    ok, per_key, lengths = await client.batch_get(keys, [memoryview(b) for b in bufs])
    assert all(per_key), f"batch_get missed: {per_key}"
    for i, k in enumerate(keys):
        want = payloads[i]
        assert lengths[i] == len(want), \
            f"{k}: length {lengths[i]} != {len(want)}"
        assert bufs[i][:lengths[i]] == want, f"{k}: payload mismatch"
    print(f"[ok] batch_get: hits={sum(per_key)}/{len(keys)}, "
          f"lengths={lengths} (full+partial verified byte-for-byte)")

    # A miss in the batch reports hit=False, length=0.
    mbuf = [bytearray(full)]
    ok, per_key, lengths = await client.batch_get(
        [b"absent@1@0@deadbeef"], [memoryview(mbuf[0])])
    assert per_key == [False] and lengths == [0], (per_key, lengths)
    print("[ok] batch_get miss: hit=False, length=0")

    client.ping_sync()
    print("[ok] ping_sync")


def main():
    lib = os.environ.get("DFKV_LIB", os.path.join(HERE, "..", "..", "build", "libdfkv.so"))
    server = os.environ.get("DFKV_SERVER", os.path.join(HERE, "..", "..", "build", "dfkv_server"))
    lib = os.path.abspath(lib)
    server = os.path.abspath(server)
    assert os.path.exists(lib), f"missing {lib}"
    assert os.path.exists(server), f"missing {server}"

    with tempfile.TemporaryDirectory(prefix="dfkv_smoke_") as cache_dir:
        proc, port = _start_server(server, cache_dir)
        print(f"[info] dfkv_server on 127.0.0.1:{port}, lib={lib}")
        try:
            async def go():
                client = DfkvNativeClient(
                    raw_endpoint=f"n1=127.0.0.1:{port}",
                    group="",
                    membership="static",
                    key_namespace=KEY_NAMESPACE,
                    lib_path=lib,
                    get_parallelism=4,
                    loop=asyncio.get_running_loop(),
                )
                try:
                    await _run(client)
                finally:
                    client.close()
            asyncio.run(go())
            print("\nALL SMOKE CHECKS PASSED")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
