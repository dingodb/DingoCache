"""P0 GPUDirect probe (validated on hd04 H100 + IB, 2026-06-18).

Verifies the design-critical primitives the connector relies on:
  * dfkv_register_memory accepts a CUDA device pointer (GPUDirect MR),
  * a GPU->RDMA->server->RDMA->GPU round-trip is byte-correct via the batch
    path (the path the connector uses).

Run inside the vLLM image (torch + CUDA) against a running dfkv_server:

    DFKV_LIB=.../build-rdma/libdfkv.so \
    DFKV_MEMBERS=c1=<ip>:<rdma-port> \
    DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 \
        python integration/vllm/tests/test_gpudirect_regmem.py

NOTE: DFKV_MEMBERS must use the server's --rdma-port, and DFKV_RDMA=1 must be set
(a GPU pointer in TCP mode segfaults on the CPU-side copy).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dfkv_vllm.dfkv_client import DfkvDeviceClient  # noqa: E402


def main() -> int:
    members = os.environ.get("DFKV_MEMBERS", "c1=127.0.0.1:18901")
    client = DfkvDeviceClient(
        members=members,
        key_namespace=b"dfkv/model/v1/test/gpudirect",
        lib_path=os.environ.get("DFKV_LIB"))
    assert client.transport_mode == "rdma", (
        f"transport={client.transport_mode}; set DFKV_RDMA=1 + DFKV_RDMA_DEV "
        "(GPU pointers require RDMA)")

    n = 2 * 1024 * 1024
    src = torch.arange(n, dtype=torch.uint8, device="cuda")
    dst = torch.zeros(n, dtype=torch.uint8, device="cuda")

    # The crux: register GPU device pointers -> GPUDirect MR.
    client.register_memory(src.data_ptr(), n)
    client.register_memory(dst.data_ptr(), n)
    print("register_memory on GPU pointers: OK (GPUDirect MR created)")

    assert client.batch_put([b"p0/probe"], [src.data_ptr()], [n]) == [0]
    hits, lens = client.batch_get([b"p0/probe"], [dst.data_ptr()], [n])
    assert hits == [1] and lens == [n], f"hits={hits} lens={lens}"

    torch.cuda.synchronize()
    assert torch.equal(src, dst), "byte mismatch after GPUDirect round-trip"
    print(f"P0 PASS: GPUDirect round-trip byte-correct on "
          f"{torch.cuda.get_device_name()}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
