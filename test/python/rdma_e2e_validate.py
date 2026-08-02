#!/usr/bin/env python3
"""End-to-end zero-copy validation against a REMOTE dfkv_server over RDMA.

Unlike test_dfkv_hicache.py (which spawns local TCP nodes), this drives the real
SGLang-facing plugin stack — DfkvHiCache (ctypes) -> libdfkv.so -> KVClient ->
RDMA scatter -> dfkv_server — against a running remote cache node, and asserts the
GET payload lands directly in the numpy host-pool buffer (zero copy).

Run on the client node:
    DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 \
    DFKV_MEMBERS='n=10.0.0.2:28001' DFKV_PAGE_BYTES=2752512 DFKV_PAGES=32 \
    python3 rdma_e2e_validate.py

The server must run with a matching --rdma-port and --rdma-dev on the same fabric.
Exits non-zero on any mismatch/miss. Env-configurable so it is fabric-agnostic.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # shim 'sglang' + FakeMlaPool
sys.path.insert(0, os.path.join(HERE, "..", "..", "integration", "hicache"))
os.environ.setdefault("DFKV_BUILD", os.path.join(HERE, "..", "..", "build"))

import numpy as np  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # noqa: E402
import dfkv_hicache  # noqa: E402
from test_dfkv_hicache import FakeMlaPool  # noqa: E402

MEMBERS = os.environ.get("DFKV_MEMBERS", "n=127.0.0.1:28001")
PAGE_BYTES = int(os.environ.get("DFKV_PAGE_BYTES", str(2752512)))  # GLM-5.1 MLA page
PAGE_SIZE = int(os.environ.get("DFKV_PAGE_SIZE", "64"))
PAGES = int(os.environ.get("DFKV_PAGES", "32"))


def _cfg():
    return HiCacheStorageConfig(
        tp_rank=0, tp_size=8, is_mla_model=True, is_page_first_layout=False,
        model_name="glm-5.1",
        extra_config={"members": MEMBERS, "interface_v1": 1})


def main():
    print("members=%s rdma=%s dev=%s pages=%d page_bytes=%d" % (
        MEMBERS, os.environ.get("DFKV_RDMA"), os.environ.get("DFKV_RDMA_DEV"),
        PAGES, PAGE_BYTES))
    cfg = _cfg()
    pool = FakeMlaPool(PAGES, PAGE_BYTES, PAGE_SIZE)
    st = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
    st.register_mem_pool_host(pool)

    keys = ["e2e-%d" % i for i in range(PAGES)]
    for i in range(PAGES):
        pool.fill_page(i, (i * 7 + 1) & 0xFF)
    host_indices = list(range(PAGES * PAGE_SIZE))

    t0 = time.time(); sset = st.batch_set_v1(keys, host_indices); dt = time.time() - t0
    set_ok = sum(bool(x) for x in sset)
    print("batch_set_v1: %d/%d  %.2f GB/s" % (set_ok, PAGES, PAGES * PAGE_BYTES / 1e9 / dt))

    expected = [pool.page_bytes_at(i) for i in range(PAGES)]
    pool.zero()  # a correct GET must refill the exact bytes via RDMA scatter
    t0 = time.time(); sget = st.batch_get_v1(keys, host_indices); dt = time.time() - t0
    hits = sum(bool(x) for x in sget)
    match = all(pool.page_bytes_at(i) == expected[i] for i in range(PAGES))
    print("batch_get_v1: %d/%d hit  %.2f GB/s  bytes_match=%s" % (
        hits, PAGES, PAGES * PAGE_BYTES / 1e9 / dt, match))

    pool.zero()
    smiss = st.batch_get_v1(["absent-%d" % i for i in range(PAGES)], host_indices)
    no_false_hit = not any(bool(x) for x in smiss)
    print("miss check: all-miss=%s" % no_false_hit)

    ok = set_ok == PAGES and hits == PAGES and match and no_false_hit
    print("RESULT:", "ZERO-COPY RDMA E2E OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
