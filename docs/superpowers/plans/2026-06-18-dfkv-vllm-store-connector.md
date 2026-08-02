# DfkvStoreConnector Implementation Plan

> **OBSOLETE HISTORICAL PLAN — do not use as current API guidance.** The
> implementation now opens with
> `dfkv_open(members, namespace_bytes, namespace_len)`, uses canonical
> `dfkv/pool/v2|...` object keys, and stores opaque raw values. References below
> to the retired identity, geometry arguments, and value format are intentionally
> preserved as a record of the June 2026 plan. Current guidance:
> [CONNECTORS.md](../../CONNECTORS.md) and
> [ARCHITECTURE.md](../../ARCHITECTURE.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a direct vLLM KV connector (`DfkvStoreConnector`, implementing vLLM 0.23.0's `KVConnectorBase_V1`) that stores/loads KV cache to/from a dfkv cluster over GPUDirect RDMA, bypassing LMCache — a drop-in for the `--kv-transfer-config` slot Mooncake occupies today.

**Architecture:** Out-of-tree vLLM plugin in the dfkv repo under `integration/vllm/`, loaded via `kv_connector_module_path`. Mirrors the production `MooncakeStoreConnector` three-part structure (connector / scheduler / worker), with all dfkv contact collapsed into one new unit, `dfkv_client.py`, that drives the existing `libdfkv.so` C-ABI on **raw GPU device pointers**. Real I/O is issued per-request in `get_finished()` (no layerwise), exactly as Mooncake does.

**Tech Stack:** Python 3.12, ctypes over `libdfkv.so`, vLLM 0.23.0 (openclaw image), PyTorch (CUDA), InfiniBand verbs + nvidia-peermem (GPUDirect), DeepSeek-V4-Flash (MLA). Dev/validate on isolated node hd04-gpu1-0065 via `tsh-go`.

**Reference template (read-only, ai_david KB, outside this repo):**
`01-工作/20260618-210410-dfkv对接vllm-MooncakeStore模板/` — full `mooncake/store/*` source + `DfkvStoreConnector-施工图.md` seam map. Throughout this plan, "TEMPLATE/<file>" means that file.

**Spec:** `docs/superpowers/specs/2026-06-18-dfkv-vllm-store-connector-design.md`

---

## File Structure

All new code under `integration/vllm/` (symmetric to `integration/lmcache/`):

```
integration/vllm/
  pyproject.toml                       # package "dfkv-vllm", deps: none at runtime beyond vllm/torch
  README.md                            # enable string + extra_config keys
  src/dfkv_vllm/
    __init__.py
    connector.py      # DfkvStoreConnector(KVConnectorBase_V1) — thin role dispatcher
    scheduler.py      # DfkvStoreScheduler — hit decision
    worker.py         # DfkvStoreWorker — register_kv_caches, put/get, async threads, LookupKeyServer
    dfkv_client.py    # NEW core: device-pointer ctypes client (register_memory/batch_put/batch_get_auto/batch_exist)
    coordinator.py    # ChunkedTokenDatabase + key geometry (ported)
    data.py           # metadata dataclasses (ported)
    protocol.py       # in-process lookup RPC (ported)
    metrics.py        # Prometheus metrics + KV-events (ported, P3)
    stats.py          # connector stats (ported, P3)
  tests/
    test_gpudirect_regmem.py   # P0 standalone de-risk
    test_dfkv_client.py        # dfkv_client unit tests vs live dfkv_server
    rdma_e2e_validate.py       # connector e2e bytes_match (P1)
```

Reuse from existing `integration/lmcache/src/dfkv_connector/`:
- `native_client.py::load_lib` — declares all ctypes signatures (reused verbatim).
- `key_mapper.py` — key derivation (imported by coordinator.py).
- `exists_cache.py` — batched exists cache (imported by scheduler/worker).

**C-ABI actually used** (`src/dfkv_c_api.h`, confirmed):
- `dfkv_open(members, model_hash, u32×8 geometry) -> handle`
- `dfkv_register_memory(handle, base, size) -> int` ← **GPUDirect registration hook**
- `dfkv_batch_put(handle, keys**, ptrs**, sizes*, n, out_rc*) -> int`
- `dfkv_batch_get_auto(handle, keys**, ptrs**, caps*, n, out_hit*, out_len*) -> int`
- `dfkv_batch_exist(handle, keys**, n, out_exist*) -> int`
- `dfkv_set_batch_concurrency`, `dfkv_stats_snapshot`, `dfkv_close`

---

## Phase P0 — GPUDirect de-risk

**Why first:** the entire GPUDirect approach hinges on one unverified fact — that `dfkv_register_memory` (→ `ibv_reg_mr`) accepts a CUDA device pointer under nvidia-peermem on 0065. Settle it before any connector code.

### Task 1: Standalone GPU-pointer reg+RDMA test

**Files:**
- Create: `integration/vllm/tests/test_gpudirect_regmem.py`

Prereq (operator action, on 0065, once): a `dfkv_server` reachable over RDMA, and `libdfkv.so` built with `-DDFKV_WITH_RDMA=ON`. Reuse dfkv's existing on-node build (`build-rdma/libdfkv.so`, `build-rdma/dfkv_server`).

- [ ] **Step 1: Write the test**

```python
# test_gpudirect_regmem.py — verify dfkv_register_memory + RDMA put/get on a GPU pointer.
# Run inside the vLLM image (torch + CUDA present) against a live dfkv_server.
import ctypes, os, sys, torch
sys.path.insert(0, os.path.expanduser(
    "~/../userdata/.../integration/lmcache/src"))  # for native_client.load_lib
from dfkv_connector.native_client import load_lib

MEMBERS = os.environ["DFKV_MEMBERS"]      # e.g. "c1=127.0.0.1:18800"
MODEL_HASH = 0xABCDEF
lib = load_lib(os.environ["DFKV_LIB"])
h = lib.dfkv_open(MEMBERS.encode(), ctypes.c_uint64(MODEL_HASH),
                  *(ctypes.c_uint32(0) for _ in range(8)))
assert h, "dfkv_open failed"

n = 2 * 1024 * 1024                        # 2 MiB ~ one MLA page
src = torch.arange(n, dtype=torch.uint8, device="cuda")    # GPU buffer with known content
dst = torch.zeros(n, dtype=torch.uint8, device="cuda")
base_src, base_dst = src.data_ptr(), dst.data_ptr()

# 1) Register both GPU regions (the crux: does ibv_reg_mr take a device ptr?)
for base in (base_src, base_dst):
    rc = lib.dfkv_register_memory(h, ctypes.c_void_p(base), ctypes.c_uint64(n))
    assert rc == 0, f"dfkv_register_memory(GPU ptr {base:#x}) FAILED rc={rc} — GPUDirect not available"

# 2) PUT from GPU src, GET into GPU dst, verify bytes
key = b"p0/gpudirect/probe"
assert lib.dfkv_put(h, key, ctypes.c_void_p(base_src), ctypes.c_uint64(n)) == 0
got_len = ctypes.c_uint64(0)
hit = lib.dfkv_get_auto(h, key, ctypes.c_void_p(base_dst), ctypes.c_uint64(n),
                        ctypes.byref(got_len))
assert hit == 1 and got_len.value == n, f"get_auto hit={hit} len={got_len.value}"
torch.cuda.synchronize()
assert torch.equal(src, dst), "byte mismatch after GPUDirect round-trip"
print("P0 PASS: GPUDirect register + RDMA put/get byte-correct on", torch.cuda.get_device_name())
lib.dfkv_close(h)
```

- [ ] **Step 2: Run on 0065 (gate)**

Run (via tsh-go, inside the vLLM container with GPUs):
```
DFKV_LIB=.../build-rdma/libdfkv.so DFKV_MEMBERS=c1=<ip>:18800 \
  python integration/vllm/tests/test_gpudirect_regmem.py
```
Expected: `P0 PASS: GPUDirect register + RDMA put/get byte-correct ...`

- [ ] **Step 3: If it FAILS — branch the decision (do not proceed blindly)**

If `dfkv_register_memory` returns non-zero on the GPU pointer:
1. Confirm peermem: `lsmod | grep -E 'nvidia_peermem|nv_peer_mem'` on the node. If absent, that is the blocker — escalate (operator loads the module) before continuing.
2. If peermem is present but reg still fails, dfkv needs the dmabuf path (`ibv_reg_dmabuf_mr`). That is a dfkv-core change — STOP and re-scope: fall back to the spec's host-bounce path for P1 and open a separate dfkv-core task for dmabuf. Record the failure mode in the spec risk table.

- [ ] **Step 4: Commit**

```bash
git add integration/vllm/tests/test_gpudirect_regmem.py
git commit -m "test(vllm-connector): P0 GPUDirect register_memory + RDMA round-trip probe"
```

---

## Phase P1 — minimal correct connector (synchronous)

Goal of phase: vLLM 0.23.0 + V4-Flash + `DfkvStoreConnector` serves requests, KV bytes round-trip correctly through dfkv, and a cold→warm replay shows an L3 hit speedup. Synchronous I/O (no async threads yet).

### Task 2: Package scaffold

**Files:**
- Create: `integration/vllm/pyproject.toml`, `integration/vllm/src/dfkv_vllm/__init__.py`, `integration/vllm/README.md`

- [ ] **Step 1: pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "dfkv-vllm"
version = "0.1.0"
description = "Direct vLLM KVConnectorBase_V1 connector for dfkv (GPUDirect RDMA)"
requires-python = ">=3.12"
# vllm + torch provided by the runtime image; not pinned here.

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: __init__.py exporting the connector**

```python
from .connector import DfkvStoreConnector
__all__ = ["DfkvStoreConnector"]
```

- [ ] **Step 3: Install editable on 0065 and verify import**

Run: `pip install -e integration/vllm && python -c "import dfkv_vllm; print(dfkv_vllm.DfkvStoreConnector)"`
Expected: prints the class (will fail until Task 5 defines it — acceptable; this step gates after Task 5).

- [ ] **Step 4: Commit**

```bash
git add integration/vllm/pyproject.toml integration/vllm/src/dfkv_vllm/__init__.py integration/vllm/README.md
git commit -m "feat(vllm-connector): package scaffold for dfkv-vllm"
```

### Task 3: `dfkv_client.py` — device-pointer client (TDD)

This is the only genuinely new logic. The lmcache `native_client` is memoryview/host-only; the connector hands raw GPU pointers, so we drive the C-ABI with integer pointers directly.

**Files:**
- Create: `integration/vllm/src/dfkv_vllm/dfkv_client.py`
- Test: `integration/vllm/tests/test_dfkv_client.py`

- [ ] **Step 1: Write the failing test (vs a live dfkv_server, host buffers first)**

```python
# test_dfkv_client.py
import ctypes, os, torch
from dfkv_vllm.dfkv_client import DfkvDeviceClient

def test_roundtrip_gpu_pointers():
    c = DfkvDeviceClient(members=os.environ["DFKV_MEMBERS"], model_hash=0x1234,
                         lib_path=os.environ["DFKV_LIB"])
    n = 1 << 20
    a = torch.arange(n, dtype=torch.uint8, device="cuda")
    b = torch.zeros(n, dtype=torch.uint8, device="cuda")
    c.register_memory(a.data_ptr(), n)
    c.register_memory(b.data_ptr(), n)
    rcs = c.batch_put(["k0"], [a.data_ptr()], [n])
    assert rcs == [0]
    hits, lens = c.batch_get_auto(["k0"], [b.data_ptr()], [n])
    torch.cuda.synchronize()
    assert hits == [1] and lens == [n]
    assert torch.equal(a, b)
    assert c.batch_exist(["k0", "missing"]) == [1, 0]
    c.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest integration/vllm/tests/test_dfkv_client.py -v`
Expected: FAIL `ModuleNotFoundError: dfkv_vllm.dfkv_client`

- [ ] **Step 3: Implement `dfkv_client.py`**

```python
"""Device-pointer dfkv client for the vLLM connector.

Unlike integration/lmcache native_client (memoryview/host buffers), every buffer
here is a raw integer pointer — a slice of the GPU paged KV cache pre-registered
via dfkv_register_memory. We reuse native_client.load_lib() only for its ctypes
signature declarations, then call the C-ABI with c_void_p(int) pointers.
"""
import ctypes
from typing import Optional, Sequence
from dfkv_connector.native_client import load_lib   # reuse signature setup

c_void_p, c_char_p = ctypes.c_void_p, ctypes.c_char_p
c_uint64, c_uint32, c_int = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int


class DfkvDeviceClient:
    def __init__(self, members: str, model_hash: int,
                 lib_path: Optional[str] = None, batch_concurrency: int = 8):
        self._lib = load_lib(lib_path)
        # geometry args (8×u32): mirror native_client — opaque to raw-pointer ops,
        # passed for handle compatibility. See native_client.py dfkv_open call.
        self._h = self._lib.dfkv_open(
            members.encode(), c_uint64(model_hash & 0xFFFFFFFFFFFFFFFF),
            *(c_uint32(0) for _ in range(8)))
        if not self._h:
            raise RuntimeError("dfkv_open failed")
        self._lib.dfkv_set_batch_concurrency(self._h, c_uint64(batch_concurrency))

    def register_memory(self, base: int, size: int) -> None:
        rc = self._lib.dfkv_register_memory(self._h, c_void_p(base), c_uint64(size))
        if rc != 0:
            raise RuntimeError(f"dfkv_register_memory(base={base:#x},size={size}) rc={rc}")

    def batch_put(self, keys: Sequence[str], ptrs: Sequence[int],
                  sizes: Sequence[int]) -> list:
        n = len(keys)
        karr = (c_char_p * n)(*[k.encode() for k in keys])
        parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
        sarr = (c_uint64 * n)(*sizes)
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_put(self._h, karr, parr, sarr, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_put rc={rc}")
        return list(out)

    def batch_get_auto(self, keys: Sequence[str], ptrs: Sequence[int],
                       caps: Sequence[int]):
        n = len(keys)
        karr = (c_char_p * n)(*[k.encode() for k in keys])
        parr = (c_void_p * n)(*[c_void_p(p) for p in ptrs])
        carr = (c_uint64 * n)(*caps)
        out_hit = (c_int * n)()
        out_len = (c_uint64 * n)()
        rc = self._lib.dfkv_batch_get_auto(self._h, karr, parr, carr, n,
                                           out_hit, out_len)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_get_auto rc={rc}")
        return list(out_hit), list(out_len)

    def batch_exist(self, keys: Sequence[str]) -> list:
        n = len(keys)
        karr = (c_char_p * n)(*[k.encode() for k in keys])
        out = (c_int * n)()
        rc = self._lib.dfkv_batch_exist(self._h, karr, n, out)
        if rc != 0:
            raise RuntimeError(f"dfkv_batch_exist rc={rc}")
        return list(out)

    def close(self):
        if getattr(self, "_h", None):
            self._lib.dfkv_close(self._h)
            self._h = None
```

- [ ] **Step 4: Run to verify it passes (on 0065, GPU + live dfkv_server)**

Run: `DFKV_LIB=... DFKV_MEMBERS=... pytest integration/vllm/tests/test_dfkv_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add integration/vllm/src/dfkv_vllm/dfkv_client.py integration/vllm/tests/test_dfkv_client.py
git commit -m "feat(vllm-connector): device-pointer dfkv client (GPUDirect batch put/get/exist)"
```

### Task 4: Port coordinator / data / protocol + key derivation

**Files:**
- Create: `coordinator.py`, `data.py`, `protocol.py` (from TEMPLATE/store/{coordinator,data,protocol}.py)

- [ ] **Step 1: Copy the three template files verbatim into `src/dfkv_vllm/`.**

- [ ] **Step 2: Apply substitutions (exact list):**
  - Class/type renames: `MooncakeStore*` → `DfkvStore*`, `Mooncake*Metadata` → `Dfkv*Metadata`.
  - `data.py`: keep `ChunkedTokenDatabase`, `KeyMetadata`, `LoadSpec`, `ReqMeta`, `RequestTracker` as-is (storage-agnostic).
  - Key generation: where `KeyMetadata` produces the store key string, route through `dfkv_connector.key_mapper` so dfkv and lmcache share one key scheme. Import: `from dfkv_connector.key_mapper import ...`.
  - Remove any `import mooncake` / Mooncake-store references (none expected in these three; verify with `grep -i mooncake`).

- [ ] **Step 3: Verify import + key determinism**

Run: `python -c "from dfkv_vllm import coordinator, data, protocol; print('ok')"`
and a unit assert that the same (model, tp_rank, block_hash) yields a stable key string.
Expected: `ok` + stable key.

- [ ] **Step 4: Commit**

```bash
git add integration/vllm/src/dfkv_vllm/{coordinator,data,protocol}.py
git commit -m "feat(vllm-connector): port token-db/coordinator/metadata, route keys via dfkv key_mapper"
```

### Task 5: connector + scheduler + worker (minimal, synchronous)

**Files:**
- Create: `connector.py` (from TEMPLATE/store/connector.py), `scheduler.py` (TEMPLATE/store/scheduler.py), `worker.py` (TEMPLATE/store/worker.py)

- [ ] **Step 1: Port `connector.py`** — copy TEMPLATE/store/connector.py; rename `MooncakeStoreConnector`→`DfkvStoreConnector`, `MooncakeStore*`→`DfkvStore*`; keep all method bodies (they are pure delegation). Keep `start_load_kv`/`save_kv_layer`/`wait_for_save`/`wait_for_layer_load` as no-ops.

- [ ] **Step 2: Port `scheduler.py`** — copy TEMPLATE/store/scheduler.py; rename classes; `get_num_new_matched_tokens` keeps its logic but its existence query resolves via the worker's lookup (Task 8 wires the real RPC; for P1, scheduler may call a synchronous `DfkvDeviceClient.batch_exist` directly on rank 0's members — note this shortcut in a comment, replaced in P2).

- [ ] **Step 3: Port `worker.py` minimal** — copy TEMPLATE/store/worker.py; then:
  - Replace the `MooncakeDistributedStore` setup block (TEMPLATE worker.py:920-1011) with `self.client = DfkvDeviceClient(members=..., model_hash=..., lib_path=...)` read from `kv_connector_extra_config`.
  - `register_kv_caches` (TEMPLATE worker.py:1119-1190): keep the stride-probe segment logic verbatim; replace `self.store.register_buffer(base_addr, region_len)` with `self.client.register_memory(base_addr, region_len)`.
  - Replace the three store primitives:
    - `self.store.batch_put_from_multi_buffers(...)` → `self.client.batch_put(keys, ptrs, sizes)`
    - `self.store.batch_get_into_multi_buffers(...)` → `self.client.batch_get_auto(keys, ptrs, caps)`
    - `self.store.batch_is_exist(keys)` → `self.client.batch_exist(keys)`
  - **P1 simplification:** do NOT start the async send/recv threads. Instead, in `get_finished`, issue `batch_put`/`batch_get_auto` synchronously and return finished ids immediately. Add a `# P2: move to async threads` marker. Strip kv-events/metrics/disk-offload code paths (delete, not stub) for P1.

- [ ] **Step 4: Verify the connector imports and registers under vLLM**

Run: `python -c "from dfkv_vllm.connector import DfkvStoreConnector; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add integration/vllm/src/dfkv_vllm/{connector,scheduler,worker}.py
git commit -m "feat(vllm-connector): minimal synchronous DfkvStoreConnector (register_memory + sync put/get/exist)"
```

### Task 6: End-to-end bytes_match + cache-hit gate on 0065

**Files:**
- Create: `integration/vllm/tests/rdma_e2e_validate.py` (adapt from dfkv's existing `tests/python/rdma_e2e_validate.py` pattern: zero a buffer, batch_get, assert `bytes_match`).

- [ ] **Step 1: Launch dfkv_server + vLLM on 0065** (operator runbook, on-node `ctr`/`nerdctl`):
```
# dfkv_server: build-rdma/dfkv_server --dir <data> --port 18800 --rdma-port 18801 --rdma-dev <rail> --cap <bytes>
# vLLM:
vllm serve /userdata/dsv4/DeepSeek-V4-Flash \
  -dp <fit-1-node> -tp <fit-1-node> --kv-cache-dtype fp8 --block-size 256 \
  --enable-expert-parallel --max-model-len 262144 --tokenizer-mode deepseek_v4 \
  --kv-transfer-config '{"kv_connector":"DfkvStoreConnector",
    "kv_connector_module_path":"dfkv_vllm.connector","kv_role":"kv_both",
    "kv_connector_extra_config":{"members":"c1=<ip>:18800","group":"g1",
      "lib":"<path>/libdfkv.so","model_hash":"<hash>"}}'
```
(DP/TP chosen to fit one 8-GPU node; V4-Flash, not Pro.)

- [ ] **Step 2: bytes_match probe** — send one request, capture the stored KV via `rdma_e2e_validate.py` reading the keys back into a zeroed host buffer, assert exact match.
Expected: `bytes_match=True`, dfkv access_log shows `batch_put`/`batch_get_auto` ok.

- [ ] **Step 3: Cold→warm cache-hit gate** — run the same 50k-input prompt set twice (same seed). Compare TTFT/throughput run-1 (cold) vs run-2 (warm).
Expected: run-2 TTFT markedly lower than run-1; 0 failed requests; dfkv access_log shows non-zero `batch_get_auto` hits on run-2.

- [ ] **Step 4: Commit**

```bash
git add integration/vllm/tests/rdma_e2e_validate.py
git commit -m "test(vllm-connector): P1 e2e bytes_match + cold/warm cache-hit validation on V4-Flash"
```

---

## Phase P2 — performance structure (async overlap)

Goal: restore Mooncake's async I/O model so transfers overlap compute, and move existence lookup to the proper rank-0 server.

### Task 7: Async send/recv threads + per-request `get_finished`

**Files:**
- Modify: `integration/vllm/src/dfkv_vllm/worker.py`

- [ ] **Step 1:** Port `KVTransferThread` / `KVCacheStoreSendingThread` / `KVCacheStoreRecvingThread` from TEMPLATE/store/worker.py (the `add_request`/queue/`_handle_request` model), substituting the store primitives with `DfkvDeviceClient` calls as in Task 5. Drop disk-offload sub-batching for now (mark `# P3 disk-offload`).
- [ ] **Step 2:** Restore `register_kv_caches` to start both threads (TEMPLATE worker.py:1192-1222).
- [ ] **Step 3:** Restore `get_finished` to enqueue loads/saves and reap finished ids with the CUDA-event save synchronization (TEMPLATE worker.py:1238-1290).
- [ ] **Step 4: Gate** — rerun Task 6 Step 3 warm run; assert correctness unchanged (0 fail, bytes still match on a probe) and TTFT/throughput improved vs P1 synchronous.
- [ ] **Step 5: Commit** `feat(vllm-connector): async send/recv threads + get_finished overlap`.

### Task 8: rank-0 LookupKeyServer + scheduler LookupKeyClient

**Files:**
- Modify: `worker.py`, `scheduler.py`; use ported `protocol.py`

- [ ] **Step 1:** Port `LookupKeyServer` (worker, rank 0) and `LookupKeyClient` (scheduler) from TEMPLATE; the server answers existence via `DfkvDeviceClient.batch_exist` (wrap with `dfkv_connector.exists_cache` to cut hot-path round-trips).
- [ ] **Step 2:** Replace the P1 scheduler shortcut (direct batch_exist) with `LookupKeyClient`.
- [ ] **Step 3: Gate** — multi-rank run (DP>1) on 0065; assert `get_num_new_matched_tokens` returns correct hit counts and warm replay still hits.
- [ ] **Step 4: Commit** `feat(vllm-connector): rank-0 lookup server for scheduler hit decisions`.

### Task 9: Confirm GPUDirect zero-copy + perf band

- [ ] **Step 1:** Verify with `dfkv_stats_snapshot` / dfkv `/metrics` that loads land directly in GPU blocks (no host bounce in the path) and rails are used as expected.
- [ ] **Step 2: Gate** — 50k-input warm throughput in the order of the LMCache reference (~90–120k tok/s band) on the single node; record numbers.
- [ ] **Step 3: Commit** `test(vllm-connector): P2 GPUDirect zero-copy + perf-band record`.

---

## Phase P3 — production hardening

### Task 10: KV-events aggregation

- [ ] **Step 1:** Port `*KVEvents` (TEMPLATE/store/connector.py:54-84) and worker event emission; wire `get_kv_connector_kv_cache_events`/`take_events`/`update_connector_output`.
- [ ] **Step 2: Gate** — with `kv_events_config` enabled, events surface and aggregate across workers; no regression with events disabled.
- [ ] **Step 3: Commit** `feat(vllm-connector): KV-cache event aggregation`.

### Task 11: Prometheus metrics + stats

**Files:** Create `metrics.py`, `stats.py` (from TEMPLATE/store/{metrics,stats}.py + TEMPLATE/mooncake/stats.py)

- [ ] **Step 1:** Port the stats/metrics classes; back per-op counters with `dfkv_stats_snapshot` (already exposed) + connector-level counters (hits/misses/bytes/errors).
- [ ] **Step 2:** Wire `get_kv_connector_stats`/`build_kv_connector_stats`/`build_prom_metrics`.
- [ ] **Step 3: Gate** — vLLM `/metrics` exposes `dfkv_*` connector counters; values move under load.
- [ ] **Step 4: Commit** `feat(vllm-connector): Prometheus metrics + connector stats`.

### Task 12: Error handling / degradation hardening

**Files:** Modify `worker.py`, `scheduler.py`

- [ ] **Step 1:** Implement the spec invariant: read miss/error → `get_block_ids_with_load_errors` populated → vLLM recomputes; write error → count + drop save, never block; dfkv peer-down → treated as miss. Port `get_block_ids_with_load_errors` (TEMPLATE worker.py KVCacheStoreRecvingThread error-block path).
- [ ] **Step 2: Test** — fault injection: point the connector at a dead dfkv member; assert inference continues (0 user-visible failures), errors counted in metrics, no hang (explicitly assert no `wait_complete`-style block).
- [ ] **Step 3: Commit** `feat(vllm-connector): fail-soft degradation (miss-not-crash invariant)`.

### Task 13: Head-to-head bench + final V4-Flash validation

**Files:** Create `integration/vllm/tests/bench_compare.md` (runbook + recorded results)

- [ ] **Step 1:** Same workload (model, input size, concurrency sweep 10/30/60, same seed), three configs: nocache / MooncakeStoreConnector / DfkvStoreConnector. Record throughput + TTFT (cold and warm).
- [ ] **Step 2: Gate (definition of done):** capability parity with MooncakeStoreConnector + V4-Flash 0 failures + dfkv ≈ mooncake (within a stated band) at equal workload, with the warm cache-hit advantage over nocache demonstrated.
- [ ] **Step 3: Commit** `docs(vllm-connector): head-to-head bench results + V4-Flash validation`.

---

## Self-Review

**Spec coverage:** §3 packaging → Task 2,5; §4 components → Tasks 3 (dfkv_client), 4 (coordinator/data/protocol), 5 (connector/scheduler/worker), 10/11 (metrics/stats); §5A registration → Task 5 Step 3; §5B hit decision → Task 5/8; §5C get_finished async → Task 7; §5D error handling → Task 12; §5E MLA → Task 5 (stride probe kept verbatim); §6 phases P0-P3 → Tasks 1 / 2-6 / 7-9 / 10-13; §7 risk 1 (reg_mr) → Task 1; risk 3 (weights) → Task 6 Step 1 prereq. No uncovered spec sections.

**Placeholder scan:** P2/P3 tasks reference TEMPLATE files with explicit line ranges and named substitutions rather than inlining ~3000 lines of ported code — this is the honest unit of work (copy-named-file + named-substitution), not a "TODO". New/critical code (P0 probe, dfkv_client.py) is shown in full. No "TBD"/"add error handling"-style placeholders remain.

**Type consistency:** `DfkvDeviceClient` methods (`register_memory`, `batch_put`, `batch_get_auto`, `batch_exist`, `close`) are defined in Task 3 and called with the same names/signatures in Task 5 and Tasks 7/8/12. C-ABI symbols match `src/dfkv_c_api.h`. Class rename scheme (`Mooncake*`→`Dfkv*`) is uniform across Tasks 4/5/10/11.

## Notes for the executor

- Everything runs on hd04-gpu1-0065 via `tsh-go ssh-exec hd04.enc hd04-gpu1-0065 "..."`; the node has a `nodepool=fault` taint — launch via on-node `ctr`/`nerdctl`, do not rely on normal scheduling. Production (lm-dsv4 / 0052) is strictly read-only.
- Build `libdfkv.so` on the node (`-DDFKV_WITH_RDMA=ON`); the CI artifact needs GLIBC_2.38 and is not portable to the node.
- Upstream-repo discipline: no internal cluster names / IPs / product names in committed code, comments, or docs (parameterize via env/extra_config).
</content>
