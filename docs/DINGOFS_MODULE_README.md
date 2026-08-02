# DingoFS KV cache for SGLang HiCache (`src/cache/kvclient`)

Branch `feat/kvcache-sglang`. A KV-by-hash cache path so SGLang's HiCache can use
DingoFS as its L3 external KV store (GLM-5.1 / MLA). Two layers:

1. **Portable semantic core (this dir)** — no brpc/MDS deps, builds on a plain
   Linux box (gcc + cmake, no GPU/RDMA):
   - `kv_types.h` / `key_map.h` — length-frame the explicit namespace and
     canonical object key, SHA-256 them, and retain a 128-bit native `BlockKey`
     for wire/storage identity.
   - Values are opaque raw bytes. Their authoritative stored length is metadata
     returned separately; no model, dtype, page, layer, geometry, or checksum
     envelope is prepended.
   - `con_hash.{h,cc}` — Ketama ring (client-side routing).
   - `store_engine.h` / `disk_slab_store.{h,cc}` — the default single-disk
     extent-slab store: bounded disk + LRU + **cache-only / no S3** (miss =
     clean NotFound); `Cache()` is synchronous and durable-visible.
     `kv_store.{h,cc}` is the explicitly selected file-per-block diagnostic
     backend.
   - `disk_cache_group.{h,cc}` — **multi-NVMe per node** (like dingo-cache
     `--cache_dir=d1,d2,d3`): one selected `StoreEngine` per disk, intra-node
     Ketama routes a block to one disk, and total capacity is split across
     disks. With no option/env the selection is slab; invalid slab geometry
     refuses startup rather than choosing file. `dfkv_server --dir` accepts
     comma-separated paths.
   - `transport.h` + `tcp_transport.{h,cc}` — transport abstraction + a real TCP
     loopback impl used by the standalone harness.
   - `kv_node_server.{h,cc}` + `dfkv_server_main.cc` — a cache-node daemon
     (`dfkv_server`) over the wire protocol.
   - `kv_client.{h,cc}` — routes the canonical identity and passes raw bytes to
     Put/Get/Exist unchanged.
   - `dfkv_c_api.{h,cc}` — ctypes ABI; construct with one immutable
     `dfkv_client_options_v2` via `dfkv_open_v2`.
   - `python/dingofs_hicache.py` — the SGLang `HiCacheStorage` plugin
     (`--hicache-storage-backend dynamic`). MLA: single object/page, no rank
     suffix, `backup_skip` (only tp_rank 0 writes).

2. **Full-build brpc integration (see `INTEGRATION.md`)** — wires the same core
   into the production `dingo-cache` (brpc `BlockCacheService` Exist/SyncCache
   handlers, `--kv_cache_only`, `NullStorageClientPool`, a `DingofsTransport`
   that routes via the existing `RemoteBlockCache`/MDS PeerGroup, and a nanobind
   module). RDMA data-path is intentionally deferred (separate task).

## Build & test the portable core (no GPU)
```
cmake -S /home/ketor/dfkv-dev -B /home/ketor/dfkv-dev/build && cmake --build /home/ketor/dfkv-dev/build
ctest --test-dir /home/ketor/dfkv-dev/build --output-on-failure   # 33 C++ tests
cd test/unit/cache/kvclient/python && python3 -m unittest test_dingofs_hicache  # 6 plugin tests
```

## Deploy for pre-prod FUNCTIONAL validation (no GPU needed to test the path)
- Run one `dfkv_server` per node (`--dir <nvme path> --port <p> --cap <bytes>`); this intentionally omits the engine option and therefore starts slab.
- Launch SGLang with `--hicache-storage-backend dynamic` +
  `--hicache-storage-backend-extra-config '{"backend_name":"dfkv",
  "module_path":"dfkv_hicache","class_name":"DfkvHiCache","interface_v1":1,
  "members":"n1=ip:port,...",
  "key_namespace":"<optional-coordinated-schema-override>"}'` and
  `PYTHONPATH`/`DFKV_LIB` set. SGLang supplies the exact runtime `model_name`;
  omit `key_namespace` for the safe automatic model + `sglang-hicache/raw-v1`
  namespace.
