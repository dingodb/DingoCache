# dfkv connector for LMCache

An LMCache `RemoteConnector` that stores KV-cache chunks in a **dfkv** cluster.
It talks to dfkv through the C ABI (`libdfkv.so`) via Python **ctypes** — there
is no native CPython extension to compile, so the wheel is pure Python.

Ported from the dingofs LMCache connector with two changes:

1. **Arbitrary block size.** The dingofs connector hard-capped a block at 4 MiB
   (its cache node used fixed io_uring buffers). dfkv has no such cap; this
   connector handles whatever `full_chunk_size_bytes` LMCache computes and reads
   back variable-size ("unfull") chunks at their true stored length via the
   `dfkv_get_auto` / `dfkv_batch_get_auto` C ABI (which the dfkv build adds).
2. **ctypes backend.** Replaces the dingofs pybind11 `_dingofs_native` module
   with direct `libdfkv.so` calls dispatched to a thread-pool executor (ctypes
   releases the GIL during foreign calls, and one `dfkv_open` handle is
   thread-safe to share).

See [docs/CONNECTORS.md](../../docs/CONNECTORS.md) §4 for the design,
implementation, and deployment guide.

## Build & install

```bash
# 1) Build libdfkv.so (RDMA transport) from the top-level dfkv CMake.
make lib                       # -> ../../build-rdma/libdfkv.so
export DFKV_LIB=$(pwd)/../../build-rdma/libdfkv.so

# 2) Install the exact local dfkv-common dependency and connector into the same
#    venv as vLLM + LMCache.
python -m pip install ../common .
# Editable development install:
# python -m pip install -e ../common -e .
```

## Configure LMCache (plugin mode)

```yaml
chunk_size: 16
local_cpu: false
save_chunk_meta: false
remote_storage_plugins: ["dfkv"]
extra_config:
  remote_storage_plugin.dfkv.module_path: dfkv_connector.adapter
  remote_storage_plugin.dfkv.class_name:  DfkvConnectorAdapter
  remote_storage_plugin.dfkv.url:         dfkv://<mds_ip:port,...>/<group>
  remote_storage_plugin.dfkv.membership:  mds            # or "static"
  remote_storage_plugin.dfkv.lib:         /path/to/libdfkv.so
```

- **mds membership** (default): the URL host part is a comma-separated list of
  dfkv MDS `ip:port` endpoints; the ring is discovered for `<group>`.
- **static membership**: the URL host part is a literal member string,
  `dfkv://n1=10.0.0.1:12000,n2=10.0.0.2:12000/unused`.

The library is found via (highest first) `remote_storage_plugin.dfkv.lib` →
env `DFKV_LIB` → `$DFKV_BUILD/libdfkv.so`.

## Configure LMCache (MP-server mode — L2 adapter)

LMCache's multiprocess server (`lmcache server` + `LMCacheMPConnector` on the
vLLM side, the path used by models that split the KV cache into multiple groups
such as **GLM-5.1/5.2 DSA** and **DeepSeek-V4-Flash**) drives its remote tier
through `L2AdapterInterface`, **not** the in-process `remote_storage_plugins`
mechanism above. `dfkv_connector.l2_adapter.DfkvL2Adapter` implements that
interface and is loaded through LMCache's built-in `plugin` L2 adapter:

```bash
# 1) Start the MP server with dfkv as the remote (L2) tier:
lmcache server --port 6555 --max-workers 8 --l1-size-gb 80 \
  --eviction-policy LRU --chunk-size 256 \
  --l2-adapter '{"type":"plugin",
    "module_path":"dfkv_connector.l2_adapter",
    "class_name":"DfkvL2Adapter",
    "config_class_name":"DfkvL2AdapterConfig",
    "adapter_params":{
      "url":"dfkv://<mds_ip:port,...>/<group>",
      "membership":"mds",
      "lib":"/path/to/libdfkv.so",
      "model_name":"<exact-model-or-deployment-identity>"}}'

# 2) Point vLLM at the MP server (NOTE: --no-enable-prefix-caching routes all
#    KV reuse through LMCache):
vllm serve <model> --tensor-parallel-size 8 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"lmcache.mp.port":6555}}'
```

`adapter_params` keys: `url` (required, same grammar as in-process),
`membership` (`mds`|`static`), `lib` (else `DFKV_LIB`), required `model_name`
(MP-server does not supply runtime model metadata), `mds_poll_ms` (3000),
`num_workers` (8), and `max_capacity_gb` (0 = dfkv manages capacity; >0 enables
aggregate L2 eviction). The MP-server API does not expose enough model/PP
metadata to prove byte-identical MLA replication, so object keys always retain
the `world_size/global_rank` identity and the adapter rejects manual folding.
The server's pinned L1 arena is auto-registered for RDMA zero-copy when LMCache
passes an `l1_memory_desc`. Only `dfkv_register_memory` return code `0` counts
as registered; any native MR rejection fails startup instead of silently using
the arena. Validated on GLM-5.2 (vLLM 0.23.0 + LMCache 0.4.7):
store → restart (L1 wiped) → reload from dfkv with prefill skipped.

## Identity and raw-value contract

The in-process path gets the exact `model_name` from LMCache runtime metadata;
the MP-server path gets it from `adapter_params`. The binary namespace always
binds that identity to the source-controlled `lmcache/raw-v1` layout ID.

The in-process `RemoteConnector` preserves the `world_size` and `worker_id`
chosen by LMCache. Replicated-MLA normalization and single-writer selection
belong to LMCache's `RemoteBackend`; dfkv applies no second folding or striping
policy. The MP-server path likewise fails closed by retaining rank identity
because its L2 API does not expose equivalent model/PP semantics.

Object keys use the shared self-delimiting binary form: `DFKVPOOL\x02`,
uint32-LE length-framed pool/hash, fixed uint32-size/int32-rank pairs for
DP/TP/PCP/DCP/PP, uint32 group, and a length-framed component. Fields may
contain NUL, delimiters, and non-UTF-8 bytes. SG users append `DFKVSG\x02` plus
uint32-LE width/group. The namespace is separate binary identity.

All scalar calls pass `(key pointer, exact uint64 length)` and batch calls keep
parallel pointer/length arrays aligned and alive through native return. There
is no C-string truncation, text decode/re-encode, delimiter/Python-hash
identity, or legacy ABI fallback.

dfkv stores exactly the LMCache chunk bytes and returns the stored length
separately; it adds no geometry/dtype envelope. A namespace or object-key
difference is a cold miss. Reusing the same namespace+key for a different
dtype, chunk shape, serialization order, or layout is a type-safety violation:
bump the source-controlled layout ID and deploy every writer/reader together.
Cross-runtime sharing additionally requires byte-compatible object keys and
payloads.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DFKV_LIB` | `$DFKV_BUILD/libdfkv.so` | path to `libdfkv.so` |
| `DFKV_CONNECTOR_GET_PARALLELISM` | 1 | concurrent batched-get groups (executor workers) |
| `DFKV_CONNECTOR_BATCH_MAX_KEYS` | 512 | max keys per native batch call |
| `DFKV_CONNECTOR_ASSUME_EXISTS` | 0 | skip remote contains checks (debug) |
| `DFKV_ACCESS_LOG_ENABLED` | 0 | per-op access log |
| `DFKV_ACCESS_LOG_PATH` | (stderr) | access-log file path |

## Limitations

- Two integration paths, pick by LMCache mode: the in-process
  `remote_storage_plugin` path (`adapter.py`, `RemoteConnector`) for the legacy
  in-process connector, and the **MP-server L2-adapter path** (`l2_adapter.py`,
  `DfkvL2Adapter`) for `LMCacheMPConnector`. The L2 adapter bridges dfkv's
  synchronous ctypes client to LMCache's eventfd model with a background asyncio
  loop (dfkv has no native eventfd, so no pybind/`NativeConnectorL2Adapter`
  path is used).
- **L2 eviction is supported** (dfkv gained a `remove` RPC): set
  `max_capacity_gb > 0` on the L2 adapter to enable LMCache's L2EvictionController,
  which calls `DfkvL2Adapter.delete()` → `dfkv_batch_remove` to drop blocks when
  the configured capacity is exceeded. The in-process connector's `remove_sync`
  is likewise backed by `dfkv_remove`. Requires a `libdfkv.so` / `dfkv_server`
  built with the remove RPC (older libs are detected via `supports_remove()` and
  the delete path degrades to a logged no-op). Default `max_capacity_gb = 0`
  leaves capacity management to dfkv's own per-node LRU.
- No enumeration (`list()` returns `[]`) — dfkv has no listing RPC.
