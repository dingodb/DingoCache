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

# 2) Install the connector into the same venv as vLLM + LMCache.
make install                   # or: make install-dev   (editable)
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
      "model_name":"<deployment-name>"}}'

# 2) Point vLLM at the MP server (NOTE: --no-enable-prefix-caching routes all
#    KV reuse through LMCache):
vllm serve <model> --tensor-parallel-size 8 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"lmcache.mp.port":6555}}'
```

`adapter_params` keys: `url` (required, same grammar as in-process),
`membership` (`mds`|`static`), `lib` (else `DFKV_LIB`), `model_name`
(isolation namespace → stable dfkv `model_hash`), `mds_poll_ms` (3000),
`page_size` (0 = geometry guard off), `num_workers` (8), `max_capacity_gb`
(0 = dfkv manages its own capacity; >0 enables aggregate L2 eviction),
`mla_canonical_keys` (bool, default false): MLA-only opt-in — folds the
kv_rank rank fields so replicated KV shares one key and stores are
deduped with an exists probe (fixes the 8x storage/write inflation;
MLA+PP=1 deployments only — enabling under a sharded or PP model would
collapse distinct content; flipping later = cold cache; or env
`DFKV_L2ADAPTER_MLA_CANONICAL_KEYS=1`). The
server's pinned L1 arena is auto-registered for RDMA zero-copy when LMCache
passes an `l1_memory_desc`. Validated on GLM-5.2 (vLLM 0.23.0 + LMCache 0.4.7):
store → restart (L1 wiped) → reload from dfkv with prefill skipped.

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
